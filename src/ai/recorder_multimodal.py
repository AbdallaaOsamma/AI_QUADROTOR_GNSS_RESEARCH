# src/ai/recorder_multimodal.py
"""
Enhanced Data Recorder with Multi-Modal Perception

Uses the new multi-modal perception system that automatically falls back
from edge detection to depth-based navigation when lines aren't visible.
"""
import os
import csv
import time
import argparse
from pathlib import Path

import yaml
import cv2
import numpy as np
import airsim

from src.utils.airsim_cam import grab_rgb_frame
from src.ai.perception import MultiModalPerception, grab_all_frames


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def connect(host):
    client = airsim.MultirotorClient(ip=host)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def takeoff(client, alt):
    client.takeoffAsync().join()
    client.moveToZAsync(-alt, 1.0).join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/record_centerline.yaml")
    parser.add_argument("--debug", action="store_true", help="Show debug windows")
    parser.add_argument("--duration", type=float, help="Override recording duration in seconds")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    host = cfg["sim"]["host"]
    rate_hz = int(cfg["sim"]["control_rate_hz"])
    dt = 1.0 / rate_hz
    alt = float(cfg["uav"]["takeoff_alt_m"])
    out_root = cfg["record"]["out_dir"]
    cam = cfg["record"]["camera_name"]
    save_images = bool(cfg["record"]["save_images"])
    save_debug = cfg["record"].get("save_debug", False)
    
    # Duration priority: CLI arg > Config file
    if args.duration:
        duration_s = args.duration
    else:
        duration_s = float(cfg["record"]["duration_s"])

    # Control gains
    vx_fwd = float(cfg["control"]["vx_fwd"])
    k_lat = float(cfg["control"]["k_lat"])
    k_yaw = float(cfg["control"]["k_yaw"])
    vy_max = float(cfg["control"]["vy_max"])
    rz_max = float(cfg["control"]["rz_max"])

    # Initialize multi-modal perception
    perception_cfg = {
        "canny_low": cfg["vision"].get("canny_low", 40),
        "canny_high": cfg["vision"].get("canny_high", 120),
        "hough_thresh": cfg["vision"].get("hough_thresh", 20),
        "roi_ymin_frac": cfg["vision"].get("roi_ymin_frac", 0.5),
        "min_lines_threshold": cfg["vision"].get("min_lines", 4),
    }
    perception = MultiModalPerception(perception_cfg)

    # Output directories
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(out_root, f"run_{stamp}"))
    frames_dir = ensure_dir(os.path.join(run_dir, "frames"))
    debug_dir = ensure_dir(os.path.join(run_dir, "debug")) if save_debug else None
    labels_csv = os.path.join(run_dir, "labels.csv")

    print(f"[INFO] Connecting to AirSim at {host}...")
    client = connect(host)
    print("[OK] Connected.")

    print(f"[INFO] Taking off to {alt} m...")
    takeoff(client, alt)
    print("[OK] Takeoff complete.")

    print(f"[INFO] Recording with multi-modal perception -> {run_dir}")
    print(f"[INFO] Camera: {cam}, Duration: {duration_s}s")

    # Perception mode stats
    mode_counts = {}

    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "t", "img_path", "alt_m",
            "vx", "vy", "vz", "r_z_rad",
            "lat_err", "heading_err",
            "confidence", "mode", "lines", "obs_dist"
        ])
        writer.writeheader()

        t0 = time.time()
        idx = 0

        try:
            while time.time() - t0 < duration_s:
                # Grab all image types
                images = grab_all_frames(client, cam)
                rgb = images.get("rgb")
                depth = images.get("depth")

                if rgb is None:
                    print("[WARN] No RGB image, waiting...")
                    time.sleep(0.5)
                    continue

                # Run multi-modal perception
                result = perception.process(rgb, depth_image=depth)

                # Track modes used
                mode_counts[result.mode_used] = mode_counts.get(result.mode_used, 0) + 1

                # Convert perception to control commands
                vx = vx_fwd

                # Scale commands by confidence
                confidence_scale = max(0.3, result.confidence)

                vy = float(np.clip(-k_lat * result.lateral_error * confidence_scale, -vy_max, vy_max))
                rz = float(np.clip(-k_yaw * result.heading_error * confidence_scale, -rz_max, rz_max))

                # Emergency slowdown if obstacle too close
                # Emergency slowdown if obstacle too close
                if result.obstacle_distance < 3.0:
                    vx = max(0.1, vx * (result.obstacle_distance / 3.0))

                # Use moveByVelocityZBodyFrameAsync to HOLD ALTITUDE
                # Note: 'z' is negative of altitude (NED coordinate system)
                vz = 0.0  # Placeholder for logging
                client.moveByVelocityZBodyFrameAsync(
                    vx, vy, -alt, dt,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=rz)
                )

                # Resize for dataset
                w = int(cfg["record"]["img_size"]["width"])
                h = int(cfg["record"]["img_size"]["height"])
                img_ds = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

                # Save image
                img_path = ""
                if save_images:
                    img_path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
                    cv2.imwrite(img_path, img_ds)

                # Save debug overlay
                if save_debug and result.debug_overlay is not None:
                    debug_path = os.path.join(debug_dir, f"debug_{idx:05d}.png")
                    cv2.imwrite(debug_path, result.debug_overlay)

                # Get altitude
                st = client.getMultirotorState()
                alt_now = max(0.0, -st.kinematics_estimated.position.z_val)

                # Log row
                writer.writerow({
                    "t": round(time.time() - t0, 3),
                    "img_path": os.path.relpath(img_path, run_dir) if img_path else "",
                    "alt_m": round(float(alt_now), 3),
                    "vx": round(vx, 3),
                    "vy": round(vy, 3),
                    "vz": round(vz, 3),
                    "r_z_rad": round(rz, 4),
                    "lat_err": round(result.lateral_error, 3),
                    "heading_err": round(result.heading_error, 3),
                    "confidence": round(result.confidence, 2),
                    "mode": result.mode_used,
                    "lines": result.lines_detected,
                    "obs_dist": round(result.obstacle_distance, 2)
                })

                # Debug visualization
                if args.debug and result.debug_overlay is not None:
                    cv2.imshow("Perception", result.debug_overlay)
                    cv2.imshow("Camera", rgb)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Print status periodically
                if idx % 15 == 0:
                    print(f"[{idx:04d}] mode={result.mode_used:12s} conf={result.confidence:.2f} "
                          f"lat={result.lateral_error:+.2f} lines={result.lines_detected} "
                          f"vy={vy:+.2f} rz={rz:+.2f}")

                idx += 1
                time.sleep(dt)

        except KeyboardInterrupt:
            print("\n[INFO] User interrupted.")
        finally:
            print("[INFO] Landing...")
            client.moveByVelocityBodyFrameAsync(0, 0, 0, 1).join()
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
            if args.debug:
                cv2.destroyAllWindows()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Recording Complete!")
    print(f"{'='*60}")
    print(f"Frames saved: {idx}")
    print(f"Output directory: {run_dir}")
    print(f"\nPerception Mode Usage:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = (count / idx * 100) if idx > 0 else 0
        print(f"  {mode:15s}: {count:5d} ({pct:5.1f}%)")

    # Recommendations based on mode usage
    edge_modes = sum(c for m, c in mode_counts.items() if "edge" in m)
    if edge_modes < idx * 0.5:
        print("\n[WARNING] Edge detection worked <50% of time")
        print("Recommendations:")
        print("  - Environment may lack visible road edges")
        print("  - Consider training on depth-based features")
        print("  - Check camera angle and lighting")


if __name__ == "__main__":
    main()
