# src/ai/recorder.py
import os
import csv
import time
import math
import argparse
from pathlib import Path

import yaml
import airsim
import numpy as np
import cv2

from src.utils.airsim_cam import grab_rgb_frame, save_frame_bgr


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def connect(host: str):
    client = airsim.MultirotorClient(ip=host)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def takeoff(client, alt_m: float):
    client.takeoffAsync().join()
    client.moveToZAsync(z=-alt_m, velocity=1.0).join()


class ScriptedExpert:
    """
    Simple time-based expert: cycles through segments that specify desired
    body-frame (vx, vy, vz) [m/s] and yaw-rate r_z [deg/s -> rad/s].
    """
    def __init__(self, segments):
        self.plan = []
        for seg in segments:
            self.plan.append({
                "name": seg.get("name", ""),
                "seconds": float(seg["seconds"]),
                "vx": float(seg["vx"]),
                "vy": float(seg["vy"]),
                "vz": float(seg["vz"]),
                "r_z_rad": math.radians(float(seg.get("r_z_deg", 0.0)))
            })
        self.total_dur = sum(s["seconds"] for s in self.plan)

    def command_at(self, t: float):
        """
        Returns (vx, vy, vz, r_z_rad, seg_name) for elapsed time t.
        If t exceeds total plan, return zeros.
        """
        acc = 0.0
        for seg in self.plan:
            if t < acc + seg["seconds"]:
                return seg["vx"], seg["vy"], seg["vz"], seg["r_z_rad"], seg["name"]
            acc += seg["seconds"]
        return 0.0, 0.0, 0.0, 0.0, "done"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/record.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    host = cfg["sim"]["host"]
    rate_hz = int(cfg["sim"]["control_rate_hz"])
    dt = 1.0 / rate_hz

    out_root = cfg["record"]["out_dir"]
    cam_name = cfg["record"]["camera_name"]
    alt_m = float(cfg["uav"]["takeoff_alt_m"])
    duration_s = float(cfg["record"]["duration_s"])
    segments = cfg["script"]["segments"]

    # Prepare output dirs
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(out_root, f"run_{stamp}"))
    frames_dir = ensure_dir(os.path.join(run_dir, "frames"))
    labels_csv = os.path.join(run_dir, "labels.csv")

    # Connect and takeoff
    print(f"[INFO] Connecting to AirSim at {host} ...")
    client = connect(host)
    print("[OK] Connected, armed, API control enabled.")

    print(f"[INFO] Taking off to {alt_m} m ...")
    takeoff(client, alt_m)
    print("[OK] Takeoff complete.")

    # Build the scripted expert
    expert = ScriptedExpert(segments)
    print(f"[INFO] Recording for ~{duration_s} s at {rate_hz} Hz into {run_dir}")

    # Open CSV
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t", "seg_name",
                "img_path", "alt_m",
                "vx", "vy", "vz", "r_z_rad"
            ],
        )
        writer.writeheader()

        t0 = time.time()
        idx = 0
        try:
            while True:
                t = time.time() - t0
                if t > duration_s:
                    break

                vx, vy, vz, r_z, seg_name = expert.command_at(t)

                # Send body-frame velocity command for dt
                client.moveByVelocityBodyFrameAsync(
                    vx, vy, vz, dt,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=r_z)
                )

                # Grab image
                img = grab_rgb_frame(client, camera_name=cam_name)
                if img is None:
                    # If no image, still log label (rare)
                    img_path = ""
                else:
                    img_path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
                    cv2.imwrite(img_path, img)

                # Altitude from state (NED -> -z)
                st = client.getMultirotorState()
                alt_now = -st.kinematics_estimated.position.z_val

                # Log row
                writer.writerow({
                    "t": round(t, 3),
                    "seg_name": seg_name,
                    "img_path": os.path.relpath(img_path, run_dir) if img_path else "",
                    "alt_m": round(float(alt_now), 3),
                    "vx": vx, "vy": vy, "vz": vz, "r_z_rad": r_z
                })

                idx += 1
                time.sleep(dt)

        except KeyboardInterrupt:
            print("\n[INFO] Recording interrupted by user.")
        finally:
            client.moveByVelocityBodyFrameAsync(0, 0, 0, 1).join()
            client.armDisarm(False)
            client.enableApiControl(False)

    print(f"[OK] Saved {idx} frames and labels to: {run_dir}")
    print(f"[OK] Labels CSV: {labels_csv}")


if __name__ == "__main__":
    main()
