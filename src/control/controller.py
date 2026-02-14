# src/control/controller.py
import time
import argparse
import yaml
import os
import airsim
import cv2  # make sure opencv-python is installed

from src.utils.logging import EpisodeLogger
from src.utils.airsim_cam import grab_rgb_frame, save_frame_bgr
from src.control.pid import PID, PIDGains


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
    # AirSim uses NED coordinates: up is negative Z
    client.moveToZAsync(z=-alt_m, velocity=1.0).join()


def hover_loop(client, cfg):
    """
    Keeps altitude at target_alt using a vertical PID, logs telemetry to CSV.
    """
    rate_hz = cfg["sim"]["control_rate_hz"]
    dt = 1.0 / rate_hz

    # PID for vertical (altitude) hold
    gz = cfg["pid"]["vz"]
    pid_z = PID(PIDGains(**gz))

    logger = EpisodeLogger(cfg["logging"]["out_dir"])
    print_every = cfg["logging"]["print_every"]

    target_alt = cfg["uav"]["takeoff_alt_m"]

    print(f"[INFO] Hover loop at {rate_hz} Hz. Press Ctrl+C to stop.")
    t0 = time.time()
    tick = 0
    try:
        while True:
            st = client.getMultirotorState()
            pos = st.kinematics_estimated.position
            alt_now = -pos.z_val  # meters
            err_alt = target_alt - alt_now

            vz_cmd = pid_z.update(err_alt, dt)

            # Hold x/y position and yaw rate steady for this test
            client.moveByVelocityBodyFrameAsync(
                0.0, 0.0, -vz_cmd, dt,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0)
            )

            tick += 1
            logger.log({
                "t": time.time() - t0,
                "alt_target_m": target_alt,
                "alt_m": alt_now,
                "vz_cmd": vz_cmd,
                "vx_cmd": 0.0,
                "vy_cmd": 0.0,
                "rz_cmd": 0.0
            })

            if tick % print_every == 0:
                print(f"[{tick:05d}] alt={alt_now:.2f} m  err={err_alt:+.2f}  vz_cmd={vz_cmd:+.2f}")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping hover and disarming...")
    finally:
        logger.close()
        client.moveByVelocityBodyFrameAsync(0, 0, 0, 1).join()
        client.armDisarm(False)
        client.enableApiControl(False)


def hover_capture(client, cfg, seconds=5.0):
    """
    Same altitude hold as hover_loop, but also saves camera frames to:
      data/expert/hover_capture/frame_00000.png, ...
    """
    rate_hz = cfg["sim"]["control_rate_hz"]
    dt = 1.0 / rate_hz

    gz = cfg["pid"]["vz"]
    pid_z = PID(PIDGains(**gz))
    target_alt = cfg["uav"]["takeoff_alt_m"]

    img_dir = os.path.join(cfg["logging"]["out_dir"], "hover_capture")
    os.makedirs(img_dir, exist_ok=True)

    print(f"[INFO] Capturing ~{seconds} s of frames while hovering...")
    t0 = time.time()
    idx = 0
    try:
        while time.time() - t0 < seconds:
            st = client.getMultirotorState()
            pos = st.kinematics_estimated.position
            alt_now = -pos.z_val
            err_alt = target_alt - alt_now
            vz_cmd = pid_z.update(err_alt, dt)

            client.moveByVelocityBodyFrameAsync(
                0.0, 0.0, -vz_cmd, dt,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0)
            )

            img = grab_rgb_frame(client, camera_name="0")
            if img is not None:
                save_frame_bgr(img, img_dir, idx)
                idx += 1

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        client.moveByVelocityBodyFrameAsync(0, 0, 0, 1).join()
        print(f"[OK] Saved {idx} frames to {img_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", default="hover_test", choices=["hover_test", "hover_capture"])
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    host = cfg["sim"]["host"]
    alt = cfg["uav"]["takeoff_alt_m"]

    print(f"[INFO] Connecting to AirSim at {host} ...")
    client = connect(host)
    print("[OK] Connected, armed, API control enabled.")

    print(f"[INFO] Taking off to {alt} m ...")
    takeoff(client, alt)
    print("[OK] Takeoff complete.")

    if args.mode == "hover_test":
        hover_loop(client, cfg)
    elif args.mode == "hover_capture":
        hover_capture(client, cfg, seconds=5.0)


if __name__ == "__main__":
    main()
