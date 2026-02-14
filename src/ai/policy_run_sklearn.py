# src/ai/policy_run_sklearn.py
import time, argparse, joblib
import numpy as np
import cv2, yaml, airsim

from src.utils.airsim_cam import grab_rgb_frame
from src.control.pid import PID, PIDGains

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def connect(host: str):
    client = airsim.MultirotorClient(ip=host)
    client.confirmConnection()
    client.simPause(False)
    client.enableApiControl(True)
    client.armDisarm(True)
    return client

def safe_takeoff(client, target_alt_m: float, settle_tol_m: float = 0.15, timeout_s: float = 12.0):
    """
    Non-blocking takeoff: fire-and-poll (no join timeouts).
    Skips if already near target altitude.
    """
    st = client.getMultirotorState()
    alt_now = max(0.0, -st.kinematics_estimated.position.z_val)  # print positive meters
    if st.landed_state == 1 and abs(alt_now - target_alt_m) < settle_tol_m:
        print(f"[INFO] Already flying near {target_alt_m:.2f} m (alt={alt_now:.2f}). Skipping takeoff.")
        return

    print("[INFO] Takeoff sequence (non-blocking)...")
    try:
        client.takeoffAsync()  # fire-and-forget
    except Exception as e:
        print(f"[WARN] takeoffAsync error: {e}")

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = client.getMultirotorState()
        if st.landed_state == 1:
            break
        time.sleep(0.1)

    try:
        client.moveToZAsync(z=-target_alt_m, velocity=1.0)  # fire-and-forget
    except Exception as e:
        print(f"[WARN] moveToZAsync error: {e}")

    t1 = time.time()
    while time.time() - t1 < timeout_s:
        st = client.getMultirotorState()
        alt_now = max(0.0, -st.kinematics_estimated.position.z_val)
        if abs(alt_now - target_alt_m) <= settle_tol_m:
            print(f"[OK] Takeoff complete at alt={alt_now:.2f} m.")
            return
        time.sleep(0.1)
    print(f"[WARN] Altitude not within tolerance after {timeout_s}s (alt={alt_now:.2f}). Continuing.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to sklearn .joblib")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--rate_hz", type=int, default=10)
    ap.add_argument("--duration_s", type=float, default=20)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    client = connect(cfg["sim"]["host"])
    print("[OK] Connected, armed.")

    target_alt = cfg["uav"]["takeoff_alt_m"]
    safe_takeoff(client, target_alt)

    pkg = joblib.load(args.model)
    pipeline = pkg["pipeline"]
    img_w, img_h = pkg["img_size"]
    grayscale = pkg["grayscale"]
    clamp_cfg = pkg["clamp"]
    print(f"[INFO] Loaded model {args.model} (img={img_w}x{img_h}, gray={grayscale})")

    # Runtime clamps (keep motions gentle)
    rt_clamp = {
        "vx": (-0.8, 0.8),
        "vy": (-0.4, 0.4),
        "vz": (clamp_cfg["vz"][0], clamp_cfg["vz"][1]),  # body z command clamp
        "rz": (-0.4, 0.4),
    }

    pid_z = PID(PIDGains(**cfg["pid"]["vz"]))
    dt = max(1.0 / max(1, args.rate_hz), 0.02)

    t0 = time.time()
    tick = 0
    try:
        while time.time() - t0 < args.duration_s:
            # grab image
            img = grab_rgb_frame(client, camera_name="0")
            if img is None:
                vx_m, vy_m, vz_m, rz_m = 0.0, 0.0, 0.0, 0.0
            else:
                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
                feat = (img.reshape(1, -1).astype(np.float32) / 255.0)
                vx_m, vy_m, vz_m, rz_m = pipeline.predict(feat)[0]

            # altitude hold — AirSim body z is DOWN => command = -vz_pid
            st = client.getMultirotorState()
            alt_now = max(0.0, -st.kinematics_estimated.position.z_val)
            err_alt = target_alt - alt_now
            vz_pid = pid_z.update(err_alt, dt)
            vz_cmd_body = -vz_pid  # <-- key fix (invert sign for NED/body z)

            # collision brake
            coll = client.simGetCollisionInfo()
            if coll.has_collided:
                vx_cmd = vy_cmd = vz_cmd = rz_cmd = 0.0
            else:
                vx_cmd = float(np.clip(vx_m, rt_clamp["vx"][0], rt_clamp["vx"][1]))
                vy_cmd = float(np.clip(vy_m, rt_clamp["vy"][0], rt_clamp["vy"][1]))
                vz_cmd = float(np.clip(vz_cmd_body, rt_clamp["vz"][0], rt_clamp["vz"][1]))
                rz_cmd = float(np.clip(rz_m, rt_clamp["rz"][0], rt_clamp["rz"][1]))

            client.moveByVelocityBodyFrameAsync(
                vx_cmd, vy_cmd, vz_cmd, dt,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=rz_cmd)
            )

            tick += 1
            if tick % 10 == 0:
                print(f"[{tick:04d}] alt={alt_now:.2f} vx={vx_cmd:+.2f} vy={vy_cmd:+.2f} vz={vz_cmd:+.2f} rz={rz_cmd:+.2f} collided={coll.has_collided}")
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping policy run.")
    finally:
        try:
            client.moveByVelocityBodyFrameAsync(0, 0, 0, 1).join()
        except Exception:
            pass
        client.armDisarm(False)
        client.enableApiControl(False)
        print("[OK] Disarmed, API control released.")

if __name__ == "__main__":
    main()
