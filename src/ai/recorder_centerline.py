# src/ai/recorder_centerline.py
import os, csv, time, math, argparse
from pathlib import Path
import yaml, cv2, numpy as np, airsim

from src.utils.airsim_cam import grab_rgb_frame

def load_cfg(p): 
    with open(p,"r") as f: return yaml.safe_load(f)

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def connect(host):
    c = airsim.MultirotorClient(ip=host); c.confirmConnection()
    c.enableApiControl(True); c.armDisarm(True); return c

def takeoff(c, alt): c.takeoffAsync().join(); c.moveToZAsync(-alt,1.0).join()

def centerline_control(img, cfg_v):
    """
    Return (lat_err_px, heading_err_rad, overlay) using simple lane/aisle detection.
    """
    h, w = img.shape[:2]
    y0 = int(h*cfg_v["roi_ymin_frac"])
    roi = img[y0:h, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg_v["canny_low"], cfg_v["canny_high"])
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=cfg_v["hough_thresh"], minLineLength=25, maxLineGap=15)

    overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cx = w//2
    lat_err = 0.0
    heading_err = 0.0

    if lines is not None and len(lines)>0:
        # separate left/right lines by slope sign relative to image coords
        left, right = [], []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if x2==x1: slope = 1e9
            else: slope = (y2-y1)/(x2-x1+1e-6)
            # only near-vertical-ish lines (ignore horizontals)
            if abs(slope) < 0.3: 
                continue
            if slope < 0: left.append((x1,y1,x2,y2))
            else: right.append((x1,y1,x2,y2))
            cv2.line(overlay, (x1,y1), (x2,y2), (0,255,0), 2)

        # estimate lane/aisle center by averaging bottom x of left/right
        bottoms = []
        for segs in (left, right):
            for (x1,y1,x2,y2) in segs:
                if y1>y2:
                    xb = x1 + (h-y0-y1) * (x2-x1) / (y2-y1+1e-6)
                else:
                    xb = x2 + (h-y0-y2) * (x1-x2) / (y1-y2+1e-6)
                bottoms.append(xb)
        if len(bottoms)>=2:
            lane_cx = int(np.mean(bottoms))
            lat_err = (lane_cx - cx)  # +ve means center is to the right
            cv2.circle(overlay, (lane_cx, (h-y0)-5), 5, (0,0,255), -1)

        # heading from best-fit line over all segments
        pts=[]
        for segs in (left,right):
            for (x1,y1,x2,y2) in segs:
                pts.append([x1,y1]); pts.append([x2,y2])
        if len(pts)>=4:
            pts = np.array(pts, dtype=np.float32)
            [vx,vy,x0,y0fit] = cv2.fitLine(pts, cv2.DIST_L2,0,0.01,0.01)
            angle = math.atan2(vy[0], vx[0])
            # relative to vertical (downwards); convert to yaw error
            heading_err = float(angle - (-np.pi/2.0))
    # scale lat_err into [-1,1] approx by image half-width
    lat_err_norm = float(np.clip(lat_err / (w*0.5), -1.0, 1.0))
    return lat_err_norm, heading_err, overlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/record_centerline.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    host = cfg["sim"]["host"]
    rate_hz = int(cfg["sim"]["control_rate_hz"])
    dt = 1.0 / rate_hz
    alt = float(cfg["uav"]["takeoff_alt_m"])
    out_root = cfg["record"]["out_dir"]
    cam = cfg["record"]["camera_name"]
    save_images = bool(cfg["record"]["save_images"])

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(out_root, f"run_{stamp}"))
    frames_dir = ensure_dir(os.path.join(run_dir, "frames"))
    labels_csv = os.path.join(run_dir, "labels.csv")

    vx_fwd = float(cfg["control"]["vx_fwd"])
    k_lat  = float(cfg["control"]["k_lat"])
    k_yaw  = float(cfg["control"]["k_yaw"])
    vy_max = float(cfg["control"]["vy_max"])
    rz_max = float(cfg["control"]["rz_max"])
    vis = cfg["vision"]

    print(f"[INFO] Connect {host}...")
    client = connect(host)
    print("[OK] Connected.")
    print(f"[INFO] Takeoff to {alt} m...")
    takeoff(client, alt)
    print("[OK] Takeoff complete.")

    print(f"[INFO] Centerline recording -> {run_dir}")
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t","img_path","alt_m","vx","vy","vz","r_z_rad","lat_err","heading_err"])
        writer.writeheader()

        t0 = time.time()
        idx = 0
        try:
            while time.time() - t0 < cfg["record"]["duration_s"]:
                img = grab_rgb_frame(client, camera_name=cam)
                if img is None:
                    break
                lat_err, heading_err, _ = centerline_control(img, vis)

                # convert errors to commands
                vx = vx_fwd
                vy = float(np.clip(-k_lat * lat_err, -vy_max, vy_max))
                rz = float(np.clip(-k_yaw * heading_err, -rz_max, rz_max))
                vz = 0.0

                client.moveByVelocityBodyFrameAsync(vx, vy, vz, dt, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=rz))

                # downsample/resize for dataset if needed
                w = int(cfg["record"]["img_size"]["width"])
                h = int(cfg["record"]["img_size"]["height"])
                img_ds = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)

                img_path = ""
                if save_images:
                    img_path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
                    cv2.imwrite(img_path, img_ds)

                st = client.getMultirotorState()
                alt_now = max(0.0, -st.kinematics_estimated.position.z_val)

                writer.writerow({
                    "t": round(time.time()-t0,3),
                    "img_path": os.path.relpath(img_path, run_dir) if img_path else "",
                    "alt_m": round(float(alt_now),3),
                    "vx": vx, "vy": vy, "vz": vz, "r_z_rad": rz,
                    "lat_err": round(lat_err,3), "heading_err": round(heading_err,3)
                })

                idx += 1
                time.sleep(dt)
        except KeyboardInterrupt:
            print("\n[INFO] User stop.")
        finally:
            print("[INFO] Landing...")
            client.moveByVelocityBodyFrameAsync(0,0,0,1).join()
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)

    print(f"[OK] Saved {idx} frames + labels to {run_dir}")

if __name__ == "__main__":
    main()