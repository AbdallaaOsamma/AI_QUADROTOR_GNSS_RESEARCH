# src/utils/airsim_cam.py
import airsim, numpy as np, cv2, os

def grab_rgb_frame(client, camera_name="0", image_type=airsim.ImageType.Scene):
    resp = client.simGetImage(camera_name, image_type)
    if resp is None:
        return None
    img_bytes = np.frombuffer(resp, dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)  # BGR
    return img

def save_frame_bgr(img_bgr, out_dir, idx):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"frame_{idx:05d}.png")
    cv2.imwrite(path, img_bgr)
    return path
