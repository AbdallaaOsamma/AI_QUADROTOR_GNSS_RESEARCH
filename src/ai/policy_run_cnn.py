import argparse
import time
import yaml
import numpy as np
import cv2
import torch
import airsim

from src.ai.model_cnn import NavigationCNN

def detect_road_center(img_gray):
    """
    Detect road center from grayscale image.
    Returns horizontal offset from center (-1 = far left, 0 = center, +1 = far right)
    Uses thresholding to find road surface, then calculates center of mass.
    """
    h, w = img_gray.shape
    
    # The road is typically darker than surroundings - threshold to find it
    # Adjust threshold based on your lighting conditions
    _, road_mask = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Focus on lower half of image (where road is in front of drone)
    road_mask[0:h//3, :] = 0
    
    # Find center of mass of road pixels
    moments = cv2.moments(road_mask)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        # Normalize to [-1, 1] where 0 = image center
        offset = 2.0 * (cx - w/2) / w
        return np.clip(offset, -1, 1)
    return 0.0

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def safe_takeoff(client, target_alt, timeout=15.0):
    print("[INFO] Takeoff sequence...")
    client.takeoffAsync().join()
    
    # Actively go to target altitude
    print(f"[INFO] Moving to altitude {target_alt}m...")
    client.moveToZAsync(-target_alt, 1.0).join()
    
    # Hover briefly to stabilize
    client.hoverAsync().join()
    time.sleep(1.0)
    
    # verify
    state = client.getMultirotorState()
    alt = -state.kinematics_estimated.position.z_val
    print(f"[INFO] Takeoff complete. Current altitude: {alt:.2f}m")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--model', help='Path to CNN model .pth file')
    parser.add_argument('--rate_hz', type=int, default=10)
    parser.add_argument('--duration_s', type=float, default=30)
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)
    
    # Determine model path
    model_path = args.model
    if not model_path and 'model' in cfg and 'path' in cfg['model']:
        model_path = cfg['model']['path']
        
    if not model_path:
        raise ValueError("Model path must be provided via --model arg or config file 'model.path'")
    
    # Connect
    client = airsim.MultirotorClient(ip=cfg['sim']['host'])
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[OK] Connected, armed.")
    
    # Takeoff
    target_alt = cfg['uav']['takeoff_alt_m']
    safe_takeoff(client, target_alt)
    
    # Load CNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    img_size = checkpoint['img_size']
    grayscale = checkpoint.get('grayscale', True)
    
    model = NavigationCNN(img_height=img_size[0], img_width=img_size[1], num_outputs=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"[INFO] Loaded model {model_path} (img={img_size[0]}x{img_size[1]}, gray={grayscale})")
    
    # PID for altitude
    vz_pid_kp = cfg['pid']['vz']['kp']
    vz_pid_ki = cfg['pid']['vz']['ki']
    vz_pid_kd = cfg['pid']['vz']['kd']
    vz_int_err = 0.0
    vz_prev_err = 0.0
    
    # Runtime clamps - RELAXED to allow proper street navigation
    vx_clamp = [-2.0, 2.0]  # Forward/backward
    vy_clamp = [-1.5, 1.5]  # Left/right (CRITICAL for staying on street)
    vz_clamp = [-1.0, 1.0]  # Up/down
    rz_clamp = [-0.8, 0.8]  # Yaw rate
    
    # Steering correction gains for road center tracking
    road_center_kp = 0.3  # Proportional gain for steering correction
    road_center_max_correction = 0.5  # Max correction to apply to vy
    
    dt = 1.0 / args.rate_hz
    t_end = time.time() + args.duration_s
    step = 0
    
    try:
        with torch.no_grad():
            while time.time() < t_end:
                step += 1
                
                # Get camera image
                cam_name = cfg['model'].get('camera_name', '0')
                responses = client.simGetImages([
                    airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False)
                ])
                
                if not responses or not responses[0].height:
                    time.sleep(dt)
                    continue
                
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3)
                
                # Preprocess
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, tuple(img_size), interpolation=cv2.INTER_AREA)
                img_norm = img_resized.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
                
                # Predict
                outputs = model(img_tensor)
                pred = outputs.cpu().numpy()[0]
                
                vx_raw, vy_raw, vz_raw, rz_raw = pred
                
                # Use AI predictions with proper clamping
                vx = np.clip(vx_raw, vx_clamp[0], vx_clamp[1])
                vy = np.clip(vy_raw, vy_clamp[0], vy_clamp[1])
                rz = np.clip(rz_raw, rz_clamp[0], rz_clamp[1])
                
                # Apply road center correction to keep drone on street
                road_offset = detect_road_center(img_gray)
                # If drone is off-center, steer back towards road
                steering_correction = -road_offset * road_center_kp
                steering_correction = np.clip(steering_correction, -road_center_max_correction, road_center_max_correction)
                vy_corrected = np.clip(vy + steering_correction, vy_clamp[0], vy_clamp[1])
                
                # DEBUG: Log raw model outputs and corrections
                if step % 10 == 0:
                    print(f"[{step:04d}] RAW_MODEL: vx={vx_raw:+.3f} vy={vy_raw:+.3f} vz={vz_raw:+.3f} rz={rz_raw:+.3f}")
                
                # Use moveByVelocityZBodyFrameAsync to HOLD ALTITUDE
                # Note: 'z' is negative of altitude (NED coordinate system)
                # We ignore the manual PID loop and trust AirSim's internal controller for Z
                
                # Check collision
                collision_info = client.simGetCollisionInfo()
                if collision_info.has_collided:
                    print(f"[{step:04d}] COLLISION DETECTED - stopping")
                    vx, vy, vz, rz = 0, 0, 0, 0
                
                # Send command
                client.moveByVelocityZBodyFrameAsync(
                    float(vx), float(vy_corrected), float(-target_alt), dt,
                    airsim.YawMode(is_rate=True, yaw_or_rate=float(rz))
                )
                
                if step % 10 == 0:
                    print(f"[{step:04d}] CONTROL: vx={vx:+.2f} vy={vy:+.2f}->{vy_corrected:+.2f} (road_offset={road_offset:+.2f}) vz={vz:+.2f} rz={rz:+.2f} | alt={alt_now:.2f}m")
                
                time.sleep(dt)
    
    except KeyboardInterrupt:
        print("\n[INFO] User stop")
    
    finally:
        print("[INFO] Landing...")
        client.moveByVelocityBodyFrameAsync(0, 0, 0, 1).join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print("[OK] Disarmed, API control released.")

if __name__ == "__main__":
    main()