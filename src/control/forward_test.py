import time
import airsim

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("[INFO] Taking off...")
    client.takeoffAsync().join()
    
    print("[INFO] Moving forward 10m...")
    client.moveByVelocityAsync(2, 0, 0, 5).join()  # 2 m/s forward for 5 seconds
    
    print("[INFO] Hovering...")
    time.sleep(2)
    
    print("[INFO] Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("[OK] Done")

if __name__ == "__main__":
    main()