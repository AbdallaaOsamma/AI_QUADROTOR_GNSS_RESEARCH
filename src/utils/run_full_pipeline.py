"""
Master Automation Script for AirSim Vision-Based Navigation FYP.

Orchestrates the entire end-to-end pipeline:
1. Environment Management (Launch/Kill AirSim)
2. Data Collection (Vision-based)
3. Model Training (CNN with Augmentation)
4. Deployment & Verification
5. Report Generation

Usage:
    python -m src.utils.run_full_pipeline
"""
import os
import sys
import time
import subprocess
import yaml
import signal
# import psutil  <-- Removed dependency
from pathlib import Path
from datetime import datetime

# ================= Configuration =================
AIRSIM_EXE_PATH = r"C:\AirSim\AirSimNH\WindowsNoEditor\AirSimNH.exe"
DATA_DURATION_S = 180.0  # 3 minutes of data collection
TRAIN_EPOCHS = 100
DEPLOY_DURATION_S = 60.0

# Paths
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
CONFIG_RECORD = WORKSPACE_ROOT / "configs/record_centerline.yaml"
CONFIG_TRAIN = WORKSPACE_ROOT / "configs/train_cnn_fixed.yaml"
CONFIG_DEPLOY = WORKSPACE_ROOT / "configs/run_cnn_deployment.yaml"
REPORT_PATH = WORKSPACE_ROOT / "fyp_report.txt"

# ================= Helpers =================
def log(msg):
    print(f"\n[PIPELINE] {msg}")
    with open(REPORT_PATH, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

def launch_airsim():
    log("Launching AirSim environment...")
    if not os.path.exists(AIRSIM_EXE_PATH):
        log(f"ERROR: AirSim executable not found at {AIRSIM_EXE_PATH}")
        sys.exit(1)
        
    # Launch in background
    subprocess.Popen(AIRSIM_EXE_PATH, cwd=os.path.dirname(AIRSIM_EXE_PATH))
    log("Waiting 15s for environment to load...")
    time.sleep(15)

def kill_airsim():
    log("Closing AirSim environment...")
    img_name = os.path.basename(AIRSIM_EXE_PATH)
    # Use Windows taskkill instead of psutil to avoid dependency issues
    try:
        subprocess.run(f"taskkill /F /IM {img_name}", shell=True, check=False)
        log(f"Sent kill signal to {img_name}")
    except Exception as e:
        log(f"Error killing AirSim: {e}")

def run_command(cmd, desc):
    log(f"START: {desc}")
    log(f"Command: {cmd}")
    try:
        # Run command and stream output
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            cwd=str(WORKSPACE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
                
        if process.returncode != 0:
            log(f"FAILED: {desc} (Exit Code: {process.returncode})")
            return False
            
        log(f"COMPLETED: {desc}")
        return True
        
    except Exception as e:
        log(f"EXCEPTION: {e}")
        return False

# ================= Main Pipeline =================
def main():
    # Initialize Report
    with open(REPORT_PATH, "w") as f:
        f.write("=== FYP Vision-Based Navigation Automation Report ===\n")
        f.write(f"Date: {datetime.now()}\n\n")
        
    try:
        # 1. Environment Launch
        launch_airsim()
        
        # 2. Data Collection
        # Update config duration if needed (automated edit or just rely on default)
        # We assume record_centerline.yaml is configured for multimodal perception
        success = run_command(
            f"python -m src.ai.recorder_multimodal --duration {DATA_DURATION_S}",
            "Data Collection"
        )
        if not success: raise RuntimeError("Data collection failed")
        
        # 3. Model Training
        # Note: train_cnn_fixed.yaml points to specific run dir. 
        # We need to update it to point to the *LATEST* run dir we just created.
        # But wait! train_cnn.py's `latest_run_dir` utility AUTOMATICALLY finds the latest run 
        # if output_dir is passed. Let's rely on the config system or simple logic.
        
        # Actually, let's update the train config dynamically to be safe
        # Find latest run folder
        import glob
        run_root = WORKSPACE_ROOT / "data/expert/centerline"
        list_runs = glob.glob(str(run_root / "run_*"))
        if not list_runs: raise RuntimeError("No data found!")
        latest_run = max(list_runs, key=os.path.getctime)
        latest_run = latest_run.replace(os.sep, '/') # standardize path
        log(f"Latest data found: {latest_run}")
        
        # Update train config
        with open(CONFIG_TRAIN, 'r') as f:
            train_cfg = yaml.safe_load(f)
        train_cfg['data']['run_dir'] = latest_run
        train_cfg['train']['num_epochs'] = TRAIN_EPOCHS
        with open(CONFIG_TRAIN, 'w') as f:
            yaml.dump(train_cfg, f)
            
        success = run_command(
            f"python -m src.ai.train_cnn --config configs/train_cnn_fixed.yaml",
            "Model Training (CNN + Augmentation)"
        )
        if not success: raise RuntimeError("Training failed")
        
        # 4. Deployment
        # Find latest model
        model_root = WORKSPACE_ROOT / "models"
        list_models = glob.glob(str(model_root / "cnn_nav_*.pth"))
        if not list_models: raise RuntimeError("No model found!")
        latest_model = max(list_models, key=os.path.getctime)
        latest_model = latest_model.replace(os.sep, '/')
        log(f"Latest model found: {latest_model}")
        
        # Update deploy config
        with open(CONFIG_DEPLOY, 'r') as f:
            deploy_cfg = yaml.safe_load(f)
        deploy_cfg['model']['path'] = latest_model
        # Ensure PID is set (we fixed this earlier but good to be safe)
        if 'pid' not in deploy_cfg:
             deploy_cfg['pid'] = {
                 'vz': {'kp': 0.9, 'ki': 0.08, 'kd': 0.3, 'limit': 1.8},
                 'vx': {'kp': 0.8, 'ki': 0.05, 'kd': 0.2, 'limit': 2.5},
                 'vy': {'kp': 0.8, 'ki': 0.05, 'kd': 0.2, 'limit': 2.5},
                 'rz': {'kp': 0.7, 'ki': 0.00, 'kd': 0.12, 'limit': 1.2}
             }
        
        with open(CONFIG_DEPLOY, 'w') as f:
            yaml.dump(deploy_cfg, f)
            
        success = run_command(
            f"python -m src.ai.policy_run_cnn --config configs/run_cnn_deployment.yaml --duration_s {DEPLOY_DURATION_S}",
            "Autonomous Deployment"
        )
        if not success: raise RuntimeError("Deployment failed")
        
        log("SUCCESS: End-to-end pipeline completed!")
        
    except Exception as e:
        log(f"CRITICAL FAILURE: {e}")
        
    finally:
        kill_airsim()
        log("Report saved to fyp_report.txt")

if __name__ == "__main__":
    main()
