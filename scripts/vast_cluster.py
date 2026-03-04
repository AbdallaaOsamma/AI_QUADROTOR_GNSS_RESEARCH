"""Vast.ai GPU Cluster Orchestrator — run all experiments in parallel.

Rents N GPU instances, bootstraps each one, assigns one experiment per GPU,
and collects results. Turns 22-44 hours of sequential training into ~3 hours.

Usage:
    # Dry run — show what would happen
    python scripts/vast_cluster.py --dry-run

    # Launch all 11 experiments across 11 GPUs
    python scripts/vast_cluster.py --launch

    # Check status of running cluster
    python scripts/vast_cluster.py --status

    # Collect results from all instances
    python scripts/vast_cluster.py --collect

    # Tear down all instances
    python scripts/vast_cluster.py --destroy
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VASTAI = "C:/Users/bedor/AppData/Roaming/Python/Python313/Scripts/vastai.exe"
REPO_URL = "https://github.com/Abood204/AI_QUADROTOR_GNSS_RESEARCH.git"
LOCAL_PROJECT = Path(__file__).resolve().parent.parent

# All independent experiments that can run in parallel
EXPERIMENTS = [
    # --- Ablation 1: Reward components ---
    {
        "name": "abl1_full_reward",
        "cmd": "python -m src.training.train --config configs/train_ppo.yaml --reward_config configs/rewards/default.yaml --run_name abl1_full_reward --total_timesteps {steps}",
    },
    {
        "name": "abl1_no_smoothness",
        "cmd": "python -m src.training.train --config configs/train_ppo.yaml --reward_config configs/ablations/no_smoothness.yaml --run_name abl1_no_smoothness --total_timesteps {steps}",
    },
    {
        "name": "abl1_progress_only",
        "cmd": "python -m src.training.train --config configs/train_ppo.yaml --reward_config configs/ablations/progress_only.yaml --run_name abl1_progress_only --total_timesteps {steps}",
    },
    # --- Ablation 2: Frame stack (uses base config, 4-frame is default) ---
    {
        "name": "abl2_frames_4",
        "cmd": "python -m src.training.train --config configs/train_ppo.yaml --run_name abl2_frames_4 --total_timesteps {steps}",
    },
    # Note: abl2_frames_1 requires --overrides support (Known Issue #1), skipped for now
    # --- Ablation 3: Domain randomization ---
    {
        "name": "abl3_with_dr",
        "cmd": "python -m src.training.train --config configs/train_ppo_dr.yaml --run_name abl3_with_dr --total_timesteps {steps}",
    },
    {
        "name": "abl3_no_dr",
        "cmd": "python -m src.training.train --config configs/train_ppo.yaml --run_name abl3_no_dr --total_timesteps {steps}",
    },
    # --- Reward sweep ---
    {
        "name": "reward_aggressive",
        "cmd": "python -m src.training.train --config configs/train_ppo.yaml --reward_config configs/rewards/aggressive.yaml --run_name reward_aggressive --total_timesteps {steps}",
    },
    {
        "name": "reward_cautious",
        "cmd": "python -m src.training.train --config configs/train_ppo.yaml --reward_config configs/rewards/cautious.yaml --run_name reward_cautious --total_timesteps {steps}",
    },
    # --- Waypoint navigation ---
    {
        "name": "waypoint_v1",
        "cmd": "python -m src.training.train --config configs/train_ppo_waypoint.yaml --run_name waypoint_v1 --total_timesteps {steps}",
    },
]

# Bootstrap script that runs on each fresh instance
BOOTSTRAP_SCRIPT = r"""#!/bin/bash
set -e

echo "[1/8] Installing system deps..."
apt-get update -qq && apt-get install -y -qq xvfb libglu1-mesa libglib2.0-0 git tmux wget unzip > /dev/null 2>&1

echo "[2/8] Creating airuser..."
useradd -m airuser 2>/dev/null || true

echo "[3/8] Cloning repo..."
rm -rf /home/airuser/AI_QUADROTOR_GNSS_RESEARCH
git clone --branch master --single-branch {repo_url} /home/airuser/AI_QUADROTOR_GNSS_RESEARCH

echo "[4/8] Creating conda env..."
conda create -n airsim_rl python=3.11 -y -q > /dev/null 2>&1

echo "[5/8] Installing PyTorch + deps..."
export PIP_ROOT_USER_ACTION=ignore
/opt/conda/envs/airsim_rl/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
/opt/conda/envs/airsim_rl/bin/pip install -q msgpack-rpc-python
/opt/conda/envs/airsim_rl/bin/pip install -q --no-build-isolation airsim==1.8.1
cd /home/airuser/AI_QUADROTOR_GNSS_RESEARCH && /opt/conda/envs/airsim_rl/bin/pip install -q -e ".[dev]"

echo "[6/8] Downloading AirSim NH..."
wget -q -O /home/airuser/NH.zip "https://github.com/microsoft/AirSim/releases/download/v1.8.1/AirSimNH.zip"
cd /home/airuser && unzip -q -o NH.zip
chmod +x /home/airuser/AirSimNH/LinuxNoEditor/AirSimNH.sh

echo "[7/8] Configuring AirSim settings..."
mkdir -p /home/airuser/Documents/AirSim
cp /home/airuser/AI_QUADROTOR_GNSS_RESEARCH/configs/settings_training.json /home/airuser/Documents/AirSim/settings.json
chown -R airuser:airuser /home/airuser

echo "[8/8] Starting Xvfb + AirSim..."
Xvfb :99 -screen 0 1280x720x24 &>/dev/null &
sleep 2
su airuser -c "HOME=/home/airuser DISPLAY=:99 /home/airuser/AirSimNH/LinuxNoEditor/AirSimNH.sh -RenderOffScreen &>/tmp/airsim_startup.log &"
sleep 45

# Verify AirSim is running
if ps aux | grep -q "[A]irSimNH.*Binaries"; then
    echo "[OK] AirSim is running!"
else
    echo "[FAIL] AirSim did not start"
    tail -20 /tmp/airsim_startup.log
    exit 1
fi

echo "[BOOTSTRAP COMPLETE]"
"""


def run_vastai(*args):
    """Run a vastai command and return stdout."""
    cmd = [VASTAI] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] vastai {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def search_instances(count):
    """Find cheapest RTX 4090 instances."""
    raw = run_vastai(
        "search", "offers",
        "gpu_name=RTX_4090 num_gpus=1 disk_space>=80 dph<=0.50 reliability>0.95",
        "--storage", "80", "-o", "dph", "--raw",
    )
    try:
        offers = json.loads(raw)
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse offers")
        return []
    return offers[:count]


def rent_instance(offer_id):
    """Rent a single instance. Returns contract ID."""
    out = run_vastai(
        "create", "instance", str(offer_id),
        "--image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
        "--disk", "80",
    )
    try:
        data = json.loads(out.replace("'", '"'))
        return data.get("new_contract")
    except (json.JSONDecodeError, AttributeError):
        # Try to extract contract ID from string
        if "new_contract" in out:
            import re
            m = re.search(r"new_contract['\"]?\s*:\s*(\d+)", out)
            if m:
                return int(m.group(1))
    print(f"[ERROR] Could not parse contract from: {out}")
    return None


def get_instances():
    """Get all running instances as JSON."""
    raw = run_vastai("show", "instances", "--raw")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def wait_for_instances(instance_ids, timeout=600):
    """Wait until all instances are running."""
    print(f"[cluster] Waiting for {len(instance_ids)} instances to start...")
    start = time.time()
    while time.time() - start < timeout:
        instances = get_instances()
        running = [i for i in instances if i["id"] in instance_ids and i["actual_status"] == "running"]
        loading = [i for i in instances if i["id"] in instance_ids and i["actual_status"] == "loading"]
        print(f"  Running: {len(running)}/{len(instance_ids)}  Loading: {len(loading)}  ({int(time.time()-start)}s)")
        if len(running) == len(instance_ids):
            return True
        time.sleep(15)
    print("[TIMEOUT] Not all instances started")
    return False


def ssh_cmd(host, port, command, timeout=600):
    """Run a command on a remote instance via SSH."""
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-p", str(port),
        f"root@{host}",
        command,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr, result.returncode


def attach_ssh_key(instance_id):
    """Attach local SSH public key to instance."""
    ssh_pub = Path.home() / ".ssh" / "id_ed25519.pub"
    if not ssh_pub.exists():
        print("[ERROR] No SSH key found at ~/.ssh/id_ed25519.pub")
        return False
    key = ssh_pub.read_text().strip()
    out = run_vastai("attach ssh", str(instance_id), key)
    return "success" in out.lower() or "True" in out


def bootstrap_instance(host, port, experiment_name):
    """Bootstrap an instance with all dependencies."""
    script = BOOTSTRAP_SCRIPT.format(repo_url=REPO_URL)
    print(f"  [{experiment_name}] Bootstrapping {host}:{port}...")

    # Upload and run bootstrap script
    upload_cmd = f"cat << 'ENDSCRIPT' > /tmp/bootstrap.sh\n{script}\nENDSCRIPT\nbash /tmp/bootstrap.sh"
    out, code = ssh_cmd(host, port, upload_cmd, timeout=1200)

    if "[BOOTSTRAP COMPLETE]" in out:
        print(f"  [{experiment_name}] Bootstrap complete!")
        return True
    else:
        print(f"  [{experiment_name}] Bootstrap FAILED (exit {code})")
        # Print last 10 lines for debugging
        for line in out.strip().split("\n")[-10:]:
            print(f"    {line}")
        return False


def launch_experiment(host, port, experiment):
    """Launch a training experiment in tmux on the remote instance."""
    name = experiment["name"]
    cmd = experiment["cmd"]
    train_cmd = f"cd /home/airuser/AI_QUADROTOR_GNSS_RESEARCH && /opt/conda/envs/airsim_rl/bin/{cmd}"
    tmux_cmd = f'tmux new-session -d -s train "{train_cmd} 2>&1 | tee /tmp/train_{name}.log"'

    out, code = ssh_cmd(host, port, tmux_cmd)
    if code == 0:
        print(f"  [{name}] Training launched!")
        return True
    else:
        print(f"  [{name}] Launch FAILED: {out}")
        return False


def check_status():
    """Check training status on all running instances."""
    instances = get_instances()
    if not instances:
        print("[cluster] No running instances found.")
        return

    print(f"\n{'='*80}")
    print(f"  CLUSTER STATUS — {len(instances)} instance(s)")
    print(f"{'='*80}")

    total_cost = 0
    for inst in instances:
        iid = inst["id"]
        host = inst.get("ssh_host", "")
        port = inst.get("ssh_port", "")
        dph = inst.get("dph_total", 0)
        label = inst.get("label", "")
        status = inst.get("actual_status", "unknown")
        total_cost += dph

        print(f"\n  Instance {iid} ({label or 'unlabeled'}) — {status} — ${dph:.3f}/hr")

        if status == "running" and host and port:
            # Check if training is active
            out, _ = ssh_cmd(host, port, "tmux capture-pane -t train -p 2>/dev/null | tail -5", timeout=15)
            if out.strip():
                for line in out.strip().split("\n")[-3:]:
                    print(f"    {line.strip()}")
            else:
                print(f"    (no tmux session or output)")

    print(f"\n  Total cluster cost: ${total_cost:.3f}/hr (${total_cost*24:.2f}/day)")
    print(f"{'='*80}\n")


def collect_results():
    """Rsync logs from all instances to local."""
    instances = get_instances()
    local_logs = LOCAL_PROJECT / "logs" / "ppo"
    local_logs.mkdir(parents=True, exist_ok=True)

    for inst in instances:
        host = inst.get("ssh_host", "")
        port = inst.get("ssh_port", "")
        if not host or not port:
            continue

        print(f"  Collecting from {host}:{port}...")
        cmd = [
            "rsync", "-avz", "--progress",
            "-e", f"ssh -o StrictHostKeyChecking=no -p {port}",
            f"root@{host}:/home/airuser/AI_QUADROTOR_GNSS_RESEARCH/logs/ppo/",
            str(local_logs) + "/",
        ]
        subprocess.run(cmd)

    print(f"\n[collect] Results synced to {local_logs}")


def destroy_all():
    """Destroy all running instances."""
    instances = get_instances()
    if not instances:
        print("[cluster] No instances to destroy.")
        return

    for inst in instances:
        iid = inst["id"]
        print(f"  Destroying instance {iid}...")
        run_vastai("destroy", "instance", str(iid))

    print(f"[cluster] {len(instances)} instance(s) destroyed.")


def launch_cluster(experiments, timesteps, budget_per_hour=5.0):
    """Main launch: rent GPUs, bootstrap, assign experiments, go."""
    n = len(experiments)
    print(f"\n{'='*80}")
    print(f"  VAST.AI GPU CLUSTER LAUNCH")
    print(f"  {n} experiments × {timesteps:,} steps each")
    print(f"  Estimated time: ~2-4 hours (vs ~{n*3} hours sequential)")
    print(f"  Estimated cost: ~${n * 0.30 * 3:.2f} (${0.30:.2f}/hr × {n} GPUs × ~3hrs)")
    print(f"{'='*80}\n")

    # 1. Search for offers
    print(f"[1/5] Searching for {n} RTX 4090 instances...")
    offers = search_instances(n + 2)  # grab extras in case some fail
    if len(offers) < n:
        print(f"[ERROR] Only found {len(offers)} offers, need {n}")
        return

    est_cost = sum(o.get("dph_total", 0.5) for o in offers[:n])
    print(f"  Found {len(offers)} offers. Top {n} cost: ~${est_cost:.2f}/hr total")

    # 2. Rent instances
    print(f"\n[2/5] Renting {n} instances...")
    instance_ids = []
    for i, offer in enumerate(offers[:n]):
        oid = offer["id"]
        contract = rent_instance(oid)
        if contract:
            instance_ids.append(contract)
            print(f"  [{i+1}/{n}] Rented offer {oid} -> contract {contract} (${offer.get('dph_total', '?')}/hr)")
        else:
            print(f"  [{i+1}/{n}] FAILED to rent offer {oid}")

    if len(instance_ids) < n:
        print(f"[WARN] Only rented {len(instance_ids)}/{n} instances")

    # 3. Wait for all to start
    print(f"\n[3/5] Waiting for instances to come online...")
    wait_for_instances(instance_ids, timeout=600)

    # Get SSH details
    instances = get_instances()
    running = [i for i in instances if i["id"] in instance_ids and i["actual_status"] == "running"]

    # Attach SSH keys
    print(f"\n  Attaching SSH keys...")
    for inst in running:
        attach_ssh_key(inst["id"])
    time.sleep(10)  # wait for keys to propagate

    # 4. Bootstrap ALL instances in parallel (saves ~90 min vs sequential)
    print(f"\n[4/5] Bootstrapping {len(running)} instances IN PARALLEL...")
    assignments = []
    for i, inst in enumerate(running):
        exp = experiments[i] if i < len(experiments) else None
        if exp:
            assignments.append((inst, exp))

    ready = []
    def _bootstrap_one(inst_exp):
        inst, exp = inst_exp
        host = inst.get("ssh_host", "")
        port = inst.get("ssh_port", "")
        ok = bootstrap_instance(host, port, exp["name"])
        return (inst, exp, ok)

    with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
        futures = {pool.submit(_bootstrap_one, a): a for a in assignments}
        for future in as_completed(futures):
            inst, exp, ok = future.result()
            if ok:
                ready.append((inst, exp))

    print(f"\n  {len(ready)}/{len(assignments)} instances bootstrapped successfully")

    # 5. Launch experiments on all ready instances
    print(f"\n[5/5] Launching {len(ready)} experiments...")
    for inst, exp in ready:
        host = inst.get("ssh_host", "")
        port = inst.get("ssh_port", "")
        exp_with_steps = {**exp, "cmd": exp["cmd"].format(steps=timesteps)}
        launch_experiment(host, port, exp_with_steps)

        # Label instance with experiment name
        run_vastai("label", "instance", str(inst["id"]), exp["name"])

    print(f"\n{'='*80}")
    print(f"  CLUSTER LAUNCHED!")
    print(f"  {len(ready)}/{n} experiments running on {len(ready)} GPUs")
    print(f"  Monitor: python scripts/vast_cluster.py --status")
    print(f"  Collect: python scripts/vast_cluster.py --collect")
    print(f"  Destroy: python scripts/vast_cluster.py --destroy")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Vast.ai GPU Cluster Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true", help="Rent GPUs and launch all experiments")
    group.add_argument("--status", action="store_true", help="Check cluster status")
    group.add_argument("--collect", action="store_true", help="Sync results to local")
    group.add_argument("--destroy", action="store_true", help="Tear down all instances")
    group.add_argument("--dry-run", action="store_true", help="Show plan without executing")

    parser.add_argument("--timesteps", type=int, default=500_000, help="Steps per experiment (default: 500K)")
    parser.add_argument("--only", nargs="+", help="Run only these experiments by name")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in EXPERIMENTS if e["name"] in args.only]

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch {len(experiments)} experiments:\n")
        for i, exp in enumerate(experiments, 1):
            cmd = exp["cmd"].format(steps=args.timesteps)
            print(f"  GPU {i}: {exp['name']}")
            print(f"         {cmd}\n")
        est = len(experiments) * 0.30 * 3
        print(f"  Estimated cost: ~${est:.2f} ({len(experiments)} GPUs × $0.30/hr × ~3hrs)")
        print(f"  Estimated time: ~3 hours (vs ~{len(experiments)*3} hours sequential)")
        print(f"  Speedup: ~{len(experiments)}x\n")
        return

    if args.launch:
        launch_cluster(experiments, args.timesteps)
    elif args.status:
        check_status()
    elif args.collect:
        collect_results()
    elif args.destroy:
        destroy_all()


if __name__ == "__main__":
    sys.exit(main() or 0)
