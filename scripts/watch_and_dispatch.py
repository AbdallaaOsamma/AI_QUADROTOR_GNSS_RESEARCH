"""Training completion watcher — auto-dispatches next steps when a run finishes.

Polls the most-recent ppo log directory every N seconds.  When final_model.zip
appears (indicating the training loop exited cleanly) it prints the exact
commands to run next, ordered by priority.

Usage:
    python scripts/watch_and_dispatch.py                  # watches latest run
    python scripts/watch_and_dispatch.py logs/ppo/abl3_with_dr  # specific run
    python scripts/watch_and_dispatch.py --interval 60    # poll every 60s (default 30)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_ppo_run(log_root: str = "logs/ppo") -> Path | None:
    """Return the most recently modified run directory under log_root."""
    dirs = [
        d for d in Path(log_root).iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    zips = sorted(run_dir.glob("checkpoints/ppo_*_steps.zip"))
    return zips[-1] if zips else None


def _step_count(run_dir: Path) -> int:
    ckpt = _latest_checkpoint(run_dir)
    if not ckpt:
        return 0
    try:
        return int(ckpt.stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _is_finished(run_dir: Path) -> bool:
    return (run_dir / "final_model.zip").exists()


def _status_line(run_dir: Path) -> str:
    steps = _step_count(run_dir)
    ckpts = len(list(run_dir.glob("checkpoints/*.zip")))
    finished = "[DONE]" if _is_finished(run_dir) else "[running]"
    return (
        f"  {finished} {run_dir.name}  |  "
        f"steps: {steps:,}  |  checkpoints: {ckpts}"
    )


# ---------------------------------------------------------------------------
# Dispatch table — what to do when a run finishes
# ---------------------------------------------------------------------------

NEXT_STEPS = [
    {
        "label": "1. [PRIORITY] Start SAC training (longest lead time)",
        "cmd": "python -m src.training.train --algo sac --config configs/train_sac.yaml",
    },
    {
        "label": "2. Run 20-episode evaluation on best PPO model",
        "cmd": (
            "python scripts/run_full_eval.py "
            "--model logs/ppo/abl3_with_dr/best_model/best_model.zip"
        ),
        "note": "Replace abl3_with_dr with whichever run produced the best reward.",
    },
    {
        "label": "3. Run statistical analysis (after BOTH eval results exist)",
        "cmd": (
            "python scripts/run_statistical_analysis.py "
            "--results_dir logs/eval_results "
            "--metric average_speed_ms "
            "--output logs/statistical_results.json"
        ),
    },
]


def _print_dispatch(run_dir: Path) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  TRAINING COMPLETE: {run_dir.name}")
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)
    print("\n  AUTO-DISPATCH — run these commands next:\n")
    for step in NEXT_STEPS:
        print(f"  {step['label']}")
        print(f"    $ {step['cmd']}")
        if "note" in step:
            print(f"    NOTE: {step['note']}")
        print()
    print(sep)
    print("  Training monitor: python scripts/monitor_training.py <run_dir>")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Watch training and dispatch next steps")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Path to training run dir (default: auto-detect latest)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--log_root",
        default="logs/ppo",
        help="Root log directory to scan for runs (default: logs/ppo)",
    )
    args = parser.parse_args()

    # Resolve the run directory to watch
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = _latest_ppo_run(args.log_root)
        if not run_dir:
            print(f"[watch] No run directories found under '{args.log_root}'. Exiting.")
            sys.exit(1)

    if not run_dir.exists():
        print(f"[watch] Run dir not found: {run_dir}")
        sys.exit(1)

    print(f"\n[watch] Monitoring: {run_dir}")
    print(f"[watch] Poll interval: {args.interval}s  |  Ctrl+C to stop\n")

    # If already finished, dispatch immediately
    if _is_finished(run_dir):
        print("[watch] final_model.zip already present — run was already complete.")
        _print_dispatch(run_dir)
        return

    notified = False
    try:
        while True:
            print(_status_line(run_dir), flush=True)

            if _is_finished(run_dir) and not notified:
                notified = True
                _print_dispatch(run_dir)
                # Keep polling so user sees it if they glance back at the terminal
                print("[watch] Waiting for you to start the next steps...")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[watch] Stopped.")


if __name__ == "__main__":
    main()
