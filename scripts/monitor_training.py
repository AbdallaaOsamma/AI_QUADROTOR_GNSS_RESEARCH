"""Read SB3 TensorBoard events and print a compact training status line.

Usage:
    python scripts/monitor_training.py [run_dir]

Default run_dir: logs/ppo/parking_finetune
"""
import glob
import os
import sys
from pathlib import Path


def read_tb_events(event_file: str) -> dict:
    """Parse TF events file and return latest scalar values."""
    try:
        # Try tensorboard's event file reader (lightweight, no full TF needed)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(Path(event_file).parent))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        latest = {}
        for tag in tags:
            events = ea.Scalars(tag)
            if events:
                latest[tag] = (events[-1].step, events[-1].value)
        return latest
    except Exception:
        pass

    # Fallback: try tensorflow directly
    try:
        import tensorflow as tf
        data = {}
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                data[v.tag] = (e.step, v.simple_value)
        return data
    except Exception:
        pass

    return {}


def format_status(run_dir: str) -> str:
    run_dir = Path(run_dir)
    lines = [f"=== Training Monitor: {run_dir.name} ==="]

    # Find event files
    event_files = sorted(glob.glob(str(run_dir / "**" / "*.tfevents.*"), recursive=True))
    if not event_files:
        return "\n".join(lines + ["  No TF event files found yet."])

    # Use the most recently modified
    event_file = max(event_files, key=os.path.getmtime)
    size_kb = os.path.getsize(event_file) / 1024
    lines.append(f"  Event file: {Path(event_file).name} ({size_kb:.1f} KB)")

    scalars = read_tb_events(event_file)

    if not scalars:
        lines.append("  No scalar data yet (training may still be initializing).")
    else:
        # Key SB3 metrics
        key_map = {
            "rollout/ep_rew_mean":    "Ep reward (mean)",
            "rollout/ep_len_mean":    "Ep length (mean)",
            "train/loss":             "Loss",
            "train/policy_gradient_loss": "Policy grad loss",
            "train/value_loss":       "Value loss",
            "train/entropy_loss":     "Entropy loss",
            "train/approx_kl":        "Approx KL",
            "train/clip_fraction":    "Clip fraction",
            "time/fps":               "FPS",
            "time/total_timesteps":   "Total steps",
        }
        step = 0
        for tag, (s, v) in scalars.items():
            step = max(step, s)

        lines.append(f"  Steps: {step:,}")
        lines.append("")
        for sb3_tag, label in key_map.items():
            if sb3_tag in scalars:
                s, v = scalars[sb3_tag]
                lines.append(f"  {label:<25} {v:>10.4f}  (step {s:,})")

        # Convergence signal
        if "rollout/ep_rew_mean" in scalars:
            _, rew = scalars["rollout/ep_rew_mean"]
            if rew > 5:
                signal = "LEARNING [OK]"
            elif rew > 0:
                signal = "improving (early)"
            elif rew > -20:
                signal = "flat / slow start"
            else:
                signal = "struggling (many collisions)"
            lines.append(f"\n  Convergence signal: {signal}")

    # Checkpoints
    ckpts = sorted(glob.glob(str(run_dir / "checkpoints" / "*.zip")))
    lines.append(f"\n  Checkpoints saved: {len(ckpts)}")
    if ckpts:
        lines.append(f"  Latest: {Path(ckpts[-1]).name}")

    # Best model
    best = run_dir / "best_model" / "best_model.zip"
    lines.append(f"  Best model saved: {'yes' if best.exists() else 'not yet'}")

    return "\n".join(lines)


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/ppo/parking_finetune"
    print(format_status(run_dir))
