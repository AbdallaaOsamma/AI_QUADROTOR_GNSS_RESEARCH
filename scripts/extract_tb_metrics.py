"""Extract training metrics from TensorBoard event files for all runs.

Produces a JSON summary and a Markdown table suitable for pasting into the report.

Usage:
    python scripts/extract_tb_metrics.py
    python scripts/extract_tb_metrics.py --output logs/metrics_summary.json
"""
import argparse
import json
import os

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    raise

# Map run directory → canonical label for the paper
RUN_REGISTRY = [
    # name, ppo_subdir, label
    ("abl1_full_reward",    "PPO_1", "Full Reward (baseline)"),
    ("abl1_no_smoothness",  "PPO_1", "No Smoothness"),
    ("abl1_progress_only",  "PPO_2", "Progress Only"),
    ("abl2_frames_1",       "PPO_1", "1-Frame Stack"),
    ("abl2_frames_4",       "PPO_1", "4-Frame Stack (baseline)"),
    ("abl3_with_dr",        "PPO_1", "With DR (baseline)"),
    # Add abl3_no_dr once it has been trained:
    # ("abl3_no_dr",        "PPO_1", "No DR"),
    # Main baselines
    ("ppo_base",            "PPO_1", "PPO Base (500k)"),
]

TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "reward/r_collision",
    "reward/r_progress",
    "reward/r_smoothness",
    "reward/r_drift",
    "time/fps",
]

TAIL_N = 20  # average over last N TensorBoard data points


def extract_run(log_dir: str, ppo_subdir: str) -> dict | None:
    tb_dir = os.path.join("logs", "ppo", log_dir, ppo_subdir)
    if not os.path.isdir(tb_dir):
        return None
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    if "rollout/ep_rew_mean" not in available:
        return None

    result = {"log_dir": log_dir, "tb_subdir": ppo_subdir}

    for tag in TAGS:
        if tag not in available:
            result[tag] = None
            result[tag + "_std"] = None
            continue
        data = ea.Scalars(tag)
        vals = [s.value for s in data]
        steps = [s.step for s in data]
        tail = vals[-TAIL_N:]
        result["total_steps"] = steps[-1]
        result["n_eval_points"] = len(vals)
        result[tag] = float(np.mean(tail))
        result[tag + "_std"] = float(np.std(tail))
        result[tag + "_final"] = float(vals[-1])

    return result


def collision_rate_approx(r_collision_per_step: float, ep_len: float) -> float:
    """Approximate episode-level collision rate from per-step collision reward.

    r_collision = w_collision * indicator(collision) logged as mean across rollout steps.
    For w_collision = -100 and mean episode length ep_len:
        collision_rate ≈ -r_collision_per_step * ep_len / 100
    """
    if r_collision_per_step is None or ep_len is None:
        return None
    return min(1.0, abs(r_collision_per_step) * ep_len / 100.0)


def format_table(rows: list[dict]) -> str:
    headers = [
        "Run", "Steps", "Ep Reward", "Ep Length (steps)", "~Collision Rate",
        "Progress r/step", "Smoothness r/step", "FPS",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join([" --- "] * len(headers)) + "|")

    for r in rows:
        rew = r.get("rollout/ep_rew_mean")
        rew_std = r.get("rollout/ep_rew_mean_std")
        ep_len = r.get("rollout/ep_len_mean")
        r_col = r.get("reward/r_collision")
        r_prog = r.get("reward/r_progress")
        r_smooth = r.get("reward/r_smoothness")
        fps = r.get("time/fps")
        steps = r.get("total_steps", 0)

        col_rate = collision_rate_approx(r_col, ep_len)

        def fmt(v, fmt_str=".2f"):
            return f"{v:{fmt_str}}" if v is not None else "N/A"

        row = [
            r["label"],
            f"{steps:,}" if steps else "N/A",
            f"{fmt(rew)} ± {fmt(rew_std)}" if rew_std else fmt(rew),
            fmt(ep_len),
            f"{col_rate:.1%}" if col_rate is not None else "N/A",
            fmt(r_prog, ".4f") if r_prog else "N/A",
            fmt(r_smooth, ".4f") if r_smooth else "N/A",
            fmt(fps, ".0f") if fps else "N/A",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/metrics_summary.json")
    parser.add_argument("--table", action="store_true", help="Print Markdown table")
    args = parser.parse_args()

    results = []
    for log_dir, ppo_subdir, label in RUN_REGISTRY:
        data = extract_run(log_dir, ppo_subdir)
        if data is None:
            print(f"[SKIP] {log_dir}/{ppo_subdir} — not found or no data")
            continue
        data["label"] = label
        results.append(data)
        col_rate = collision_rate_approx(
            data.get("reward/r_collision"),
            data.get("rollout/ep_len_mean"),
        )
        print(
            f"[OK] {label:30s} steps={data.get('total_steps',0):>7,} "
            f"rew={data.get('rollout/ep_rew_mean', 0):>8.2f}±{data.get('rollout/ep_rew_mean_std', 0):.2f} "
            f"col_rate~{col_rate:.1%} ep_len={data.get('rollout/ep_len_mean', 0):.1f}"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    if args.table:
        print("\n\n" + "=" * 80)
        print("MARKDOWN TABLE (paste into report)")
        print("=" * 80 + "\n")
        print(format_table(results))


if __name__ == "__main__":
    main()
