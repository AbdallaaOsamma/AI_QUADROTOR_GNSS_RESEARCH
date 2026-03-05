"""Run statistical analysis (ANOVA + paired t-tests) over RL evaluation results.

Compares episode-level metrics across multiple training runs to determine
whether observed performance differences are statistically significant.

Supported comparisons:
  - One-way ANOVA across all discovered run groups
  - Paired t-tests for known pairs: (ppo, sac), (ppo_dr, ppo_no_dr), (ppo_vio, ppo_gt)

Usage:
    python -m scripts.run_statistical_analysis --results_dir logs/eval_results
    python -m scripts.run_statistical_analysis --results_dir logs/eval_results \\
        --metric average_speed_ms --output logs/statistical_results.json
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
from scipy.stats import f_oneway, ttest_rel


# Pairs of run subdirectory names to compare with paired t-tests.
# Each element is (name_a, name_b). Pairs with missing runs are skipped.
COMPARISON_PAIRS = [
    ("ppo", "sac"),
    ("ppo_dr", "ppo_no_dr"),
    ("ppo_vio", "ppo_gt"),
]


def load_episode_summaries(eval_dir: str) -> list[dict]:
    """Load all episode summary JSON files from a directory (recursive).

    Parameters
    ----------
    eval_dir:
        Path to directory (or nested subdirectory) containing ``*.json`` files.

    Returns
    -------
    list[dict]
        Parsed JSON objects, one per file found.
    """
    pattern = os.path.join(eval_dir, "**", "*.json")
    paths = glob.glob(pattern, recursive=True)
    summaries = []
    for p in sorted(paths):
        try:
            with open(p, encoding="utf-8") as fh:
                summaries.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Could not load {p}: {exc}")
    return summaries


def extract_metric(summaries: list[dict], metric: str) -> np.ndarray:
    """Extract scalar metric values from episode summaries, dropping None.

    Parameters
    ----------
    summaries:
        List of episode summary dicts loaded by :func:`load_episode_summaries`.
    metric:
        Key to extract from each summary dict.

    Returns
    -------
    np.ndarray
        1-D float array of non-None values.
    """
    values = [s[metric] for s in summaries if s.get(metric) is not None]
    n_dropped = len(summaries) - len(values)
    if n_dropped > 0:
        warnings.warn(
            f"extract_metric('{metric}'): dropped {n_dropped}/{len(summaries)} "
            f"summaries with missing or None values",
            stacklevel=2,
        )
    return np.array(values, dtype=float)


def run_anova(*groups: np.ndarray) -> dict:
    """One-way ANOVA across N groups.

    Parameters
    ----------
    *groups:
        Two or more 1-D arrays of observations, one per group.

    Returns
    -------
    dict
        ``{"f_statistic": float, "p_value": float, "n_groups": int,
           "group_sizes": list[int], "group_means": list[float]}``
    """
    for i, g in enumerate(groups):
        if len(g) < 2:
            return {
                "f_statistic": None,
                "p_value": None,
                "error": "insufficient_samples",
                "n_groups": len(groups),
                "group_sizes": [int(len(g)) for g in groups],
                "group_means": [float(np.mean(g)) if len(g) > 0 else None for g in groups],
            }
    f_stat, p_val = f_oneway(*groups)
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "n_groups": len(groups),
        "group_sizes": [int(len(g)) for g in groups],
        "group_means": [float(np.mean(g)) for g in groups],
    }


def run_paired_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test between two equal-length arrays.

    Parameters
    ----------
    a, b:
        1-D arrays of paired observations (must be the same length).

    Returns
    -------
    dict
        ``{"statistic": float, "p_value": float, "significant_at_0.05": bool,
           "mean_a": float, "mean_b": float, "mean_diff": float}``
    """
    if len(a) != len(b):
        n = min(len(a), len(b))
        warnings.warn(
            f"run_paired_ttest: arrays have unequal lengths ({len(a)} vs {len(b)}), "
            f"truncating to {n} pairs. Pairing validity may be compromised.",
            stacklevel=2,
        )
        a = a[:n]
        b = b[:n]
    stat, p_val = ttest_rel(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant_at_0.05": bool(p_val < 0.05),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(a) - np.mean(b)),
    }


def _discover_run_dirs(results_dir: str) -> dict[str, str]:
    """Return ``{run_name: run_path}`` for immediate subdirectories of *results_dir*."""
    if not os.path.isdir(results_dir):
        return {}
    entries = {}
    for entry in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, entry)
        if os.path.isdir(full):
            entries[entry] = full
    return entries


def main() -> None:
    """CLI entry point for statistical analysis."""
    parser = argparse.ArgumentParser(
        description="ANOVA + paired t-test analysis of RL evaluation results",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="logs/eval_results",
        help="Directory containing per-run subdirectories with episode JSON files",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="average_speed_ms",
        help="Episode-level metric key to analyse (default: average_speed_ms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/statistical_results.json",
        help="Path to write JSON results (default: logs/statistical_results.json)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  STATISTICAL ANALYSIS")
    print(f"  results_dir : {args.results_dir}")
    print(f"  metric      : {args.metric}")
    print(f"  output      : {args.output}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 1. Discover run subdirectories and load episode summaries           #
    # ------------------------------------------------------------------ #
    run_dirs = _discover_run_dirs(args.results_dir)
    if not run_dirs:
        print(f"[WARN] No subdirectories found in '{args.results_dir}'. "
              "Results will be empty.")

    groups: dict[str, np.ndarray] = {}
    for run_name, run_path in run_dirs.items():
        summaries = load_episode_summaries(run_path)
        values = extract_metric(summaries, args.metric)
        if len(values) == 0:
            print(f"[WARN] No valid '{args.metric}' values found for run '{run_name}'. "
                  "Skipping.")
            continue
        groups[run_name] = values
        print(f"  [{run_name}]  n={len(values)}  mean={np.mean(values):.4f}  "
              f"std={np.std(values):.4f}")

    results: dict = {
        "metric": args.metric,
        "results_dir": args.results_dir,
        "run_names": list(groups.keys()),
        "anova": None,
        "paired_ttests": [],
    }

    # ------------------------------------------------------------------ #
    # 2. One-way ANOVA across all groups                                  #
    # ------------------------------------------------------------------ #
    print(f"\n--- One-Way ANOVA ({args.metric}) ---")
    if len(groups) >= 2:
        anova_result = run_anova(*groups.values())
        results["anova"] = anova_result
        if anova_result.get("error"):
            print(f"  [SKIP] ANOVA error: {anova_result['error']}")
        else:
            print(f"  F={anova_result['f_statistic']:.4f}  "
                  f"p={anova_result['p_value']:.4e}  "
                  f"groups={anova_result['n_groups']}")
        if anova_result.get("p_value") is not None and anova_result["p_value"] < 0.05:
            print("  ** Significant difference detected (p < 0.05) **")
        else:
            print("  No significant difference (p >= 0.05)")
    else:
        print(f"  [SKIP] Need >= 2 groups for ANOVA; found {len(groups)}.")

    # ------------------------------------------------------------------ #
    # 3. Paired t-tests for known comparison pairs                        #
    # ------------------------------------------------------------------ #
    print(f"\n--- Paired t-Tests ({args.metric}) ---")
    for name_a, name_b in COMPARISON_PAIRS:
        if name_a not in groups or name_b not in groups:
            print(f"  [SKIP] ({name_a}, {name_b}) — one or both runs not found.")
            continue
        a = groups[name_a]
        b = groups[name_b]
        # Trim to same length for paired test
        n = min(len(a), len(b))
        if n < 2:
            print(f"  [SKIP] ({name_a}, {name_b}) — fewer than 2 paired observations.")
            continue
        if len(a) != len(b):
            print(f"  [WARN] ({name_a}, {name_b}) — unequal sample sizes "
                  f"({len(a)} vs {len(b)}), truncating to {n} pairs.")
        ttest_result = run_paired_ttest(a[:n], b[:n])
        ttest_result["pair"] = f"{name_a}_vs_{name_b}"
        ttest_result["n_pairs"] = n
        results["paired_ttests"].append(ttest_result)
        sig = "**SIGNIFICANT**" if ttest_result["significant_at_0.05"] else "not significant"
        print(f"  ({name_a} vs {name_b})  t={ttest_result['statistic']:.4f}  "
              f"p={ttest_result['p_value']:.4e}  [{sig}]")
        print(f"    mean_{name_a}={ttest_result['mean_a']:.4f}  "
              f"mean_{name_b}={ttest_result['mean_b']:.4f}  "
              f"diff={ttest_result['mean_diff']:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Save results                                                     #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[OK] Results written to '{args.output}'")


if __name__ == "__main__":
    main()
