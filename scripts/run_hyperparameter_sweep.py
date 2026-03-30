"""Optuna hyperparameter sweep for the PPO training pipeline.

Samples reward weights, learning rate, and n_steps via Bayesian optimisation,
launching each trial as a subprocess and reading the EvalCallback evaluations.npz
to extract the mean reward.

Usage:
    python scripts/run_hyperparameter_sweep.py
    python scripts/run_hyperparameter_sweep.py --n_trials 20 --study_name ppo_sweep
    python scripts/run_hyperparameter_sweep.py --storage sqlite:///logs/optuna.db
"""

import argparse
import json
import os
import re
import sys
import tempfile

import optuna


def run_trial(config_overrides: dict, total_timesteps: int = 50_000) -> float:
    """Launch train.py as a subprocess and return the mean training reward.

    Parameters
    ----------
    config_overrides:
        Key-value pairs deep-merged into the training config (highest priority).
        Passed to train.py as ``--overrides <json_string>``.
    total_timesteps:
        Number of environment steps for this trial.  Kept short (default 50 k)
        so the sweep completes in a reasonable wall-clock budget.

    Returns
    -------
    float
        Mean episode reward from the last logged rollout (``ep_rew_mean`` in
        SB3's verbose output).  Returns ``float("-inf")`` on subprocess failure
        or if no rollout data was captured.

    Notes
    -----
    Eval is disabled (``--no_eval``) because the Optuna sweep typically runs
    with a single AirSim instance on port 41451.  The eval port (41452) is not
    available, so ``EvalCallback`` would never write ``evaluations.npz`` and
    every trial would silently score ``-inf``.  Using training reward from
    stdout is simpler and avoids requiring a second AirSim instance.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmp_log_dir:
        # Inject log_dir via overrides (train.py has no --log_dir flag)
        overrides_with_dir = {
            **config_overrides,
            "output": {"log_dir": tmp_log_dir},
        }
        cmd = [
            sys.executable,
            "-m",
            "src.training.train",
            "--config",
            "configs/train_ppo_fast.yaml",
            "--total_timesteps",
            str(total_timesteps),
            "--run_name",
            "optuna_trial",
            "--no_eval",   # eval port 41452 not available during sweep
            "--overrides",
            json.dumps(overrides_with_dir),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return float("-inf")

        # SB3 with verbose=1 prints a table containing "| ep_rew_mean  |  <value> |"
        # after each rollout.  Extract all occurrences and return the last one.
        matches = re.findall(
            r"\|\s*ep_rew_mean\s*\|\s*([-\d.eE+]+)\s*\|",
            result.stdout,
        )
        if not matches:
            return float("-inf")

        return float(matches[-1])


def main() -> None:
    """CLI entry point for the Optuna hyperparameter sweep."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep for PPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_hyperparameter_sweep.py
  python scripts/run_hyperparameter_sweep.py --n_trials 20 --study_name ppo_sweep
  python scripts/run_hyperparameter_sweep.py --storage sqlite:///logs/optuna.db
        """,
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of Optuna trials to run (default: 10)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="ppo_hyperparameter_sweep",
        help="Optuna study name (default: ppo_hyperparameter_sweep)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///logs/optuna.db (default: in-memory)",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=50_000,
        help="Training timesteps per trial (default: 50_000)",
    )

    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )

    # Wrap objective to inject total_timesteps from CLI args.
    def _objective(trial: optuna.Trial) -> float:
        w_progress = trial.suggest_float("w_progress", 0.1, 2.0)
        w_smoothness = trial.suggest_float("w_smoothness", -1.0, 0.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])

        config_overrides = {
            "reward": {
                "w_progress": w_progress,
                "w_smoothness": w_smoothness,
            },
            "ppo": {
                "learning_rate": learning_rate,
                "n_steps": n_steps,
            },
        }

        return run_trial(config_overrides, total_timesteps=args.total_timesteps)

    study.optimize(_objective, n_trials=args.n_trials)

    print("\n" + "=" * 70)
    print("  HYPERPARAMETER SWEEP COMPLETE")
    print("=" * 70)
    print(f"\n  Study name : {study.study_name}")
    print(f"  Best trial : #{study.best_trial.number}")
    print(f"  Best value : {study.best_trial.value:.4f}")
    print("\n  Best hyperparameters:")
    for key, val in study.best_trial.params.items():
        print(f"    {key}: {val}")

    # Persist results to JSON for downstream analysis.
    os.makedirs("logs", exist_ok=True)
    output_path = os.path.join("logs", f"optuna_{args.study_name}.json")

    results_data = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results_data, fh, indent=2)

    print(f"\n  Results saved to: {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
