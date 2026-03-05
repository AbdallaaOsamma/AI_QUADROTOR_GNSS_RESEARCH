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
import sys
import tempfile

import numpy as np
import optuna


def run_trial(config_overrides: dict, total_timesteps: int = 50_000) -> float:
    """Launch train.py as a subprocess and return the mean eval reward.

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
        Mean reward from the last evaluation checkpoint recorded in
        ``evaluations.npz``.  Returns ``float("-inf")`` on subprocess failure
        or if the evaluation file is missing.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmp_log_dir:
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
            "--log_dir",
            tmp_log_dir,
            "--overrides",
            json.dumps(config_overrides),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return float("-inf")

        # SB3 EvalCallback writes evaluations.npz with shape (n_evals, n_episodes)
        # under <log_dir>/<run_name>/evaluations.npz
        eval_path = os.path.join(tmp_log_dir, "optuna_trial", "evaluations.npz")
        if not os.path.isfile(eval_path):
            return float("-inf")

        data = np.load(eval_path)
        # ``results`` has shape (n_evals, n_episodes); take mean of the last eval.
        results = data["results"]
        if results.size == 0:
            return float("-inf")

        return float(results[-1].mean())


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function.

    Samples:
    - ``w_progress``    — forward-progress reward weight [0.1, 2.0]
    - ``w_smoothness``  — action-smoothness penalty weight [-1.0, 0.0]
    - ``learning_rate`` — PPO learning rate (log-uniform) [1e-5, 1e-3]
    - ``n_steps``       — PPO rollout buffer size, categorical {512, 1024, 2048}

    Returns the mean eval reward from ``run_trial``.
    """
    w_progress = trial.suggest_float("w_progress", 0.1, 2.0)
    w_smoothness = trial.suggest_float("w_smoothness", -1.0, 0.0)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])

    config_overrides = {
        "rewards": {
            "w_progress": w_progress,
            "w_smoothness": w_smoothness,
        },
        "learning_rate": learning_rate,
        "n_steps": n_steps,
    }

    return run_trial(config_overrides)


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
            "rewards": {
                "w_progress": w_progress,
                "w_smoothness": w_smoothness,
            },
            "learning_rate": learning_rate,
            "n_steps": n_steps,
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
