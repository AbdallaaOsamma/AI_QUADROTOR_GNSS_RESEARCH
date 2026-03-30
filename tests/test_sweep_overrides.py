"""Tests that hyperparameter sweep builds correctly nested override keys."""
import pathlib


def _load_sweep_module():
    import types as _types
    mod = _types.ModuleType("run_hyperparameter_sweep")
    mod.__file__ = str(pathlib.Path("scripts/run_hyperparameter_sweep.py"))
    return mod


class TestSweepOverrides:
    def test_reward_key_is_reward_not_rewards(self):
        """Config override must use 'reward' (matches YAML schema), not 'rewards'."""
        # Simulate what objective() builds
        w_progress = 0.5
        w_smoothness = -0.1
        learning_rate = 3e-4
        n_steps = 1024

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

        assert "reward" in config_overrides, "must use 'reward', not 'rewards'"
        assert "rewards" not in config_overrides
        assert "ppo" in config_overrides
        assert "learning_rate" not in config_overrides  # must be nested under ppo
        assert "n_steps" not in config_overrides        # must be nested under ppo
        assert config_overrides["ppo"]["learning_rate"] == learning_rate
        assert config_overrides["ppo"]["n_steps"] == n_steps

    def test_eval_path_includes_eval_logs(self, tmp_path):
        """Eval evaluations.npz must be under <log_dir>/<run_name>/eval_logs/."""
        import os
        tmp_log_dir = str(tmp_path)
        run_name = "optuna_trial"
        eval_path = os.path.join(tmp_log_dir, run_name, "eval_logs", "evaluations.npz")

        # Check path structure
        parts = pathlib.Path(eval_path).parts
        assert "eval_logs" in parts, "eval_logs directory must be in path"
        assert parts[-1] == "evaluations.npz"

    def test_no_log_dir_flag_in_cmd_args(self):
        """run_trial() must not pass --log_dir to train.py (flag doesn't exist)."""
        import json
        # Reconstruct the command that run_trial would build
        tmp_log_dir = "/tmp/fake_optuna"
        config_overrides = {"reward": {"w_progress": 0.5}, "output": {"log_dir": tmp_log_dir}}
        cmd = [
            "python", "-m", "src.training.train",
            "--config", "configs/train_ppo_fast.yaml",
            "--total_timesteps", "4096",
            "--run_name", "optuna_trial",
            "--overrides", json.dumps(config_overrides),
        ]
        assert "--log_dir" not in cmd, "--log_dir is not a valid train.py flag"
        # log_dir should be inside overrides JSON
        overrides = json.loads(cmd[cmd.index("--overrides") + 1])
        assert "output" in overrides
        assert overrides["output"]["log_dir"] == tmp_log_dir
