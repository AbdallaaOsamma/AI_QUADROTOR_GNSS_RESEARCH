"""Tests for the step-ceiling guard in train.main().

Verifies that main() raises SystemExit when a resumed checkpoint has already
reached or exceeded total_timesteps, rather than silently running zero steps.

Uses the same module-loading pattern as test_train_resume_hyperparams.py to
avoid importing heavy deps (torch, stable_baselines3, matplotlib) at test time.
"""
import importlib.util
import pathlib
import sys
import unittest.mock as mock

import pytest


def _load_train_module():
    """Load train.py with heavy deps mocked. Returns the module object."""
    spec = importlib.util.spec_from_file_location(
        "train_mod_ceiling",
        pathlib.Path("src/training/train.py"),
    )
    fake_schedule_fn = mock.MagicMock(side_effect=lambda x: (lambda _: x) if not callable(x) else x)
    fake_utils = mock.MagicMock()
    fake_utils.get_schedule_fn = fake_schedule_fn

    with mock.patch.dict("sys.modules", {
        "yaml": mock.MagicMock(),
        "stable_baselines3": mock.MagicMock(),
        "stable_baselines3.common": mock.MagicMock(),
        "stable_baselines3.common.callbacks": mock.MagicMock(),
        "stable_baselines3.common.monitor": mock.MagicMock(),
        "stable_baselines3.common.vec_env": mock.MagicMock(),
        "stable_baselines3.common.utils": fake_utils,
        "src.environments.airsim_env": mock.MagicMock(),
        "src.training.callbacks": mock.MagicMock(),
        "src.training.env_scheduler": mock.MagicMock(),
    }):
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def _run_main_with_resume(mod, ckpt_steps: int, total_timesteps: int):
    """Call mod.main() patched to reach the step-ceiling check."""
    minimal_cfg = {
        "ppo": {
            "total_timesteps": total_timesteps,
            "learning_rate": 3e-4,
            "n_steps": 64,
            "batch_size": 16,
            "n_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "output": {
            "log_dir": "/tmp/test_logs",
            "checkpoint_freq": 10000,
            "eval_freq": 5000,
            "eval_episodes": 1,
        },
        "frame_stack": 4,
    }

    fake_model = mock.MagicMock()
    fake_model.num_timesteps = ckpt_steps

    fake_vec_env = mock.MagicMock()

    test_argv = [
        "train.py",
        "--config", "configs/train_ppo.yaml",
        "--resume", "fake_checkpoint.zip",
        "--no_eval",
    ]

    # yaml.safe_load is already mocked at module import, but we need to set its
    # return_value on the instance that train.py actually calls.
    fake_yaml = mock.MagicMock()
    fake_yaml.safe_load.return_value = minimal_cfg

    with mock.patch.object(sys, "argv", test_argv), \
         mock.patch.object(mod, "yaml", fake_yaml), \
         mock.patch("builtins.open", mock.mock_open(read_data="")), \
         mock.patch.object(mod, "launch_airsim_if_needed"), \
         mock.patch.object(mod, "make_vec_env", return_value=fake_vec_env), \
         mock.patch.object(mod, "VecFrameStack", return_value=fake_vec_env), \
         mock.patch.object(mod, "build_model", return_value=fake_model), \
         mock.patch.object(mod, "_is_port_open", return_value=False), \
         mock.patch("os.makedirs"):
        mod.main()


def test_resume_past_ceiling_exits():
    """main() must raise SystemExit when checkpoint >= total_timesteps."""
    mod = _load_train_module()
    with pytest.raises(SystemExit) as exc_info:
        _run_main_with_resume(mod, ckpt_steps=1_000_000, total_timesteps=975_000)

    assert "total_timesteps" in str(exc_info.value), (
        "SystemExit message should mention total_timesteps"
    )


def test_resume_exactly_at_ceiling_exits():
    """main() must raise SystemExit even when checkpoint == total_timesteps exactly."""
    mod = _load_train_module()
    with pytest.raises(SystemExit):
        _run_main_with_resume(mod, ckpt_steps=500_000, total_timesteps=500_000)


def test_resume_below_ceiling_prints_remaining(capsys):
    """main() must print remaining steps and NOT raise SystemExit when steps remain."""
    mod = _load_train_module()
    try:
        _run_main_with_resume(mod, ckpt_steps=400_000, total_timesteps=975_000)
    except SystemExit:
        pytest.fail("main() must not raise SystemExit when there are remaining steps")
    except Exception:
        # Training loop will error out without a real environment — that's fine.
        pass

    captured = capsys.readouterr()
    assert "Remaining: 575000 steps" in captured.out, (
        f"main() should print remaining steps after resume. Got:\n{captured.out}"
    )
