"""Tests that build_model() applies YAML hyperparams after resuming from checkpoint.

Without this, fine-tuning configs (e.g. parking_finetune) silently use the
base checkpoint's LR/clip_range instead of the conservative fine-tune values.
"""
import importlib.util
import pathlib
import unittest.mock as mock

import pytest


def _load_train_module():
    """Load train.py with heavy deps mocked. Returns the module object."""
    spec = importlib.util.spec_from_file_location(
        "train_mod",
        pathlib.Path("src/training/train.py"),
    )
    # get_schedule_fn must be a real identity-like callable so schedule keys work
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


def _make_fake_model(**attrs):
    m = mock.MagicMock()
    defaults = {
        "lr_schedule": lambda _: 3e-4,   # SB3 internal attr — used by _update_learning_rate
        "learning_rate": 3e-4,           # legacy attr (should NOT be touched for LR override)
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps": 2048,
    }
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


class TestResumeHyperparamOverride:
    def test_learning_rate_overridden(self):
        """YAML learning_rate must update lr_schedule (the SB3 internal attr) on resume."""
        mod = _load_train_module()
        fake_model = _make_fake_model()
        mod.PPO.load.return_value = fake_model

        mod.build_model(
            algo="ppo",
            env=mock.MagicMock(),
            cfg={"learning_rate": 5e-5, "total_timesteps": 500_000},
            log_dir="/tmp/test",
            resume="fake_checkpoint.zip",
        )

        # build_model must set lr_schedule (not learning_rate) — SB3 reads lr_schedule
        lr_sched = fake_model.lr_schedule
        actual = lr_sched(1.0) if callable(lr_sched) else lr_sched
        assert actual == pytest.approx(5e-5), (
            f"Expected lr_schedule(1.0)=5e-5 (parking fine-tune value), got {actual}"
        )

    def test_clip_range_overridden(self):
        """YAML clip_range must override the checkpoint value on resume."""
        mod = _load_train_module()
        fake_model = _make_fake_model(clip_range=0.2)
        mod.PPO.load.return_value = fake_model

        result = mod.build_model(
            algo="ppo",
            env=mock.MagicMock(),
            cfg={"clip_range": 0.1, "total_timesteps": 500_000},
            log_dir="/tmp/test",
            resume="fake_checkpoint.zip",
        )

        cr = result.clip_range
        actual = cr(1.0) if callable(cr) else cr
        assert actual == pytest.approx(0.1)

    def test_n_steps_not_overridden(self):
        """n_steps must NOT be overridden — rollout buffer was allocated at load time."""
        mod = _load_train_module()
        fake_model = _make_fake_model(n_steps=2048)
        mod.PPO.load.return_value = fake_model

        mod.build_model(
            algo="ppo",
            env=mock.MagicMock(),
            cfg={"n_steps": 1024, "total_timesteps": 500_000},
            log_dir="/tmp/test",
            resume="fake_checkpoint.zip",
        )

        # n_steps should still be the checkpoint value — setattr was never called
        assert fake_model.n_steps == 2048, (
            "n_steps should NOT be overridden (rollout buffer already allocated)"
        )

    def test_multiple_params_overridden_together(self):
        """All safe override keys from parking_finetune.yaml must apply."""
        mod = _load_train_module()
        fake_model = _make_fake_model()
        mod.PPO.load.return_value = fake_model

        mod.build_model(
            algo="ppo",
            env=mock.MagicMock(),
            cfg={
                "learning_rate": 5e-5,
                "clip_range": 0.1,
                "ent_coef": 0.02,
                "n_steps": 1024,       # must be excluded
                "total_timesteps": 500_000,
            },
            log_dir="/tmp/test",
            resume="fake_checkpoint.zip",
        )

        lr = fake_model.lr_schedule
        assert (lr(1.0) if callable(lr) else lr) == pytest.approx(5e-5)
        cr = fake_model.clip_range
        assert (cr(1.0) if callable(cr) else cr) == pytest.approx(0.1)
        assert fake_model.ent_coef == pytest.approx(0.02)
        # n_steps must NOT have been set to 1024
        assert fake_model.n_steps == 2048

    def test_no_resume_uses_yaml_cfg_normally(self):
        """Without --resume, YAML cfg is passed to PPO constructor as before."""
        mod = _load_train_module()
        mod.PPO.reset_mock()

        mod.build_model(
            algo="ppo",
            env=mock.MagicMock(),
            cfg={"learning_rate": 3e-4, "n_steps": 2048, "total_timesteps": 1_000_000},
            log_dir="/tmp/test",
            resume=None,
        )

        call_kwargs = mod.PPO.call_args[1]
        assert call_kwargs["learning_rate"] == pytest.approx(3e-4)
        assert call_kwargs["n_steps"] == 2048
        assert "total_timesteps" not in call_kwargs  # filtered out by _caller_keys
