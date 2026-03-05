"""Tests for the build_model() factory and --algo CLI flag in train.py."""
import gymnasium
import numpy as np
import pytest
from gymnasium import spaces


class _MinimalEnv(gymnasium.Env):
    """Minimal Gymnasium-compliant environment for testing build_model()."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0.0, 1.0, shape=(84, 84, 4), dtype=np.float32),
                "velocity": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs = {
            "image": np.zeros((84, 84, 4), dtype=np.float32),
            "velocity": np.zeros(3, dtype=np.float32),
        }
        return obs, {}

    def step(self, action):
        obs = {
            "image": np.zeros((84, 84, 4), dtype=np.float32),
            "velocity": np.zeros(3, dtype=np.float32),
        }
        return obs, 0.0, False, False, {}


def test_build_model_ppo(tmp_path):
    """build_model returns a PPO instance for algo='ppo'."""
    from src.training.train import build_model

    env = _MinimalEnv()
    cfg = {
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
    }
    from stable_baselines3 import PPO

    model = build_model("ppo", env, cfg, log_dir=str(tmp_path))
    assert isinstance(model, PPO)


def test_build_model_sac(tmp_path):
    """build_model returns a SAC instance for algo='sac'."""
    from src.training.train import build_model

    env = _MinimalEnv()
    from stable_baselines3 import SAC

    cfg = {
        "learning_rate": 3e-4,
        "buffer_size": 1000,
        "batch_size": 16,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": "auto",
    }
    model = build_model("sac", env, cfg, log_dir=str(tmp_path))
    assert isinstance(model, SAC)


def test_build_model_invalid():
    """build_model raises ValueError for unknown algorithm."""
    from src.training.train import build_model

    with pytest.raises(ValueError, match="Unknown algorithm"):
        build_model("ddpg", None, {}, log_dir="/tmp")
