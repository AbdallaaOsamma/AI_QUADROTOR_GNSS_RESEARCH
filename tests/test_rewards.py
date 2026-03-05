"""Tests for pluggable reward functions.

Runs without AirSim — tests pure reward computation logic.
"""
import numpy as np
import pytest

from src.environments.rewards import RewardFunction


class TestRewardFunction:
    def setup_method(self):
        self.reward_fn = RewardFunction({
            "w_progress": 0.5,
            "w_collision": -100.0,
            "w_smoothness": -0.1,
        })

    def test_forward_progress_positive(self):
        """Positive vx_body should yield positive progress reward."""
        reward, info = self.reward_fn(
            vx_body=2.0,
            has_collided=False,
            action=np.array([0.5, 0.0, 0.0]),
            prev_action=np.array([0.5, 0.0, 0.0]),
        )
        assert info["r_progress"] == pytest.approx(1.0)  # 0.5 * 2.0
        assert info["r_collision"] == 0.0
        assert reward > 0

    def test_backward_motion_negative(self):
        """Negative vx_body should yield negative progress reward."""
        reward, info = self.reward_fn(
            vx_body=-1.0,
            has_collided=False,
            action=np.array([-0.3, 0.0, 0.0]),
            prev_action=np.array([-0.3, 0.0, 0.0]),
        )
        assert info["r_progress"] == pytest.approx(-0.5)

    def test_collision_penalty(self):
        """Collision should apply large negative penalty."""
        reward, info = self.reward_fn(
            vx_body=2.0,
            has_collided=True,
            action=np.array([0.5, 0.0, 0.0]),
            prev_action=np.array([0.5, 0.0, 0.0]),
        )
        assert info["r_collision"] == pytest.approx(-100.0)
        assert reward < 0

    def test_smoothness_penalty(self):
        """Large action change should incur smoothness penalty."""
        reward, info = self.reward_fn(
            vx_body=0.0,
            has_collided=False,
            action=np.array([1.0, 0.0, 0.0]),
            prev_action=np.array([-1.0, 0.0, 0.0]),
        )
        # ||[1,0,0] - [-1,0,0]|| = 2.0, penalty = -0.1 * 2.0 = -0.2
        assert info["r_smoothness"] == pytest.approx(-0.2)

    def test_zero_action_no_smoothness_penalty(self):
        """Identical consecutive actions should have zero smoothness penalty."""
        action = np.array([0.3, 0.2, 0.1])
        _, info = self.reward_fn(
            vx_body=1.0,
            has_collided=False,
            action=action,
            prev_action=action.copy(),
        )
        assert info["r_smoothness"] == pytest.approx(0.0)

    def test_default_config(self):
        """RewardFunction with no config should use sensible defaults."""
        rf = RewardFunction()
        assert rf.w_progress == 0.5
        assert rf.w_collision == -100.0
        assert rf.w_smoothness == -0.1

    def test_custom_weights(self):
        """Custom weights should override defaults."""
        rf = RewardFunction({"w_progress": 1.0, "w_collision": -50.0})
        reward, info = rf(
            vx_body=3.0,
            has_collided=True,
            action=np.zeros(3),
            prev_action=np.zeros(3),
        )
        assert info["r_progress"] == pytest.approx(3.0)
        assert info["r_collision"] == pytest.approx(-50.0)

    def test_reward_decomposition_sums(self):
        """Total reward should equal sum of components."""
        reward, info = self.reward_fn(
            vx_body=1.5,
            has_collided=False,
            action=np.array([0.4, 0.1, 0.2]),
            prev_action=np.array([0.3, 0.0, 0.1]),
        )
        expected = info["r_progress"] + info["r_collision"] + info["r_smoothness"]
        assert reward == pytest.approx(expected)


def test_drift_penalty_nonzero_when_drift_present():
    """Reward should include drift penalty when drift_error > 0 and w_drift != 0."""
    rf = RewardFunction({"w_progress": 0.5, "w_collision": -100.0,
                         "w_smoothness": -0.1, "w_drift": -0.5})
    action = np.zeros(3, dtype=np.float32)
    reward, info = rf(
        vx_body=1.0, has_collided=False,
        action=action, prev_action=action, drift_error=0.5
    )
    assert info.get("r_drift", 0.0) != 0.0
    assert info["r_drift"] == pytest.approx(-0.25)  # -0.5 * 0.5


def test_drift_penalty_zero_when_weight_zero():
    """With w_drift=0, drift penalty is absent regardless of drift_error."""
    rf = RewardFunction({"w_progress": 0.5, "w_collision": -100.0,
                         "w_smoothness": -0.1, "w_drift": 0.0})
    action = np.zeros(3, dtype=np.float32)
    reward, info = rf(
        vx_body=1.0, has_collided=False,
        action=action, prev_action=action, drift_error=2.0
    )
    assert info.get("r_drift", 0.0) == 0.0
