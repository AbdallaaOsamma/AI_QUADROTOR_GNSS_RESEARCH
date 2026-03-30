"""Tests for pluggable reward functions.

Runs without AirSim — tests pure reward computation logic.
"""
import numpy as np
import pytest

from src.environments.rewards import RewardFunction, WaypointRewardFunction


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


def test_proximity_penalty_activates_when_close():
    """Proximity penalty fires when min_depth is below threshold."""
    rf = RewardFunction({"w_proximity": 2.0, "proximity_threshold": 0.25})
    action = np.zeros(3, dtype=np.float32)
    # min_depth=0.125 => 1m at 8m clip => penalty = -2.0 * (0.25 - 0.125) = -0.25
    _, info = rf(vx_body=0.0, has_collided=False, action=action,
                 prev_action=action, min_depth=0.125)
    assert info["r_proximity"] == pytest.approx(-0.25)


def test_proximity_penalty_zero_when_far():
    """No proximity penalty when min_depth is above threshold."""
    rf = RewardFunction({"w_proximity": 2.0, "proximity_threshold": 0.25})
    action = np.zeros(3, dtype=np.float32)
    _, info = rf(vx_body=0.0, has_collided=False, action=action,
                 prev_action=action, min_depth=0.5)
    assert info["r_proximity"] == pytest.approx(0.0)


def test_proximity_penalty_zero_when_weight_zero():
    """No proximity penalty when w_proximity=0 (default, backward-compatible)."""
    rf = RewardFunction()
    action = np.zeros(3, dtype=np.float32)
    _, info = rf(vx_body=1.0, has_collided=False, action=action,
                 prev_action=action, min_depth=0.0)
    assert info["r_proximity"] == pytest.approx(0.0)


def test_proximity_included_in_total():
    """Total reward must equal sum of all components including proximity."""
    rf = RewardFunction({"w_proximity": 2.0, "proximity_threshold": 0.25})
    action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
    reward, info = rf(vx_body=1.0, has_collided=False, action=action,
                      prev_action=action, min_depth=0.1)
    expected = (info["r_progress"] + info["r_collision"] + info["r_smoothness"]
                + info["r_drift"] + info["r_proximity"])
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


def test_waypoint_reward_backward_motion_zero_progress():
    """WaypointRewardFunction must clamp backward-flight progress reward to 0.

    The base RewardFunction gives negative progress for negative vx_body, but
    WaypointRewardFunction uses max(0, vx_body) to prevent rewarding backward
    flight as an avoidance strategy. If this guard is ever removed, catastrophic
    forgetting of forward-navigation behaviour can occur.
    """
    rf = WaypointRewardFunction({"w_progress": 0.3})
    action = np.zeros(3, dtype=np.float32)
    _, info = rf(
        vx_body=-1.0,
        has_collided=False,
        action=action,
        prev_action=action,
    )
    assert info["r_progress"] == pytest.approx(0.0), (
        "WaypointRewardFunction must clamp backward-flight progress reward to 0 (not negative)"
    )
