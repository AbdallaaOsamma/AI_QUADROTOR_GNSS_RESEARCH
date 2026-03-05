"""Tests for evaluation metrics.

Runs without AirSim — tests pure computation on trajectory data.
"""
import pytest

from src.evaluation.metrics import (
    average_speed,
    collision_rate,
    compute_episode_summary,
    distance_before_collision,
    localisation_drift,
    path_smoothness,
    survival_time,
)


def _make_trajectory(n=10, dx=1.0, collision_step=None):
    """Generate a simple straight-line trajectory."""
    traj = []
    for i in range(n):
        reward = -100.0 if (collision_step is not None and i == collision_step) else 0.5
        traj.append({"x": i * dx, "y": 0.0, "z": -3.0, "reward": reward})
    return traj


class TestDistanceBeforeCollision:
    def test_no_collision(self):
        traj = _make_trajectory(10, dx=1.0)
        dbc = distance_before_collision(traj)
        assert dbc == pytest.approx(9.0)  # 9 segments of 1m each

    def test_collision_at_step_5(self):
        traj = _make_trajectory(10, dx=1.0, collision_step=5)
        dbc = distance_before_collision(traj)
        assert dbc == pytest.approx(5.0)  # stops at step 5

    def test_empty_trajectory(self):
        assert distance_before_collision([]) == pytest.approx(0.0)


class TestCollisionRate:
    def test_no_collisions(self):
        episodes = [{"collided": False}] * 5
        assert collision_rate(episodes) == pytest.approx(0.0)

    def test_all_collisions(self):
        episodes = [{"collided": True}] * 5
        assert collision_rate(episodes) == pytest.approx(1.0)

    def test_half_collisions(self):
        episodes = [{"collided": True}, {"collided": False}] * 3
        assert collision_rate(episodes) == pytest.approx(0.5)

    def test_empty_episodes(self):
        assert collision_rate([]) == pytest.approx(0.0)


class TestAverageSpeed:
    def test_constant_speed(self):
        traj = _make_trajectory(11, dx=1.0)  # 10 segments, 11 points
        speed = average_speed(traj, dt=0.1)
        # 10m in 1.1s = 9.09 m/s
        assert speed == pytest.approx(10.0 / 1.1, abs=0.01)

    def test_stationary(self):
        traj = _make_trajectory(10, dx=0.0)
        assert average_speed(traj, dt=0.1) == pytest.approx(0.0)


class TestPathSmoothness:
    def test_straight_line_low_jerk(self):
        """Constant velocity should have zero jerk."""
        traj = _make_trajectory(20, dx=1.0)
        jerk = path_smoothness(traj, dt=0.1)
        assert jerk == pytest.approx(0.0, abs=1e-6)

    def test_short_trajectory(self):
        traj = _make_trajectory(2, dx=1.0)
        assert path_smoothness(traj, dt=0.1) == pytest.approx(0.0)


class TestSurvivalTime:
    def test_basic(self):
        traj = _make_trajectory(100)
        assert survival_time(traj, dt=0.1) == pytest.approx(10.0)


class TestLocalisationDrift:
    def test_localisation_drift_nonzero_when_data_present(self):
        """localisation_drift returns nonzero when est != gt."""
        trajectory = [
            {"x_gt": 0.0, "y_gt": 0.0, "x_est": 0.1, "y_est": 0.05},
            {"x_gt": 1.0, "y_gt": 0.0, "x_est": 1.2, "y_est": 0.1},
            {"x_gt": 2.0, "y_gt": 0.0, "x_est": 2.5, "y_est": 0.2},
        ]
        result = localisation_drift(trajectory)
        assert result > 0.0

    def test_localisation_drift_zero_when_perfect(self):
        """localisation_drift returns 0 when est == gt."""
        trajectory = [
            {"x_gt": 0.0, "y_gt": 0.0, "x_est": 0.0, "y_est": 0.0},
            {"x_gt": 1.0, "y_gt": 0.5, "x_est": 1.0, "y_est": 0.5},
        ]
        assert localisation_drift(trajectory) == 0.0


class TestEpisodeSummary:
    def test_summary_keys(self):
        traj = _make_trajectory(50, dx=0.5)
        summary = compute_episode_summary(traj, dt=0.1, collided=False)
        assert "distance_before_collision_m" in summary
        assert "average_speed_ms" in summary
        assert "path_smoothness_jerk" in summary
        assert "survival_time_s" in summary
        assert summary["collided"] is False
        assert summary["total_steps"] == 50
