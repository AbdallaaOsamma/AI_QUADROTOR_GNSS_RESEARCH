"""Tests for distance_before_collision with has_collided flag and w_collision=0."""
import pytest

from src.evaluation.metrics import distance_before_collision


def _row(x: float, y: float, reward: float = 0.0, has_collided: bool = False) -> dict:
    return {"x": x, "y": y, "reward": reward, "has_collided": has_collided}


class TestDistanceBeforeCollision:
    def test_stops_on_has_collided_flag(self):
        """has_collided=True should stop integration even when reward > -50."""
        traj = [
            _row(0, 0),
            _row(1, 0),
            _row(2, 0, has_collided=True),  # collision at step 3; reward = 0
            _row(3, 0),  # should NOT be counted
        ]
        result = distance_before_collision(traj)
        # Distance: (0→1) + (1→2) = 2.0; step 3 row triggers stop
        assert result == pytest.approx(2.0)

    def test_stops_on_reward_fallback(self):
        """Large negative reward (<-50) triggers collision without has_collided."""
        traj = [
            _row(0, 0),
            _row(3, 4),          # distance = 5
            _row(3, 4, reward=-100.0),  # collision — large negative reward
            _row(6, 8),
        ]
        result = distance_before_collision(traj)
        assert result == pytest.approx(5.0)

    def test_progress_only_ablation_w_collision_zero(self):
        """When w_collision=0, reward never goes below -50; has_collided is the
        only reliable indicator — the fix must handle this correctly."""
        traj = [
            _row(0, 0),
            _row(1, 0),
            _row(1, 1, reward=0.0, has_collided=True),  # collision, reward=0
            _row(5, 5),  # beyond collision — must not be counted
        ]
        result = distance_before_collision(traj)
        # (0→1) = 1.0 + (1→(1,1)) = 1.0 → total 2.0; collision on 3rd row stops it
        assert result == pytest.approx(2.0)

    def test_no_collision_returns_full_distance(self):
        traj = [_row(0, 0), _row(3, 0), _row(3, 4)]
        result = distance_before_collision(traj)
        assert result == pytest.approx(7.0)

    def test_empty_trajectory(self):
        assert distance_before_collision([]) == pytest.approx(0.0)
