"""Standardized evaluation metrics for navigation performance.

All functions take trajectory data (lists of dicts) and return scalar metrics.
Designed to be used without AirSim — pure computation on recorded telemetry.
"""
from __future__ import annotations

import math

import numpy as np


def distance_before_collision(trajectory: list[dict]) -> float:
    """Total distance traveled (meters) before first collision.

    If no collision occurred, returns total distance for the trajectory.
    """
    total = 0.0
    prev = None
    for row in trajectory:
        x, y = row["x"], row["y"]
        if prev is not None:
            total += math.sqrt((x - prev[0]) ** 2 + (y - prev[1]) ** 2)
        prev = (x, y)
        # Collision detected by large negative reward
        if row.get("reward", 0) < -50:
            break
    return total


def collision_rate(episodes: list[dict]) -> float:
    """Fraction of episodes that ended in collision.

    Args:
        episodes: list of episode summary dicts with 'collided' bool key
    """
    if not episodes:
        return 0.0
    return sum(1 for ep in episodes if ep.get("collided", False)) / len(episodes)


def average_speed(trajectory: list[dict], dt: float = 0.1) -> float:
    """Average ground speed (m/s) over the trajectory."""
    total_dist = 0.0
    prev = None
    for row in trajectory:
        x, y = row["x"], row["y"]
        if prev is not None:
            total_dist += math.sqrt((x - prev[0]) ** 2 + (y - prev[1]) ** 2)
        prev = (x, y)
    duration = len(trajectory) * dt
    return total_dist / duration if duration > 0 else 0.0


def path_smoothness(trajectory: list[dict], dt: float = 0.1) -> float:
    """Mean absolute jerk (m/s^3) — lower is smoother.

    Computed from velocity changes across consecutive steps.
    """
    if len(trajectory) < 3:
        return 0.0

    velocities = []
    prev = None
    for row in trajectory:
        x, y = row["x"], row["y"]
        if prev is not None:
            vx = (x - prev[0]) / dt
            vy = (y - prev[1]) / dt
            velocities.append((vx, vy))
        prev = (x, y)

    if len(velocities) < 2:
        return 0.0

    jerks = []
    for i in range(1, len(velocities)):
        ax = (velocities[i][0] - velocities[i - 1][0]) / dt
        ay = (velocities[i][1] - velocities[i - 1][1]) / dt
        jerks.append(math.sqrt(ax ** 2 + ay ** 2))

    return float(np.mean(jerks))


def survival_time(trajectory: list[dict], dt: float = 0.1) -> float:
    """Time (seconds) before episode termination."""
    return len(trajectory) * dt


def time_to_goal(trajectory: list[dict], dt: float = 0.1, reached: bool = False) -> float | None:
    """Time (seconds) to reach the goal waypoint, or None if not reached.

    When goal navigation is active, this measures elapsed time from episode
    start until the final waypoint is reached.  Episodes that end in collision
    or timeout return None so that aggregation can distinguish succeeded runs.

    Args:
        trajectory: list of step dicts recorded during the episode
        dt: control period in seconds
        reached: True when the mission_success flag is set for this episode
    """
    if not reached:
        return None
    return round(len(trajectory) * dt, 2)


def trajectory_rmse(
    trajectory: list[dict],
    goal_x: float,
    goal_y: float,
) -> float:
    """Root Mean Square Error of positions from the ideal straight-line path to goal.

    Computes the perpendicular distance of each recorded position from the
    straight line drawn between the episode start position and the goal
    waypoint, then returns the RMSE of those distances.  A lower value means
    the drone flew a more direct path with less lateral deviation.

    Args:
        trajectory: list of position dicts with 'x' and 'y' keys
        goal_x: x coordinate of the goal waypoint (metres, AirSim NED)
        goal_y: y coordinate of the goal waypoint (metres, AirSim NED)
    """
    if len(trajectory) < 2:
        return 0.0

    start_x = trajectory[0]["x"]
    start_y = trajectory[0]["y"]

    dx = goal_x - start_x
    dy = goal_y - start_y
    path_length = math.sqrt(dx ** 2 + dy ** 2)

    if path_length < 1e-6:
        return 0.0

    # Unit vector along the ideal path
    ux, uy = dx / path_length, dy / path_length

    # Perpendicular distance from each position to the ideal line
    errors_sq = []
    for row in trajectory:
        px = row["x"] - start_x
        py = row["y"] - start_y
        perp = abs(px * uy - py * ux)  # cross-product magnitude
        errors_sq.append(perp ** 2)

    return round(float(math.sqrt(np.mean(errors_sq))), 3)


def localisation_drift(trajectory: list[dict]) -> float:
    """Mean absolute localisation drift (metres) relative to ground truth.

    Measures the GNSS-denied state estimation error: the average L2 distance
    between VIO-estimated position (from optical flow / inertial odometry) and
    AirSim ground-truth position at each step.

    Args:
        trajectory: list of step dicts with 'x_gt', 'y_gt' (AirSim ground truth)
                    and 'x_est', 'y_est' (VIO-estimated position) keys.

    Returns:
        Mean positional error in metres.  Returns 0.0 if estimation data is
        absent (e.g. when running on AirSim ground-truth kinematics directly).

    Note:
        This metric requires a VIO pipeline to produce 'x_est'/'y_est' values.
        Until Known Issue #10 (VIO pipeline) is implemented, trajectory dicts
        will not contain estimation data and this function will return 0.0.
    """
    errors = []
    for row in trajectory:
        if "x_est" in row and "y_est" in row and "x_gt" in row and "y_gt" in row:
            dx = row["x_est"] - row["x_gt"]
            dy = row["y_est"] - row["y_gt"]
            errors.append(math.sqrt(dx ** 2 + dy ** 2))
    return round(float(np.mean(errors)), 3) if errors else 0.0


def goal_completion_rate(episodes: list[dict]) -> float:
    """Fraction of waypoints reached across all episodes.

    Args:
        episodes: list of episode summary dicts with 'goals_reached' and
                  'total_goals' int keys (added by waypoint evaluation).
                  Episodes without these keys contribute 0 numerator/denominator.
    """
    total_reached = sum(ep.get("goals_reached_count", 0) for ep in episodes)
    total_spawned = sum(ep.get("total_goals_count", 0) for ep in episodes)
    if total_spawned == 0:
        return 0.0
    return total_reached / total_spawned


def compute_episode_summary(
    trajectory: list[dict],
    dt: float = 0.1,
    collided: bool = False,
    goals_reached_count: int = 0,
    total_goals_count: int = 0,
    mission_success_flag: bool = False,
    goal_x: float | None = None,
    goal_y: float | None = None,
) -> dict:
    """Compute all metrics for a single episode.

    Backward-compatible: existing callers without waypoint args are unaffected.
    Returns a dict suitable for JSON serialization.

    Args:
        trajectory: list of step dicts with at least 'x', 'y', 'reward' keys
        dt: control period in seconds
        collided: whether the episode ended in a collision
        goals_reached_count: number of waypoints reached this episode
        total_goals_count: total waypoints spawned this episode
        mission_success_flag: True when all waypoints were reached
        goal_x: x coordinate of the final goal waypoint (for trajectory RMSE)
        goal_y: y coordinate of the final goal waypoint (for trajectory RMSE)
    """
    summary = {
        "distance_before_collision_m": round(distance_before_collision(trajectory), 2),
        "average_speed_ms": round(average_speed(trajectory, dt), 3),
        "path_smoothness_jerk": round(path_smoothness(trajectory, dt), 3),
        "survival_time_s": round(survival_time(trajectory, dt), 2),
        "collided": collided,
        "total_steps": len(trajectory),
    }
    # Waypoint metrics — only included when goal navigation is active
    if total_goals_count > 0:
        summary["goals_reached_count"] = goals_reached_count
        summary["total_goals_count"] = total_goals_count
        summary["mission_success"] = mission_success_flag
        # time_to_goal: seconds to complete all waypoints (None if not reached)
        summary["time_to_goal_s"] = time_to_goal(trajectory, dt, mission_success_flag)
        # trajectory_rmse: lateral deviation from ideal straight-line path
        if goal_x is not None and goal_y is not None:
            summary["trajectory_rmse_m"] = trajectory_rmse(trajectory, goal_x, goal_y)
    return summary
