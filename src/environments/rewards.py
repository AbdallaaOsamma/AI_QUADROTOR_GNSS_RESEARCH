"""Pluggable reward functions for the AirSim drone environment."""
from __future__ import annotations

from typing import Optional

import numpy as np


class RewardFunction:
    """Computes per-step reward from environment state.

    Loaded from YAML config under the `reward` key. All weights default
    to values proven in initial training runs.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.w_progress = cfg.get("w_progress", 0.5)
        self.w_collision = cfg.get("w_collision", -100.0)
        self.w_smoothness = cfg.get("w_smoothness", -0.1)
        self.w_drift = cfg.get("w_drift", 0.0)
        self.w_proximity = cfg.get("w_proximity", 0.0)
        # Normalized depth threshold below which proximity penalty activates.
        # At depth_clip_m=8m: 0.25 = 2m, 0.1875 = 1.5m, 0.125 = 1m.
        self.proximity_threshold = cfg.get("proximity_threshold", 0.25)

    def __call__(
        self,
        vx_body: float,
        has_collided: bool,
        action: np.ndarray,
        prev_action: np.ndarray,
        drift_error: float = 0.0,
        min_depth: float = 1.0,
    ) -> tuple[float, dict]:
        """Return (total_reward, info_dict) for a single step."""
        r_progress = self.w_progress * vx_body
        r_collision = self.w_collision if has_collided else 0.0
        r_smoothness = self.w_smoothness * float(np.linalg.norm(action - prev_action))
        r_drift = self.w_drift * drift_error
        r_proximity = (
            -self.w_proximity * max(0.0, self.proximity_threshold - min_depth)
            if self.w_proximity != 0.0 else 0.0
        )

        total = r_progress + r_collision + r_smoothness + r_drift + r_proximity

        info = {
            "r_progress": r_progress,
            "r_collision": r_collision,
            "r_smoothness": r_smoothness,
            "r_drift": r_drift,
            "r_proximity": r_proximity,
        }
        return total, info


class WaypointRewardFunction:
    """Goal-directed reward function for multi-waypoint navigation.

    Reward components:
    - heading:   cos(bearing_to_goal) — continuous 360° gradient for yaw learning
    - dist:      delta-based (prev_dist_norm - curr_dist_norm) — penalises hovering
    - goal:      per-waypoint arrival bonus
    - mission:   all-waypoints completion bonus
    - collision: terminal collision penalty
    - smoothness: action-change penalty

    All weights configurable via YAML config under the `reward` key.
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.w_heading    = cfg.get("w_heading",    0.5)
        self.w_progress   = cfg.get("w_progress",   0.3)
        self.w_dist       = cfg.get("w_dist",       0.3)
        self.w_goal       = cfg.get("w_goal",      10.0)
        self.w_mission    = cfg.get("w_mission",   50.0)
        self.w_collision  = cfg.get("w_collision", -100.0)
        self.w_smoothness = cfg.get("w_smoothness", -0.1)

        # Tracks previous normalised distance for delta computation.
        # Reset to None after waypoint is reached to avoid spurious delta.
        self._prev_dist_norm: Optional[float] = None

    def __call__(
        self,
        vx_body: float,
        has_collided: bool,
        action: np.ndarray,
        prev_action: np.ndarray,
        goal_reached: bool = False,
        dist_norm: float = 1.0,
        cos_theta: float = 0.0,
        all_goals_done: bool = False,
    ) -> tuple[float, dict]:
        """Return (total_reward, info_dict) for a single step."""
        # Heading reward: cos(bearing) in body frame — max when facing goal
        r_heading = self.w_heading * cos_theta
        # Progress reward: only forward flight rewarded — backward gives zero
        r_progress = self.w_progress * max(0.0, vx_body)

        # Distance reward: delta-based shaping
        if self._prev_dist_norm is None:
            r_dist = 0.0
        else:
            r_dist = self.w_dist * (self._prev_dist_norm - dist_norm)

        # Update prev dist — reset on waypoint arrival so next step gets no delta
        if goal_reached:
            self._prev_dist_norm = None
        else:
            self._prev_dist_norm = dist_norm

        # Goal / mission bonuses
        r_goal = self.w_goal if goal_reached else 0.0
        r_mission = self.w_mission if all_goals_done else 0.0

        # Collision and smoothness (same as baseline)
        r_collision = self.w_collision if has_collided else 0.0
        r_smoothness = self.w_smoothness * float(np.linalg.norm(action - prev_action))

        total = r_heading + r_progress + r_dist + r_goal + r_mission + r_collision + r_smoothness

        info = {
            "r_heading":    r_heading,
            "r_progress":   r_progress,
            "r_dist":       r_dist,
            "r_goal":       r_goal,
            "r_mission":    r_mission,
            "r_collision":  r_collision,
            "r_smoothness": r_smoothness,
        }
        return total, info
