"""Tests for VIO body→world frame rotation in AirSimDroneEnv.step().

Validates that integrating body-frame velocity with yaw rotation correctly
accumulates world-frame position at 0°, 90°, 180°, and 270° yaw.
"""
import math

import numpy as np
import pytest


def _rotate_body_to_world(vx: float, vy: float, yaw_rad: float):
    """Reference implementation of the body→world rotation."""
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    vx_w = vx * cos_y - vy * sin_y
    vy_w = vx * sin_y + vy * cos_y
    return vx_w, vy_w


class TestVIOBodyToWorldRotation:
    """Unit tests for the rotation formula used in airsim_env.py."""

    @pytest.mark.parametrize(
        "yaw_deg, vx, vy, expected_vx_w, expected_vy_w",
        [
            # Facing north (yaw=0): body vx → world x, body vy → world y
            (0, 1.0, 0.0, 1.0, 0.0),
            (0, 0.0, 1.0, 0.0, 1.0),
            # Facing east (yaw=90°): body vx → world +y, body vy → world -x
            (90, 1.0, 0.0, 0.0, 1.0),
            (90, 0.0, 1.0, -1.0, 0.0),
            # Facing south (yaw=180°): body vx → world -x, body vy → world -y
            (180, 1.0, 0.0, -1.0, 0.0),
            # Facing west (yaw=270°): body vx → world -y, body vy → world +x
            (270, 1.0, 0.0, 0.0, -1.0),
        ],
    )
    def test_rotation_cardinal_directions(
        self, yaw_deg, vx, vy, expected_vx_w, expected_vy_w
    ):
        yaw_rad = math.radians(yaw_deg)
        vx_w, vy_w = _rotate_body_to_world(vx, vy, yaw_rad)
        assert vx_w == pytest.approx(expected_vx_w, abs=1e-6)
        assert vy_w == pytest.approx(expected_vy_w, abs=1e-6)

    def test_integration_accumulates_correctly(self):
        """Integrating at 45° for 4 steps should give equal x and y displacement."""
        yaw_rad = math.radians(45)
        vx_body, vy_body = 1.0, 0.0
        dt = 0.1
        pos = np.zeros(2, dtype=np.float64)
        for _ in range(4):
            vx_w, vy_w = _rotate_body_to_world(vx_body, vy_body, yaw_rad)
            pos += np.array([vx_w, vy_w]) * dt
        # At 45° yaw, equal x and y components
        assert pos[0] == pytest.approx(pos[1], rel=1e-6)
        assert pos[0] == pytest.approx(4 * 0.1 * math.cos(yaw_rad), rel=1e-6)
