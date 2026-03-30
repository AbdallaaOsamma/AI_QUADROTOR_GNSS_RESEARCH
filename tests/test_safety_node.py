"""Tests for safety_monitor.safety_node — no ROS2 required.

Tests focus on the pure apply_safety() function which contains all
the safety logic. The ROS2 node wrapper is import-tested separately.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_module_importable():
    """SafetyNode class is importable when ROS2 packages are mocked."""
    ros_mocks = {
        "rclpy": MagicMock(),
        "rclpy.node": MagicMock(),
        "sensor_msgs": MagicMock(),
        "sensor_msgs.msg": MagicMock(),
        "geometry_msgs": MagicMock(),
        "geometry_msgs.msg": MagicMock(),
        "std_msgs": MagicMock(),
        "std_msgs.msg": MagicMock(),
    }
    with patch.dict(sys.modules, ros_mocks):
        mod = importlib.import_module(
            "ros_ws.src.safety_monitor.safety_monitor.safety_node"
        )
        assert hasattr(mod, "SafetyNode")
        assert hasattr(mod, "apply_safety")
        assert hasattr(mod, "main")


def test_estop_zeroes_all_commands():
    """E-stop signal must zero all velocity outputs regardless of input."""
    from ros_ws.src.safety_monitor.safety_monitor.safety_node import apply_safety

    vx, vy, yaw = apply_safety(3.0, 1.0, 0.5, depth_m=None, estop=True)
    assert vx == 0.0
    assert vy == 0.0
    assert yaw == 0.0


def test_velocity_clamping():
    """Velocities exceeding limits must be clamped to max values."""
    from ros_ws.src.safety_monitor.safety_monitor.safety_node import apply_safety

    vx, vy, yaw = apply_safety(
        10.0,
        5.0,
        10.0,
        depth_m=None,
        estop=False,
        max_vx=3.0,
        max_vy=1.0,
        max_yaw_rate_rad=np.deg2rad(45.0),
    )
    assert vx == pytest.approx(3.0)
    assert vy == pytest.approx(1.0)
    assert yaw == pytest.approx(np.deg2rad(45.0))


def test_velocity_clamping_negative():
    """Negative velocities exceeding limits must be clamped symmetrically."""
    from ros_ws.src.safety_monitor.safety_monitor.safety_node import apply_safety

    vx, vy, yaw = apply_safety(
        -10.0,
        -5.0,
        -10.0,
        depth_m=None,
        estop=False,
        max_vx=3.0,
        max_vy=1.0,
        max_yaw_rate_rad=np.deg2rad(45.0),
    )
    assert vx == pytest.approx(-3.0)
    assert vy == pytest.approx(-1.0)
    assert yaw == pytest.approx(-np.deg2rad(45.0))


def test_proximity_braking_triggers():
    """Forward vx scales linearly when centre ROI depth < 1.5m threshold.

    At 1.0m obstacle with 1.5m threshold:
      t = 1.0 / 1.5 = 0.6667
      scale = 0.2 + 0.6667 * (1.0 - 0.2) = 0.7333
      vx = 2.0 * 0.7333 = 1.4667
    """
    from ros_ws.src.safety_monitor.safety_monitor.safety_node import (
        _PROXIMITY_SCALE,
        _PROXIMITY_THRESH_M,
        apply_safety,
    )

    min_depth = 1.0  # obstacle depth in centre ROI
    depth = np.full((84, 84), 5.0, dtype=np.float32)
    depth[30:54, 30:54] = min_depth
    vx, vy, yaw = apply_safety(2.0, 0.0, 0.0, depth_m=depth, estop=False)

    t = min_depth / _PROXIMITY_THRESH_M
    expected_scale = _PROXIMITY_SCALE + t * (1.0 - _PROXIMITY_SCALE)
    assert vx == pytest.approx(2.0 * expected_scale, rel=1e-5)


def test_proximity_braking_no_trigger():
    """No braking when all depth values exceed threshold."""
    from ros_ws.src.safety_monitor.safety_monitor.safety_node import apply_safety

    depth = np.full((84, 84), 5.0, dtype=np.float32)
    vx, vy, yaw = apply_safety(2.0, 0.0, 0.0, depth_m=depth, estop=False)
    assert vx == pytest.approx(2.0)


def test_proximity_braking_only_on_positive_vx():
    """Backward flight (vx < 0) must NOT trigger proximity braking."""
    from ros_ws.src.safety_monitor.safety_monitor.safety_node import apply_safety

    depth = np.full((84, 84), 0.5, dtype=np.float32)
    vx, vy, yaw = apply_safety(-2.0, 0.0, 0.0, depth_m=depth, estop=False)
    assert vx == pytest.approx(-2.0)


def test_no_depth_image_no_braking():
    """When depth image is None, no proximity braking is applied."""
    from ros_ws.src.safety_monitor.safety_monitor.safety_node import apply_safety

    vx, vy, yaw = apply_safety(2.0, 0.5, 0.1, depth_m=None, estop=False)
    assert vx == pytest.approx(2.0)
    assert vy == pytest.approx(0.5)
    assert yaw == pytest.approx(0.1)


def test_within_limits_passes_through():
    """Values within limits pass through unchanged."""
    from ros_ws.src.safety_monitor.safety_monitor.safety_node import apply_safety

    vx, vy, yaw = apply_safety(1.0, 0.5, 0.1, depth_m=None, estop=False)
    assert vx == pytest.approx(1.0)
    assert vy == pytest.approx(0.5)
    assert yaw == pytest.approx(0.1)
