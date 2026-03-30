"""Tests for rl_inference package -- no ROS2 or ONNX needed."""
import sys
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np

ROS_MOCKS = {
    "rclpy": MagicMock(),
    "rclpy.node": MagicMock(),
    "sensor_msgs": MagicMock(),
    "sensor_msgs.msg": MagicMock(),
    "geometry_msgs": MagicMock(),
    "geometry_msgs.msg": MagicMock(),
    "rcl_interfaces": MagicMock(),
    "rcl_interfaces.msg": MagicMock(),
}

# Import the module once at module level with ROS mocks active
with patch.dict(sys.modules, ROS_MOCKS):
    from ros_ws.src.rl_inference.rl_inference import inference_node as _mod


def test_inference_node_importable():
    """Module imports without rclpy; InferenceNode and main exist."""
    assert hasattr(_mod, "InferenceNode")
    assert hasattr(_mod, "main")


def test_depth_cb_normalizes_16uc1():
    """_depth_cb decodes 16UC1 and normalizes: 5000mm -> 5m -> 5/20 = 0.25."""
    fake_self = MagicMock()
    fake_self.IMAGE_SIZE = 84
    fake_self._depth_clip_m = 20.0
    fake_self._frame_buffer = deque(
        [np.zeros((84, 84, 1), dtype=np.float32)] * 4, maxlen=4
    )
    fake_self._node = MagicMock()

    raw = np.full((84, 84), 5000, dtype=np.uint16)
    msg = MagicMock()
    msg.encoding = "16UC1"
    msg.height = 84
    msg.width = 84
    msg.data = raw.tobytes()

    # cv2 may be a MagicMock in the test environment; make resize return a
    # proper numpy array so the normalization math is exercised correctly.
    with patch.object(_mod.cv2, "resize", return_value=np.full((84, 84), 5.0, dtype=np.float32)):
        _mod.InferenceNode._depth_cb(fake_self, msg)

    last = fake_self._frame_buffer[-1]
    assert last.shape == (84, 84, 1)
    np.testing.assert_allclose(last, 0.25, atol=1e-5)
