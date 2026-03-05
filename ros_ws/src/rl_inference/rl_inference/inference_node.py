"""ROS2 RL inference node for QDrone 2.

Subscribes to:
  /camera/depth/image_raw   (sensor_msgs/Image, 16UC1 or 32FC1)
  /drone/state              (geometry_msgs/TwistStamped -- body-frame velocity)

Publishes:
  /drone/cmd_vel            (geometry_msgs/Twist -- vx, vy, yaw_rate)

Run: ros2 run rl_inference inference_node --ros-args -p model_path:=model.onnx
"""
from collections import deque

import cv2
import numpy as np

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


class InferenceNode:
    """Runs ONNX policy inference and publishes velocity commands.

    Imports rclpy lazily so this file is importable on dev machines
    without ROS2 installed (needed for unit testing).
    """

    FRAME_STACK = 4
    IMAGE_SIZE = 84

    def __init__(self):
        import rclpy
        from rclpy.node import Node  # noqa: F811
        from geometry_msgs.msg import Twist, TwistStamped
        from rcl_interfaces.msg import ParameterDescriptor
        from sensor_msgs.msg import Image

        # Store message types for use in callbacks
        self._Twist = Twist
        self._TwistStamped = TwistStamped
        self._Image = Image

        # Wrap a Node instance (cannot subclass with lazy imports)
        self._node = rclpy.create_node("rl_inference")

        self._node.declare_parameter(
            "model_path",
            "model.onnx",
            ParameterDescriptor(description="Path to ONNX model"),
        )
        self._node.declare_parameter("max_vx", 3.0)
        self._node.declare_parameter("max_vy", 1.0)
        self._node.declare_parameter("max_yaw_rate_deg", 45.0)

        model_path = self._node.get_parameter("model_path").value
        self.max_vx = self._node.get_parameter("max_vx").value
        self.max_vy = self._node.get_parameter("max_vy").value
        self.max_yaw_rate = np.deg2rad(
            self._node.get_parameter("max_yaw_rate_deg").value
        )

        if not _ORT_AVAILABLE:
            self._node.get_logger().error(
                "onnxruntime not installed. pip install onnxruntime"
            )
            raise RuntimeError("onnxruntime required")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._node.get_logger().info(f"Loaded ONNX model: {model_path}")

        self._frame_buffer: deque = deque(
            [np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, 1), dtype=np.float32)]
            * self.FRAME_STACK,
            maxlen=self.FRAME_STACK,
        )
        self._latest_velocity = np.zeros(3, dtype=np.float32)

        self._sub_depth = self._node.create_subscription(
            Image, "/camera/depth/image_raw", self._depth_cb, 10
        )
        self._sub_state = self._node.create_subscription(
            TwistStamped, "/drone/state", self._state_cb, 10
        )
        self._pub_cmd = self._node.create_publisher(Twist, "/drone/cmd_vel", 10)

        self._node.create_timer(0.1, self._inference_step)
        self._node.get_logger().info("rl_inference node ready (10 Hz)")

    def _depth_cb(self, msg):
        """Decode depth image, resize to 84x84, normalize to [0, 1]."""
        if msg.encoding == "16UC1":
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width
            )
            depth_m = arr.astype(np.float32) / 1000.0
        elif msg.encoding == "32FC1":
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width
            )
            depth_m = arr
        else:
            self._node.get_logger().warning(
                f"Unsupported encoding: {msg.encoding}"
            )
            return

        resized = cv2.resize(depth_m, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        clipped = np.clip(resized, 0.0, 20.0) / 20.0
        frame = clipped[:, :, np.newaxis].astype(np.float32)
        self._frame_buffer.append(frame)

    def _state_cb(self, msg):
        """Store latest body-frame velocity from state topic."""
        self._latest_velocity = np.array(
            [
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.angular.z,
            ],
            dtype=np.float32,
        )

    def _inference_step(self):
        """Run ONNX inference and publish velocity command at 10 Hz."""
        image = np.concatenate(list(self._frame_buffer), axis=-1)[
            np.newaxis
        ]  # (1, 84, 84, 4)
        velocity = self._latest_velocity[np.newaxis]  # (1, 3)

        action = self.session.run(
            ["action"],
            {"image": image, "velocity": velocity},
        )[0][0]  # (3,)

        action = np.clip(action, -1.0, 1.0)
        cmd = self._Twist()
        cmd.linear.x = float(action[0] * self.max_vx)
        cmd.linear.y = float(action[1] * self.max_vy)
        cmd.angular.z = float(action[2] * self.max_yaw_rate)
        self._pub_cmd.publish(cmd)


def main(args=None):
    """ROS2 entry point."""
    import rclpy

    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node._node)
    except KeyboardInterrupt:
        pass
    finally:
        node._node.destroy_node()
        rclpy.shutdown()
