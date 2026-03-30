"""ROS2 safety monitor node for QDrone 2.

Sits between rl_inference and the flight controller. Applies:
1. Velocity clamping (hard limits)
2. Proximity braking (depth-based forward speed reduction)
3. Emergency stop (zero command on /estop signal)

Subscribes to:
  /drone/cmd_vel         (geometry_msgs/Twist — raw RL command)
  /camera/depth/image_raw (sensor_msgs/Image — for proximity check)
  /estop                 (std_msgs/Bool — emergency stop signal)

Publishes:
  /drone/cmd_vel_safe    (geometry_msgs/Twist — safety-filtered command)

Run: ros2 run safety_monitor safety_node
"""

import numpy as np

try:
    # Available when running from the project root with src/ on the path
    from src.safety.roi_utils import centre_roi_min_depth as _centre_roi_min_depth
except ImportError:
    # Fallback for ROS2 colcon builds where src/ is not on sys.path
    def _centre_roi_min_depth(depth_m: np.ndarray, roi_frac: float = 0.3) -> float:
        h, w = depth_m.shape[:2]
        r0 = int(h * (1 - roi_frac) / 2)
        r1 = int(h * (1 + roi_frac) / 2)
        c0 = int(w * (1 - roi_frac) / 2)
        c1 = int(w * (1 + roi_frac) / 2)
        roi = depth_m[r0:r1, c0:c1]
        return float(roi.min()) if roi.size > 0 else float("inf")


# Safety defaults — overridable via ROS2 parameters
_DEFAULT_MAX_VX = 3.0  # m/s
_DEFAULT_MAX_VY = 1.0  # m/s
_DEFAULT_MAX_YAW_RATE = 45.0  # deg/s
_PROXIMITY_THRESH_M = 1.5  # metres — trigger proximity braking
_PROXIMITY_SCALE = 0.2  # scale vx to this fraction when obstacle close


def apply_safety(
    vx: float,
    vy: float,
    yaw_rate: float,
    depth_m: np.ndarray | None,
    estop: bool,
    max_vx: float = _DEFAULT_MAX_VX,
    max_vy: float = _DEFAULT_MAX_VY,
    max_yaw_rate_rad: float = np.deg2rad(_DEFAULT_MAX_YAW_RATE),
) -> tuple[float, float, float]:
    """Apply all safety layers. Returns (safe_vx, safe_vy, safe_yaw_rate)."""
    if estop:
        return 0.0, 0.0, 0.0

    # 1. Velocity clamping
    vx = float(np.clip(vx, -max_vx, max_vx))
    vy = float(np.clip(vy, -max_vy, max_vy))
    yaw_rate = float(np.clip(yaw_rate, -max_yaw_rate_rad, max_yaw_rate_rad))

    # 2. Proximity braking — only when moving forward
    if depth_m is not None and vx > 0:
        min_depth = _centre_roi_min_depth(depth_m)
        if min_depth < _PROXIMITY_THRESH_M:
            # Linear interpolation: at threshold → scale=1.0, at 0m → scale=_PROXIMITY_SCALE
            t = max(0.0, min_depth / _PROXIMITY_THRESH_M)
            scale = _PROXIMITY_SCALE + t * (1.0 - _PROXIMITY_SCALE)
            vx *= scale

    return vx, vy, yaw_rate


class SafetyNode:
    """ROS2 safety monitor node (lazy ROS2 imports for testability)."""

    def __init__(self):
        import rclpy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Image
        from std_msgs.msg import Bool

        self._Twist = Twist
        self._node = rclpy.create_node("safety_monitor")

        self._node.declare_parameter("max_vx", _DEFAULT_MAX_VX)
        self._node.declare_parameter("max_vy", _DEFAULT_MAX_VY)
        self._node.declare_parameter("max_yaw_rate_deg", _DEFAULT_MAX_YAW_RATE)

        self.max_vx = self._node.get_parameter("max_vx").value
        self.max_vy = self._node.get_parameter("max_vy").value
        self.max_yaw_rate_rad = np.deg2rad(
            self._node.get_parameter("max_yaw_rate_deg").value
        )

        self._estop = False
        self._latest_depth: np.ndarray | None = None

        self._sub_cmd = self._node.create_subscription(
            Twist, "/drone/cmd_vel", self._cmd_cb, 10
        )
        self._sub_depth = self._node.create_subscription(
            Image, "/camera/depth/image_raw", self._depth_cb, 10
        )
        self._sub_estop = self._node.create_subscription(
            Bool, "/estop", self._estop_cb, 10
        )
        self._pub_safe = self._node.create_publisher(
            Twist, "/drone/cmd_vel_safe", 10
        )

        self._node.get_logger().info("safety_monitor node ready")

    def _cmd_cb(self, msg):
        safe_vx, safe_vy, safe_yaw = apply_safety(
            vx=msg.linear.x,
            vy=msg.linear.y,
            yaw_rate=msg.angular.z,
            depth_m=self._latest_depth,
            estop=self._estop,
            max_vx=self.max_vx,
            max_vy=self.max_vy,
            max_yaw_rate_rad=self.max_yaw_rate_rad,
        )
        cmd = self._Twist()
        cmd.linear.x = safe_vx
        cmd.linear.y = safe_vy
        cmd.angular.z = safe_yaw
        self._pub_safe.publish(cmd)

    def _depth_cb(self, msg):
        if msg.encoding == "16UC1":
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width
            )
            self._latest_depth = arr.astype(np.float32) / 1000.0
        elif msg.encoding == "32FC1":
            self._latest_depth = np.frombuffer(
                msg.data, dtype=np.float32
            ).reshape(msg.height, msg.width)
        else:
            self._node.get_logger().warning(
                f"safety_monitor: unsupported depth encoding '{msg.encoding}' — "
                "resetting depth to zeros — assuming obstacle at 0m, proximity braking active"
            )
            self._latest_depth = np.zeros((msg.height, msg.width), dtype=np.float32)

    def _estop_cb(self, msg):
        self._estop = msg.data
        if self._estop:
            self._node.get_logger().warning("E-STOP ACTIVE — zeroing all commands")
        else:
            self._node.get_logger().info("E-stop cleared")


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node._node)
    except KeyboardInterrupt:
        pass
    finally:
        node._node.destroy_node()
        rclpy.shutdown()
