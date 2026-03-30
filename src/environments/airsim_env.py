"""Gymnasium environment for AirSim Quadrotor RL training.

Observation (Dict):
    - image: Depth (H, W, 1) normalised to [0, 1]
    - velocity: [vx, vy, yaw_rate] in body frame

Action (Box[-1, 1]):
    - [target_vx, target_vy, target_yaw_rate] scaled to physical limits

Uses simContinueForTime for lockstep simulation stepping (no wall-clock
sleeps) and moveByVelocityZBodyFrameAsync for active altitude hold.
"""
from __future__ import annotations

import math
import time

import airsim
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environments.rewards import RewardFunction, WaypointRewardFunction


class SimulatedVIO:
    """Dead-reckoning VIO estimator that accumulates realistic drift.

    Simulates GNSS-denied state estimation: velocity estimates drift from
    ground truth over time via Gaussian noise, plus a constant per-episode
    bias injection (simulating IMU bias / calibration error).

    Args:
        drift_std_per_step: Gaussian sigma added to each velocity component per step.
        bias_std: sigma for per-episode constant bias (m/s per axis).
        rng: numpy random generator.
    """

    def __init__(self, drift_std_per_step: float, bias_std: float, rng: np.random.Generator):
        self._drift_std = drift_std_per_step
        self._bias_std = bias_std
        self._rng = rng
        self._bias = np.zeros(3, dtype=np.float32)
        self._accumulated_drift = np.zeros(3, dtype=np.float32)

    def reset(self, rng: np.random.Generator) -> None:
        """Re-sample per-episode bias and clear accumulated drift."""
        self._rng = rng
        self._bias = (
            self._rng.normal(0, self._bias_std, 3).astype(np.float32)
            if self._bias_std > 0 else np.zeros(3, dtype=np.float32)
        )
        self._accumulated_drift = np.zeros(3, dtype=np.float32)

    def update(self, gt_velocity: np.ndarray) -> np.ndarray:
        """Return VIO-estimated velocity given ground-truth velocity."""
        if self._drift_std > 0:
            self._accumulated_drift += self._rng.normal(0, self._drift_std, 3).astype(np.float32)
        return gt_velocity + self._accumulated_drift + self._bias


class AirSimDroneEnv(gym.Env):

    metadata = {"render_modes": ["human"]}
    MAX_RESET_RETRIES = 5

    def __init__(self, cfg: dict | None = None):
        super().__init__()

        cfg = cfg or {}
        env_cfg = cfg.get("env", {})
        reward_cfg = cfg.get("reward", {})
        self._dr_cfg = cfg.get("domain_randomization", {})
        self._vio_enabled = self._dr_cfg.get("vio_enabled", False)
        self._vio: SimulatedVIO | None = None
        self._depth_noise_std = 0.0  # set by _apply_domain_randomization; 0 = no noise
        self._flow_noise_std = 0.0
        self._vio_pos_est = np.zeros(2, dtype=np.float32)  # [x_est, y_est] integrated from VIO vel
        # Motor lag (first-order low-pass on commanded velocity)
        self._tau_motor = 0.0  # 0 = no lag (disabled unless DR sets it)
        self._cmd_vel_filtered = np.zeros(3, dtype=np.float32)  # [vx, vy, yaw_rate] filtered

        self.ip = env_cfg.get("ip", "")
        self.port = env_cfg.get("port", 41451)
        self.image_shape = tuple(env_cfg.get("image_shape", [84, 84, 1]))
        self.target_alt = env_cfg.get("target_alt", 3.0)
        self.max_vx = env_cfg.get("max_vx", 3.0)
        self.max_vy = env_cfg.get("max_vy", 1.0)
        self.max_yaw_rate = np.deg2rad(env_cfg.get("max_yaw_rate_deg", 45))
        self.dt = env_cfg.get("dt", 0.1)
        self.max_steps = env_cfg.get("max_steps", 1024)
        self.depth_clip_m = env_cfg.get("depth_clip_m", 20.0)
        self._depth_clip_base = self.depth_clip_m  # base value; per-episode jitter applied on top

        # Goal navigation (backward-compatible — off by default)
        self.goal_navigation = env_cfg.get("goal_navigation", False)
        self.num_waypoints = env_cfg.get("num_waypoints", 3)
        self.goal_radius_m = env_cfg.get("goal_radius_m", 3.0)
        self.max_goal_dist_m = env_cfg.get("max_goal_dist_m", 30.0)
        self.waypoint_arena_half = env_cfg.get("waypoint_arena_half_m", 20.0)
        # Exploration mode: continuously sample new goals instead of terminating
        self.exploration_mode = env_cfg.get("exploration_mode", False)

        # Pluggable reward — select function based on mode
        if self.goal_navigation:
            self.reward_fn = WaypointRewardFunction(reward_cfg)
        else:
            self.reward_fn = RewardFunction(reward_cfg)

        # AirSim client
        self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)
        self.client.confirmConnection()

        # Velocity observation dimension: 3 (baseline) or 6 (goal navigation)
        # Goal nav adds: [cos_theta, sin_theta, dist_norm] to [vx, vy, yaw_rate]
        vel_dim = 6 if self.goal_navigation else 3

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0, shape=self.image_shape, dtype=np.float32
            ),
            "velocity": spaces.Box(
                low=-np.inf, high=np.inf, shape=(vel_dim,), dtype=np.float32
            ),
        })

        # Internal state
        self.state = {
            "image": np.zeros(self.image_shape, dtype=np.float32),
            "velocity": np.zeros(vel_dim, dtype=np.float32),
        }
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self._last_col_ts = 0  # updated in reset()

        # Goal navigation internal state (initialised in reset)
        self._waypoint_queue: list[tuple[float, float]] = []
        self._goals_reached_this_episode = 0
        self._total_goals_this_episode = 0

    # ------------------------------------------------------------------
    # Config update (for EnvironmentScheduler with SubprocVecEnv)
    # ------------------------------------------------------------------
    def update_config(self, env_cfg: dict) -> None:
        """Update mutable env parameters from a new config dict.

        Called by EnvironmentScheduler via env_method() for SubprocVecEnv
        compatibility. Only updates parameters that make sense to change
        between episodes (not port/ip/image_shape).
        """
        self.max_vx = env_cfg.get("max_vx", self.max_vx)
        self.max_vy = env_cfg.get("max_vy", self.max_vy)
        self.target_alt = env_cfg.get("target_alt", self.target_alt)
        self.max_steps = env_cfg.get("max_steps", self.max_steps)
        self.depth_clip_m = env_cfg.get("depth_clip_m", self.depth_clip_m)

        # Clear stale delta-distance state so the first step after a config
        # rotation doesn't produce a spurious distance reward from the old layout.
        if hasattr(self.reward_fn, "_prev_dist_norm"):
            self.reward_fn._prev_dist_norm = None

    # ------------------------------------------------------------------
    # Domain Randomization
    # ------------------------------------------------------------------
    def _apply_domain_randomization(self):
        """Apply domain randomization on episode reset.

        Hooks for sensor noise, spawn position, and future texture variation.
        Controlled via `domain_randomization` key in config YAML.
        """
        if not self._dr_cfg.get("enabled", False):
            # Reset noise levels so a rotation from DR-enabled → DR-disabled
            # config (via EnvironmentScheduler) stops applying noise.
            self._depth_noise_std = 0.0
            self._flow_noise_std = 0.0
            self._tau_motor = 0.0
            self._cmd_vel_filtered = np.zeros(3, dtype=np.float32)
            self.depth_clip_m = self._depth_clip_base
            return

        # Depth noise: Gaussian noise injected per-step in _get_depth_image
        self._depth_noise_std = self._dr_cfg.get("depth_noise_std", 0.0)

        # Optical flow noise: per-step Gaussian velocity perturbation
        self._flow_noise_std = self._dr_cfg.get("flow_noise_std", 0.0)

        # Spawn position randomization
        spawn_radius = self._dr_cfg.get("spawn_radius_m", 0.0)
        if spawn_radius > 0:
            dx = float(self.np_random.uniform(-spawn_radius, spawn_radius))
            dy = float(self.np_random.uniform(-spawn_radius, spawn_radius))
            random_yaw = float(self.np_random.uniform(-math.pi, math.pi))
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(dx, dy, -self.target_alt),
                    airsim.to_quaternion(0, 0, random_yaw),
                ),
                ignore_collision=True,
            )

        if self._vio_enabled:
            drift_std = self._dr_cfg.get("vio_drift_std_per_step", 0.01)
            bias_std = self._dr_cfg.get("vio_bias_std", 0.005)
            if self._vio is None:
                self._vio = SimulatedVIO(drift_std, bias_std, self.np_random)
            self._vio.reset(self.np_random)

        # Wind disturbance: sample a random horizontal wind vector each episode
        wind_max = self._dr_cfg.get("wind_max_ms", 0.0)
        if wind_max > 0:
            wx = float(self.np_random.uniform(-wind_max, wind_max))
            wy = float(self.np_random.uniform(-wind_max, wind_max))
            try:
                self.client.simSetWind(airsim.Vector3r(wx, wy, 0.0))
            except Exception:
                pass  # graceful degradation if AirSim version doesn't support simSetWind

        # Motor lag: sample a per-episode time constant from [min, max] range
        lag_range = self._dr_cfg.get("motor_lag_tau", None)
        if lag_range is not None:
            try:
                tau_min, tau_max = lag_range[0], lag_range[1]
            except (TypeError, IndexError):
                tau_min = tau_max = float(lag_range)
            self._tau_motor = float(self.np_random.uniform(tau_min, tau_max))
        else:
            self._tau_motor = 0.0
        self._cmd_vel_filtered = np.zeros(3, dtype=np.float32)

        # Depth clip jitter: vary clip range slightly per episode
        clip_jitter = self._dr_cfg.get("depth_clip_jitter", 0.0)
        if clip_jitter > 0:
            self.depth_clip_m = float(
                self._depth_clip_base + self.np_random.uniform(-clip_jitter, clip_jitter)
            )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_depth_image(self) -> np.ndarray:
        """Capture and process depth image from AirSim."""
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ])
        if not responses:
            return np.zeros(self.image_shape, dtype=np.float32)

        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        h, w = responses[0].height, responses[0].width
        if h == 0 or w == 0 or img1d.size == 0:
            return np.zeros(self.image_shape, dtype=np.float32)
        img1d = img1d.reshape(h, w)
        img_depth = cv2.resize(img1d, (self.image_shape[1], self.image_shape[0]))
        img_depth = np.clip(img_depth, 0, self.depth_clip_m) / self.depth_clip_m

        # Domain randomization: sensor noise
        if hasattr(self, "_depth_noise_std") and self._depth_noise_std > 0:
            noise = self.np_random.normal(0, self._depth_noise_std, img_depth.shape)
            img_depth = np.clip(img_depth + noise, 0.0, 1.0).astype(np.float32)

        if len(self.image_shape) == 3:
            img_depth = np.expand_dims(img_depth, axis=-1)

        return img_depth

    def _get_body_velocity(self) -> np.ndarray:
        """Get body-frame velocity [vx, vy, yaw_rate]."""
        kin = self.client.getMultirotorState().kinematics_estimated
        v_global = np.array([
            kin.linear_velocity.x_val,
            kin.linear_velocity.y_val,
            kin.linear_velocity.z_val,
        ])
        yaw = airsim.to_eularian_angles(kin.orientation)[2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        v_body = R_yaw @ v_global

        gt_vel = np.array([
            v_body[0],
            v_body[1],
            kin.angular_velocity.z_val,
        ], dtype=np.float32)

        # Optical flow noise: per-step sensor perturbation
        if self._flow_noise_std > 0:
            gt_vel = gt_vel + self.np_random.normal(
                0, self._flow_noise_std, 3
            ).astype(np.float32)

        if self._vio_enabled and self._vio is not None:
            return self._vio.update(gt_vel)
        return gt_vel

    def _sample_waypoints(self) -> list[tuple[float, float]]:
        """Sample num_waypoints positions with minimum spacing constraint.

        Returns a list of (x, y) world-frame tuples. Retries up to 200 times
        to satisfy the 2 * goal_radius_m minimum inter-waypoint spacing.
        """
        waypoints: list[tuple[float, float]] = []
        min_spacing = 2.0 * self.goal_radius_m
        half = self.waypoint_arena_half

        for _ in range(200 * self.num_waypoints):
            if len(waypoints) >= self.num_waypoints:
                break
            x = float(self.np_random.uniform(-half, half))
            y = float(self.np_random.uniform(-half, half))
            # Check spacing against all already-accepted waypoints
            if all(
                math.hypot(x - wx, y - wy) >= min_spacing
                for wx, wy in waypoints
            ):
                waypoints.append((x, y))

        return waypoints

    def _sample_one_waypoint(self) -> tuple[float, float]:
        """Sample a single random waypoint (no inter-waypoint spacing constraint)."""
        half = self.waypoint_arena_half
        x = float(self.np_random.uniform(-half, half))
        y = float(self.np_random.uniform(-half, half))
        return (x, y)

    def _get_goal_obs(self) -> np.ndarray:
        """Compute goal-relative observation [cos_theta, sin_theta, dist_norm].

        Uses current AirSim state. cos/sin encoding avoids ±π discontinuity,
        providing a smooth 360° gradient for yaw control learning.

        Returns zeros if waypoint queue is empty (mission complete).
        """
        if not self._waypoint_queue:
            return np.zeros(3, dtype=np.float32)

        gx, gy = self._waypoint_queue[0]
        kin = self.client.getMultirotorState().kinematics_estimated
        px = kin.position.x_val
        py = kin.position.y_val
        yaw = airsim.to_eularian_angles(kin.orientation)[2]

        # World-frame delta
        dx_w = gx - px
        dy_w = gy - py

        # Rotate to body frame using negative yaw
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx_b = c * dx_w - s * dy_w
        dy_b = s * dx_w + c * dy_w

        bearing = math.atan2(dy_b, dx_b)
        dist = math.hypot(dx_w, dy_w)
        dist_norm = float(np.clip(dist / self.max_goal_dist_m, 0.0, 1.0))

        return np.array([math.cos(bearing), math.sin(bearing), dist_norm], dtype=np.float32)

    def _check_goal_reached(self) -> bool:
        """Check if the drone has reached the current waypoint.

        Pops the waypoint from the queue and increments the counter when reached.
        Returns True if a waypoint was reached this step.
        """
        if not self._waypoint_queue:
            return False

        gx, gy = self._waypoint_queue[0]
        kin = self.client.getMultirotorState().kinematics_estimated
        px = kin.position.x_val
        py = kin.position.y_val

        if math.hypot(px - gx, py - gy) <= self.goal_radius_m:
            self._waypoint_queue.pop(0)
            self._goals_reached_this_episode += 1
            return True
        return False

    def _get_obs(self) -> dict:
        depth = self._get_depth_image()
        vel = self._get_body_velocity()

        if self.goal_navigation:
            goal_obs = self._get_goal_obs()
            velocity = np.concatenate([vel, goal_obs])
        else:
            velocity = vel

        # Keep self.state updated for deploy.py compatibility (reads state["image"]
        # for the safety depth check — one-step-stale by design).
        self.state["image"] = depth
        self.state["velocity"] = velocity

        # Return a new dict so the caller holds an independent reference;
        # prevents latent corruption if observation access patterns change.
        return {"image": depth, "velocity": velocity.copy()}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        last_exc: Exception | None = None
        for _attempt in range(self.MAX_RESET_RETRIES):
            try:
                return self._reset_inner()
            except Exception as exc:
                last_exc = exc
                if _attempt < self.MAX_RESET_RETRIES - 1:
                    print(
                        f"[AirSimDroneEnv] reset() attempt {_attempt + 1} failed: {exc}. "
                        "Retrying...",
                        flush=True,
                    )
                    time.sleep(1.0)

        raise RuntimeError(
            f"AirSimDroneEnv.reset() failed after {self.MAX_RESET_RETRIES} attempts"
        ) from last_exc

    def _reset_inner(self):
        """Inner reset logic — called by reset() with retry wrapping."""
        # Unpause sim — step() leaves it paused for lockstep,
        # and blocking calls (takeoff, moveToZ) hang on a paused sim.
        self.client.simPause(False)
        self.client.reset()
        self.client.simPause(False)  # reset() re-pauses internally; force unpause
        time.sleep(0.3)

        # Force clean spawn so we never start inside geometry
        self.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(0, 0, -self.target_alt),
                airsim.to_quaternion(0, 0, 0),
            ),
            ignore_collision=True,
        )
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-self.target_alt, 2.0).join()

        # Record stale collision timestamp — step() uses delta to filter
        # real in-episode collisions from pre-reset stale flags.
        self._last_col_ts = self.client.simGetCollisionInfo().time_stamp

        # Enter lockstep mode for deterministic training
        self.client.simPause(True)

        self._apply_domain_randomization()
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        # Initialise VIO position estimate to the actual spawn position so that
        # domain-randomized spawn jitter doesn't immediately create drift error.
        spawn_kin = self.client.getMultirotorState().kinematics_estimated
        self._vio_pos_est = np.array(
            [spawn_kin.position.x_val, spawn_kin.position.y_val],
            dtype=np.float32,
        )

        if self.goal_navigation:
            self._waypoint_queue = self._sample_waypoints()
            self._goals_reached_this_episode = 0
            self._total_goals_this_episode = len(self._waypoint_queue)
            # Clear prev_dist so first step gets no spurious delta reward
            self.reward_fn._prev_dist_norm = None

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step  (lockstep: fire command -> advance sim by dt -> read state)
    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_vx = action[0] * self.max_vx
        target_vy = action[1] * self.max_vy
        target_yaw_rate = action[2] * self.max_yaw_rate

        # Motor lag: first-order low-pass filter on commanded velocity.
        # alpha = dt / (dt + tau); alpha=1 when tau=0 (no lag).
        if self._tau_motor > 0:
            alpha = self.dt / (self.dt + self._tau_motor)
            cmd = np.array([target_vx, target_vy, target_yaw_rate], dtype=np.float32)
            self._cmd_vel_filtered = self._cmd_vel_filtered + alpha * (cmd - self._cmd_vel_filtered)
            target_vx, target_vy, target_yaw_rate = (
                float(self._cmd_vel_filtered[0]),
                float(self._cmd_vel_filtered[1]),
                float(self._cmd_vel_filtered[2]),
            )

        self.client.moveByVelocityZBodyFrameAsync(
            float(target_vx),
            float(target_vy),
            float(-self.target_alt),
            self.dt,
            yaw_mode=airsim.YawMode(
                is_rate=True,
                yaw_or_rate=float(math.degrees(target_yaw_rate)),
            ),
        )

        self.client.simContinueForTime(self.dt)
        self.client.simPause(True)

        obs = self._get_obs()
        self.step_count += 1

        # Ground-truth kinematics — fetched once and reused for VIO integration,
        # drift error, and the info dict (avoids a second RPC call).
        kin_for_info = self.client.getMultirotorState().kinematics_estimated

        # Integrate VIO velocity to track estimated position (dead-reckoning).
        # Rotate body-frame [vx, vy] to world-frame before integrating into
        # the world-frame position estimate.
        if self._vio_enabled:
            vio_vel_body = obs["velocity"][:2]  # [vx, vy] in body frame
            q = kin_for_info.orientation
            yaw = math.atan2(
                2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
                1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2),
            )
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            vx_w = vio_vel_body[0] * cos_y - vio_vel_body[1] * sin_y
            vy_w = vio_vel_body[0] * sin_y + vio_vel_body[1] * cos_y
            self._vio_pos_est = self._vio_pos_est + np.array([vx_w, vy_w], dtype=np.float32) * self.dt

        # Collision detection
        vx_body = obs["velocity"][0]
        col_info = self.client.simGetCollisionInfo()
        has_collided = (
            col_info.has_collided
            and col_info.time_stamp != self._last_col_ts
        )

        # Drift error: L2 distance between VIO-estimated and ground-truth position
        drift_error = 0.0
        if self._vio_enabled:
            drift_error = float(np.hypot(
                self._vio_pos_est[0] - kin_for_info.position.x_val,
                self._vio_pos_est[1] - kin_for_info.position.y_val,
            ))

        if self.goal_navigation:
            goal_reached = self._check_goal_reached()
            # Exploration mode: immediately queue a new random goal so the drone never stops
            if self.exploration_mode and goal_reached:
                self._waypoint_queue.append(self._sample_one_waypoint())
            all_goals_done = (len(self._waypoint_queue) == 0)
            # dist_norm and cos_theta are in obs["velocity"][5] and [3]
            dist_norm = float(obs["velocity"][5]) if not all_goals_done else 0.0
            cos_theta = float(obs["velocity"][3])
            reward, reward_info = self.reward_fn(
                vx_body=vx_body,
                has_collided=has_collided,
                action=action,
                prev_action=self.prev_action,
                goal_reached=goal_reached,
                dist_norm=dist_norm,
                cos_theta=cos_theta,
                all_goals_done=all_goals_done,
            )
        else:
            goal_reached = False
            all_goals_done = False
            # Use centre ROI to avoid ground-pixel false triggers.
            # Global image.min() fires the proximity penalty every step because
            # bottom pixels of the 90° FOV camera see the ground at <2m altitude.
            # Squeeze channel dim if present: (H,W,1) → (H,W) or (H,W) → (H,W)
            depth_frame = obs["image"][..., 0] if obs["image"].ndim == 3 else obs["image"]
            _rf = 0.3
            _h, _w = depth_frame.shape
            _r0, _r1 = int(_h * (1 - _rf) / 2), int(_h * (1 + _rf) / 2)
            _c0, _c1 = int(_w * (1 - _rf) / 2), int(_w * (1 + _rf) / 2)
            min_depth = float(depth_frame[_r0:_r1, _c0:_c1].min())
            reward, reward_info = self.reward_fn(
                vx_body, has_collided, action, self.prev_action,
                drift_error=drift_error,
                min_depth=min_depth,
            )

        self.prev_action = action.copy()

        terminated = has_collided or (self.goal_navigation and all_goals_done)
        truncated = self.step_count >= self.max_steps

        info = {
            **reward_info,
            "vx_body": vx_body,
            "step_count": self.step_count,
            "has_collided": has_collided,
            "x_gt": float(kin_for_info.position.x_val),
            "y_gt": float(kin_for_info.position.y_val),
        }
        if self._vio_enabled:
            info["x_est"] = float(self._vio_pos_est[0])
            info["y_est"] = float(self._vio_pos_est[1])

        if self.goal_navigation:
            info.update({
                "goals_reached": self._goals_reached_this_episode,
                "total_goals": self._total_goals_this_episode,
                "mission_success": all_goals_done and not has_collided,
                "goal_reached_this_step": goal_reached,
            })

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            self.client.simPause(False)
        except RuntimeError:
            pass
        except Exception:
            pass
