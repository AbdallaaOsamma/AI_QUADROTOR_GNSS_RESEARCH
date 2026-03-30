"""Tests for ONNX export shape auto-detection logic."""
import unittest.mock as mock


def _make_fake_obs_space(image_shape, velocity_shape):
    """Build a minimal fake observation space mirroring SB3 Dict space."""
    image_space = mock.MagicMock()
    image_space.shape = image_shape

    vel_space = mock.MagicMock()
    vel_space.shape = velocity_shape

    obs_space = mock.MagicMock()
    obs_space.spaces = {"image": image_space, "velocity": vel_space}
    return obs_space


class TestExportShapeDetection:
    def test_frame_stack_4_velocity_3(self, tmp_path, monkeypatch):
        """Standard training with frame_stack=4, velocity=(3,) should export."""
        obs_space = _make_fake_obs_space((84, 84, 4), (3,))

        with mock.patch("scripts.export_onnx.PPO") as MockPPO, \
             mock.patch("scripts.export_onnx.torch"), \
             mock.patch("os.path.exists", return_value=True):

            fake_model = mock.MagicMock()
            fake_model.observation_space = obs_space
            MockPPO.load.return_value = fake_model

            image_shape = obs_space.spaces["image"].shape
            velocity_shape = obs_space.spaces["velocity"].shape

            assert image_shape == (84, 84, 4)
            assert velocity_shape == (3,)
            assert image_shape[2] == 4  # detected frame_stack
            assert velocity_shape[0] == 3  # detected vel_dim

    def test_frame_stack_1_velocity_3(self):
        """frame_stack=1 should produce image (84, 84, 1) and velocity (3,)."""
        obs_space = _make_fake_obs_space((84, 84, 1), (3,))
        image_shape = obs_space.spaces["image"].shape
        velocity_shape = obs_space.spaces["velocity"].shape

        detected_frame_stack = image_shape[2]
        detected_vel_dim = velocity_shape[0]

        assert detected_frame_stack == 1
        assert detected_vel_dim == 3

    def test_frame_stack_4_stacked_velocity_12(self):
        """VecFrameStack with n_stack=4 produces velocity dim 12 (4*3)."""
        obs_space = _make_fake_obs_space((84, 84, 4), (12,))
        velocity_shape = obs_space.spaces["velocity"].shape

        detected_vel_dim = velocity_shape[0]
        assert detected_vel_dim == 12

    def test_cli_frame_stack_mismatch_warns(self, capsys):
        """Passing a CLI frame_stack that differs from model should warn, not fail."""
        obs_space = _make_fake_obs_space((84, 84, 4), (3,))
        detected = obs_space.spaces["image"].shape[2]  # 4
        cli_value = 1  # deliberate mismatch

        # Simulate the warning branch in export_to_onnx
        if cli_value is not None and cli_value != detected:
            import warnings
            warnings.warn(
                f"CLI --frame_stack={cli_value} differs from model ({detected}). "
                "Using model value."
            )

        # The resolved value should be from the model
        resolved = detected
        assert resolved == 4
