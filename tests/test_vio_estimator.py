"""Tests for SimulatedVIO dead-reckoning estimator. Runs without AirSim."""
import numpy as np


def test_vio_drift_accumulates_over_steps():
    """VIO estimate should diverge from ground truth over many steps."""
    from src.environments.airsim_env import SimulatedVIO
    vio = SimulatedVIO(drift_std_per_step=0.05, bias_std=0.0, rng=np.random.default_rng(0))
    gt_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    errors = [np.linalg.norm(vio.update(gt_vel) - gt_vel) for _ in range(100)]
    assert errors[-1] > errors[0]


def test_vio_bias_is_constant_per_episode():
    """Bias is fixed per episode — repeated calls with same gt produce same output."""
    from src.environments.airsim_env import SimulatedVIO
    vio = SimulatedVIO(drift_std_per_step=0.0, bias_std=0.1, rng=np.random.default_rng(7))
    gt = np.zeros(3, dtype=np.float32)
    est1 = vio.update(gt)
    est2 = vio.update(gt)
    np.testing.assert_allclose(est1, est2, atol=1e-5)


def test_vio_zero_drift_and_zero_bias_returns_gt():
    """With zero drift and zero bias, estimate equals ground truth."""
    from src.environments.airsim_env import SimulatedVIO
    vio = SimulatedVIO(drift_std_per_step=0.0, bias_std=0.0, rng=np.random.default_rng(0))
    gt = np.array([1.5, -0.3, 0.2], dtype=np.float32)
    np.testing.assert_allclose(vio.update(gt), gt, atol=1e-6)


def test_vio_reset_clears_accumulated_drift():
    """After reset, VIO state starts fresh (drift is cleared)."""
    from src.environments.airsim_env import SimulatedVIO
    vio = SimulatedVIO(drift_std_per_step=0.1, bias_std=0.0, rng=np.random.default_rng(0))
    gt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(50):
        vio.update(gt)
    # Reset with a zero-bias rng
    vio.reset(np.random.default_rng(1))
    est = vio.update(gt)
    # After reset, accumulated drift is cleared — error should be small
    assert np.linalg.norm(est - gt) < 0.5
