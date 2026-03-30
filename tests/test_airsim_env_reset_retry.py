"""Tests for reset() retry logic in AirSimDroneEnv.

These tests mock AirSim at the boundary so they run without a live AirSim
instance. The retry loop and RuntimeError propagation are tested by patching
_reset_inner on a bare env instance (constructed via __new__ to skip __init__).
"""
import sys
import unittest.mock as mock

import pytest

# ── Isolated import of AirSimDroneEnv ─────────────────────────────────────────
# We need to mock cv2 and airsim during the import, then restore sys.modules so
# the mock cv2 does NOT leak into other test modules (e.g. test_rl_inference.py).
# conftest.py stubs airsim/msgpackrpc before collection; cv2 may not be in
# sys.modules yet.  Add a lightweight mock so that `import cv2` in
# airsim_env.py (and stable_baselines3) does not trigger the broken Anaconda
# cv2 binary.  We only add cv2 if it is not already present so we don't
# overwrite a functional installation.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = mock.MagicMock()

from src.environments.airsim_env import AirSimDroneEnv  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _bare_env() -> AirSimDroneEnv:
    """Return a bare AirSimDroneEnv with only reset() and MAX_RESET_RETRIES set."""
    env = AirSimDroneEnv.__new__(AirSimDroneEnv)
    env.MAX_RESET_RETRIES = 5
    return env


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_reset_retries_on_airsim_error():
    """reset() must retry up to MAX_RESET_RETRIES times before raising RuntimeError."""
    env = _bare_env()
    call_count = 0

    def _always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectionRefusedError("AirSim not available")

    env._reset_inner = _always_fail

    with mock.patch("time.sleep"):
        with pytest.raises(RuntimeError, match="reset\\(\\) failed after 5 attempts"):
            env.reset()

    assert call_count == 5, (
        f"Expected exactly 5 attempts (MAX_RESET_RETRIES=5), got {call_count}"
    )


def test_reset_succeeds_on_second_attempt():
    """reset() must succeed if _reset_inner() fails once then succeeds."""
    env = _bare_env()
    fake_obs = ({"image": None, "velocity": None}, {})
    call_count = 0

    def _fail_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("transient blip")
        return fake_obs

    env._reset_inner = _fail_once

    with mock.patch("time.sleep"):
        result = env.reset()

    assert result == fake_obs, "reset() should return the successful attempt's result"
    assert call_count == 2, (
        f"Expected 2 attempts (fail once, succeed on second), got {call_count}"
    )
