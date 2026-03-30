"""Tests for scripts/run_hyperparameter_sweep.py."""


def test_sweep_module_importable():
    """Sweep script must import without error and expose required callables."""
    import importlib

    mod = importlib.import_module("scripts.run_hyperparameter_sweep")
    assert hasattr(mod, "run_trial")
    assert hasattr(mod, "main")


def test_run_trial_returns_float(tmp_path):
    """run_trial() must return a float reward value on subprocess success."""
    from unittest.mock import patch, MagicMock

    fake_result = MagicMock()
    fake_result.returncode = 0
    # Simulate SB3 verbose output containing ep_rew_mean
    fake_result.stdout = "| ep_rew_mean          | 42.5     |\n"

    with patch("subprocess.run", return_value=fake_result):
        from scripts.run_hyperparameter_sweep import run_trial

        result = run_trial(config_overrides={}, total_timesteps=50_000)
        assert isinstance(result, float)
        assert result == 42.5


def test_run_trial_returns_neg_inf_on_failure():
    """run_trial() must return -inf when the subprocess exits non-zero."""
    from unittest.mock import patch, MagicMock

    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stdout = ""

    with patch("subprocess.run", return_value=fake_result):
        from scripts.run_hyperparameter_sweep import run_trial

        result = run_trial(config_overrides={}, total_timesteps=50_000)
        assert result == float("-inf")
