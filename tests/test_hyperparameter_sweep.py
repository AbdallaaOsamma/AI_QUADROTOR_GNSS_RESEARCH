"""Tests for scripts/run_hyperparameter_sweep.py."""


def test_sweep_module_importable():
    """Sweep script must import without error."""
    import importlib

    mod = importlib.import_module("scripts.run_hyperparameter_sweep")
    assert hasattr(mod, "objective")
    assert hasattr(mod, "main")


def test_objective_returns_float(tmp_path):
    """objective() must return a float reward value."""
    import optuna
    from unittest.mock import patch

    with patch("scripts.run_hyperparameter_sweep.run_trial", return_value=42.5):
        from scripts.run_hyperparameter_sweep import objective

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = objective(trial)
        assert isinstance(result, float)
