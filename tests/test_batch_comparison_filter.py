"""Tests that run_batch_comparison.py discovery correctly includes ppo_2026 runs."""


def _load_discover_models():
    """Import discover_models without running main()."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "run_batch_comparison",
        pathlib.Path("scripts/run_batch_comparison.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSkipPatterns:
    def test_ppo_2026_not_skipped(self):
        mod = _load_discover_models()
        patterns = mod._SKIP_PATTERNS
        assert not any(p in "ppo_2026_20260307_123456" for p in patterns), (
            f"ppo_2026 runs should NOT be skipped; patterns={patterns}"
        )

    def test_smoke_is_skipped(self):
        mod = _load_discover_models()
        patterns = mod._SKIP_PATTERNS
        assert any(p in "smoke_test_run" for p in patterns)

    def test_optuna_trial_is_skipped(self):
        mod = _load_discover_models()
        patterns = mod._SKIP_PATTERNS
        assert any(p in "optuna_trial_42" for p in patterns)

    def test_tmp_is_skipped(self):
        mod = _load_discover_models()
        patterns = mod._SKIP_PATTERNS
        assert any(p in "tmp_ablation_run" for p in patterns)

    def test_ppo_2025_not_skipped(self):
        """ppo_2025 runs should now also be discoverable (removed from patterns)."""
        mod = _load_discover_models()
        patterns = mod._SKIP_PATTERNS
        assert not any(p in "ppo_2025_old_run" for p in patterns)
