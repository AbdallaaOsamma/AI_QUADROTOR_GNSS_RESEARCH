"""Tests for statistical analysis script. AirSim-free."""
import json


def test_module_importable():
    """Script must import cleanly."""
    import importlib
    mod = importlib.import_module("scripts.run_statistical_analysis")
    assert hasattr(mod, "load_episode_summaries")
    assert hasattr(mod, "run_anova")
    assert hasattr(mod, "run_paired_ttest")


def test_load_episode_summaries(tmp_path):
    """load_episode_summaries reads all JSON files from a directory."""
    from scripts.run_statistical_analysis import load_episode_summaries
    d = tmp_path / "eval"
    d.mkdir()
    for i in range(3):
        (d / f"ep_{i}.json").write_text(json.dumps(
            {"average_speed_ms": float(i), "collided": False}
        ))
    summaries = load_episode_summaries(str(d))
    assert len(summaries) == 3


def test_paired_ttest_significant():
    """t-test should detect significant difference between clearly separated groups."""
    import numpy as np
    from scripts.run_statistical_analysis import run_paired_ttest
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    b = np.array([2.0, 2.1, 1.9, 2.05, 1.95])
    result = run_paired_ttest(a, b)
    assert result["p_value"] < 0.05
    assert "statistic" in result


def test_anova_two_groups():
    """ANOVA should return f_statistic and p_value for two clearly separated groups."""
    import numpy as np
    from scripts.run_statistical_analysis import run_anova
    a = np.array([1.0, 1.1, 0.9])
    b = np.array([3.0, 3.1, 2.9])
    result = run_anova(a, b)
    assert result["p_value"] < 0.05
    assert "f_statistic" in result


def test_paired_ttest_unequal_lengths():
    """Unequal-length arrays should return valid results (Welch's t-test handles this)."""
    import numpy as np
    from scripts.run_statistical_analysis import run_paired_ttest
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    b = np.array([2.0, 2.1, 1.9])
    result = run_paired_ttest(a, b)
    assert "p_value" in result
    assert "statistic" in result
    assert result["mean_a"] == float(np.mean(a))
    assert result["mean_b"] == float(np.mean(b))


def test_anova_insufficient_samples():
    """ANOVA with a group of size 1 should return error dict, not NaN."""
    import numpy as np
    from scripts.run_statistical_analysis import run_anova
    a = np.array([1.0])
    b = np.array([2.0, 2.1, 1.9])
    result = run_anova(a, b)
    assert result["f_statistic"] is None
    assert result["p_value"] is None
    assert result["error"] == "insufficient_samples"
