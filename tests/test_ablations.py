"""Tests for run_ablations.py: command construction and CLI behaviour."""
import json
import subprocess
import sys


def test_ablation_dry_run_exits_zero():
    """Full dry-run must succeed (exit 0) — verifies the --overrides JSON path."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_ablations", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_ablation_dry_run_overrides_json_valid():
    """Commands containing --overrides must embed valid JSON strings."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_ablations", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    # Extract every --overrides value from the printed commands and parse as JSON.
    for line in result.stdout.splitlines():
        if "--overrides" in line:
            # The printed command line looks like:
            #   Command: python -m src.training.train ... --overrides {"key": val}
            # We need the token after "--overrides ".
            idx = line.index("--overrides ") + len("--overrides ")
            overrides_str = line[idx:].strip()
            try:
                parsed = json.loads(overrides_str)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"--overrides value is not valid JSON: {overrides_str!r}"
                ) from exc
            assert isinstance(parsed, dict), (
                f"--overrides value must be a JSON object, got: {parsed!r}"
            )


def test_ablation_build_command_overrides_structure():
    """build_command() must pass --overrides as a single valid-JSON argument."""
    # Import the helper directly to unit-test without subprocess.
    import importlib
    import sys as _sys

    # Ensure the scripts package is importable.
    spec = importlib.util.find_spec("scripts.run_ablations")
    assert spec is not None, "scripts.run_ablations not importable"

    run_ablations = importlib.import_module("scripts.run_ablations")

    frames_1_exp = next(
        exp for exp in run_ablations.ABLATIONS if exp["name"] == "abl2_frames_1"
    )
    cmd = run_ablations.build_command(frames_1_exp, total_timesteps=4096)

    assert "--overrides" in cmd, "--overrides flag must appear in command"
    overrides_idx = cmd.index("--overrides")
    overrides_value = cmd[overrides_idx + 1]

    # Must be a single argument containing valid JSON.
    parsed = json.loads(overrides_value)
    assert parsed == {"frame_stack": 1}, f"Unexpected overrides value: {parsed!r}"


def test_ablation_no_dr_overrides_structure():
    """abl3_no_dr --overrides must correctly encode domain_randomization keys."""
    import importlib

    run_ablations = importlib.import_module("scripts.run_ablations")

    no_dr_exp = next(
        exp for exp in run_ablations.ABLATIONS if exp["name"] == "abl3_no_dr"
    )
    cmd = run_ablations.build_command(no_dr_exp, total_timesteps=4096)

    assert "--overrides" in cmd
    overrides_value = cmd[cmd.index("--overrides") + 1]
    parsed = json.loads(overrides_value)

    assert "domain_randomization" in parsed
    dr = parsed["domain_randomization"]
    assert dr["enabled"] is False
    assert dr["depth_noise_std"] == 0.0
    assert dr["spawn_radius_m"] == 0.0


def test_ablation_only_full_names():
    """--only with full experiment names must succeed (exit 0)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_ablations",
            "--dry-run",
            "--only",
            "abl2_frames_1",
            "abl3_no_dr",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_ablation_only_shorthand_names():
    """--only with shorthand suffixes (documented in CLAUDE.md) must succeed (exit 0)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_ablations",
            "--dry-run",
            "--only",
            "no_smoothness",
            "progress_only",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    # Both experiments should appear in output.
    assert "abl1_no_smoothness" in result.stdout
    assert "abl1_progress_only" in result.stdout
