# Contributing to AI-Augmented Quadrotor Navigation

Thank you for your interest in contributing! This is an active FYP research project — contributions that help advance the science or improve the engineering are very welcome.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Branch Workflow](#branch-workflow)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [How to Contribute](#how-to-contribute)
- [Reporting Bugs](#reporting-bugs)

---

## Development Setup

```bash
# Fork and clone
git clone https://github.com/<your-username>/AI_QUADROTOR_GNSS_RESEARCH.git
cd AI_QUADROTOR_GNSS_RESEARCH

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify everything works (no AirSim required)
make test
make lint
```

---

## Branch Workflow

| Branch | Purpose |
|--------|---------|
| `main` | Stable, always passing CI — direct commits discouraged |
| `develop` | Integration branch — merge PRs here first |
| `feature/<name>` | New features or algorithms |
| `fix/<name>` | Bug fixes |
| `experiment/<name>` | Research experiments |

**Standard flow:**

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
# ... make changes ...
git push origin feature/your-feature-name
# Open PR targeting develop
```

---

## Running Tests

All 58 tests run without AirSim (the AirSim client is mocked at the boundary):

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_rewards.py -v

# Run specific test
pytest tests/test_rewards.py::test_progress_reward -v
```

**Important**: Every new feature must include tests. PRs without tests will not be merged.

---

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
make lint
# or
ruff check src/ tests/ scripts/
```

Key conventions:
- Line length: 100 characters
- Python 3.11+ type hints on all public functions
- All ML tensors: `(N, C, H, W)` format, normalized `[0, 1]`
- AirSim coordinates: NED frame (Down is +Z), altitude in negative meters
- Hyperparameters in YAML configs under `configs/` — never hardcoded

---

## How to Contribute

### Good first issues

- Improve test coverage for edge cases
- Add type hints to uncovered functions
- Improve docstrings in `src/environments/`
- Add a new reward function variant under `configs/rewards/`

### Research contributions

The following known gaps are documented in [CLAUDE.md](CLAUDE.md):

| Issue | Description | Difficulty |
|-------|-------------|------------|
| Physics DR | Vary mass, inertia, motor gains per episode | Hard |
| Full VIO pipeline | Replace ground-truth kinematics with optical-flow VIO | Hard |
| Curriculum learning | Progressive difficulty in `configs/curriculum/` | Medium |
| SAC implementation | Already in progress on `feature/sac-training` | Medium |
| Perception module | Extract depth processing into `src/perception/` | Easy |

### Adding a new algorithm

1. Add a config under `configs/train_<algo>.yaml`
2. Implement training in `src/training/train.py` (or add alongside it)
3. Ensure eval works with `scripts/run_full_eval.py`
4. Add statistical comparison support in `scripts/run_statistical_analysis.py`
5. Update the results table in `README.md`

### Adding a new reward function

1. Subclass `RewardFunction` in `src/environments/rewards.py`
2. Add a config under `configs/rewards/<name>.yaml`
3. Add an ablation config under `configs/ablations/`
4. Write tests in `tests/test_rewards.py`

---

## Reporting Bugs

Use the [GitHub issue tracker](https://github.com/Abood204/AI_QUADROTOR_GNSS_RESEARCH/issues).

Please include:
- Python version (`python --version`)
- Package version (`pip show ai-quadrotor-rl`)
- AirSim version (if relevant)
- Minimal reproduction steps
- Expected vs actual behaviour
- Full traceback

---

## Pull Request Checklist

Before opening a PR, confirm:

- [ ] Tests pass: `make test`
- [ ] Linting passes: `make lint`
- [ ] New code has tests
- [ ] YAML configs used for any new hyperparameters
- [ ] PR targets `develop`, not `main`
- [ ] PR description explains the *why*, not just the *what*

---

## Questions?

Open a [GitHub Discussion](https://github.com/Abood204/AI_QUADROTOR_GNSS_RESEARCH/discussions) or file an issue tagged `question`.
