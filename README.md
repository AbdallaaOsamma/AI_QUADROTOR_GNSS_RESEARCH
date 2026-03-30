<div align="center">

# 🚁 AI-Augmented Quadrotor Navigation

**Deep Reinforcement Learning for Autonomous Flight in GNSS-Denied Environments**

[![CI](https://github.com/Abood204/AI_QUADROTOR_GNSS_RESEARCH/actions/workflows/ci.yml/badge.svg)](https://github.com/Abood204/AI_QUADROTOR_GNSS_RESEARCH/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-58%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AirSim](https://img.shields.io/badge/simulator-AirSim%201.8-blueviolet)](https://github.com/microsoft/AirSim)
[![SB3](https://img.shields.io/badge/RL-Stable--Baselines3-orange)](https://github.com/DLR-RM/stable-baselines3)

[Overview](#overview) · [Architecture](#architecture) · [Results](#results) · [Quick Start](#quick-start) · [Experiments](#experiments--reproducibility) · [Citation](#citation)

</div>

---

## Overview

Modern drones depend on GPS for positioning. In **GNSS-denied environments** — indoors, urban canyons, tunnels, or GPS-jammed zones — conventional flight controllers fail entirely.

This project develops a **Deep RL agent** that navigates autonomously using only a forward-facing depth camera and inertial measurements — **no GPS, no pre-built maps**. The agent is trained in [Microsoft AirSim](https://github.com/microsoft/AirSim) with aggressive domain randomization for sim-to-real transfer, and validated across multiple environments with a comprehensive 7-metric evaluation protocol.

### Key Contributions

| # | Contribution | Status |
|---|---|---|
| 1 | **GPS-free navigation** — depth + simulated VIO state estimation, no GNSS signals | ✅ Complete |
| 2 | **PPO baseline** with NatureCNN + MLP fusion, domain randomization, multi-env rotation | ✅ Complete |
| 3 | **Ablation study** — 7 experiments across frame stack depth, DR, reward shaping | ✅ Complete |
| 4 | **SAC comparative study** — off-policy vs on-policy algorithm comparison | 🔄 In Progress |
| 5 | **4-layer safety monitor** — velocity clamp, proximity scaling, altitude guard, e-stop | ✅ Complete |
| 6 | **Statistical validation** — ANOVA + paired t-tests across all experimental conditions | ✅ Complete |
| 7 | **Optuna HPO** — automated hyperparameter optimization with 50+ trial sweep | ✅ Complete |

---

## Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │                 AirSim Simulation                 │
                    │                                                    │
  Depth Camera ────►│  84×84 depth frame  ──►  4-frame stack            │
  (forward-facing)  │                              │                    │
                    │                              ▼                    │
                    │                    NatureCNN Encoder              │
                    │                              │                    │
                    │                              ├──────► PPO Policy ─┼──► [vx, vy, yaw] ──► AirSim
                    │                              │                    │         │
  IMU / Kinematics ►│   SimulatedVIO Estimator     │                    │         │
                    │   (dead-reckoning + drift)   │                    │         ▼
                    │   [vx, vy, yaw_rate] ───────►┘                    │   Safety Monitor
                    │                                                    │   (always-on)
                    └──────────────────────────────────────────────────┘
```

### Observation Space

| Component | Shape | Description |
|-----------|-------|-------------|
| Depth image | `(84, 84, 4)` | 4-frame stack, normalized [0,1], clipped at 20 m |
| Body velocity | `(3,)` | VIO-estimated `[vx, vy, yaw_rate]` with realistic drift |

### Action Space

| Component | Normalized Range | Physical Scale |
|-----------|-----------------|----------------|
| Forward velocity `vx` | `[-1, 1]` | ± 3.0 m/s |
| Lateral velocity `vy` | `[-1, 1]` | ± 1.0 m/s |
| Yaw rate | `[-1, 1]` | ± 45 °/s |

### Reward Function

| Term | Weight | Description |
|------|--------|-------------|
| Progress | `+0.5` | Forward velocity in body frame (`vx_body`) |
| Collision | `-100` | Terminal penalty on impact |
| Smoothness | `-0.1` | Action jerk penalty `‖aₜ − aₜ₋₁‖` |
| Drift divergence | `-0.05` | VIO estimation error growth |

### Safety Monitor (4 layers)

1. **Velocity clamp** — hard physical limits on all axes
2. **Proximity scaling** — center 30% ROI depth < 1.5 m → linearly scale `vx` → 0.2×
3. **Altitude guard** — flag if deviation > 1.0 m from target altitude
4. **Emergency stop** — zero all commands on collision detection or comms timeout

---

## Results

### Ablation Study

Seven experiments across four ablation axes reveal that **single-frame observation without smoothness penalty** consistently achieves the best balance of forward progress and collision avoidance.

| Experiment | Frame Stack | Smoothness | Domain Rand. | Finding |
|------------|-------------|------------|--------------|---------|
| `baseline` | 4 | ✅ | ✅ | Reference |
| `frame_stack_1` | **1** | ✅ | ✅ | ✅ Best overall |
| `no_smoothness` | 1 | ❌ | ✅ | Comparable, less jerk |
| `no_dr` | 1 | ✅ | ❌ | Overfits to training env |
| `progress_only` | 1 | ❌ | ✅ | Unsafe (no collision term) |

> Training curves available via `tensorboard --logdir logs/`. Full numerical results in `docs/`.

### Algorithm Comparison (PPO vs SAC)

| Algorithm | Type | Steps | Advantage |
|-----------|------|-------|-----------|
| PPO | On-policy | 300k | Stable, well-understood |
| SAC | Off-policy | 200k | Higher sample efficiency |

> SAC training in progress on [`feature/sac-training`](https://github.com/Abood204/AI_QUADROTOR_GNSS_RESEARCH/tree/feature/sac-training). Results pending.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `success_rate` | % episodes reaching goal without collision |
| `collision_rate` | % episodes ending in collision |
| `avg_episode_reward` | Mean total reward per episode |
| `avg_forward_progress` | Mean distance traveled forward (m) |
| `avg_episode_length` | Mean timesteps before termination |
| `goal_distance` | Final distance from target at episode end (m) |
| `localisation_drift` | VIO position error accumulation (m) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Microsoft AirSim](https://github.com/microsoft/AirSim) binary (Unreal Engine not required for pre-built binaries)
- CUDA-capable GPU (recommended; CPU training is ~10× slower)

### Installation

```bash
git clone https://github.com/Abood204/AI_QUADROTOR_GNSS_RESEARCH.git
cd AI_QUADROTOR_GNSS_RESEARCH
pip install -e ".[dev]"
```

### 1. Verify Setup

```bash
# Run all 58 tests (no AirSim required)
make test

# Verify AirSim connection (AirSim must be running)
make check-env
```

### 2. Train

```bash
# Quick smoke test — confirms pipeline works (~2 min)
python -m src.training.train --total_timesteps 4096

# Recommended: PPO with domain randomization (~8–10 h on GPU)
python -m src.training.train --config configs/train_ppo_dr.yaml

# Fast iteration config (~3–4 h)
python -m src.training.train --config configs/train_ppo_fast.yaml

# Multi-environment rotation (AirSimNH + Blocks)
python -m src.training.train --config configs/train_ppo_multienv.yaml

# Resume from checkpoint
python -m src.training.train --resume logs/ppo/checkpoints/rl_model_XXXXX_steps.zip
```

### 3. Evaluate

```bash
# 20-episode comprehensive evaluation
python scripts/run_full_eval.py --model logs/ppo/best_model/best_model.zip

# View training curves
tensorboard --logdir logs/ppo/
```

### 4. Deploy

```bash
# Live inference with safety monitor (AirSim must be running)
python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip

# Disable safety monitor (research use only)
python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip --no_safety
```

---

## Experiments & Reproducibility

### Training Profiles

| Profile | Config | Steps | Purpose |
|---------|--------|-------|---------|
| Smoke test | *(CLI flag)* | 4k | Pipeline verification |
| Fast | `train_ppo_fast.yaml` | 500k | Rapid iteration |
| Base | `train_ppo.yaml` | 1M | Standard PPO baseline |
| DR | `train_ppo_dr.yaml` | 1M | Domain randomization |
| Multi-env | `train_ppo_multienv.yaml` | 1M | Environment generalization |
| SAC | `train_sac.yaml` | 200k | Off-policy comparison |

### Full Ablation Suite

```bash
# All 7 ablation experiments
python scripts/run_ablations.py

# Preview without running
python scripts/run_ablations.py --dry-run

# Specific experiments
python scripts/run_ablations.py --only frame_stack_1 no_smoothness no_dr
```

### Hyperparameter Optimization (Optuna)

```bash
python scripts/run_hyperparameter_sweep.py --n_trials 50 --study_name ppo_baseline
```

### Statistical Analysis

```bash
# ANOVA + paired t-tests across experimental conditions
python scripts/run_statistical_analysis.py \
  --baseline_dir logs/ppo/baseline \
  --comparison_dir logs/ppo/ablation_no_smoothness
```

### Parallel Training (Multi-GPU / Cluster)

```bash
# Launch 4 AirSim instances, then train with SubprocVecEnv
export AIRSIM_BIN=/path/to/AirSimNH.sh
bash scripts/launch_airsim_cluster.sh 4
python -m src.training.train --num_envs 4 --base_port 41451 --config configs/train_ppo_dr.yaml
bash scripts/launch_airsim_cluster.sh 4 --stop
```

---

## Project Structure

```
AI_QUADROTOR_GNSS_RESEARCH/
│
├── src/
│   ├── environments/        # Gymnasium env wrapper + VIO estimator + domain randomization
│   │   ├── airsim_env.py    # Main AirSimDroneEnv (lockstep, depth, safety)
│   │   └── rewards.py       # Pluggable reward functions
│   ├── training/            # PPO/SAC training loop, callbacks, multi-env scheduler
│   ├── evaluation/          # 7-metric eval protocol, statistical tests, plotting
│   ├── deployment/          # Live inference pipeline
│   ├── safety/              # Hard safety envelope (always-on)
│   └── control/             # PID baseline, AirSim interface wrappers
│
├── configs/
│   ├── train_ppo*.yaml      # PPO training configs
│   ├── train_sac.yaml       # SAC training config
│   ├── safety.yaml          # Safety monitor thresholds
│   ├── rewards/             # default, aggressive, cautious reward weights
│   ├── ablations/           # 7 ablation experiment configs
│   └── environments/        # AirSimNH (outdoor) and Blocks (indoor) configs
│
├── scripts/
│   ├── run_full_eval.py          # 20-episode comprehensive evaluation
│   ├── run_ablations.py          # Ablation study runner
│   ├── run_hyperparameter_sweep.py  # Optuna HPO
│   ├── run_statistical_analysis.py  # ANOVA + paired t-tests
│   ├── run_reward_sweep.py       # 3-config reward weight sweep
│   └── launch_airsim_cluster.sh  # Parallel training launcher
│
├── tests/                   # 58 unit tests — all AirSim-free (mock boundary)
├── ros_ws/                  # ROS2 workspace (rl_inference, safety_monitor nodes)
├── docs/                    # FYP documents, architecture, presentation
├── configs/                 # All YAML hyperparameter configs
│
├── Makefile                 # Common commands
├── pyproject.toml           # Package + dependency definition
└── CONTRIBUTING.md          # How to contribute
```

---

## Configuration

All hyperparameters are in YAML files under `configs/` — nothing hardcoded.

```yaml
# configs/train_ppo_dr.yaml (excerpt)
algorithm: ppo
total_timesteps: 1_000_000
n_steps: 2048
batch_size: 64
learning_rate: 3.0e-4
frame_stack: 4

domain_randomization:
  enabled: true
  depth_noise_std: 0.05
  spawn_radius: 5.0
  spawn_yaw_range: 180

reward:
  w_progress: 0.5
  w_collision: -100.0
  w_smoothness: -0.1
  w_drift: -0.05
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `make install` | Install package + dev deps (editable) |
| `make test` | Run all 58 unit tests |
| `make lint` | Ruff linting |
| `make train` | Train PPO with default config (1M steps) |
| `make check-env` | Verify AirSim + Gymnasium setup |
| `python scripts/run_full_eval.py --model <path>` | 20-episode evaluation |
| `python scripts/run_ablations.py` | Full ablation suite (7 experiments) |
| `python scripts/run_hyperparameter_sweep.py` | Optuna HPO sweep |
| `python scripts/run_statistical_analysis.py` | ANOVA + paired t-tests |
| `tensorboard --logdir logs/ppo/` | Training curves |

---

## Branch & Release Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, tested code — always passing CI |
| `develop` | Integration branch for ongoing work |
| `feature/sac-training` | SAC algorithm implementation and experiments |

| Tag | Description |
|-----|-------------|
| `v1.0-ppo-baseline` | PPO complete: training, ablations, evaluation, statistical analysis |
| `v1.1-sac-comparison` | *(upcoming)* PPO vs SAC comparative results |

---

## Roadmap

- [x] AirSim Gymnasium environment wrapper with lockstep simulation
- [x] Simulated VIO pipeline with drift and bias injection
- [x] PPO training with domain randomization and multi-environment rotation
- [x] 4-layer safety monitor
- [x] Comprehensive ablation study (7 experiments)
- [x] Statistical analysis (ANOVA + paired t-tests)
- [x] Optuna hyperparameter optimization
- [ ] SAC training and PPO vs SAC comparison (`feature/sac-training`)
- [ ] Physics domain randomization (mass, inertia, motor gains, wind)
- [ ] Full optical-flow VIO pipeline (replacing ground-truth kinematics proxy)
- [ ] Curriculum learning implementation

---

## Citation

If you use this work, please cite:

```bibtex
@misc{shoaeb2026quadrotor,
  author       = {Shoaeb, Abdalla},
  title        = {AI-Augmented Flight Control for Quadrotor UAV using Reinforcement Learning
                  in GNSS-Denied Environments},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Abood204/AI_QUADROTOR_GNSS_RESEARCH}},
  institution  = {Heriot-Watt University Dubai},
  note         = {FYP Thesis, B50PR}
}
```

---

## Acknowledgements

- [Microsoft AirSim](https://github.com/microsoft/AirSim) — high-fidelity UAV simulation
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO and SAC implementations
- [Optuna](https://optuna.org/) — hyperparameter optimization framework
- Heriot-Watt University Dubai — FYP supervision and support

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>FYP Thesis · Heriot-Watt University Dubai · 2025–2026</sub>
</div>
