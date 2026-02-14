# GNSS-Denied Quadrotor Autonomy

**AI-Augmented Flight Control for UAV in GNSS-Denied Conditions**

A depth-only PPO agent learns obstacle avoidance in AirSim, generalizing across environments via domain randomization and multi-environment training. Simulation-first architecture designed for later sim-to-real transfer.

**Student**: Abdalla Shoaeb (H00404752)
**Institution**: Heriot-Watt University Dubai
**Supervisor**: Dr. Mounis Shawgi

## Quick Start

```bash
# 1. Install (editable, with dev dependencies)
pip install -e ".[dev]"

# 2. Start AirSim simulator (must be running before any commands)

# 3. Verify environment
make check-env

# 4. Train PPO agent (default: 1M steps)
make train

# 5. Evaluate trained model
python -m src.evaluation.evaluate --model logs/ppo/best_model/best_model.zip

# 6. Deploy with safety monitor
python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip
```

## Architecture

```
Depth Camera (84x84) --> Depth Processing --> Frame Stack (4) --+
                                                                |--> PPO Policy --> Safety Monitor --> AirSim
IMU / Kinematics -----> Body-Frame Velocity [vx, vy, yaw] ----+
```

- **Observation**: Dict{depth: (84,84,4), velocity: (3,)} after frame stacking
- **Policy**: PPO with MultiInputPolicy (NatureCNN + MLP)
- **Action**: Continuous [-1,1]^3 scaled to [vx, vy, yaw_rate]
- **Safety**: Velocity clamp, proximity scaling, altitude guard, e-stop
- **Simulation**: Lockstep via `simContinueForTime(dt)` + `simPause(True)`

## Repository Structure

```
src/
  environments/     Gymnasium env wrapper + pluggable rewards
  training/         PPO training loop, callbacks, env scheduler
  evaluation/       Metrics, comparison tools, plotting
  deployment/       Live inference with safety monitor
  safety/           Hard safety envelope (velocity, proximity, altitude, e-stop)
  control/          PID controller, AirSim interface
  utils/            Logging, camera helpers
scripts/
  check_env.py      Gymnasium compliance checker
  run_reward_sweep.py   3-config reward weight experiments
  run_ablations.py      4 ablation experiments (reward, frames, DR, safety)
  run_full_eval.py      20-episode comprehensive evaluation
  export_onnx.py        ONNX export for sim-to-real transfer
configs/
  train_ppo.yaml        Base PPO training config
  train_ppo_dr.yaml     Training with domain randomization
  train_ppo_multienv.yaml  Multi-environment rotation training
  safety.yaml           Safety monitor parameters
  rewards/              Reward weight variants (default, aggressive, cautious)
  environments/         Per-environment configs (AirSimNH, Blocks)
  ablations/            Ablation experiment configs
tests/                  Unit tests (58 tests)
docs/                   FYP proposal, interim report, architecture
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `make install` | Install package in editable mode |
| `make check-env` | Verify AirSim + Gymnasium |
| `make train` | Train PPO (default config) |
| `make eval MODEL=<path>` | Evaluate trained model |
| `make deploy MODEL=<path>` | Deploy with safety monitor |
| `make test` | Run unit tests |
| `make lint` | Run ruff linter |
| `python -m scripts.run_reward_sweep` | Reward weight sweep (3 configs) |
| `python -m scripts.run_ablations` | 4 ablation experiments |
| `python -m scripts.run_full_eval --model <path>` | Full 20-episode evaluation |
| `python -m scripts.export_onnx --model <path>` | Export to ONNX |
| `tensorboard --logdir logs/ppo/` | View training curves |

## Requirements

- Python 3.11+
- AirSim simulator (with Unreal Engine)
- CUDA-capable GPU (recommended for training)
- See `pyproject.toml` for Python dependencies

## Key Design Decisions

1. **RL over Imitation Learning** — IL failed due to scripted expert ceiling; RL discovers policies from scratch
2. **Depth-only observation** — Invariant to lighting/texture, directly addresses "anywhere in the world"
3. **Lockstep simulation** — Deterministic training, no wall-clock bottleneck
4. **Safety as separate module** — Always-on envelope independent of policy exploration
5. **AirSim** — Existing pipeline, multiple environments, good depth simulation
