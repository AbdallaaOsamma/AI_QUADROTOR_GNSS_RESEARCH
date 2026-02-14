# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

**AirSim Quadrotor Autonomy via Reinforcement Learning**.
Transitions the system from legacy Imitation Learning (scripted expert) to an **End-to-End Reinforcement Learning (PPO)** stack. The drone learns to navigate complex environments from raw Depth/RGB inputs without human capability constraints.

## Environment Setup

- **Python 3.11.x** with virtual environment `.venv311`
- **Core ML**: `torch`, `stable-baselines3`, `gymnasium`, `tensorboard`
- **Simulation**: `airsim==1.8.1`, `msgpack-rpc-python`
- **Legacy**: `scikit-learn`, `pandas`, `opencv-contrib-python`

## Commands

All modules are run as packages from the project root:

```bash
# Verify AirSim environment and RL wrapper
python -m src.rl.check_env

# Train PPO Agent (End-to-End)
python -m src.rl.train_ppo --total_timesteps 1000000

# Visualize Training Progress
tensorboard --logdir logs/ppo/

# Deploy Trained Policy (Inference)
python -m src.rl.deploy_ppo --model logs/ppo/best_model.zip

# [LEGACY] Record expert data (Imitation Learning utils)
python -m src.ai.recorder_multimodal --config configs/record_centerline.yaml
```

## Architecture (New RL Stack)

### 1. RL Environment (`src/rl/env_airsim.py`)
- **Type**: `gymnasium.Env`
- **Observation**:
  - `Depth`: Stacked 84x84 frames (temporal awareness)
  - `Kinematics`: `[vx, vy, yaw_rate]` (proprioception)
- **Action**: Continuous `[target_vx, target_vy, target_yaw_rate]`
- **Reward**: `Progress(v_x) - Collision(100) - Smoothness(accel)`

### 2. Training (`src/rl/train_ppo.py`)
- **Algorithm**: PPO (Proximal Policy Optimization) from `stable-baselines3`
- **Policy**: `CnnPolicy` (NatureCNN for visual feature extraction)
- **Logging**: Tensorboard (`logs/`) and Checkpoints (`models/rl/`)

### Source Layout

- `src/rl/` - **(Active)** RL logic, environment wrappers, training/eval scripts.
- `src/ai/` - **(Legacy)** Old Imitation Learning stack (recorder, perception, behavior cloning).
- `src/control/` - Low-level AirSim PID and connection utilities.
- `configs/` - YAML operational configs.

## Coding Conventions

- **Inputs**: All ML models input `(N, C, H, W)` tensors normalized to `[0, 1]`.
- **Coordinates**: AirSim uses **NED** (Down is +Z). Altitude commands are negative.
- **Safety**: Do not remove `client.enableApiControl(True)` checks. Always include `try/finally` blocks to land the drone on error.

