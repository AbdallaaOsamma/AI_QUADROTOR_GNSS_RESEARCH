# GNSS-Denied Quadrotor Autonomy: Architecture & Implementation Plan

**Project**: AI-Augmented Flight Control for UAV in GNSS-Denied Conditions
**Student**: Abdalla Shoaeb (H00404752), Heriot-Watt University Dubai
**Date**: 2026-02-14
**Revision**: 1.0

---

## DELIVERABLE 1: REPO TRIAGE REPORT

### Classification Legend
- **KEEP**: File is sound, fits new architecture with minor updates
- **REFACTOR**: Core idea is good, needs significant rewrite for new system
- **REGENERATE**: Concept needed, but current implementation must be rewritten from scratch
- **DELETE**: Removes clutter, not needed in new system

### Root Files

| File | Verdict | Rationale |
|------|---------|-----------|
| `CLAUDE.md` | REGENERATE | Must reflect new architecture, not the old IL/RL hybrid |
| `README.md` | REGENERATE | Placeholder scaffold; needs real project documentation |
| `.gitignore` | KEEP | Add entries for `logs/`, `models/`, `*.onnx`, `wandb/` |
| `requirements.txt` | REGENERATE | Unpinned versions, missing new deps (wandb, onnxruntime, etc.) |
| `FYP Proposal.pdf` | KEEP | Reference document, move to `docs/` |
| `Interim Report H00404752.pdf` | KEEP | Reference document, move to `docs/` |
| `autonomy_audit.md` | KEEP | Valuable self-assessment, move to `docs/` |
| `fyp_report.txt` | DELETE | Auto-generated pipeline log, no value |
| `steup_notes.txt` | DELETE | Typo-named scratch notes |
| `merge_all_data.py` | DELETE | Root-level script, IL-specific dataset merge |
| `merge_datasets.py` | DELETE | Root-level script, IL-specific dataset merge |

### configs/

| File | Verdict | Rationale |
|------|---------|-----------|
| `default.yaml` | REFACTOR | PID gains and sim config are reusable; restructure as `configs/control/pid_gains.yaml` |
| `rl_ppo.yaml` | REFACTOR | Core RL config is solid; extend with domain randomization and multi-env settings |
| `rl_ppo_pretrained.yaml` | DELETE | Pretrained CNN approach is abandoned in favor of cleaner feature extractors |
| `pretrain_kaggle.yaml` | DELETE | Kaggle pretraining pipeline not needed with proper RL training |
| `record.yaml` | DELETE | Time-based scripted expert recording; superseded by RL |
| `record_centerline.yaml` | DELETE | Centerline-following expert; superseded by RL |
| `train_sklearn.yaml` | DELETE | sklearn MLP config; legacy IL |
| `train_cnn.yaml` | DELETE | CNN BC training config; legacy IL |
| `train_cnn_centerline.yaml` | DELETE | Variant CNN config; legacy IL |
| `train_cnn_fixed.yaml` | DELETE | Variant CNN config; legacy IL |
| `run_cnn_deployment.yaml` | DELETE | CNN deployment config; legacy IL |

### src/control/

| File | Verdict | Rationale |
|------|---------|-----------|
| `pid.py` | KEEP | Clean, correct PID with anti-windup. Reuse directly. |
| `controller.py` | REFACTOR | `connect()`, `takeoff()` are reusable patterns. `hover_loop` and `hover_capture` become test utilities. Remove `moveByVelocityBodyFrameAsync` (altitude drift bug). |
| `forward_test.py` | DELETE | Simple forward flight test, replaced by proper test harness |
| `__init__.py` | KEEP | Update exports |

### src/utils/

| File | Verdict | Rationale |
|------|---------|-----------|
| `logging.py` | REFACTOR | EpisodeLogger is useful but needs structured logging (JSON, TensorBoard integration) |
| `airsim_cam.py` | KEEP | `grab_rgb_frame` and `save_frame_bgr` are clean helpers |
| `vision_diagnostic.py` | DELETE | Canny/Hough diagnostic tool; legacy |
| `run_full_pipeline.py` | DELETE | Hardcoded Windows paths, IL-specific pipeline orchestration |
| `__init__.py` | KEEP | Update exports |

### src/ai/ (Legacy IL Stack)

| File | Verdict | Rationale |
|------|---------|-----------|
| `model_cnn.py` | DELETE | NavigationCNN was for IL; RL uses SB3's built-in feature extractors or custom ones |
| `recorder.py` | DELETE | ScriptedExpert time-based recording; superseded |
| `recorder_centerline.py` | DELETE | Centerline-following recorder; superseded |
| `recorder_multimodal.py` | DELETE | Multi-modal recorder; superseded. The PerceptionResult pattern is worth noting but will be redesigned |
| `perception.py` | DELETE | Canny/Hough/depth fallback pipeline; the "Rube Goldberg" correctly identified in audit. Depth zone navigation concept reappears in RL reward shaping |
| `train_sklearn.py` | DELETE | sklearn MLP training; legacy |
| `train_cnn.py` | DELETE | CNN BC training; legacy. `NavigationDataset` pattern is standard PyTorch but not needed |
| `policy_run_sklearn.py` | DELETE | sklearn deployment; legacy |
| `policy_run_cnn.py` | DELETE | CNN deployment with hard-coded overrides; the exact anti-pattern identified in audit |
| `synthetic_expert.py` | DELETE | Waypoint trajectory generator; replaced by RL exploration |
| `synthetic_expert_v2.py` | DELETE | Advanced trajectory generator; same |
| `dataset_kaggle.py` | DELETE | External dataset loader; not needed |
| `pretrain_corner_cnn.py` | DELETE | Pretraining on Kaggle data; not needed |
| `__init__.py` | DELETE | Entire package deleted |

### src/rl/

| File | Verdict | Rationale |
|------|---------|-----------|
| `env_airsim.py` | REFACTOR | Core Gymnasium env is well-built. Needs: (1) domain randomization hooks, (2) configurable reward functions, (3) multi-sensor observation space, (4) curriculum support |
| `train_ppo.py` | REFACTOR | Clean SB3 training loop. Extend with: WandB logging, curriculum callbacks, multi-env configs |
| `deploy_ppo.py` | REFACTOR | Good deployment pattern. Needs: safety watchdog, telemetry recording, graceful degradation |
| `test_full_nav.py` | REFACTOR | Trajectory logging and summary stats are valuable. Extend with standardized metrics |
| `check_env.py` | KEEP | Gymnasium compliance checker; useful as-is |
| `train_ppo_pretrained.py` | DELETE | Pretrained extractor approach abandoned; custom feature extractors built differently |
| `__init__.py` | KEEP | Update exports |

### sim_env/

| File | Verdict | Rationale |
|------|---------|-----------|
| `create_parking.py` | DELETE | Unreal Engine parking lot generator; environment generation handled differently |
| `README.md` | REGENERATE | Need proper simulator setup instructions |

### data/

| File | Verdict | Rationale |
|------|---------|-----------|
| `expert/` (all contents) | DELETE | 81 hover frames + 1 CSV; no value for RL training. Training data generated during RL episodes |
| `diagnostic/` | DELETE | Vision diagnostic output |

### Summary Counts
- **KEEP**: 7 files
- **REFACTOR**: 8 files
- **REGENERATE**: 4 files
- **DELETE**: 28 files

---

## DELIVERABLE 2: TARGET ARCHITECTURE

### System-Level Architecture

```
===================================================================================
                    GNSS-DENIED QUADROTOR AUTONOMY STACK
===================================================================================

 SENSORS (AirSim)           PERCEPTION              STATE ESTIMATION
+------------------+    +-------------------+    +---------------------+
| Forward Depth    |--->| Depth Processing  |--->|                     |
| Camera (84x84)   |    | - Normalize 0-20m |    | Velocity Estimator  |
+------------------+    | - Resize/Stack    |    | (AirSim kinematics  |
                        +-------------------+    |  as ground truth;   |
+------------------+                             |  VIO placeholder    |
| IMU / Kinematics |---------------------------->|  for physical)      |
| vx, vy, vz,     |                             +---------------------+
| yaw_rate, orient |                                       |
+------------------+                                       v
                                                 +---------------------+
                                                 | Observation Builder |
                                                 | Dict{               |
                                                 |   depth: (4,84,84) |
                                                 |   velocity: (3,)   |
                                                 | }                   |
                                                 +---------------------+
                                                           |
                                                           v
                                              +--------------------------+
                                              |     POLICY (PPO)         |
                                              | MultiInputPolicy         |
                                              | - NatureCNN for depth    |
                                              | - MLP for velocity       |
                                              | Output: [vx, vy, yaw]   |
                                              +--------------------------+
                                                           |
                                                           v
                                              +--------------------------+
                                              |    SAFETY MONITOR        |
                                              | - Collision proximity    |
                                              | - Velocity limits        |
                                              | - Altitude bounds        |
                                              | - Emergency stop         |
                                              +--------------------------+
                                                           |
                                                           v
                                              +--------------------------+
                                              |   LOW-LEVEL CONTROL      |
                                              | moveByVelocityZBody      |
                                              | AsyncFrame(vx,vy,z,dt)  |
                                              | - Altitude hold (z_cmd) |
                                              | - Yaw rate mode         |
                                              +--------------------------+
                                                           |
                                                           v
                                              +--------------------------+
                                              |     AIRSIM PHYSICS       |
                                              | Lockstep simulation      |
                                              | simContinueForTime(dt)   |
                                              | simPause(True)           |
                                              +--------------------------+

===================================================================================
         TRAINING INFRASTRUCTURE               EVALUATION
===================================================================================

+---------------------------+           +---------------------------+
| Domain Randomization      |           | Metrics Engine            |
| - Lighting variation      |           | - Distance before crash   |
| - Texture randomization   |           | - Collision rate          |
| - Object placement        |           | - Path smoothness         |
| - Weather effects         |           | - Speed efficiency        |
| - Sensor noise injection  |           | - Exploration coverage    |
+---------------------------+           +---------------------------+
            |                                       |
            v                                       v
+---------------------------+           +---------------------------+
| Curriculum Manager        |           | Visualization             |
| - Phase 1: Open space     |           | - Trajectory plots        |
| - Phase 2: Wide corridors |           | - Reward curves           |
| - Phase 3: Obstacles      |           | - Heatmaps                |
| - Phase 4: Tight spaces   |           | - Video recordings        |
+---------------------------+           +---------------------------+
            |
            v
+---------------------------+
| Multi-Environment Pool    |
| - AirSimNH (neighborhood) |
| - Blocks (indoor)         |
| - Custom UE environments  |
| - Randomized variants     |
+---------------------------+
```

### Module Descriptions

#### 1. Observation Pipeline (`src/perception/`)
**Purpose**: Transform raw simulator outputs into policy-ready observations.

- **Depth Processor**: Captures AirSim DepthPerspective, clips to [0, 20m], normalizes to [0, 1], resizes to 84x84. Frame-stacking (4 frames) provides temporal/motion cues.
- **Kinematics Extractor**: Reads body-frame velocity [vx, vy] and yaw_rate from AirSim state. Global-to-body rotation via yaw angle.
- **Observation Space**: `Dict{"depth": Box(0,1, shape=(4,84,84)), "velocity": Box(-inf,inf, shape=(3,))}` after VecFrameStack.

#### 2. Policy (`src/policy/`)
**Purpose**: Map observations to continuous velocity commands.

- **Architecture**: PPO with MultiInputPolicy (SB3). NatureCNN processes depth stack. MLP processes velocity. Outputs 3D continuous action [-1, 1] scaled to [max_vx, max_vy, max_yaw_rate].
- **Why PPO**: On-policy, stable, well-tested for continuous control. Alternative SAC considered as Plan B for sample efficiency.

#### 3. Safety Monitor (`src/safety/`)
**Purpose**: Hard safety envelope around policy outputs.

- **Velocity Clamp**: Enforce |vx| <= 3.0, |vy| <= 1.0, |yaw_rate| <= 45 deg/s
- **Proximity Alert**: If min_depth < 1.5m in center ROI, scale down vx proportionally
- **Altitude Guard**: If |altitude - target| > 1.0m, override to altitude recovery
- **Emergency Stop**: If collision detected or comms timeout > 500ms, execute hover-then-land

#### 4. Low-Level Control (`src/control/`)
**Purpose**: Interface to AirSim flight controller.

- **Altitude Hold**: `moveByVelocityZBodyFrameAsync` with target_z = -target_alt (NED)
- **PID**: Retained for altitude hold verification and optional outer-loop control
- **Lockstep**: `simContinueForTime(dt)` + `simPause(True)` for deterministic training

#### 5. Training Infrastructure (`src/training/`)
**Purpose**: Manage RL training with curriculum and domain randomization.

- **Curriculum**: 4-phase difficulty ladder (open -> corridors -> obstacles -> tight spaces)
- **Domain Randomization**: Lighting, textures, object positions, sensor noise
- **Logging**: WandB + TensorBoard for training curves, episode stats, video rollouts
- **Checkpointing**: Best model by eval reward, periodic saves

#### 6. Evaluation (`src/evaluation/`)
**Purpose**: Standardized assessment of trained policies.

- **Metrics**: Distance before collision, collision rate, path smoothness (jerk), average speed, exploration area
- **Protocols**: Fixed-seed reproducible episodes across multiple environments
- **Outputs**: CSV telemetry, JSON summaries, trajectory plots, comparison tables

### Design Decisions and Justifications

**Decision 1: RL (PPO) over Imitation Learning**
- IL failed because the scripted expert was a Canny edge detector, not a real navigation policy
- RL discovers policies from scratch via reward maximization, no expert ceiling
- PPO is stable, well-documented, and works for continuous control

**Decision 2: Depth-only (no RGB) for primary observation**
- Depth is invariant to lighting, texture, color -- directly addresses "anywhere in the world"
- Reduces observation complexity (1 channel vs 3)
- Depth encodes obstacle geometry directly
- RGB can be added later as auxiliary input if needed (Plan B)

**Decision 3: Keep AirSim as primary simulator**
- Existing working pipeline and institutional knowledge
- Multiple built-in environments (AirSimNH, Blocks, etc.)
- Good depth sensor simulation
- Domain randomization via Unreal Engine material swaps
- Limitation: AirSim is deprecated by Microsoft (last release 2022), but still functional for research
- Plan B: Evaluate Isaac Sim or Habitat if AirSim becomes blocking

**Decision 4: Lockstep simulation over real-time**
- Deterministic training (critical for reproducibility)
- No wall-clock bottleneck on slow hardware
- Clean separation of sim time from real time
- Already proven in existing `env_airsim.py`

**Decision 5: Safety Monitor as separate module, not embedded in policy**
- Policy can be unsafe during exploration; safety monitor is always-on
- Clean separation of concerns
- Easy to tune safety margins independently
- Maps directly to physical system safety requirements

---

## DELIVERABLE 3: IMPLEMENTATION PLAN (4 WEEKS)

> **Constraint**: 28 calendar days. Phase 3 (goal-conditioning, observation enrichment) is CUT.
> Advanced features (Rungs 5-7) become "if time permits" stretch goals, not commitments.
> Focus: clean foundation -> working baseline -> generalization evidence -> evaluation + report.

### Week 1: Foundation (Days 1-7)

**Days 1-2: Clean Repo + Skeleton**
- [ ] Execute all 28 file deletions per triage report
- [ ] Create new folder structure (Deliverable 4)
- [ ] Write `pyproject.toml` with pinned dependencies
- [ ] Write `Makefile` with `install`, `check-env`, `train`, `eval`, `deploy`
- [ ] Move retained files to new locations, fix all imports
- [ ] **Gate**: `make install && make check-env` passes

**Days 3-5: Refactor Core Environment + Safety**
- [ ] Refactor `env_airsim.py` -> `src/environments/airsim_env.py` with pluggable reward classes and domain randomization hooks
- [ ] Create `src/safety/monitor.py` (velocity clamp, proximity scaling, altitude guard, e-stop)
- [ ] Create `src/control/airsim_interface.py` (connect, takeoff, land wrappers)
- [ ] Write unit tests: `tests/test_rewards.py`, `tests/test_safety.py`, `tests/test_pid.py`
- [ ] **Gate**: env resets/steps 100x without error, all unit tests pass

**Days 6-7: Training + Evaluation Infrastructure**
- [ ] Refactor `train_ppo.py` -> `src/training/train.py` with TensorBoard + optional WandB, curriculum callback, best-model checkpoint
- [ ] Create `src/evaluation/evaluate.py` with standardized 20-episode protocol, JSON summary, trajectory plots
- [ ] Create `src/evaluation/metrics.py` (DBC, collision rate, speed, smoothness, survival, coverage)
- [ ] Kick off first overnight training run (500k steps) to validate pipeline
- [ ] **Gate**: 10k-step smoke train completes, eval produces JSON + plots

### Week 2: Baseline + Multi-Environment (Days 8-14)

**Days 8-10: PPO Baseline (Rung 1)**
- [ ] Analyze overnight training results, tune reward weights if needed
- [ ] Run 2-3 reward weight experiments in parallel (adjust w_progress, w_collision, w_smoothness)
- [ ] Train best config for 500k-1M steps
- [ ] Run full evaluation (20 episodes, fixed seeds)
- [ ] **Gate**: >50m average DBC in AirSimNH, documented reward curves

**Days 11-12: Safety Integration + PID-only Baseline**
- [ ] Integrate SafetyMonitor into `src/deployment/deploy.py`
- [ ] Record PID-only baseline (no AI, just hover + slow forward) for ablation comparison
- [ ] Run eval with safety on vs off, document collision rate difference
- [ ] **Gate**: Safety reduces collision rate by >30%, PID-only baseline documented

**Days 13-14: Multi-Environment Setup + Training**
- [ ] Set up 2nd AirSim environment (Blocks indoor)
- [ ] Implement environment rotation in training (switch every N episodes)
- [ ] Kick off multi-env training run overnight (500k+ steps)
- [ ] Add sensor noise injection (Gaussian noise on depth, ±5%)
- [ ] **Gate**: Training runs stable across 2 environments without crashes

### Week 3: Generalization + Ablations (Days 15-21)

**Days 15-17: Multi-Env Results + Domain Randomization**
- [ ] Evaluate multi-env policy on both training envs + measure cross-env transfer ratio
- [ ] Add domain randomization: spawn position randomization, depth noise variation
- [ ] Train DR variant (500k steps)
- [ ] **Gate**: Cross-env transfer ratio >0.5 (stretch: >0.7)

**Days 18-21: Ablation Experiments**
Run these in parallel (each is a separate training run):
- [ ] **Ablation 1**: Reward components (full vs no-smoothness vs progress-only)
- [ ] **Ablation 2**: Frame stack depth (1 vs 4 frames)
- [ ] **Ablation 3**: Safety monitor effect (with vs without)
- [ ] **Ablation 4**: Domain randomization (DR vs no-DR on held-out conditions)
- [ ] Collect all ablation results into comparison tables and plots
- [ ] **Gate**: 4 ablation experiments completed with documented results

### Week 4: Evaluation + Report (Days 22-28)

**Days 22-24: Comprehensive Evaluation**
- [ ] Run full evaluation protocol on best model: 20 episodes x each environment
- [ ] Generate all required plots (training curves, trajectories, ablation comparisons, failure analysis)
- [ ] Export best policy to ONNX format
- [ ] Document worst-3 episodes with failure taxonomy
- [ ] Record 2-3 demo videos (best run, typical run, failure case)
- [ ] **Gate**: All figures and tables generated

**Days 25-27: Final Report**
- [ ] Write FYP report sections: Introduction, Background, Methodology, Results, Discussion, Conclusion
- [ ] Insert all figures, tables, ablation results
- [ ] Honest limitations section (what "anywhere" means and doesn't mean)
- [ ] Clean repo: update README, verify `make install && make train && make eval` reproduces

**Day 28: Polish + Submit**
- [ ] Proofread report
- [ ] Final git tag `v2.0-submission`
- [ ] Verify all artifacts (models, logs, plots) are reproducible or documented
- [ ] **Gate**: Repository is self-contained, report is complete

### What Was Cut (vs. 12-week plan)

| Cut Item | Why | Recovery Path |
|----------|-----|---------------|
| Curriculum learning (Rung 5) | 1-week feature, not essential for FYP | Mention as future work |
| Goal-conditioned navigation (Rung 6) | Research-grade feature, too risky for 4 weeks | Mention as future work |
| Observation enrichment (RGB, goal vector) | Diminishing returns vs. time cost | Mention as future work |
| ONNX physical deployment guide | Nice-to-have, not graded | Keep ONNX export, cut the guide |
| WandB (required -> optional) | TensorBoard is sufficient | Use WandB only if already set up |

### Critical Path (4-week)

```
Week 1 (Foundation) --> Week 2 Days 8-10 (Baseline) --> Week 4 (Eval + Report)
                                    |
                                    +--> Week 2-3 (Multi-env + Ablations) --> Week 4
```

**Minimum viable in 4 weeks**: Week 1 + Baseline (Days 8-10) + Week 4 = Rung 1 with full evaluation. This is achievable even if multi-env and ablations slip.

**Realistic target**: Rungs 1-3 (baseline + multi-env + domain randomization) + 4 ablation experiments. This is a Merit/Distinction-level submission.

---

## DELIVERABLE 4: FOLDER STRUCTURE + INTERFACES

### Proposed Repository Structure

```
airsim-gnss-denied-quad/
|
|-- pyproject.toml              # Project metadata + dependencies (replaces requirements.txt)
|-- Makefile                    # One-command: install, train, eval, deploy
|-- CLAUDE.md                   # Updated project instructions
|-- README.md                   # Comprehensive project documentation
|-- .gitignore                  # Updated ignore rules
|
|-- docs/
|   |-- FYP Proposal.pdf
|   |-- Interim Report.pdf
|   |-- autonomy_audit.md       # Preserved self-assessment
|   |-- architecture.md         # This document (or generated from it)
|
|-- configs/
|   |-- default.yaml            # Master config (inheritable)
|   |-- train_ppo.yaml          # PPO hyperparameters
|   |-- environments/
|   |   |-- airsim_nh.yaml      # AirSimNH environment config
|   |   |-- airsim_blocks.yaml  # Blocks environment config
|   |-- rewards/
|   |   |-- standard.yaml       # Default reward weights
|   |   |-- exploration.yaml    # Exploration-heavy rewards
|   |-- curriculum/
|       |-- 4phase.yaml         # 4-phase curriculum definition
|
|-- src/
|   |-- __init__.py
|   |
|   |-- environments/           # Gymnasium environments
|   |   |-- __init__.py
|   |   |-- airsim_env.py       # Core AirSimDroneEnv (refactored from src/rl/env_airsim.py)
|   |   |-- wrappers.py         # Domain randomization, curriculum, observation wrappers
|   |   |-- rewards.py          # Pluggable reward functions
|   |
|   |-- control/                # Low-level control
|   |   |-- __init__.py
|   |   |-- pid.py              # PID controller (kept from src/control/pid.py)
|   |   |-- airsim_interface.py # AirSim connection, takeoff, land, command helpers
|   |
|   |-- safety/                 # Safety monitoring
|   |   |-- __init__.py
|   |   |-- monitor.py          # SafetyMonitor: velocity clamp, proximity, altitude, e-stop
|   |
|   |-- training/               # Training scripts and callbacks
|   |   |-- __init__.py
|   |   |-- train.py            # Main PPO training entry point
|   |   |-- callbacks.py        # Curriculum, logging, domain randomization callbacks
|   |   |-- schedulers.py       # Learning rate and curriculum schedulers
|   |
|   |-- evaluation/             # Evaluation and metrics
|   |   |-- __init__.py
|   |   |-- evaluate.py         # Run evaluation episodes, compute metrics
|   |   |-- metrics.py          # Metric definitions and computation
|   |   |-- visualize.py        # Trajectory plots, reward curves, heatmaps
|   |
|   |-- deployment/             # Inference and export
|   |   |-- __init__.py
|   |   |-- deploy.py           # Deploy trained policy (real-time inference loop)
|   |   |-- export_onnx.py      # Export SB3 model to ONNX
|   |
|   |-- utils/                  # Shared utilities
|       |-- __init__.py
|       |-- logging.py          # Structured logging (refactored from src/utils/logging.py)
|       |-- airsim_cam.py       # Camera helpers (kept from src/utils/airsim_cam.py)
|       |-- config.py           # YAML config loading and merging
|
|-- tests/                      # Unit and integration tests
|   |-- __init__.py
|   |-- test_env.py             # Environment compliance tests
|   |-- test_rewards.py         # Reward function unit tests
|   |-- test_safety.py          # Safety monitor tests
|   |-- test_pid.py             # PID controller tests
|
|-- scripts/                    # Convenience scripts
|   |-- check_env.py            # Quick environment verification
|   |-- record_video.py         # Record deployment video
|
|-- logs/                       # Training logs (gitignored)
|-- models/                     # Saved models (gitignored)
|-- data/                       # Any generated data (gitignored)
```

### Key Interfaces

#### AirSimDroneEnv (`src/environments/airsim_env.py`)

```python
class AirSimDroneEnv(gymnasium.Env):
    """
    Gymnasium environment for quadrotor navigation in AirSim.

    Observation Space (Dict):
        "depth": Box(0, 1, shape=(H, W, 1), float32)  # Single depth frame
        "velocity": Box(-inf, inf, shape=(3,), float32)  # [vx_body, vy_body, yaw_rate]

    Action Space (Box):
        [-1, 1]^3 -> scaled to [vx, vy, yaw_rate] physical limits

    Frame stacking handled by VecFrameStack wrapper (n_stack=4).
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Full config dict with keys: env, reward, domain_randomization
        """

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        """Reset environment. Retries on spawn collision."""

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute action via lockstep simulation.
        Returns: obs, reward, terminated, truncated, info
        Info keys: r_progress, r_collision, r_smoothness, vx_body,
                   step_count, position_xyz, yaw_deg
        """

    def close(self):
        """Disarm and release API control."""
```

#### RewardFunction (`src/environments/rewards.py`)

```python
class RewardFunction(ABC):
    """Abstract base for reward computation."""

    @abstractmethod
    def compute(self, obs: dict, action: np.ndarray, prev_action: np.ndarray,
                collision: bool, info: dict) -> tuple[float, dict]:
        """
        Returns: (total_reward, reward_components_dict)
        """

class StandardReward(RewardFunction):
    """Progress + collision + smoothness reward."""

class ExplorationReward(RewardFunction):
    """Adds coverage area bonus to standard reward."""

class ProximityReward(RewardFunction):
    """Adds continuous wall-proximity penalty."""
```

#### SafetyMonitor (`src/safety/monitor.py`)

```python
class SafetyMonitor:
    """
    Always-on safety envelope around policy outputs.

    Enforces:
        1. Velocity magnitude limits
        2. Proximity-based speed reduction
        3. Altitude deviation recovery
        4. Emergency stop on collision/timeout
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Safety config with keys: max_vx, max_vy, max_yaw_rate,
                 proximity_threshold_m, altitude_tolerance_m, comms_timeout_ms
        """

    def filter_action(self, action: np.ndarray, depth_image: np.ndarray,
                      altitude: float, target_altitude: float) -> np.ndarray:
        """
        Apply safety constraints to raw policy action.
        Returns: safe_action (same shape as input)
        """

    def check_emergency(self, collision: bool, last_comms_time: float) -> bool:
        """Returns True if emergency landing should be triggered."""
```

#### Evaluation Protocol (`src/evaluation/evaluate.py`)

```python
class EvaluationProtocol:
    """
    Standardized evaluation of navigation policies.

    Runs N episodes with fixed seeds, computes metrics, generates outputs.
    """

    def __init__(self, env_cfg: dict, n_episodes: int = 20, seeds: list = None):
        pass

    def evaluate(self, model_path: str) -> EvaluationResult:
        """
        Run evaluation episodes.
        Returns: EvaluationResult with metrics, trajectories, summary.
        """

    def compare(self, results: list[EvaluationResult]) -> ComparisonReport:
        """Compare multiple models/configs."""

@dataclass
class EvaluationResult:
    model_path: str
    episodes: list[EpisodeData]

    # Aggregate metrics
    mean_distance_m: float
    mean_collision_rate: float      # collisions per 100m
    mean_speed_ms: float
    mean_smoothness: float          # mean jerk magnitude
    mean_episode_reward: float
    coverage_area_m2: float         # bounding box of all positions

    # Per-episode
    distances: list[float]
    rewards: list[float]
    durations: list[float]
```

#### Config Schema (`configs/train_ppo.yaml`)

```yaml
# Master training configuration
env:
  ip: ""
  image_shape: [84, 84, 1]
  target_alt: 3.0
  max_vx: 3.0
  max_vy: 1.0
  max_yaw_rate_deg: 45
  dt: 0.1
  max_steps: 1024

reward:
  type: "standard"  # standard | exploration | proximity
  w_progress: 0.5
  w_collision: -100.0
  w_smoothness: -0.1
  w_proximity: -0.05     # only for proximity reward
  w_exploration: 0.01    # only for exploration reward

ppo:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  total_timesteps: 1000000

frame_stack: 4

domain_randomization:
  enabled: false
  lighting_range: [0.5, 2.0]    # multiplier
  texture_swap_prob: 0.1
  noise_std: 0.02               # Gaussian noise on depth

curriculum:
  enabled: false
  phases:
    - name: "open"
      reward_scale: 1.0
      max_steps: 512
      trigger_reward: 100.0
    - name: "corridors"
      reward_scale: 1.0
      max_steps: 768
      trigger_reward: 150.0
    - name: "obstacles"
      reward_scale: 1.0
      max_steps: 1024
      trigger_reward: 200.0
    - name: "tight"
      reward_scale: 1.0
      max_steps: 1024
      trigger_reward: null  # final phase

safety:
  max_vx: 3.0
  max_vy: 1.0
  max_yaw_rate_deg: 45
  proximity_threshold_m: 1.5
  altitude_tolerance_m: 1.0
  comms_timeout_ms: 500

output:
  log_dir: "logs/ppo"
  checkpoint_freq: 10000
  eval_freq: 5000
  eval_episodes: 5
  wandb_project: "gnss-denied-quad"
  wandb_enabled: false
```

---

## DELIVERABLE 5: SIMULATION SETUP

### Primary Simulator: AirSim (Microsoft)

**Justification**:
1. Existing working pipeline with proven lockstep simulation
2. Multiple built-in environments (AirSimNH, Blocks, LandscapeMountains)
3. High-fidelity depth sensor simulation
4. Python API with full vehicle control
5. Unreal Engine integration for custom environments
6. Institutional knowledge in this project (bugs found, workarounds documented)

**Known Limitations**:
- Deprecated by Microsoft (last release: v1.8.1, 2022)
- No active maintenance; bugs must be worked around
- Limited dynamic obstacle support
- Domain randomization requires Unreal Engine scripting

**Plan B**: If AirSim becomes a blocker:
- **Flightmare** (ETH Zurich): Fast RL training, Unity-based, but less realistic rendering
- **Isaac Sim** (NVIDIA): Excellent physics and DR, but heavy GPU requirements
- **Gymnasium-based mock env**: For unit testing and fast iteration without simulator

### Setup Steps

#### Prerequisites
1. Windows 10/11 with NVIDIA GPU (GTX 1060+ for training)
2. Conda or Python 3.11 environment
3. Unreal Engine 4.27 (only if creating custom environments)

#### Step 1: Install AirSim Binary

```bash
# Option A: Pre-built binary (recommended for getting started)
# Download from: https://github.com/microsoft/AirSim/releases/tag/v1.8.1
# Extract AirSimNH (Neighborhood environment)

# Option B: Multiple environments
# Also download: Blocks, LandscapeMountains, ZhangJiaJie
```

#### Step 2: Configure AirSim Settings

Create/update `~/Documents/AirSim/settings.json`:

```json
{
    "SeeDocsAt": "https://microsoft.github.io/AirSim/settings/",
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "ClockType": "SteppableClock",
    "ViewMode": "NoDisplay",
    "Vehicles": {
        "Drone0": {
            "VehicleType": "SimpleFlight",
            "X": 0, "Y": 0, "Z": -3,
            "Yaw": 0,
            "Cameras": {
                "front_center": {
                    "CaptureSettings": [
                        {
                            "Width": 256,
                            "Height": 256,
                            "FOV_Degrees": 90,
                            "ImageType": 0
                        },
                        {
                            "Width": 256,
                            "Height": 256,
                            "FOV_Degrees": 90,
                            "ImageType": 2
                        }
                    ],
                    "X": 0.25, "Y": 0, "Z": 0,
                    "Pitch": -10, "Roll": 0, "Yaw": 0
                }
            }
        }
    }
}
```

Key settings:
- `ClockType: "SteppableClock"` -- enables lockstep simulation
- `ViewMode: "NoDisplay"` -- headless for training (faster)
- Depth camera (ImageType 2) at 256x256 (downsampled to 84x84 in env)

#### Step 3: Install Python Environment

```bash
# Create conda environment
conda create -n airsim python=3.11 -y
conda activate airsim

# Install project
pip install -e ".[dev]"

# Or with make
make install
```

#### Step 4: Verify Setup

```bash
# 1. Launch AirSim environment (in separate terminal)
./AirSimNH/AirSimNH.exe

# 2. Check environment connectivity
make check-env
# Expected: "Environment is compliant!" with 10 successful steps

# 3. Run smoke test (short training)
make train ARGS="--total_timesteps 4096"
# Expected: PPO trains 4096 steps, saves model
```

#### Step 5: Multi-Environment Setup (Phase 2+)

```bash
# Download additional environments
# Blocks: indoor grid environment
# LandscapeMountains: outdoor with varied terrain

# Configure environment selector in configs/environments/
# Each environment has its own YAML with spawn points, difficulty, etc.
```

### Makefile Commands

```makefile
.PHONY: install check-env train eval deploy clean

install:
    pip install -e ".[dev]"

check-env:
    python -m scripts.check_env

train:
    python -m src.training.train --config configs/train_ppo.yaml $(ARGS)

eval:
    python -m src.evaluation.evaluate --config configs/train_ppo.yaml $(ARGS)

deploy:
    python -m src.deployment.deploy $(ARGS)

test:
    pytest tests/ -v

clean:
    rm -rf logs/ models/ data/ __pycache__ .pytest_cache
```

---

## DELIVERABLE 6: BASELINE TO SOTA LADDER

### Rung 0: Sanity Check (PID-Only Hover)
**What**: Drone holds altitude and position using PID only, no AI.
**Purpose**: Verify simulator connectivity, PID tuning, telemetry logging.
**Metric**: Altitude hold error < 0.3m over 60 seconds.
**Status**: Already achieved in existing codebase.

### Rung 1: PPO Baseline -- Single Environment Forward Flight
**What**: PPO learns to fly forward without collisions in AirSimNH.
**Observation**: 4-frame stacked depth (84x84) + body velocity (3,).
**Action**: Continuous [vx, vy, yaw_rate].
**Reward**: Progress (vx) - Collision (100) - Smoothness (jerk).
**Target**: >50m average distance before collision, >1.0 m/s average speed.
**Training**: 500k-1M timesteps, single environment.
**This is the minimum viable result for the FYP.**

### Rung 2: Reward Engineering -- Proximity + Exploration
**What**: Add continuous proximity penalty and exploration bonus to reward.
**Why**: Forward-only reward leads to "highway hugging" -- drone avoids walls reactively but never explores corridors or makes turns.
**Reward additions**:
- Proximity: `-0.05 * (1.0 / min_center_depth)` -- gentle push away from walls
- Exploration: `+0.01 * new_area_covered` -- incentivize visiting new grid cells
**Target**: >100m distance, >70% of reachable area visited in 1024 steps.

### Rung 3: Multi-Environment Generalization
**What**: Train across 2-3 AirSim environments (AirSimNH outdoor, Blocks indoor, custom).
**Why**: Single-environment training overfits to specific geometry and textures.
**Method**: Environment rotation every episode or every N episodes.
**Target**: Policy trained on {A, B} achieves >70% of single-env performance when tested on C.
**Measured by**: Cross-environment transfer ratio.

### Rung 4: Domain Randomization
**What**: Randomize visual appearance during training.
**Variations**:
- Lighting intensity and direction (AirSim time-of-day API)
- Gaussian noise on depth sensor (simulates sensor imperfections)
- Spawn position randomization
**Why**: Makes policy robust to visual domain shift.
**Target**: <20% performance drop when testing with unseen lighting/noise conditions.

### Rung 5: Curriculum Learning
**What**: Progressive difficulty increase during training.
**Phases**:
1. Open space (max_steps=512, sparse obstacles)
2. Wide corridors (max_steps=768, walls at 5m+ spacing)
3. Cluttered (max_steps=1024, obstacles in path)
4. Tight spaces (max_steps=1024, walls at 2-3m spacing)
**Why**: Direct training on hard environments fails (collision rate too high for learning signal).
**Target**: Curriculum reaches higher final reward or reaches Rung 1 threshold 2x faster.

### Rung 6: Goal-Conditioned Navigation
**What**: Policy receives a target direction vector and navigates toward it.
**Observation addition**: `goal_direction: Box(-1, 1, shape=(2,))` -- unit vector in body frame.
**Reward addition**: `+0.3 * cos(angle_to_goal)` -- reward alignment with goal.
**Why**: Transforms "fly forward and avoid" into "navigate to a destination".
**Target**: >80% of episodes reach goal within 2x optimal-path distance.

### Rung 7: "Anywhere in the World" -- Credible Scope Definition
**Credible Claim**: The system navigates autonomously in previously-unseen GNSS-denied environments that fall within the training distribution of depth-observable indoor/outdoor spaces.

**What "anywhere" means in practice**:
- Indoor: corridors, warehouses, parking structures, office buildings
- Outdoor urban: narrow streets, alleys, urban canyons
- Semi-structured: construction sites, industrial facilities

**What "anywhere" does NOT mean (honest limitations)**:
- Featureless environments (white walls, fog) -- depth sensor returns flat readings
- Dynamic environments with fast-moving obstacles -- no dynamic obstacle training
- Long-range outdoor (>100m visibility) -- depth sensor limited to ~20m

**How to approach credibility**:
- Train on 5+ diverse environments with domain randomization
- Test on held-out environment (never seen during training)
- Report transfer performance with confidence intervals
- Present failure cases honestly

### Plan B Alternatives

**If PPO fails to learn (reward stays flat after 500k steps)**:
1. Switch to SAC (off-policy, better sample efficiency for continuous control)
2. Add behavioral cloning pretraining (record a few manual episodes, pretrain policy weights, then fine-tune with PPO)
3. Simplify observation to pure depth center-column (reduce input dimensionality)

**If AirSim becomes a blocker**:
1. Create simplified 2D gridworld environment for rapid policy iteration
2. Port to Flightmare (Unity-based, faster rendering)
3. Use gym-pybullet-drones for physics-only training

---

## DELIVERABLE 7: EVALUATION FRAMEWORK

### Primary Metrics

| Metric | Definition | Unit | Target (Rung 1) | Target (Rung 5+) |
|--------|-----------|------|-----------------|-------------------|
| **Distance Before Collision (DBC)** | Total distance traveled before first collision or episode end | meters | >50m | >200m |
| **Collision Rate** | Collisions per 100m traveled | count/100m | <2.0 | <0.5 |
| **Mean Episode Reward** | Average total reward per episode | scalar | >50 | >300 |
| **Mean Speed** | Average forward velocity during episode | m/s | >1.0 | >2.0 |
| **Path Smoothness** | Mean absolute jerk (derivative of acceleration) | m/s^3 | <5.0 | <2.0 |
| **Survival Rate** | Fraction of episodes reaching max_steps without collision | % | >20% | >60% |
| **Coverage Area** | Convex hull area of all visited positions | m^2 | >100 | >500 |
| **Cross-Env Transfer** | Performance on unseen env / performance on training env | ratio | N/A | >0.7 |

### Evaluation Protocol

```
For each model checkpoint:
    For each environment in {train_envs + held_out_envs}:
        Run 20 episodes with fixed seeds [42, 43, ..., 61]
        Record per-step: position, velocity, action, reward, collision flag
        Compute per-episode: DBC, total_reward, duration, collision_count
    Aggregate: mean +/- std for all metrics
    Generate: trajectory plots, reward distribution, metric comparison table
```

### Required Plots

1. **Training Curves** (per run):
   - Episode reward vs timestep (with smoothing window)
   - Episode length vs timestep
   - Collision rate vs timestep (rolling window)
   - Learning rate and entropy coefficient over time

2. **Evaluation Plots** (per checkpoint):
   - Bird's-eye trajectory plot (X-Y) with collision markers (already in test_full_nav.py)
   - Altitude profile over time
   - Speed profile over time
   - Action distribution histograms (vx, vy, yaw_rate)

3. **Comparison Plots** (across experiments):
   - Bar chart: DBC across environments for each model variant
   - Box plot: Episode reward distribution across environments
   - Radar chart: Multi-metric comparison (DBC, speed, smoothness, survival, coverage)
   - Learning curve overlay: curriculum vs flat, DR vs no-DR

4. **Ablation Plots**:
   - Reward component ablation: full reward vs each component removed
   - Observation ablation: depth+vel vs depth-only vs vel-only
   - Safety monitor effect: with safety vs without safety

### Ablation Experiments

| Experiment | Variable | Configurations | Key Metric |
|-----------|----------|----------------|------------|
| **Reward Ablation** | Reward components | Full / No-smoothness / No-proximity / Progress-only | DBC, Smoothness |
| **Observation Ablation** | Input modalities | Depth+Vel / Depth-only / Vel-only | DBC, Speed |
| **Frame Stack Ablation** | Temporal depth | 1 / 2 / 4 / 8 frames | DBC, Collision Rate |
| **Safety Monitor** | Safety on/off | With safety / Without safety | Collision Rate |
| **Domain Rand.** | DR on/off | DR training / Standard training | Cross-env Transfer |
| **Curriculum** | Curriculum on/off | 4-phase / Flat difficulty | Final Reward, Training Speed |
| **Algorithm** | PPO vs SAC | PPO / SAC (same hyperparams where applicable) | Sample Efficiency, Final DBC |
| **Action Space** | Velocity range | max_vx={1,2,3,5} | DBC, Speed |

### Failure Case Documentation

Every evaluation report must include:
1. **Worst 3 episodes**: Trajectory plot with annotations of what went wrong
2. **Failure taxonomy**: Categorize failures as {frontal collision, lateral collision, altitude deviation, timeout, oscillation}
3. **Failure rate by category**: Pie chart showing distribution
4. **Identified blind spots**: Environments or conditions where policy consistently fails

### Success Criteria (for FYP)

**Minimum Viable (Pass)**:
- PPO baseline (Rung 1) with >50m DBC in single environment
- Documented training curves and evaluation metrics
- Working code with reproducible results

**Good (Merit)**:
- Multi-environment training (Rung 3) with cross-env transfer >0.7
- At least 2 ablation experiments documented
- Safety monitor integrated and evaluated

**Excellent (Distinction)**:
- Domain randomization + curriculum learning (Rungs 4-5)
- Comprehensive ablation study (4+ experiments)
- Goal-conditioned navigation (Rung 6)
- Honest failure analysis with credible "anywhere" scope definition (Rung 7)

---

## DELIVERABLE 8: RISK REGISTER + MITIGATIONS

### Technical Risks

| # | Risk | Probability | Impact | Mitigation | Fallback |
|---|------|------------|--------|------------|----------|
| T1 | **PPO fails to learn** (reward plateau) | Medium | Critical | Tune reward weights, increase entropy, verify env with random policy baseline | Switch to SAC; add BC pretraining warmstart |
| T2 | **Reward hacking** (policy exploits reward without navigating) | Medium | High | Monitor velocity profiles and trajectory coverage alongside reward; add exploration reward | Redesign reward with trajectory-level metrics; add human evaluation |
| T3 | **Sim-to-sim transfer failure** (policy overfits to one environment) | High | High | Domain randomization from Rung 4; multi-env training from Rung 3 | Accept single-env result and document limitation honestly |
| T4 | **AirSim stability issues** (crashes, memory leaks, API bugs) | Medium | Medium | Checkpoint frequently; wrap all AirSim calls in try/except; auto-restart on crash | Develop lightweight mock env for rapid iteration; port to Flightmare |
| T5 | **Lockstep timing issues** (simContinueForTime unreliable) | Low | High | Verify step timing with counters; add timeout to simPause | Fall back to real-time with sleep-based control loop |
| T6 | **Depth sensor artifacts** (NaN, inf, zero-depth) | Medium | Medium | Clip and sanitize all depth inputs; add NaN/inf checks in observation pipeline | Add depth validity mask to observation |
| T7 | **Altitude drift** | Low | Medium | Use moveByVelocityZBodyFrameAsync (already fixed); add altitude PID as backup | Fall back to moveToZ with explicit altitude commands |
| T8 | **Training compute insufficient** (1M steps takes too long) | Medium | Medium | Use headless mode; optimize env step time; reduce image resolution | Train on cloud GPU (Colab, Vast.ai); reduce to 500k steps |

### Project Risks

| # | Risk | Probability | Impact | Mitigation | Fallback |
|---|------|------------|--------|------------|----------|
| P1 | **Scope creep** ("anywhere in the world" is infinite) | High | High | Defined credible scope in Rung 7; fixed rung ladder with clear stopping points | Deliver best rung achieved with honest assessment |
| P2 | **Time overrun** on foundation work | Medium | High | Sprint structure with hard deadlines; Phase 0 is 2 weeks max | Cut Phase 3 (advanced features); focus on Rungs 1-3 only |
| P3 | **Reproducibility failure** (results differ between runs) | Medium | Medium | Fixed seeds; deterministic lockstep; log all hyperparameters and git commit hashes | Report confidence intervals across 3+ runs with different seeds |
| P4 | **Environment availability** (can't get custom UE environments) | Medium | Medium | Focus on AirSim's built-in environments (NH, Blocks, etc.) | Use spawn position randomization within single environment |
| P5 | **GPU availability for training** | Low | High | Training runs on local GPU (GTX 1060+); 1M steps ~ 8-24 hours | Use Google Colab with AirSim in headless mode; reduce training budget |
| P6 | **Report writing time underestimated** | High | Medium | Automate figure generation; write evaluation section as experiments complete | Template report sections early; fill in results incrementally |

### Dependency Map

```
T1 (PPO fails) --> blocks all of Phase 1+
    Mitigation: Detect within 100k steps (reward trend analysis)
    Decision point: If no reward improvement by 200k steps, switch to SAC

T3 (Transfer failure) --> blocks Phase 2+
    Mitigation: Start multi-env training early in Phase 2
    Decision point: If transfer ratio <0.3, focus on single-env + DR instead

P2 (Time overrun) --> compresses Phase 3-4
    Mitigation: Phase 0 hard deadline; daily progress checks
    Decision point: If Phase 0 takes >3 weeks, cut curriculum and goal-conditioning

T4 (AirSim crashes) --> blocks all phases
    Mitigation: Frequent checkpointing; auto-restart wrapper
    Decision point: If >5 crashes per training run, invest in stability fixes or switch sim
```

### Critical Path (4-Week Plan)

```
Week 1 (Foundation) --> Week 2 Days 8-10 (Baseline) --> Week 4 (Eval + Report)
                                    |
                                    +--> Week 2-3 (Multi-env + Ablations) --> Week 4
```

Week 1 and the baseline (Days 8-10) are on the critical path. If Week 1 slips past Day 7, everything compresses.

**If Week 1 takes 9 days instead of 7**: Cut ablation 4 (DR), compress eval to 2 days. Still delivers Rungs 1-3.
**If baseline fails by Day 12**: Switch to SAC immediately. Still have 16 days for training + eval.
**Absolute minimum**: Week 1 + Baseline + Week 4 = Rung 1 with evaluation. Passing grade.

---

## APPENDIX A: GIT OPERATIONS FOR CLEAN REPO

### Step-by-Step Clean Procedure

```bash
# 1. Create new branch for clean architecture
git checkout -b v2-clean-architecture

# 2. Delete legacy IL stack
rm -rf src/ai/
rm src/utils/vision_diagnostic.py
rm src/utils/run_full_pipeline.py
rm src/control/forward_test.py
rm sim_env/create_parking.py

# 3. Delete legacy configs
rm configs/record.yaml
rm configs/record_centerline.yaml
rm configs/train_sklearn.yaml
rm configs/train_cnn.yaml
rm configs/train_cnn_centerline.yaml
rm configs/train_cnn_fixed.yaml
rm configs/run_cnn_deployment.yaml
rm configs/rl_ppo_pretrained.yaml
rm configs/pretrain_kaggle.yaml

# 4. Delete legacy data
rm -rf data/expert/
rm -rf data/diagnostic/

# 5. Delete root-level junk
rm merge_all_data.py merge_datasets.py
rm fyp_report.txt steup_notes.txt

# 6. Delete legacy RL files
rm src/rl/train_ppo_pretrained.py

# 7. Move reference docs
mkdir -p docs
mv "FYP Proposal.pdf" docs/
mv "Interim Report H00404752.pdf" docs/
mv autonomy_audit.md docs/

# 8. Create new structure
mkdir -p src/environments src/safety src/training src/evaluation src/deployment
mkdir -p configs/environments configs/rewards configs/curriculum
mkdir -p tests scripts

# 9. Move and refactor retained files
mv src/rl/env_airsim.py src/environments/airsim_env.py
mv src/rl/train_ppo.py src/training/train.py
mv src/rl/deploy_ppo.py src/deployment/deploy.py
mv src/rl/test_full_nav.py src/evaluation/evaluate.py
mv src/rl/check_env.py scripts/check_env.py
# pid.py and controller.py stay in src/control/
# logging.py and airsim_cam.py stay in src/utils/

# 10. Clean up old directories
rm -rf src/rl/
rm -rf src/ai/

# 11. Commit
git add -A
git commit -m "v2: Clean architecture for GNSS-denied quadrotor autonomy

- Deleted 28 legacy IL files (scripted expert, Canny/Hough perception, CNN BC)
- Retained and reorganized 7 core files (env, PID, training, eval, utils)
- New module structure: environments/, control/, safety/, training/, evaluation/, deployment/
- Reference docs preserved in docs/"
```

### Commit Plan (After Initial Clean)

1. `v2-foundation`: New pyproject.toml, Makefile, updated CLAUDE.md, README.md
2. `v2-env-refactor`: Refactored AirSimDroneEnv with pluggable rewards and DR hooks
3. `v2-safety`: SafetyMonitor module
4. `v2-training`: Refactored training with WandB, curriculum callbacks
5. `v2-evaluation`: Standardized evaluation protocol with metrics engine
6. `v2-tests`: Unit tests for env, rewards, safety, PID
7. `v2-baseline`: First successful training run with logged results

---

## APPENDIX B: DEPENDENCY SPECIFICATION (pyproject.toml)

```toml
[project]
name = "gnss-denied-quad"
version = "2.0.0"
description = "GNSS-Denied Quadrotor Autonomy via Reinforcement Learning"
requires-python = ">=3.11,<3.12"

dependencies = [
    "airsim==1.8.1",
    "gymnasium>=0.29,<1.0",
    "stable-baselines3>=2.2,<3.0",
    "torch>=2.1,<3.0",
    "numpy>=1.24,<2.0",
    "opencv-python>=4.8,<5.0",
    "pyyaml>=6.0,<7.0",
    "tensorboard>=2.15,<3.0",
    "matplotlib>=3.8,<4.0",
    "pandas>=2.1,<3.0",
    "tqdm>=4.66,<5.0",
    "shimmy>=0.2.1,<1.0",
    "msgpack-rpc-python>=0.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "wandb>=0.16",
]
export = [
    "onnx>=1.15",
    "onnxruntime>=1.16",
]
```

---

## APPENDIX C: QUICK REFERENCE -- WHAT TO DO FIRST

### The 10 Immediate Actions (in order)

1. **Create a git branch** `v2-clean-architecture` from current state
2. **Delete the 28 files** marked DELETE in the triage report
3. **Create the new folder structure** per Deliverable 4
4. **Move the 7 KEEP/REFACTOR files** to their new locations
5. **Write `pyproject.toml`** with pinned dependencies
6. **Write `Makefile`** with install/check-env/train/eval targets
7. **Update `src/environments/airsim_env.py`** imports and module paths
8. **Update `src/training/train.py`** imports and module paths
9. **Run `make install && make check-env`** to verify everything works
10. **Commit** the clean foundation as the first v2 commit

After these 10 actions, you have a clean, working skeleton ready for Phase 1 development.
