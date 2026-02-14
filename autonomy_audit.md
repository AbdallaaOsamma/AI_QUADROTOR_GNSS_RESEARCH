# Drone Autonomy Audit & Transformation Plan

## A) CURRENT SYSTEM ARCHITECTURE (AS-IS)

### 1. Dataflow Diagram
```
[Sensor: Camera (RGB/Depth)] 
       ⬇
[Module: MultiModalPerception (src/ai/perception.py)]
       ⬇ (Extracts Lines/Depth/Ground) -- RULE BASED
[Module: Recorder (src/ai/recorder_multimodal.py)]
       ⬇ (Calculates "Expert" Velocity) -- PID/LOGIC
[Dataset: (Image, Velocity)]
       ⬇
[Model: CNN (src/ai/model_cnn.py)] -- SUPERVISED LEARNING
       ⬇
[Deployment: (src/ai/policy_run_cnn.py)]
       ⬇ (CNN Prediction)
       ⬇ <OVERRIDE: OpenCV Blob Detector (lines 157-161)>
[Control: AirSim API (moveByVelocityZ)]
```

### 2. Component Table
| Module | Inputs | Outputs | Method | "Known Path" Source |
| :--- | :--- | :--- | :--- | :--- |
| `perception.py` | RGB/Depth | Lateral/Heading Err | **Scripted** (Canny/Hough) | Explicitly hunts for "lines" or "open space" |
| `recorder...py` | Perception Result | Velocity Cmds | **Scripted** (PID) | Directly maps perception error to velocity |
| `train_cnn.py` | Dataset | Weights | **Supervised** (BC) | Clones the scripted logic above |
| `policy_run_cnn` | RGB | Velocity Cmds | **Hybrid** | **Hard-coded Override**: Checks road center via thresholding and overwrites NN |

### 3. Failure Modes
*   **Domain Shift**: Canny edge detection depends heavily on lighting and texture. Shadows or different road colors break it.
*   **Novel Obstacles**: The "Expert" is brittle; if it fails during training generation, the model learns the failure.
*   **Override limit**: The deployment script uses a simple threshold checks loops (`detect_road_center`). If the road isn't "darker than surroundings", the drone crashes even if the NN works.

---

## B) WHY IT IS BEING CALLED “NOT AI”

This system is a **Rube Goldberg machine wrapping a Canny Edge Detector**.

1.  **The "Teacher" is a Script**: The neural network is not learning to "drive"; it is minimizing the error between its output and the output of `src/ai/perception.py`. You are essentially compressing the `perception.py` script into a CNN.
2.  **Deployment Override**: In `src/ai/policy_run_cnn.py`, lines 157-161 explicitly discard the "AI" decision if it disagrees with a simple OpenCV threshold check:
    ```python
    road_offset = detect_road_center(img_gray)
    steering_correction = -road_offset * road_center_kp # <--- HARD CODED CONTROL
    vy_corrected = np.clip(vy + steering_correction, ...)
    ```
    The "AI" is effectively a dummy variable for lateral control.

---

## C) TARGET ARCHITECTURE (TO-BE)

**Selected Option: Option 2 - RL-based Policy Learning (End-to-End)**

*Justification*: The goal is "autonomous navigation". Generative/Self-Supervised perception (Option 1) is great for representation, but you still need a controller. RL solves the decision-making problem directly and allows the drone to discover policies (like "hugging the inside curve") that the handwritten script never knew.

*   **Observation**: Stacked Depth Images (for temporal info) + Current Velocity.
*   **Action**: Continuous Velocity Control `[vx, vy, yaw_rate]`.
*   **Reward Function**:
    *   Dense: `+1 * speed` (progress).
    *   Sparse: `-100` (collision).
    *   Shaping: `-0.1 * abs(lateral_accel)` (smoothness), `-penalty` for getting too close to walls.
*   **Why AI?**: The policy is not told *how* to fly (no "follow this line"). It learns to map raw depth pixels directly to thrust commands to maximize survival and speed.

---

## D) PINPOINT CODE CHANGE POINTS

### 1. Delete/Disable Known-Path Logic
**Target**: `src/ai/policy_run_cnn.py`
*   **Action**: Remove the safety wheels. Trust the policy.
*   **Patch**:
    ```python
    # BEFORE (Current)
    road_offset = detect_road_center(img_gray)
    steering_correction = -road_offset * road_center_kp
    vy_corrected = np.clip(vy + steering_correction, vy_clamp[0], vy_clamp[1])

    # AFTER (RL Deployment)
    # No overrides. The RL policy is the captain.
    vy_corrected = vy 
    ```

**Target**: `src/ai/perception.py` & `recorder_multimodal.py`
*   **Action**: Archive. These are no longer the source of truth.

### 2. Insert RL Training Loop
**Target**: `src/rl/env_airsim.py` (NEW)
*   **Task**: Wrap AirSim in `gymnasium.Env`.
*   **Logic**:
    *   `reset()`: Teleport drone to random start pose.
    *   `step(action)`: Send `moveByVelocityBodyFrame`, get `CollisionInfo`, capture `Depth`.
    *   `reward()`: `vx * dt - 100 * collision`.

**Target**: `src/rl/train_ppo.py` (NEW)
*   **Task**: Run Stable-Baselines3 PPO.
*   **Code**:
    ```python
    from stable_baselines3 import PPO
    env = AirSimGymEnv()
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=1_000_000)
    ```

---

## E) TRAINING PLAN (NO LABELS)

### 1. Data Collection (Experience Buffer)
*   **Method**: Online RL. The drone flies, crashes, resets, and learns.
*   **Curriculum**:
    *   **Phase 1 (100k steps)**: Empty world / wide street. Learn to fly straight without wobbling.
    *   **Phase 2 (500k steps)**: "Canyon" or obstacle course. Learn to avoid walls.
    *   **Phase 3 (1M+ steps)**: Domain Randomization (change lighting, weather in AirSim) to robustify vision.

### 2. Self-Supervised Objectives (Auxiliary)
If pure RL is too slow, add a **VAE Extraction** step:
1.  **Collect**: 10k random images primarily flying around.
2.  **Train**: VAE to compress 84x84 depth -> 64-dim latent vector.
3.  **RL**: Feed the 64-dim vector to PPO (instead of raw pixels). This speeds up RL convergence dramatically (State Representation Learning).

---

## F) EVALUATION: PROVE IT’S REAL AI

### 1. The "Lights Out" Test
*   **Setup**: Turn off all lights in the simulation (night mode) or change the road texture to grass.
*   **Current System**: Will fail immediately (Canny edge detection needs contrast/lines).
*   **New RL System (Depth-based)**: Will keep flying because depth sensors work in the dark/invariant to texture.

### 2. Zero-Shot Generalization
*   **Train**: Neighborhood environment.
*   **Test**: Warehouse environment.
*   **Metric**: Distance traveled before collision. The RL policy should understand "obstacle = bad" regardless of whether it looks like a house or a shelf.

---

## G) RISKS + SAFETY

### 1. Sim2Real Gap
*   **Risk**: Real depth sensors have holes/noise. AirSim depth is perfect.
*   **Mitigation**: Add **Salt-and-Pepper noise** and random "blackout" rectangles to depth images during training (Domain Randomization).

### 2. Safety Wrapper
*   **Logic**: Even an RL policy needs a "Reflex" safety layer (but distinct from "Guidance").
*   **Implementation**:
    ```python
    if perception.min_obstacle_dist < 0.5m:
        override_action = "EMERGENCY_BRAKE" # NOT "Scriped Turn", just "Stop"
    ```
