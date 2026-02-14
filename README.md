airsim-ai-quad
================

Scaffold for an AirSim-based RL/IL quadrotor project.

Structure
---------
- sim_env/                 - Unreal project or AirSim binary (placeholder)
- src/                     - Source code
  - ai_model/              - policy & training scripts
  - control/               - PID velocity interface
  - eval/                  - metrics & plotting
  - utils/                 - logging and helpers
- data/
  - expert/                - expert trajectories (images, labels)
- configs/                 - YAML configs

Next steps
----------
- Fill `sim_env/` with your AirSim build or Unreal project.
- Implement training loop in `src/ai_model/train.py`.
- Run experiments configured in `configs/default.yaml`.
