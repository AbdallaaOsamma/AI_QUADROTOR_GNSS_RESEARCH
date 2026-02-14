"""Custom SB3 callbacks for training infrastructure."""
from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    """Log reward components to TensorBoard at each rollout end.

    Reads info dicts from the Monitor wrapper and logs per-component
    reward averages to TensorBoard for analysis.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "r_progress" in info:
                self._episode_rewards.append({
                    "r_progress": info["r_progress"],
                    "r_collision": info["r_collision"],
                    "r_smoothness": info["r_smoothness"],
                })
        return True

    def _on_rollout_end(self) -> None:
        if not self._episode_rewards:
            return
        n = len(self._episode_rewards)
        for key in ("r_progress", "r_collision", "r_smoothness"):
            avg = sum(d[key] for d in self._episode_rewards) / n
            self.logger.record(f"reward/{key}", avg)
        self._episode_rewards.clear()
