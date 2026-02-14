"""
Train PPO with a pretrained NavigationCNN feature extractor.

The PretrainedCombinedExtractor loads conv_layers weights from a .pth file
produced by pretrain_corner_cnn.py, adapts them for 4-channel frame-stacked
input, freezes early layers, and fine-tunes the rest with PPO.

Usage:
    python -m src.rl.train_ppo_pretrained --pretrained models/pretrained/pretrained_corner_cnn_*.pth
    python -m src.rl.train_ppo_pretrained --pretrained models/pretrained/pretrained_corner_cnn_*.pth --total_timesteps 4096
"""

import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.ai.model_cnn import NavigationCNN
from src.rl.env_airsim import AirSimDroneEnv


class PretrainedCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that combines:
      - Pretrained NavigationCNN conv_layers for depth images (512-dim)
      - Small MLP for velocity features (3-dim)

    Handles the 1-ch → 4-ch adaptation for frame stacking and partial freezing.
    """

    def __init__(self, observation_space: gym.spaces.Dict, pretrained_path: str = "",
                 freeze_blocks: int = 3):
        # Each conv block = (Conv2d, ReLU, MaxPool2d) = 3 layers
        # 4 blocks total: layers 0-2, 3-5, 6-8, 9-11
        # Freeze first `freeze_blocks` blocks

        # Feature dim: 512 (image CNN) + 3 (velocity MLP)
        super().__init__(observation_space, features_dim=515)

        self.freeze_blocks = freeze_blocks

        # --- Image CNN (from pretrained NavigationCNN) ---
        nav_cnn = NavigationCNN(img_height=84, img_width=84, num_outputs=4)

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            state_dict = ckpt["model_state_dict"]
            nav_cnn.load_state_dict(state_dict)
            print(f"[extractor] Loaded pretrained weights from {pretrained_path}")

        # Extract conv_layers
        self.conv_layers = nav_cnn.conv_layers

        # Adapt conv1 from 1 channel → 4 channels (frame stack)
        old_conv1 = self.conv_layers[0]  # Conv2d(1, 32, 3, 1, 1)
        new_conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)

        if pretrained_path:
            # Replicate 1-ch weights across 4 channels, divide by 4
            with torch.no_grad():
                new_conv1.weight.copy_(old_conv1.weight.repeat(1, 4, 1, 1) / 4.0)
                new_conv1.bias.copy_(old_conv1.bias)
        self.conv_layers[0] = new_conv1

        # Freeze early conv blocks
        freeze_up_to = freeze_blocks * 3  # 3 layers per block
        for i, layer in enumerate(self.conv_layers):
            if i < freeze_up_to:
                for param in layer.parameters():
                    param.requires_grad = False

        n_frozen = sum(1 for p in self.conv_layers.parameters() if not p.requires_grad)
        n_total = sum(1 for _ in self.conv_layers.parameters())
        print(f"[extractor] Frozen {n_frozen}/{n_total} conv parameters "
              f"({freeze_blocks} of 4 blocks)")

        # FC head for image: 256*5*5 = 6400 → 512
        self.image_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 512),
            nn.ReLU(),
        )

        # --- Velocity MLP ---
        # velocity shape after frame stack: (3,) per frame × 4 frames = (12,)
        # SB3 VecFrameStack with Dict obs stacks only image; velocity stays (3,)
        # Actually: SB3 VecFrameStack stacks ALL keys. For "velocity" (3,) with
        # n_stack=4, it becomes (12,).
        self.velocity_mlp = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, observations):
        # SB3 with Dict obs gives us observations["image"] and observations["velocity"]
        img = observations["image"]    # (N, H, W, C) — SB3 keeps channels_order="last"
        vel = observations["velocity"]  # (N, 12) after frame stacking

        # Transpose NHWC → NCHW (SB3 doesn't auto-transpose float32 Dict obs)
        img = img.permute(0, 3, 1, 2)

        # Image features
        x = self.conv_layers(img)
        x = self.image_fc(x)  # (N, 512)

        # Velocity features
        v = self.velocity_mlp(vel)  # (N, 3)

        return torch.cat([x, v], dim=1)  # (N, 515)


def make_env(cfg: dict):
    """Return a factory that creates a Monitored AirSimDroneEnv."""
    def _init():
        return Monitor(AirSimDroneEnv(cfg))
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with pretrained CNN")
    parser.add_argument(
        "--config", type=str, default="configs/rl_ppo_pretrained.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained .pth file (overrides config)",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=None,
        help="Override total_timesteps from config",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = cfg["ppo"]
    out_cfg = cfg["output"]
    frame_stack = cfg.get("frame_stack", 4)

    total_timesteps = args.total_timesteps or ppo_cfg["total_timesteps"]

    # Pretrained path: CLI overrides config
    pretrained_path = args.pretrained or cfg.get("pretrained_path", "")

    # Timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_cfg["log_dir"], f"ppo_pretrained_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Environments ---
    train_env = DummyVecEnv([make_env(cfg)])
    train_env = VecFrameStack(train_env, n_stack=frame_stack, channels_order="last")

    eval_env = DummyVecEnv([make_env(cfg)])
    eval_env = VecFrameStack(eval_env, n_stack=frame_stack, channels_order="last")

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=out_cfg.get("checkpoint_freq", 10000),
        save_path=ckpt_dir,
        name_prefix="ppo_pretrained",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=out_cfg.get("eval_freq", 5000),
        n_eval_episodes=out_cfg.get("eval_episodes", 5),
        deterministic=True,
    )

    # --- Policy kwargs with pretrained extractor ---
    policy_kwargs = {
        "features_extractor_class": PretrainedCombinedExtractor,
        "features_extractor_kwargs": {
            "pretrained_path": pretrained_path,
            "freeze_blocks": 3,
        },
    }

    # --- Model ---
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=run_dir,
        verbose=1,
    )

    print(f"[train_ppo_pretrained] Run directory: {run_dir}")
    print(f"[train_ppo_pretrained] Pretrained weights: {pretrained_path}")
    print(f"[train_ppo_pretrained] Total timesteps: {total_timesteps}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb],
        )
    except KeyboardInterrupt:
        print("\n[train_ppo_pretrained] Training interrupted by user.")
    finally:
        final_path = os.path.join(run_dir, "final_model")
        model.save(final_path)
        print(f"[train_ppo_pretrained] Final model saved to {final_path}.zip")
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
