"""
Kaggle Drone Obstacle Avoidance dataset loader.

Expects data layout:
    root_dir/
        depth/
            000000.npy, 000001.npy, ...  (320x320 float16, raw meters, max ~125m)
        commands/
            000000.npy, 000001.npy, ...  ([vx, vy, vz, yaw_rate_deg])

Note: Depth values are raw meters (not pre-normalised).
      yaw_rate is in degrees (not radians).
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import cv2


class KaggleDepthDataset(Dataset):
    """PyTorch Dataset for Kaggle drone depth + command .npy files."""

    def __init__(self, root_dir, img_size=(84, 84), clamp=None, augment=False):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.clamp = clamp or {}
        self.augment = augment

        depth_dir = self.root_dir / "depth"
        cmd_dir = self.root_dir / "commands"

        if not depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
        if not cmd_dir.exists():
            raise FileNotFoundError(f"Commands directory not found: {cmd_dir}")

        # Build sorted file list (match depth ↔ command by filename stem)
        depth_files = sorted(depth_dir.glob("*.npy"))
        cmd_stems = {f.stem for f in cmd_dir.glob("*.npy")}

        self.samples = []
        for df in depth_files:
            if df.stem in cmd_stems:
                self.samples.append((df, cmd_dir / df.name))

        if not self.samples:
            raise RuntimeError(
                f"No matched depth/command pairs in {root_dir}. "
                f"Found {len(depth_files)} depth files, {len(cmd_stems)} command files."
            )

        print(f"[KaggleDepthDataset] {len(self.samples)} samples from {root_dir} "
              f"(augment={augment})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        depth_path, cmd_path = self.samples[idx]

        # --- Depth image ---
        depth = np.load(str(depth_path)).astype(np.float32)  # (320, 320), raw meters

        # Resize 320 → 84
        depth = cv2.resize(depth, self.img_size, interpolation=cv2.INTER_AREA)

        # Normalise: clip at 20m (matching our RL env), then scale to [0, 1]
        depth = np.clip(depth, 0.0, 20.0) / 20.0

        # --- Commands ---
        cmd = np.load(str(cmd_path)).astype(np.float32)  # [vx, vy, vz, yaw_rate_deg]
        vx, vy, vz = cmd[0], cmd[1], cmd[2]
        rz = np.deg2rad(cmd[3])  # Convert degrees → radians

        # --- Augmentation: random horizontal flip ---
        if self.augment and np.random.random() > 0.5:
            depth = np.flip(depth, axis=1).copy()
            vy = -vy
            rz = -rz

        # --- Clamp ---
        if "vx" in self.clamp:
            vx = np.clip(vx, self.clamp["vx"][0], self.clamp["vx"][1])
        if "vy" in self.clamp:
            vy = np.clip(vy, self.clamp["vy"][0], self.clamp["vy"][1])
        if "vz" in self.clamp:
            vz = np.clip(vz, self.clamp["vz"][0], self.clamp["vz"][1])
        if "r_z_rad" in self.clamp:
            rz = np.clip(rz, self.clamp["r_z_rad"][0], self.clamp["r_z_rad"][1])

        # Shape: (1, 84, 84) — single-channel depth
        depth = np.expand_dims(depth, axis=0)

        labels = np.array([vx, vy, vz, rz], dtype=np.float32)

        return torch.from_numpy(depth), torch.from_numpy(labels)

    def print_stats(self):
        """Load all commands and print dataset statistics (after conversion)."""
        all_cmds = []
        for _, cmd_path in self.samples:
            cmd = np.load(str(cmd_path)).astype(np.float32)
            # Convert yaw_rate from degrees to radians (same as __getitem__)
            cmd[3] = np.deg2rad(cmd[3])
            all_cmds.append(cmd)
        all_cmds = np.stack(all_cmds)

        print(f"\n{'='*50}")
        print(f"Dataset Statistics ({len(self.samples)} samples)")
        print(f"{'='*50}")
        names = ["vx (m/s)", "vy (m/s)", "vz (m/s)", "yaw_rate (rad/s)"]
        for i, name in enumerate(names):
            col = all_cmds[:, i]
            print(f"  {name:>20s}: mean={col.mean():+.4f}  std={col.std():.4f}  "
                  f"min={col.min():+.4f}  max={col.max():+.4f}")
        print(f"{'='*50}\n")
