"""
Pretrain NavigationCNN on Kaggle drone obstacle avoidance depth data.

Reuses the CNN architecture from model_cnn.py and training loop from train_cnn.py,
but loads data via KaggleDepthDataset instead of NavigationDataset.

Usage:
    python -m src.ai.pretrain_corner_cnn --config configs/pretrain_kaggle.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.ai.dataset_kaggle import KaggleDepthDataset
from src.ai.model_cnn import NavigationCNN
from src.ai.train_cnn import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain NavigationCNN on Kaggle depth data"
    )
    parser.add_argument(
        "--config", type=str, default="configs/pretrain_kaggle.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    out_cfg = cfg["output"]

    root_dir = data_cfg["root_dir"]
    img_size = tuple(data_cfg["img_size"])
    val_split = data_cfg["val_split"]
    random_state = data_cfg["random_state"]
    clamp = train_cfg["clamp"]

    # Build full dataset (no augmentation) to determine split indices
    full_ds = KaggleDepthDataset(root_dir, img_size=img_size, clamp=clamp, augment=False)
    full_ds.print_stats()

    # Train/val split
    n_total = len(full_ds)
    indices = np.arange(n_total)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = Subset(
        KaggleDepthDataset(root_dir, img_size=img_size, clamp=clamp, augment=True),
        train_indices,
    )
    val_ds = Subset(
        KaggleDepthDataset(root_dir, img_size=img_size, clamp=clamp, augment=False),
        val_indices,
    )

    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[pretrain] Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pretrain] Device: {device}")

    model = NavigationCNN(img_height=img_size[0], img_width=img_size[1], num_outputs=4)
    model.to(device)

    # Train (reuses train_model from train_cnn.py)
    model, val_mae = train_model(model, train_loader, val_loader, cfg, device)

    # Save
    model_dir = Path(out_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"pretrained_corner_cnn_{timestamp}.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "img_size": img_size,
        "val_mae": val_mae.tolist(),
    }, model_path)

    print(f"[OK] Val MAE: vx={val_mae[0]:.3f}, vy={val_mae[1]:.3f}, "
          f"vz={val_mae[2]:.3f}, rz={val_mae[3]:.3f}")
    print(f"[OK] Saved pretrained model to {model_path}")


if __name__ == "__main__":
    main()
