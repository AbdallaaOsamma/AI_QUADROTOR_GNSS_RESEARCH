import os
import argparse
import time
import yaml
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.ai.model_cnn import NavigationCNN

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def latest_run_dir(root):
    """Find latest run_YYYYMMDD_HHMMSS directory, or use root if it is a run dir"""
    root_path = Path(root)
    
    # Check if root itself is a run directory (has labels.csv)
    if (root_path / "labels.csv").exists():
        return str(root_path)

    # Search for subdirectories
    runs = sorted([d for d in root_path.glob("run_*") if d.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")
    return str(runs[-1])

class NavigationDataset(Dataset):
    def __init__(self, run_dir, img_size=(84, 84), grayscale=True, clamp=None, augment=False):
        self.run_dir = Path(run_dir)
        self.img_size = img_size
        self.grayscale = grayscale
        self.clamp = clamp or {}
        self.augment = augment
        
        # Load labels
        labels_csv = self.run_dir / "labels.csv"
        self.df = pd.read_csv(labels_csv)
        
        # Filter out rows with missing images
        self.df = self.df[self.df['img_path'].str.len() > 0].reset_index(drop=True)
        
        print(f"[INFO] Loaded {len(self.df)} samples from {run_dir} (Augment={augment})")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.run_dir / row['img_path']
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR)
        
        if img is None:
            # Return zeros if image missing
            img = np.zeros(self.img_size, dtype=np.uint8)
        
        # Resize
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        
        # Load labels
        vx = float(row['vx'])
        vy = float(row['vy'])
        vz = float(row['vz'])
        rz = float(row['r_z_rad'])
        
        # Data Augmentation: Random Horizontal Flip
        if self.augment and np.random.random() > 0.5:
            # Flip image horizontally
            img = cv2.flip(img, 1)
            # Invert lateral controls
            vy = -vy
            rz = -rz
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        if self.grayscale:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        
        # Clamp values
        if 'vx' in self.clamp:
            vx = np.clip(vx, self.clamp['vx'][0], self.clamp['vx'][1])
        if 'vy' in self.clamp:
            vy = np.clip(vy, self.clamp['vy'][0], self.clamp['vy'][1])
        if 'vz' in self.clamp:
            vz = np.clip(vz, self.clamp['vz'][0], self.clamp['vz'][1])
        if 'r_z_rad' in self.clamp:
            rz = np.clip(rz, self.clamp['r_z_rad'][0], self.clamp['r_z_rad'][1])
        
        labels = np.array([vx, vy, vz, rz], dtype=np.float32)
        
        return torch.from_numpy(img), torch.from_numpy(labels)

def train_model(model, train_loader, val_loader, cfg, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    
    num_epochs = cfg['train']['num_epochs']
    best_val_loss = float('inf')
    patience = cfg['train']['early_stopping_patience']
    patience_counter = 0
    
    print(f"[INFO] Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = np.zeros(4)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate MAE per output
                mae = torch.abs(outputs - labels).cpu().numpy()
                val_mae += mae.sum(axis=0)
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: vx={val_mae[0]:.3f}, vy={val_mae[1]:.3f}, vz={val_mae[2]:.3f}, rz={val_mae[3]:.3f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered after {epoch+1} epochs")
                model.load_state_dict(best_model_state)
                break
    
    return model, val_mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_cnn.yaml')
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)
    
    # Get dataset
    run_root = cfg['data']['run_dir']
    run_dir = latest_run_dir(run_root)
    print(f"[INFO] Using dataset: {run_dir}")
    
    # Create datasets
    img_size = tuple(cfg['model']['img_size'])
    grayscale = cfg['model']['grayscale']
    clamp = cfg['train']['clamp']
    
    dataset = NavigationDataset(run_dir, img_size=img_size, grayscale=grayscale, clamp=clamp)
    
    # Split train/val
    val_split = cfg['model']['val_split']
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    # Create distinct datasets for train (augmented) and val (not augmented)
    # We reload the dataset object to avoid passing augment=True to validation
    train_dataset = NavigationDataset(run_dir, img_size=img_size, grayscale=grayscale, clamp=clamp, augment=True)
    val_dataset = NavigationDataset(run_dir, img_size=img_size, grayscale=grayscale, clamp=clamp, augment=False)
    
    # We need to manually split them to ensure no leakage, but for simplicity with random_split:
    # A better approach is to wrap the subset.
    # Let's use the simplest approach: Re-instantiate based on indices? 
    # Actually, simpler: just augment the WHOLE dataset for training if we didn't split yet.
    # But we want val to be pure.
    
    # Correct approach:
    full_indices = np.arange(len(dataset))
    np.random.seed(cfg['model']['random_state'])
    np.random.shuffle(full_indices)
    
    train_indices = full_indices[:train_size]
    val_indices = full_indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(
        NavigationDataset(run_dir, img_size=img_size, grayscale=grayscale, clamp=clamp, augment=True),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        NavigationDataset(run_dir, img_size=img_size, grayscale=grayscale, clamp=clamp, augment=False),
        val_indices
    )
    
    # Create dataloaders
    batch_size = cfg['train']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    model = NavigationCNN(img_height=img_size[0], img_width=img_size[1], num_outputs=4)
    model.to(device)
    
    # Train
    model, val_mae = train_model(model, train_loader, val_loader, cfg, device)
    
    # Save model
    model_dir = Path(cfg['output']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"cnn_nav_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'img_size': img_size,
        'grayscale': grayscale,
        'val_mae': val_mae.tolist()
    }, model_path)
    
    print(f"[OK] Validation MAE: vx={val_mae[0]:.3f}, vy={val_mae[1]:.3f}, vz={val_mae[2]:.3f}, rz={val_mae[3]:.3f}")
    print(f"[OK] Saved model to {model_path}")

if __name__ == "__main__":
    main()