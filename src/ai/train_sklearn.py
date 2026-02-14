# src/ai/train_sklearn.py
import os, glob, argparse, time, csv
from pathlib import Path
import numpy as np
import cv2
import yaml
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def latest_run_dir(root):
    root = Path(root)
    runs = sorted(root.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")
    return runs[-1]

def read_labels(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def load_dataset(run_dir, img_size=(84,84), grayscale=True, clamp=None):
    labels = read_labels(Path(run_dir) / "labels.csv")
    X, Y = [], []
    for r in labels:
        img_rel = r["img_path"]
        if not img_rel:
            continue
        img_path = Path(run_dir) / img_rel
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, tuple(img_size), interpolation=cv2.INTER_AREA)

        # Feature vector
        feat = img.reshape(-1).astype(np.float32) / 255.0
        X.append(feat)

        # Targets: vx, vy, vz, r_z_rad (floats)
        vx = float(r["vx"]); vy = float(r["vy"]); vz = float(r["vz"]); rz = float(r["r_z_rad"])
        if clamp:
            vx = max(min(vx, clamp["vx"][1]), clamp["vx"][0])
            vy = max(min(vy, clamp["vy"][1]), clamp["vy"][0])
            vz = max(min(vz, clamp["vz"][1]), clamp["vz"][0])
            rz = max(min(rz, clamp["r_z_rad"][1]), clamp["r_z_rad"][0])
        Y.append([vx, vy, vz, rz])

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X, Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_sklearn.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run_root = cfg["data"]["run_dir"]
    run_dir = latest_run_dir(run_root)
    print(f"[INFO] Using dataset: {run_dir}")

    img_w, img_h = cfg["model"]["img_size"]
    grayscale = bool(cfg["model"]["grayscale"])
    clamp = cfg["train"]["clamp"]

    X, Y = load_dataset(run_dir, img_size=(img_w, img_h), grayscale=grayscale, clamp=clamp)
    print(f"[OK] Loaded: X={X.shape}, Y={Y.shape}")

    Xtr, Xval, Ytr, Yval = train_test_split(
        X, Y, test_size=cfg["model"]["val_split"], random_state=cfg["model"]["random_state"]
    )

    # Pipeline: Standardize features -> MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(cfg["model"]["hidden"]),
        activation="relu",
        solver="adam",
        max_iter=int(cfg["train"]["max_iter"]),
        random_state=cfg["model"]["random_state"],
        verbose=True
    )
    pipeline = Pipeline([("scaler", StandardScaler(with_mean=True)), ("mlp", mlp)])

    print("[INFO] Training...")
    pipeline.fit(Xtr, Ytr)

    pred_val = pipeline.predict(Xval)
    mae = mean_absolute_error(Yval, pred_val, multioutput="raw_values")
    print(f"[OK] Validation MAE: vx={mae[0]:.3f}, vy={mae[1]:.3f}, vz={mae[2]:.3f}, r_z={mae[3]:.3f}")

    out_dir = Path(cfg["output"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"sklearn_nav_{stamp}.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "img_size": (img_w, img_h),
        "grayscale": grayscale,
        "clamp": clamp
    }, model_path)
    print(f"[OK] Saved model to {model_path}")

if __name__ == "__main__":
    main()
