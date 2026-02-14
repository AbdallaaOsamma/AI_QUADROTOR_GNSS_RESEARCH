import os
import shutil
import pandas as pd

# Your 5 new runs
runs = [
    "data/expert/record/run_20260203_152244",
    "data/expert/record/run_20260203_152425",
    "data/expert/record/run_20260203_152513",
    "data/expert/record/run_20260203_152605",
    "data/expert/record/run_20260203_152700",
]

# Merged output
merged_dir = "data/expert/merged_scripted"
merged_frames = os.path.join(merged_dir, "frames")
os.makedirs(merged_frames, exist_ok=True)

all_rows = []
frame_counter = 0

for run_dir in runs:
    if not os.path.exists(run_dir):
        print(f"[WARN] Skipping {run_dir} (not found)")
        continue
        
    labels_csv = os.path.join(run_dir, "labels.csv")
    df = pd.read_csv(labels_csv)
    
    print(f"[INFO] Processing {run_dir}: {len(df)} frames")
    
    for _, row in df.iterrows():
        old_img = os.path.join(run_dir, row["img_path"])
        new_img = f"frames/frame_{frame_counter:05d}.png"
        new_img_full = os.path.join(merged_dir, new_img)
        
        if os.path.exists(old_img):
            shutil.copy2(old_img, new_img_full)
        
        row["img_path"] = new_img
        all_rows.append(row)
        frame_counter += 1

merged_df = pd.DataFrame(all_rows)
merged_csv = os.path.join(merged_dir, "labels.csv")
merged_df.to_csv(merged_csv, index=False)

print(f"[OK] Merged {len(all_rows)} frames into {merged_dir}")