import os
import shutil
import pandas as pd

# ALL runs to merge
runs = [
    # Scripted expert runs (straight flight)
    "data/expert/record/run_20260203_152244",
    "data/expert/record/run_20260203_152425",
    "data/expert/record/run_20260203_152513",
    "data/expert/record/run_20260203_152605",
    "data/expert/record/run_20260203_152700",
    
    # Synthetic expert runs (diverse maneuvers)
    "data/expert/synthetic/run_20260203_155303",
    "data/expert/synthetic/run_20260203_160026",
]

# Merged output
merged_dir = "data/expert/run_mega_dataset"
merged_frames = os.path.join(merged_dir, "frames")
os.makedirs(merged_frames, exist_ok=True)

all_rows = []
frame_counter = 0

print("[INFO] Merging comprehensive dataset...")
print("="*60)

for run_dir in runs:
    if not os.path.exists(run_dir):
        print(f"[WARN] Skipping {run_dir} (not found)")
        continue
    
    labels_csv = os.path.join(run_dir, "labels.csv")
    df = pd.read_csv(labels_csv)
    
    print(f"[INFO] Processing {run_dir}")
    print(f"       Frames: {len(df)}")
    
    for _, row in df.iterrows():
        old_img = os.path.join(run_dir, row["img_path"])
        new_img = f"frames/frame_{frame_counter:06d}.png"  # 6 digits for 10k+ frames
        new_img_full = os.path.join(merged_dir, new_img)
        
        if os.path.exists(old_img):
            shutil.copy2(old_img, new_img_full)
        
        row["img_path"] = new_img
        all_rows.append(row)
        frame_counter += 1
        
        # Progress indicator
        if frame_counter % 1000 == 0:
            print(f"       → {frame_counter} frames processed...")

merged_df = pd.DataFrame(all_rows)
merged_csv = os.path.join(merged_dir, "labels.csv")
merged_df.to_csv(merged_csv, index=False)

print("="*60)
print(f"[SUCCESS] Merged {len(all_rows)} frames into {merged_dir}")
print(f"[SUCCESS] Dataset breakdown:")
print(f"          - Scripted expert: ~1,095 frames")
print(f"          - Synthetic expert: ~12,026 frames")
print(f"          - Total: {len(all_rows)} frames")
print("="*60)