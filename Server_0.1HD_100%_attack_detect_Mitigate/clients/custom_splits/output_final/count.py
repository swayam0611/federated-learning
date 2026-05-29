import pandas as pd
import glob
import os
import numpy as np

# ================================
# CONFIGURATION
# ================================
DATASET_DIR = "/home/apeksha/Desktop/new/0.1HD_without attack/clients/custom_splits/output_final"     # folder with client CSVs
LABEL_COL = "cardio"

# ================================
# LOAD ALL CLIENT CSV FILES
# ================================
csv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))

if not csv_files:
    raise FileNotFoundError("No CSV files found!")

print("\n========== CLASS COUNTS & HELLINGER DISTANCE ==========\n")

# ================================
# COMPUTE GLOBAL DISTRIBUTION
# ================================
dfs = [pd.read_csv(f) for f in csv_files]
global_df = pd.concat(dfs, ignore_index=True)

global_probs = global_df[LABEL_COL].value_counts(normalize=True).sort_index()

q0 = global_probs.get(0, 0.0)
q1 = global_probs.get(1, 0.0)

sqrt_q0 = np.sqrt(q0)
sqrt_q1 = np.sqrt(q1)

print("🌍 GLOBAL DISTRIBUTION")
print(f"  P(C0) = {q0:.6f}, P(C1) = {q1:.6f}")
print("-" * 55)

# ================================
# PER-CLIENT COUNTS + HD
# ================================
for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    if LABEL_COL not in df.columns:
        print(f"[ERROR] {os.path.basename(csv_file)} → '{LABEL_COL}' not found")
        continue

    class_counts = df[LABEL_COL].value_counts().sort_index()
    total_samples = len(df)

    p0 = class_counts.get(0, 0) / total_samples
    p1 = class_counts.get(1, 0) / total_samples

    # ----- Hellinger Distance -----
    hd = (1 / np.sqrt(2)) * np.sqrt(
        (np.sqrt(p0) - sqrt_q0) ** 2 +
        (np.sqrt(p1) - sqrt_q1) ** 2
    )

    print(f"📁 {os.path.basename(csv_file)}")
    print(f"Total samples: {total_samples}")
    print(f"  Class 0: {class_counts.get(0, 0)}")
    print(f"  Class 1: {class_counts.get(1, 0)}")
    print(f"  P(C0) = {p0:.6f}, P(C1) = {p1:.6f}")
    print(f"  Hellinger Distance (HD) = {hd:.6f}")
    print("-" * 55)