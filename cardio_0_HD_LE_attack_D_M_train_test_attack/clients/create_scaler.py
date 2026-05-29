import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Use MASTER dataset (global distribution)
MASTER_DS = "/home/coep/Desktop/0.0hd/0.0HD_without attack/clients/custom_splits/output_final/Master_DS.csv"

df = pd.read_csv(MASTER_DS)

X = df.drop(columns=["cardio"]).values.astype(np.float32)

scaler = StandardScaler()
scaler.fit(X)

# Create shared folder if not exists
import os
SHARED_DIR = "/home/coep/Desktop/0.0hd/0.0HD_without attack/shared"
os.makedirs(SHARED_DIR, exist_ok=True)

# Save scaler parameters
np.save(f"{SHARED_DIR}/scaler_mean.npy", scaler.mean_)
np.save(f"{SHARED_DIR}/scaler_scale.npy", scaler.scale_)

print("✅ Scaler saved in:", SHARED_DIR)
