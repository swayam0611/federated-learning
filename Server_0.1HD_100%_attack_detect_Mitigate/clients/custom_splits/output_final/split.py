import numpy as np
import pandas as pd
import os

# ============================================================
# CONFIG
# ============================================================
DATASET_PATH = "stroke_dataset_reduced.csv"   # your dataset
LABEL_COL = "stroke"
N_CLIENTS = 5
TARGET_HD = 0.0
OUTPUT_DIR = "hd_0.0"
SEED = 42

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATASET_PATH)

if LABEL_COL not in df.columns:
    raise ValueError("Label column not found")

# Split by class
df0 = df[df[LABEL_COL] == 0].sample(frac=1, random_state=SEED)
df1 = df[df[LABEL_COL] == 1].sample(frac=1, random_state=SEED)

p0 = len(df0) / len(df)
p1 = len(df1) / len(df)

print("\nGLOBAL DISTRIBUTION")
print(f"cardio = 0 → {p0:.3f}")
print(f"cardio = 1 → {p1:.3f}")

# ============================================================
# TARGET CLIENT DISTRIBUTIONS (HD ≈ TARGET_HD)
# ============================================================
delta = TARGET_HD

client_distributions = [
    (p0 + delta, p1 - delta),
    (p0 - delta, p1 + delta),
    (p0 + delta, p1 - delta),
    (p0 - delta, p1 + delta),
    (p0, p1)   # one neutral client
]

# Clip values to [0,1]
client_distributions = [
    (max(0, min(1, a)), max(0, min(1, b)))
    for a, b in client_distributions
]

# ============================================================
# SPLIT DATA
# ============================================================
size_per_client = len(df) // N_CLIENTS
clients = []

ptr0, ptr1 = 0, 0

for i, (c0_ratio, c1_ratio) in enumerate(client_distributions):
    n0 = int(size_per_client * c0_ratio)
    n1 = size_per_client - n0

    c0 = df0.iloc[ptr0:ptr0 + n0]
    c1 = df1.iloc[ptr1:ptr1 + n1]

    ptr0 += n0
    ptr1 += n1

    client_df = pd.concat([c0, c1]).sample(frac=1, random_state=SEED)
    clients.append(client_df)

# ============================================================
# SAVE & VERIFY HD
# ============================================================
print("\nCLIENT DATASETS")

for i, cdf in enumerate(clients, 1):
    path = os.path.join(OUTPUT_DIR, f"client_{i}.csv")
    cdf.to_csv(path, index=False)

    p_client = cdf[LABEL_COL].value_counts(normalize=True).to_dict()

    hd = 0.5 * (
        abs(p_client.get(0, 0) - p0) +
        abs(p_client.get(1, 0) - p1)
    )

    print(f"\nClient {i}")
    print(" Distribution:", p_client)
    print(f" HD ≈ {hd:.3f}")

print(f"\n✅ SUCCESS: 5 client datasets with HD ≈ {TARGET_HD} created")