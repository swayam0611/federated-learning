import pandas as pd
import numpy as np
import os

# ================================
# CONFIG
# ================================
INPUT_CSV = "/home/coep/Desktop/0.1_wo_Attack_Eval_srvr/clients/custom_splits/merged_cardio.csv"
OUTPUT_DIR = "/home/coep/Desktop/0.1_wo_Attack_Eval_srvr/clients/custom_splits/output_final"
LABEL_COL = "cardio"
TEST_RATIO = 0.20
SEED = 42

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(INPUT_CSV)

print(f"\n✔ Loaded dataset with {len(df)} samples")

# ================================
# SHUFFLE DATA
# ================================
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ================================
# 20% TEST SPLIT
# ================================
test_size = int(len(df) * TEST_RATIO)

df_test = df.iloc[:test_size]
df_train = df.iloc[test_size:]

df_test.to_csv(os.path.join(OUTPUT_DIR, "testset.csv"), index=False)

print(f"\n✔ Test set saved: {len(df_test)} samples")
print("Test set class counts:")
print(df_test[LABEL_COL].value_counts().sort_index())

# ================================
# TRAIN SET COUNTS CHECK
# ================================
train_counts = df_train[LABEL_COL].value_counts().sort_index()

print("\nRemaining TRAIN dataset class counts:")
for k, v in train_counts.items():
    print(f"  Class {k}: {v}")
print(f"  Total TRAIN samples: {train_counts.sum()}")

# ================================
# SPLIT TRAIN DATA BY CLASS
# ================================
class0 = df_train[df_train[LABEL_COL] == 0].sample(frac=1, random_state=SEED)
class1 = df_train[df_train[LABEL_COL] == 1].sample(frac=1, random_state=SEED)

# ================================
# EXACT CLIENT DISTRIBUTION
# ================================
client_specs = [
    (7661, 3364),  # Client 1
    (3364, 7661),  # Client 2
    (7661, 3364),  # Client 3
    (3364, 7661),  # Client 4
    (5562, 4364),  # Client 5
]

clients = []
idx0 = idx1 = 0

for i, (c0, c1) in enumerate(client_specs, start=1):
    part = pd.concat([
        class0.iloc[idx0:idx0 + c0],
        class1.iloc[idx1:idx1 + c1]
    ])

    idx0 += c0
    idx1 += c1

    part = part.sample(frac=1, random_state=SEED)
    clients.append(part)

    out_path = os.path.join(OUTPUT_DIR, f"client_{i}.csv")
    part.to_csv(out_path, index=False)

# ================================
# FINAL VERIFICATION
# ================================
print("\n================ FINAL VERIFICATION ================")

total0 = total1 = total = 0

for i, c in enumerate(clients, start=1):
    counts = c[LABEL_COL].value_counts().sort_index()
    c0 = counts.get(0, 0)
    c1 = counts.get(1, 0)

    print(f"Client {i}: Class0={c0}, Class1={c1}, Total={c0 + c1}")

    total0 += c0
    total1 += c1
    total += c0 + c1

print("\nTOTAL ACROSS ALL CLIENTS")
print(f"Class 0: {total0}")
print(f"Class 1: {total1}")
print(f"Total  : {total}")

print("\n✅ DONE: testset.csv + 5 client CSVs created successfully")
