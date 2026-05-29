import pandas as pd
import os

# ======================================
# LOAD MAIN DATASET
# ======================================

input_file = "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/clients/custom_splits/output_final/main.csv"

output_dir = "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/clients/custom_splits/output_final/"

df = pd.read_csv(input_file)

label_col = df.columns[-1]

# ======================================
# SPLIT CLASS 0 AND CLASS 1
# ======================================

class0 = df[df[label_col] == 0].sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

class1 = df[df[label_col] == 1].sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ======================================
# REQUIRED DISTRIBUTION
# ======================================

distribution = {
    "client_1": (2500, 19632),
    "client_2": (19632, 2500),
    "client_3": (5000, 17132)
}

c0_idx = 0
c1_idx = 0

# ======================================
# CREATE CLIENT FILES
# ======================================

for client, (n0, n1) in distribution.items():

    part0 = class0.iloc[c0_idx:c0_idx+n0]
    part1 = class1.iloc[c1_idx:c1_idx+n1]

    c0_idx += n0
    c1_idx += n1

    client_df = pd.concat(
        [part0, part1],
        ignore_index=True
    )

    client_df = client_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    save_path = os.path.join(
        output_dir,
        f"{client}.csv"
    )

    client_df.to_csv(
        save_path,
        index=False
    )

    print(f"\n{client}")
    print(client_df[label_col].value_counts())
    print("Saved:", save_path)

print("\nDone.")