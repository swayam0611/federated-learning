import pandas as pd
import os
from sklearn.model_selection import train_test_split

# =====================================================
# LOAD DATASET
# =====================================================

dataset_path = "/home/coep/Desktop/Local Evaluation /cardio_0_HD_LE/clients/custom_splits/output_final/merged.csv"

df = pd.read_csv(dataset_path)

print("Original Dataset Shape:", df.shape)

# =====================================================
# LABEL COLUMN
# =====================================================

label_col = df.columns[-1]

print("\nOriginal Class Distribution:")
print(df[label_col].value_counts())

# =====================================================
# SHUFFLE CLASSES
# =====================================================

class0 = df[df[label_col] == 0].sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

class1 = df[df[label_col] == 1].sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# =====================================================
# IID 0.0 HD DISTRIBUTION
# =====================================================

client_distribution = [
    (8775, 8775),   # Client 1
    (8775, 8775),   # Client 2
    (8775, 8776)    # Client 3
]

# SAVE LOCATION
output_dir = "/home/coep/Desktop/Local Evaluation /cardio_0_HD_LE/clients/custom_splits/output_final/3_nodes"

os.makedirs(output_dir, exist_ok=True)

start0 = 0
start1 = 0

for client_id, (n0, n1) in enumerate(client_distribution, 1):

    part0 = class0.iloc[start0:start0+n0]
    part1 = class1.iloc[start1:start1+n1]

    start0 += n0
    start1 += n1

    client_df = pd.concat(
        [part0, part1]
    ).sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    train_df, test_df = train_test_split(
        client_df,
        test_size=0.20,
        stratify=client_df[label_col],
        random_state=42
    )

    train_df.to_csv(
        os.path.join(
            output_dir,
            f"client_{client_id}_train.csv"
        ),
        index=False
    )

    test_df.to_csv(
        os.path.join(
            output_dir,
            f"client_{client_id}_test.csv"
        ),
        index=False
    )

    print(f"\nClient {client_id}")
    print(client_df[label_col].value_counts())

    print("Train:", train_df.shape)
    print("Test :", test_df.shape)

print("\nSaved successfully at:")
print(output_dir)