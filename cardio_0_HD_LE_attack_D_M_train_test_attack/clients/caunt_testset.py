import pandas as pd

csv_path = "/home/apeksha/Desktop/0.0hd/0.0HD_without attack/clients/custom_splits/output_final/testset.csv"

df = pd.read_csv(csv_path)

# Separate classes
df_0 = df[df["cardio"] == 0]
df_1 = df[df["cardio"] == 1]

# Find minority size
min_size = min(len(df_0), len(df_1))

# Downsample both classes to equal size
df_0_bal = df_0.sample(n=min_size, random_state=42)
df_1_bal = df_1.sample(n=min_size, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([df_0_bal, df_1_bal]).sample(frac=1, random_state=42).reset_index(drop=True)

# Counts after balancing
counts = df_balanced["cardio"].value_counts().sort_index()

print("After balancing:")
print(f"Label 0 count: {counts.get(0, 0)}")
print(f"Label 1 count: {counts.get(1, 0)}")
print(f"Total samples: {len(df_balanced)}")

# OPTIONAL: save balanced dataset
balanced_path = "/home/apeksha/Desktop/0.0hd/0.0HD_without attack/clients/custom_splits/output_final/testset_balanced.csv"
df_balanced.to_csv(balanced_path, index=False)

print(f"\nBalanced dataset saved to:\n{balanced_path}")
