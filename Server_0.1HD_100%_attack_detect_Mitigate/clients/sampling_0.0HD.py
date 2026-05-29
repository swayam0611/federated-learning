import pandas as pd

# Load merged CSV
merged_df = pd.read_csv(
    "/home/coep/Desktop/HD00_final/HD00_harmless/0.0HD_without attack/clients/custom_splits/output_final/merged.csv"
)
# Print class-wise counts
class_counts = merged_df["cardio"].value_counts().sort_index()

print("merged.csv class distribution:")
print(f"Class 0 count: {class_counts.get(0, 0)}")
print(f"Class 1 count: {class_counts.get(1, 0)}")
print(f"Total samples: {len(merged_df)}")
# Partition specifications
partitions = [
    {"name": "client_1", "class0": 8632, "class1": 8632},
    {"name": "client_2", "class0": 8632, "class1": 8632},
    {"name": "client_3", "class0": 8632, "class1": 8632},
]

# Shuffle the data to randomize rows
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate by class
class0_df = merged_df[merged_df['cardio'] == 0]
class1_df = merged_df[merged_df['cardio'] == 1]

# Create partitions
for part in partitions:
    part0 = class0_df.iloc[:part["class0"]]
    part1 = class1_df.iloc[:part["class1"]]
    
    # Combine and shuffle partition
    part_df = (
        pd.concat([part0, part1])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    
    # Save partition CSV
    save_path = (
        f"/home/coep/Desktop/HD00_final/HD00_harmless/0.0HD_without attack/clients/custom_splits/output_final/3_nodes/{part['name']}.csv"
    )
    part_df.to_csv(save_path, index=False)
    
    # 🔹 PRINT COUNTS
    count_0 = (part_df['cardio'] == 0).sum()
    count_1 = (part_df['cardio'] == 1).sum()
    total = len(part_df)

    print(f"{part['name']} created:")
    print(f"  Class 0 count: {count_0}")
    print(f"  Class 1 count: {count_1}")
    print(f"  Total samples: {total}")
    print("-" * 40)

    # Remove used rows from class dataframes
    class0_df = class0_df.iloc[part["class0"]:]
    class1_df = class1_df.iloc[part["class1"]:]

print("✅ Partitions created successfully!")
