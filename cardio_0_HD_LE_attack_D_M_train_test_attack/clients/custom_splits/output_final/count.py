import pandas as pd

# ==========================================
# FILE PATHS
# ==========================================

file1 = "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/clients/custom_splits/output_final/testset.csv"

file2 = "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/clients/custom_splits/output_final/merged.csv"

output_file = "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/clients/custom_splits/output_final/main.csv"

# ==========================================
# LOAD CSV FILES
# ==========================================

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("File 1 Shape:", df1.shape)
print("File 2 Shape:", df2.shape)

# ==========================================
# COMBINE DATA
# ==========================================

combined_df = pd.concat([df1, df2], ignore_index=True)

print("Combined Shape:", combined_df.shape)

# ==========================================
# LABEL COLUMN (LAST COLUMN)
# ==========================================

label_col = combined_df.columns[-1]

zero_count = (combined_df[label_col] == 0).sum()
one_count = (combined_df[label_col] == 1).sum()

print("\nTotal Class Distribution")
print("Total 0 =", zero_count)
print("Total 1 =", one_count)

# ==========================================
# SAVE OUTPUT
# ==========================================

combined_df.to_csv(output_file, index=False)

print(f"\nSaved combined file at:\n{output_file}")