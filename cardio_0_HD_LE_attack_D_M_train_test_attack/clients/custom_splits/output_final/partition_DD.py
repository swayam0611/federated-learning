import pandas as pd

files = {
    "C1 - partition0": "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/clients/custom_splits/output_final/partition0.csv",
    "C2 - partition1": "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/clients/custom_splits/output_final/partition1.csv",
    "C3 - partition3": "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/clients/custom_splits/output_final/partition2.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)
    counts = df["cardio"].value_counts().sort_index()
    total = len(df)
    c0 = counts.get(0, 0)
    c1 = counts.get(1, 0)
    print(f"=== {name} ===")
    print(f"  Total  : {total}")
    print(f"  Class 0: {c0}  ({c0/total*100:.2f}%)")
    print(f"  Class 1: {c1}  ({c1/total*100:.2f}%)")
    print()