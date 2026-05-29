import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# Model Definition
# ============================================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


# ============================================================
# Utility Functions
# ============================================================
def compute_metrics(model, dataloader):
    model.eval()
    correct, total = 0, 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            outputs = torch.sigmoid(model(X))
            preds = (outputs > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)

            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    acc = correct / total if total else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return acc, recall, precision, f1


def compute_histogram_and_proportion(df):
    hist_series = df["cardio"].value_counts().sort_index()
    hist_series = hist_series.reindex([0, 1], fill_value=0)

    hist = hist_series.values
    total = int(hist.sum())

    proportion = (hist / total) if total > 0 else np.array([0.0, 0.0])
    return hist.tolist(), proportion.tolist()


def compute_local_alpha(local_proportion):
    balanced = np.array([0.5, 0.5])
    diffs = np.abs(np.array(local_proportion) - balanced)
    return float(np.mean(diffs))


# ============================================================
# CENTRALIZED TRAINING
# ============================================================
if __name__ == "__main__":

    filepath = "/home/coep/Desktop/cardio/dataset_gupte_heterogenous/client_datasets_noon-IID/cardio_5_features/cardio.csv"

    # Load Data
    df = pd.read_csv(filepath)
    X = df.drop(columns=["cardio"]).values.astype(np.float32)
    y = df["cardio"].values.astype(np.float32)

    # -------------------------
    # Train-Test Split (80-20)
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    trainset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    testset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    # -------------------------
    # Model, Loss, Optimizer
    # -------------------------
    model = NeuralNetwork(input_size=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # -------------------------
    # Training Loop
    # -------------------------
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in trainloader:
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}")

    # -------------------------
    # Evaluation
    # -------------------------
    acc, rec, prec, f1 = compute_metrics(model, testloader)
    print("\n=========== CENTRALIZED EVALUATION ===========")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # -------------------------
    # Compute histogram and alpha
    # -------------------------
    hist, prop = compute_histogram_and_proportion(df)
    alpha = compute_local_alpha(prop)

    print("\n=========== DATA DISTRIBUTION INFO ===========")
    print(f"Histogram:       {hist}")
    print(f"Proportion:      {prop}")
    print(f"Local Alpha:     {alpha:.4f}")

