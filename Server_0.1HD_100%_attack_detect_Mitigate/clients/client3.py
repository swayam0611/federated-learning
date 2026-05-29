import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import random

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# SCALER (SHARED)
# ============================================================
SCALER_DIR = "/home/coep/Desktop/HD01_final/HD01_harmful/Server_0.1HD_100%_attack_detect_Mitigate/shared"

scaler = StandardScaler()
scaler.mean_ = np.load(os.path.join(SCALER_DIR, "scaler_mean.npy"))
scaler.scale_ = np.load(os.path.join(SCALER_DIR, "scaler_scale.npy"))
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = scaler.mean_.shape[0]

# ============================================================
# Model Definition
# ============================================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ============================================================
# Load TRAINING data
# ============================================================
def load_train_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["cardio"]).values.astype(np.float32)
    y = df["cardio"].values.astype(np.float32)

    X = scaler.transform(X)

    trainloader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=32,
        shuffle=False,
        drop_last=True
    )

    return trainloader, X.shape[1]

# ============================================================
# Load MASTER TEST data
# ============================================================
def load_master_test_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["cardio"]).values.astype(np.float32)
    y = df["cardio"].values.astype(np.float32)

    X = scaler.transform(X)

    testloader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=32,
        shuffle=False
    )

    return testloader

# ============================================================
# Evaluation (NO HD HERE)
# ============================================================
def evaluate_model(model, dataloader):
    model.eval()
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for X, y in dataloader:
            probs = torch.sigmoid(model(X))
            preds = (probs > 0.5).float()

            tp += int(((preds == 1) & (y == 1)).sum())
            fp += int(((preds == 1) & (y == 0)).sum())
            tn += int(((preds == 0) & (y == 0)).sum())
            fn += int(((preds == 0) & (y == 1)).sum())

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0

    return acc, recall, precision, f1, tp, fp, tn, fn

# ============================================================
# Flower Client
# ============================================================
class CardioClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, client_id):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_id = client_id

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.logfile = f"{client_id.lower()}.txt"
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        round_num = config.get("round", -1)
        self.set_parameters(parameters)
        self.model.train()

        for X, y in self.trainloader:
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(X), y)
            loss.backward()
            self.optimizer.step()

        # ---- LOCAL EVALUATION (UNCHANGED) ----
        acc, recall, precision, f1, tp, fp, tn, fn = evaluate_model(
            self.model, self.testloader
        )

        print(
            f"[Client {self.client_id} | Round {round_num} | MASTER_DS]\n"
            f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}\n"
            f"  Acc={acc:.4f}, Recall={recall:.4f}, "
            f"Precision={precision:.4f}, F1={f1:.4f}"
        )

        # ---- LOG (NO HD) ----
        with open(self.logfile, "a") as f:
            f.write(
                f"ROUND {round_num} | "
                f"Acc={acc:.4f} | Recall={recall:.4f} | "
                f"Precision={precision:.4f} | F1={f1:.4f} | "
                f"TP={tp} | FP={fp} | TN={tn} | FN={fn}\n"
            )

        # 🔴 SEND CONFUSION MATRIX TO SERVER
        return self.get_parameters(), len(self.trainloader.dataset), {
            "client_id": self.client_id,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

# ============================================================
# Start Client
# ============================================================
if __name__ == "__main__":

    CLIENT_ID = "C3"
    TRAIN_CSV = "/home/coep/Desktop/HD01_final/HD01_harmful/Server_0.1HD_100%_attack_detect_Mitigate/clients/custom_splits/output_final/3_nodes/client_3.csv"
    MASTER_DS = "/home/coep/Desktop/HD01_final/HD01_harmful/Server_0.1HD_100%_attack_detect_Mitigate/clients/custom_splits/output_final/Master_DS.csv"

    trainloader, input_size = load_train_data(TRAIN_CSV)
    testloader = load_master_test_data(MASTER_DS)

    model = NeuralNetwork(input_size)

    client = CardioClient(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        client_id=CLIENT_ID
    )

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=client
    )