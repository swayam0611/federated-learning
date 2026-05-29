import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import random

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cpu"

ATTACK_ROUND = 10

# ==========================================
# SHARED SCALER
# ==========================================

SCALER_DIR = "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/shared"

scaler = StandardScaler()
scaler.mean_ = np.load(os.path.join(SCALER_DIR, "scaler_mean.npy"))
scaler.scale_ = np.load(os.path.join(SCALER_DIR, "scaler_scale.npy"))
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = scaler.mean_.shape[0]

# ==========================================
# MODEL
# ==========================================

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

# ==========================================
# LOAD DATA — single CSV, 80/20 split
# ==========================================

def load_csv(csv):
    df = pd.read_csv(csv)
    X = df.drop(columns=["cardio"]).values.astype(np.float32)
    y = df["cardio"].values.astype(np.float32)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # Scale AFTER splitting (fit on train, transform both)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    trainloader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=32,
        shuffle=False,
        drop_last=False
    )

    testloader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=32,
        shuffle=False,
        drop_last=False
    )

    input_size = X_train.shape[1]
    return trainloader, testloader, input_size

# ==========================================
# LABEL FLIPPING ATTACK
# ==========================================

def flip_labels(loader):
    X_all, y_all = [], []

    for X, y in loader:
        X_all.append(X)
        y_all.append(y)

    X_all = torch.cat(X_all)
    y_all = torch.cat(y_all)

    y_all[y_all == 1] = 0

    flipped_loader = DataLoader(
        TensorDataset(X_all, y_all),
        batch_size=32,
        shuffle=False,
        drop_last=False
    )

    return flipped_loader

# ==========================================
# EVALUATION
# ==========================================

def evaluate_model(model, loader):
    model.eval()
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred = (torch.sigmoid(model(X)) > 0.5).float()

            tp += int(((pred == 1) & (y == 1)).sum())
            fp += int(((pred == 1) & (y == 0)).sum())
            tn += int(((pred == 0) & (y == 0)).sum())
            fn += int(((pred == 0) & (y == 1)).sum())

    total = tp + fp + tn + fn
    acc       = (tp + tn) / total
    recall    = tp / (tp + fn)    if tp + fn else 0
    precision = tp / (tp + fp)    if tp + fp else 0
    f1        = (2 * precision * recall / (precision + recall)) if precision + recall else 0
    q0 = (tn + fp) / total
    q1 = (tp + fn) / total

    return acc, recall, precision, f1, tp, fp, tn, fn, q0, q1

# ==========================================
# CLIENT
# ==========================================

class CardioClient(fl.client.NumPyClient):

    def __init__(self, model, trainloader, testloader, cid):
        self.model       = model
        self.trainloader = trainloader
        self.testloader  = testloader
        self.cid         = cid
        self.loss_fn     = nn.BCEWithLogitsLoss()
        self.optimizer   = optim.Adam(self.model.parameters(), lr=0.001)
        self.current_round = 0

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state = self.model.state_dict()
        for k, v in zip(state.keys(), parameters):
            state[k] = torch.tensor(v)
        self.model.load_state_dict(state, strict=True)

# ==========================================
# PRE AGGREGATION
# ==========================================

    def fit(self, parameters, config):
        self.current_round += 1
        self.set_parameters(parameters)
        self.model.train()

        if self.current_round >= ATTACK_ROUND:
            print(f"[{self.cid}] ATTACK ACTIVE at round {self.current_round}: flipping all class 1 -> class 0")
            train_loader = flip_labels(self.trainloader)  # ← ONLY train is flipped
        else:
            train_loader = self.trainloader

        # test_loader always uses original labels
        test_loader = self.testloader

        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(X), y)
            loss.backward()
            self.optimizer.step()

        acc, recall, precision, f1, tp, fp, tn, fn, q0, q1 = evaluate_model(self.model, test_loader)

        return (
            self.get_parameters(),
            len(self.trainloader.dataset),
            {
                "client_id": self.cid,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "q0": q0, "q1": q1
            }
        )

# ==========================================
# POST AGGREGATION
# ==========================================

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Always evaluate on original (unflipped) test labels
        test_loader = self.testloader

        acc, recall, precision, f1, tp, fp, tn, fn, q0, q1 = evaluate_model(self.model, test_loader)
        loss = 1 - acc

        return (
            loss,
            len(self.testloader.dataset),
            {
                "client_id": self.cid,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "q0": q0, "q1": q1
            }
        )

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    CLIENT_ID  = "C1"
    CLIENT_CSV = "/home/coep/Desktop/Local Evaluation_0.1HD/c1/cardio_0_HD_LE_attack_D_M_train_test_attack/clients/custom_splits/output_final/partition0.csv"

    trainloader, testloader, input_size = load_csv(CLIENT_CSV)

    model = NeuralNetwork(input_size).to(DEVICE)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=CardioClient(model, trainloader, testloader, CLIENT_ID)
    )