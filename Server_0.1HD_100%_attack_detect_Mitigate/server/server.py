import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.preprocessing import StandardScaler
from collections import deque
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
# SCALER
# ============================================================
SCALER_DIR = "/home/coep/Desktop/HD01_final/HD01_harmful/Server_0.1HD_100%_attack_detect_Mitigate/shared"

scaler = StandardScaler()
scaler.mean_ = np.load(os.path.join(SCALER_DIR, "scaler_mean.npy"))
scaler.scale_ = np.load(os.path.join(SCALER_DIR, "scaler_scale.npy"))
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = scaler.mean_.shape[0]

# ============================================================
# MODEL
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
# LOAD SERVER TEST DATA
# ============================================================
def load_server_test_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["cardio"]).values.astype(np.float32)
    y = df["cardio"].values.astype(np.float32)
    X = scaler.transform(X)

    return DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=32,
        shuffle=False
    ), X.shape[1]

# ============================================================
# TESTSET DISTRIBUTION
# ============================================================
def compute_testset_probs(loader):
    y_all = torch.cat([y for _, y in loader])
    q1 = (y_all == 1).sum().item() / len(y_all)
    q0 = (y_all == 0).sum().item() / len(y_all)
    return q0, q1

# ============================================================
# HELLINGER DISTANCE
# ============================================================
def hellinger_distance(tp, fp, tn, fn, q0, q1):
    total = tp + fp + tn + fn
    if total == 0: return 0.0
    p0, p1 = (tn + fn) / total, (tp + fp) / total
    return (1 / math.sqrt(2)) * math.sqrt(
        (math.sqrt(p0) - math.sqrt(q0)) ** 2 +
        (math.sqrt(p1) - math.sqrt(q1)) ** 2
    )

# ============================================================
# GLOBAL EVALUATION
# ============================================================
def evaluate_global(model, dataloader):
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for X, y in dataloader:
            preds = (torch.sigmoid(model(X)) > 0.5).float()
            tp += int(((preds == 1) & (y == 1)).sum())
            fp += int(((preds == 1) & (y == 0)).sum())
            tn += int(((preds == 0) & (y == 0)).sum())
            fn += int(((preds == 0) & (y == 1)).sum())

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return acc, recall, precision, f1

# ============================================================
# CUSTOM STRATEGY: SINGLE PIPELINE DETECTION & MITIGATION
# ============================================================
class FedAvgWithDetection(fl.server.strategy.FedAvg):
    def __init__(self, log_file, q0, q1, **kwargs):
        super().__init__(**kwargs)
        self.log_file = log_file
        self.q0 = q0
        self.q1 = q1
        self.hd_history = {}
        self.suspicion_count = {}
        self.good_model_buffer = {}  # Per-client backup of clean parameters
        
        self.MA_WINDOW = 3
        self.RTD_THRESHOLD = 0.05
        self.PROBATION = 3
        self.START_DETECTION_ROUND = 5

    def aggregate_fit(self, rnd, results, failures):
        header = f"\n========== ROUND {rnd} =========="
        print(header)
        with open(self.log_file, "a") as f:
            f.write(header + "\n")

        mitigated_results = []

        for client, res in results:
            cid = res.metrics.get("client_id", "NA")
            tp, fp, tn, fn = res.metrics.get("tp", 0), res.metrics.get("fp", 0), res.metrics.get("tn", 0), res.metrics.get("fn", 0)
            total = tp + fp + tn + fn
            
            hd = hellinger_distance(tp, fp, tn, fn, self.q0, self.q1)
            acc = (tp + tn) / total if total else 0.0
            specificity = tn / (tn + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            p0 = (tn + fn) / total if total > 0 else 0.0
            p1 = (tp + fp) / total if total > 0 else 0.0
            dist_log = f"p0={p0:.4f} | p1={p1:.4f} | q0={self.q0:.4f} | q1={self.q1:.4f}"

            self.hd_history.setdefault(cid, deque(maxlen=self.MA_WINDOW))
            self.suspicion_count.setdefault(cid, 0)
            
            rtd = 0.0
            status = "OK"
            flip_type = "None"

            # 1. Warmup / Collection
            if rnd <= self.START_DETECTION_ROUND or len(self.hd_history[cid]) < self.MA_WINDOW:
                self.hd_history[cid].append(hd)
                self.good_model_buffer[cid] = res.parameters
                mitigated_results.append((client, res))
                msg = f"[Client {cid}] HD={hd:.4f} | COLLECTION/WARMUP"
            else:
                # 2. Detection
                ma_hd = np.mean(list(self.hd_history[cid]))
                rtd = abs(hd - ma_hd)

                if rtd > self.RTD_THRESHOLD:
                    self.suspicion_count[cid] += 1
                    
                    if self.suspicion_count[cid] >= self.PROBATION:
                        status = "🚨 ATTACK DETECTED - SWAPPING WITH BUFFER"
                        flip_type = "Attack"
                        # MITIGATION: Replace malicious parameters with last known good parameters
                        res = fl.common.FitRes(
                            parameters=self.good_model_buffer[cid],
                            num_examples=res.num_examples,
                            metrics=res.metrics,
                            status=res.status
                        )
                    else:
                        status = "⚠️ SUSPICIOUS - OBSERVE"
                    
                    mitigated_results.append((client, res))
                    msg = f"[Client {cid}] HD={hd:.4f} | RTD={rtd:.4f} | Status={status}"
                else:
                    # 3. Clean Update
                    self.suspicion_count[cid] = 0
                    self.hd_history[cid].append(hd)
                    self.good_model_buffer[cid] = res.parameters
                    mitigated_results.append((client, res))
                    msg = f"[Client {cid}] HD={hd:.4f} | RTD={rtd:.4f} | OK"

            # Console and Server Log
            print(msg)
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")

            # Client-Specific Log File (Legacy Format)
            with open(f"{cid.lower()}.txt", "a") as f:
                f.write(
                    f"ROUND {rnd} | Acc={acc:.4f} | Recall ={recall:.4f} | "
                    f"Spec={specificity:.4f} | HD={hd:.6f} | RTD={rtd:.4f} | "
                    f"Flip={flip_type} | {dist_log}\n"
                )

        # Aggregate the "Mitigated Results" (Single Pipeline)
        return super().aggregate_fit(rnd, mitigated_results, failures)

# ============================================================
# START SERVER
# ============================================================
if __name__ == "__main__":
    TEST_CSV = "/home/coep/Desktop/HD01_final/HD01_harmful/Server_0.1HD_100%_attack_detect_Mitigate/clients/custom_splits/output_final/testset.csv"
    GLOBAL_LOG = "server_metrics.txt"

    if os.path.exists(GLOBAL_LOG): os.remove(GLOBAL_LOG)
    for cid in ["c1", "c2", "c3"]:
        if os.path.exists(f"{cid}.txt"): os.remove(f"{cid}.txt")

    testloader, input_size = load_server_test_data(TEST_CSV)
    
    torch.manual_seed(SEED)
    model = NeuralNetwork(input_size)
    q0, q1 = compute_testset_probs(testloader)

    def server_evaluate_fn(server_round, parameters, config):
        state_dict = model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        model.load_state_dict(state_dict, strict=True)

        acc, recall, precision, f1 = evaluate_global(model, testloader)
        
        msg = (
            f"ROUND {server_round} | "
            f"Acc={acc:.4f} | Recall={recall:.4f} | "
            f"Precision={precision:.4f} | F1={f1:.4f}"
        )
        print(f"\n🌍 {msg}")
        with open(GLOBAL_LOG, "a") as f:
            f.write(msg + "\n")
        
        return 0.0, {"accuracy": acc}

    strategy = FedAvgWithDetection(
        log_file=GLOBAL_LOG,
        q0=q0, q1=q1,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_fn=server_evaluate_fn,
        on_fit_config_fn=lambda rnd: {"round": rnd}
    )

    fl.server.start_server(
        server_address="127.0.0.1:8081",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )