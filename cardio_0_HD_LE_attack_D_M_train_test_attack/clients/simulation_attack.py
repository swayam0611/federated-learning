import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import random
import math
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# GLOBAL SYSTEM SETUP & SEEDING
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cpu"
ATTACK_ROUND = 10
SERVER_LOG = "server_metrics.txt"

if os.path.exists(SERVER_LOG):
    os.remove(SERVER_LOG)

# MODEL DEFINITION
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

# METRICS & ALGORITHMS (EHD)
def compute_ehd(tp, fp, tn, fn, q0=0.5, q1=0.5):
    total = tp + fp + tn + fn
    if total == 0: return 0.0
    p0 = (tn + fn) / total
    p1 = (tp + fp) / total
    return (1 / math.sqrt(2)) * math.sqrt(
        (math.sqrt(p0) - math.sqrt(q0)) ** 2 +
        (math.sqrt(p1) - math.sqrt(q1)) ** 2
    )

def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    if total == 0: return 0, 0, 0, 0
    acc = (tp + tn) / total
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return acc, recall, precision, f1

def evaluate_model(model, loader):
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = (torch.sigmoid(model(X)) > 0.5).float()
            tp += int(((pred == 1) & (y == 1)).sum())
            fp += int(((pred == 1) & (y == 0)).sum())
            tn += int(((pred == 0) & (y == 0)).sum())
            fn += int(((pred == 0) & (y == 1)).sum())
            
    total = tp + fp + tn + fn
    acc, recall, precision, f1 = compute_metrics(tp, fp, tn, fn)
    q0 = (tn + fp) / total if total else 0
    q1 = (tp + fn) / total if total else 0
    return acc, recall, precision, f1, tp, fp, tn, fn, q0, q1

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

# OPTIMIZED VIRTUAL CLIENT IMPLEMENTATION
class CardioClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, testloader, input_size):
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = NeuralNetwork(input_size).to(DEVICE)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.current_round = 0

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state = self.model.state_dict()
        for k, v in zip(state.keys(), parameters):
            state[k] = torch.tensor(v)
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.current_round = config.get("current_round", 1)
        self.set_parameters(parameters)
        self.model.train()

        train_loader = self.trainloader
        if self.current_round >= ATTACK_ROUND and self.cid == "0":
            print(f"[{self.cid}] ATTACK ACTIVE at round {self.current_round}: flipping all class 1 -> class 0")
            train_loader = flip_labels(self.trainloader)  # ← ONLY train is flipped

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

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc, recall, precision, f1, tp, fp, tn, fn, q0, q1 = evaluate_model(self.model, self.testloader)
        return 1 - acc, len(self.testloader.dataset), {
            "client_id": self.cid, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "q0": q0, "q1": q1
        }

# ATTACK DETECTION STRATEGY
class FedAvgAttackDetection(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hd_history = {}
        self.suspicion_count = {}
        self.good_model_buffer = {}
        self.MA_WINDOW = 3
        self.RTD_THRESHOLD = 0.05
        self.PROBATION = 2
        self.START_DETECTION_ROUND = 5

    def aggregate_fit(self, rnd, results, failures):
        print(f"\n===== ROUND {rnd} BEFORE AGGREGATION =====")
        mitigated_results = []

        with open(SERVER_LOG, "a") as f:
            f.write(f"\nROUND {rnd}\nBEFORE AGGREGATION\n")

            for client, res in results:
                metrics = res.metrics
                cid = metrics.get("client_id", "NA")
                tp, fp, tn, fn = int(metrics.get("tp", 0)), int(metrics.get("fp", 0)), int(metrics.get("tn", 0)), int(metrics.get("fn", 0))
                
                acc, recall, precision, f1 = compute_metrics(tp, fp, tn, fn)
                ehd = compute_ehd(tp, fp, tn, fn, q0=0.5, q1=0.5)

                self.hd_history.setdefault(cid, deque(maxlen=self.MA_WINDOW))
                self.suspicion_count.setdefault(cid, 0)

                rtd, status, flip_type = 0.0, "OK", "None"

                if rnd <= self.START_DETECTION_ROUND or len(self.hd_history[cid]) < self.MA_WINDOW:
                    self.hd_history[cid].append(ehd)
                    self.good_model_buffer[cid] = res.parameters
                    mitigated_results.append((client, res))
                    status = "WARMUP"
                else:
                    ma_ehd = np.mean(list(self.hd_history[cid]))
                    rtd = abs(ehd - ma_ehd)

                    if rtd > self.RTD_THRESHOLD:
                        self.suspicion_count[cid] += 1
                        if self.suspicion_count[cid] >= self.PROBATION:
                            status = "ATTACK DETECTED - ROLLBACK TO BUFFER"
                            flip_type = "Attack"
                            res = fl.common.FitRes(parameters=self.good_model_buffer[cid], num_examples=res.num_examples, metrics=res.metrics, status=res.status)
                        else:
                            status = "SUSPICIOUS - OBSERVE"
                        mitigated_results.append((client, res))
                    else:
                        if self.suspicion_count[cid] < self.PROBATION:
                            self.suspicion_count[cid] = 0
                            self.hd_history[cid].append(ehd)
                            self.good_model_buffer[cid] = res.parameters
                        else:
                            status = "ATTACK DETECTED - ROLLBACK TO BUFFER"
                            flip_type = "Attack"
                            res = fl.common.FitRes(parameters=self.good_model_buffer[cid], num_examples=res.num_examples, metrics=res.metrics, status=res.status)
                        mitigated_results.append((client, res))

                msg = f"[{cid}] Acc={acc:.4f} EHD={ehd:.6f} RTD={rtd:.6f} Suspicion={self.suspicion_count[cid]} Status={status}"
                print(msg)
                f.write(msg + "\n")

        return super().aggregate_fit(rnd, mitigated_results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        print("\n===== AFTER AGGREGATION =====")
        global_tp = global_fp = global_tn = global_fn = 0
        for _, res in results:
            metrics = res.metrics
            global_tp += int(metrics.get("tp", 0))
            global_fp += int(metrics.get("fp", 0))
            global_tn += int(metrics.get("tn", 0))
            global_fn += int(metrics.get("fn", 0))

        global_acc, global_recall, global_precision, global_f1 = compute_metrics(global_tp, global_fp, global_tn, global_fn)
        msg = f"GLOBAL -> Acc={global_acc:.4f} Recall={global_recall:.4f} Precision={global_precision:.4f} F1={global_f1:.4f}"
        print(msg)
        with open(SERVER_LOG, "a") as f:
            f.write(f"\nAFTER AGGREGATION\n{msg}\n")
        return super().aggregate_evaluate(rnd, results, failures)

# CENTRAL GENERATOR FOR ENGINE SIMULATION
def get_client_fn(csv_dir, input_size):
    """Factory function that Flower uses to build clients dynamically on-demand"""
    def client_fn(cid: str) -> fl.client.Client:
        #partition map: map csv files to client ids (cid)
        partition_map = {
            "0": "partition0.csv", 
            "1": "partition1.csv", 
            "2": "partition2.csv"
        }
        
        csv_path = os.path.join(csv_dir, partition_map[cid])
        
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["cardio"]).values.astype(np.float32)
        y = df["cardio"].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        trainloader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True, drop_last=True)
        testloader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32, shuffle=False, drop_last=True)

        return CardioClient(cid, trainloader, testloader, input_size).to_client()
    return client_fn

# main function
if __name__ == "__main__":
    #relative pathnames
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "custom_splits", "output_final")
    sample_df = pd.read_csv(os.path.join(DATA_DIR, "partition0.csv"))
    INPUT_SIZE = sample_df.shape[1] - 1 

    print(f"🎬 Initializing Virtual Engine for 3 clients. Features detected: {INPUT_SIZE}")

    strategy = FedAvgAttackDetection(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3,
        #global round number count
        on_fit_config_fn=lambda server_round: {"current_round": server_round}
    )

    # spawns workers as independent parallel processes behind a clean interface.
    fl.simulation.start_simulation(
        client_fn=get_client_fn(DATA_DIR, INPUT_SIZE),
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
        #assign n cpus to clients
        client_resources={"num_cpus": 2}
    )

import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def export_logs_to_sheets(log_file_path, spreadsheet_title, client_start_cell, global_start_cell):
    print("\nParsing simulation records for Google Sheets upload...")
    
    # ✅ FIXED REGEX: Matches your exact active log row properties cleanly
    client_pattern = re.compile(
        r"\[(?P<cid>[A-Z0-9]+)\] Acc=(?P<acc>[\d\.]+) EHD=(?P<ehd>[\d\.]+) RTD=(?P<rtd>[\d\.]+) Suspicion=(?P<susp>\d+) Status=(?P<status>.+)"
    )
    global_pattern = re.compile(
        r"GLOBAL -> Acc=(?P<acc>[\d\.]+) Recall=(?P<rec>[\d\.]+) Precision=(?P<prec>[\d\.]+) F1=(?P<f1>[\d\.]+)"
    )

    client_rows = [["Round", "Client ID", "Accuracy", "EHD", "RTD", "Suspicion Count", "Defense Status"]]
    global_rows = [["Round", "Global Accuracy", "Global Recall", "Global Precision", "Global F1-Score"]]

    current_round = 0

    with open(log_file_path, "r") as file:
        for line in file:
            line = line.strip()
            
            # Extract current execution round
            if "ROUND" in line:
                match_round = re.search(r"ROUND (\d+)", line)
                if match_round:
                    current_round = int(match_round.group(1))
                continue
            
            # Extract client metrics row
            c_match = client_pattern.search(line)
            if c_match:
                client_rows.append([
                    current_round, f"Client {c_match.group('cid')}", float(c_match.group('acc')),
                    float(c_match.group('ehd')), float(c_match.group('rtd')),
                    int(c_match.group('susp')), c_match.group('status').strip()
                ])
                continue

            # Extract global aggregation metrics row
            g_match = global_pattern.search(line)
            if g_match:
                global_rows.append([
                    current_round, float(g_match.group('acc')), float(g_match.group('rec')),
                    float(g_match.group('prec')), float(g_match.group('f1'))
                ])

    # Authenticate via Google Sheets API Workspace
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("sheets_api_credentials.json", scope)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open(spreadsheet_title)
        
        # ──► SYNCHRONIZE TAB 1: CLIENT METRICS
        try:
            client_sheet = spreadsheet.worksheet("Client_Metrics")
        except gspread.exceptions.WorksheetNotFound:
            client_sheet = spreadsheet.add_worksheet(title="Client_Metrics", rows="100", cols="7")
        
        # Bulk matrix push
        client_sheet.update(range_name = client_start_cell, values=client_rows)
        print("✅ Successfully updated 'Client_Metrics' tab!")

        # ──► SYNCHRONIZE TAB 2: GLOBAL METRICS
        try:
            global_sheet = spreadsheet.worksheet("Global_Metrics")
        except gspread.exceptions.WorksheetNotFound:
            global_sheet = spreadsheet.add_worksheet(title="Global_Metrics", rows="100", cols="5")
        
        global_sheet.update(range_name = global_start_cell, values=global_rows)
        print("✅ Successfully updated 'Global_Metrics' tab!")

    except Exception as e:
        print(f"❌ Error uploading to Google Sheets: {e}")

# ==================================================================
# EXECUTE PARSER AFTER SIMULATION TERMINATION
# ==================================================================
if __name__ == "__main__":
    # Your existing code that triggers fl.simulation.start_simulation(...)
    # ...
    
    # Trigger the bulk exporter once the server finishes all rounds
    export_logs_to_sheets(SERVER_LOG, "fl_test", "A1", "A1")