import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import random
import math
import re
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yaml;

# global system setup
SEED = 30
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#using yaml to get config files values
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = config["device"]
ATTACK_ROUND = config["attack_round"]
SERVER_LOG = config["server_log"]
ATTACKING_CLIENT = config["attacking_client"]
FLIP_TYPE = config["flip_type"]
SHEET_START_CELL = config["sheet_start_cell"]

if os.path.exists(SERVER_LOG):
    os.remove(SERVER_LOG)

# model def
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

# metrics and algo
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

def flip_labels(loader, type):
    X_all, y_all = [], []

    for X, y in loader:
        X_all.append(X)
        y_all.append(y)

    X_all = torch.cat(X_all)
    y_all = torch.cat(y_all)

    if type=="1->0":
        y_all[y_all == 1] = 0
    if type=="0->1":
        y_all[y_all == 0] = 1

    flipped_loader = DataLoader(
        TensorDataset(X_all, y_all),
        batch_size=32,
        shuffle=False,
        drop_last=False
    )

    return flipped_loader

# optimized virtual client implementation
class CardioClient(fl.client.NumPyClient):

    def __init__(self, cid, trainloader, testloader, input_size, num_class_0, num_class_1, total_samples):
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = NeuralNetwork(input_size).to(DEVICE)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.current_round = 0
        
        # static distribution values
        self.num_class_0 = num_class_0
        self.num_class_1 = num_class_1
        self.total_samples = total_samples

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
        if self.current_round >= ATTACK_ROUND and self.cid == ATTACKING_CLIENT:
            print(f"[{self.cid}] ATTACK ACTIVE at round {self.current_round}: flipping all {FLIP_TYPE}")
            train_loader = flip_labels(self.trainloader, FLIP_TYPE)

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
                "q0": q0, "q1": q1,
                "num_class_0": self.num_class_0,
                "num_class_1": self.num_class_1,
                "total_samples": self.total_samples
            }
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc, recall, precision, f1, tp, fp, tn, fn, q0, q1 = evaluate_model(self.model, self.testloader)
        return 1 - acc, len(self.testloader.dataset), {
            "client_id": self.cid, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "q0": q0, "q1": q1
        }

# attack detection strategy
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
        
        # State repositories for caching before and after attack metrics
        self.round_9_snapshot = {}
        self.global_acc_before_attack = 0.0
        self.global_acc_after_attack = 0.0

    def aggregate_fit(self, rnd, results, failures):
        print(f"\n===== ROUND {rnd} BEFORE AGGREGATION =====")
        mitigated_results = []

        with open(SERVER_LOG, "a") as f:
            f.write(f"\nROUND {rnd}\nBEFORE AGGREGATION\n")
            tot_all = 0
            for client, res in results: 
                metrics = res.metrics
                tot_all += int(metrics.get("total_samples", 0))
            for client, res in results:
                metrics = res.metrics
                cid = metrics.get("client_id", "NA")
                tp, fp, tn, fn = int(metrics.get("tp", 0)), int(metrics.get("fp", 0)), int(metrics.get("tn", 0)), int(metrics.get("fn", 0))
                
                n_c0 = int(metrics.get("num_class_0", 0))
                n_c1 = int(metrics.get("num_class_1", 0))
                tot = int(metrics.get("total_samples", 0))

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

                # 📸 Snapshot Caching Strategy Logic
                if rnd == ATTACK_ROUND - 1:
                    self.round_9_snapshot[cid] = {
                        "acc": acc, "ehd": ehd, "c0": n_c0, "c1": n_c1, "total": tot
                    }
                elif rnd == ATTACK_ROUND:
                    r9 = self.round_9_snapshot.get(cid, {"acc": 0.0, "ehd": 0.0, "c0": n_c0, "c1": n_c1, "total": tot})
                    if (FLIP_TYPE == "1->0"):
                        attack_pct = (n_c1 / tot_all) * 100 if tot else 0.0
                    if (FLIP_TYPE == "0->1"):
                        attack_pct = (n_c0 / tot_all) * 100 if tot else 0.0

                    delta_acc = r9["acc"] - acc
                    delta_ehd = ehd - r9["ehd"]
                    
                    # Log horizontal data line string for sheet parsing
                    paper_summary = (
                        f"PAPER_ROW -> Client=C{int(cid)+1} "
                        f"Tot={tot} C0={n_c0} C1={n_c1} AtkPct={attack_pct:.2f} "
                        f"AccBf={r9['acc']:.2f} AccAf={acc:.2f} dAcc={delta_acc:.2f} "
                        f"EhdBf={r9['ehd']:.2f} EhdAf={ehd:.2f} dEhd={delta_ehd:.2f} "
                        f"RecallAf={recall:.2f}"
                    )
                    f.write(paper_summary + "\n")

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
        
        if rnd == 9:
            self.global_acc_before_attack = global_acc
        elif rnd == 10:
            self.global_acc_after_attack = global_acc
        elif rnd == 11:
            with open(SERVER_LOG, "a") as f:
                f.write(f"GLOBAL_SNAPSHOT -> Bf={self.global_acc_before_attack:.2f} Af={self.global_acc_after_attack:.2f} Mit={global_acc:.2f}\n")

        msg = f"GLOBAL -> Acc={global_acc:.4f} Recall={global_recall:.4f} Precision={global_precision:.4f} F1={global_f1:.4f}"
        print(msg)
        with open(SERVER_LOG, "a") as f:
            f.write(f"\nAFTER AGGREGATION\n{msg}\n")
        return super().aggregate_evaluate(rnd, results, failures)


def get_client_fn(csv_dir, input_size):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SCALER_DIR = os.path.join(BASE_DIR, "shared")
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(SCALER_DIR, "scaler_mean.npy"))
    scaler.scale_ = np.load(os.path.join(SCALER_DIR, "scaler_scale.npy"))
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]

    def client_fn(cid: str) -> fl.client.Client:
        partition_map = {
            "0": "partition0.csv", 
            "1": "partition1.csv", 
            "2": "partition2.csv"
        }
        
        csv_path = os.path.join(csv_dir, partition_map[cid])
        
        df = pd.read_csv(csv_path)
        
        # Parse data counts seamlessly
        class_counts = df["cardio"].value_counts().to_dict()
        num_class_0 = int(class_counts.get(0.0, 0))
        num_class_1 = int(class_counts.get(1.0, 0))
        total_samples = num_class_0 + num_class_1

        X = df.drop(columns=["cardio"]).values.astype(np.float32)
        y = df["cardio"].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=SEED,
            stratify=y
        )
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

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

        return CardioClient(cid, trainloader, testloader, input_size, num_class_0, num_class_1, total_samples).to_client()
    return client_fn

# HORIZONTAL 
def export_logs_to_sheets(log_file_path, spreadsheet_title, client_start_cell, attacking_client):
    print(f"\nCompiling 15-column horizontal paper snapshot matrix at cell location {client_start_cell}...")
    
    paper_rows = []
    g_bf = g_af = g_mit = ""

    with open(log_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "GLOBAL_SNAPSHOT" in line:
                g_match = re.search(r"Bf=(?P<bf>[\d\.]+) Af=(?P<af>[\d\.]+) Mit=(?P<mit>[\d\.]+)", line)
                if g_match:
                    g_bf = float(g_match.group("bf"))
                    g_af = float(g_match.group("af"))
                    g_mit = float(g_match.group("mit"))
            elif "PAPER_ROW" in line:
                attacking_label = f"C{int(attacking_client) + 1}"
                if f"Client={attacking_label}" not in line: 
                    continue
                c_match = re.search(
                    r"Client=(?P<client>\w+)\s+Tot=(?P<tot>\d+)\s+C0=(?P<c0>\d+)\s+C1=(?P<c1>\d+)\s+AtkPct=(?P<atkp>[\d\.]+)\s+"
                    r"AccBf=(?P<abf>[\d\.]+)\s+AccAf=(?P<aaf>[\d\.]+)\s+dAcc=(?P<da>[\d\.-]+)\s+"
                    r"EhdBf=(?P<ebf>[\d\.]+)\s+EhdAf=(?P<eaf>[\d\.]+)\s+dEhd=(?P<de>[\d\.-]+)\s+RecallAf=(?P<rec>[\d\.]+)", 
                    line
                )
                if c_match:
                    paper_rows.append({
                        "client": c_match.group("client"), "tot": int(c_match.group("tot")),
                        "c0": int(c_match.group("c0")), "c1": int(c_match.group("c1")), "atkp": float(c_match.group("atkp")),
                        "abf": float(c_match.group("abf")), "aaf": float(c_match.group("aaf")), "da": float(c_match.group("da")),
                        "ebf": float(c_match.group("ebf")), "eaf": float(c_match.group("eaf")), "de": float(c_match.group("de")),
                        "rec": float(c_match.group("rec"))
                    })

    # Ensure clean alphabetical arrangement (C1, C2, C3)
    paper_rows = sorted(paper_rows, key=lambda x: x["client"])

    final_upload_matrix = []
    for item in paper_rows:
        final_upload_matrix.append([
            f"0 ({item['tot']})",    # (1) HD and total samples
            item["client"],          # (2) Client
            item["c0"],              # (3) No. of Class 0
            item["c1"],              # (4) No. of Class 1
            item["atkp"],            # (5) Attack %
            item["abf"],             # (6) Acc Before
            item["aaf"],             # (7) Acc After
            item["da"],              # (8) Delta Acc
            item["ebf"],             # (9) EHD Before
            item["eaf"],             # (10) EHD After
            item["de"],              # (11) Delta EHD
            item["rec"],             # (12) Recall After Attack
            g_bf,                    # (13) Global Acc Before Attack
            g_af,                    # (14) Global Acc After Attack
            g_mit                    # (15) Global Acc After Mitig.
        ])

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("sheets_api_credentials.json", scope)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open(spreadsheet_title)
        try:
            client_sheet = spreadsheet.worksheet("Client_Metrics")
        except gspread.exceptions.WorksheetNotFound:
            client_sheet = spreadsheet.add_worksheet(title="Client_Metrics", rows="100", cols="15")
        
        # Pushes exactly 15 horizontal elements
        client_sheet.update(range_name=client_start_cell, values=final_upload_matrix)
        print("Paper horizontal data matrix updated cleanly!")
    except Exception as e:
        print(f"Error uploading to Google Sheets: {e}")

# main function
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data", "custom_splits", "output_final")
    sample_df = pd.read_csv(os.path.join(DATA_DIR, "partition0.csv"))
    INPUT_SIZE = sample_df.shape[1] - 1 

    print(f"Initializing Virtual Engine for 3 clients. Features detected: {INPUT_SIZE}")

    strategy = FedAvgAttackDetection(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3,
        on_fit_config_fn=lambda server_round: {"current_round": server_round}
    )

    fl.simulation.start_simulation(
        client_fn=get_client_fn(DATA_DIR, INPUT_SIZE),
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
        client_resources={"num_cpus": 2},
        ray_init_args={
        "include_dashboard": False,
        "local_mode": True,   # forces Ray to run everything in a single process, sequentially
        }
    )

    export_logs_to_sheets(SERVER_LOG, "fl_test", SHEET_START_CELL, ATTACKING_CLIENT)