import flwr as fl
import pandas as pd
import numpy as np
import os
import random
import math
import re
from collections import deque
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yaml

# ===================================================================
# GLOBAL SYSTEM SETUP & SEEDING (LOCK PARENT THREAD)
# ===================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = config["device"]
ATTACK_ROUND = config["attack_round"]
SERVER_LOG = config["server_log"]
ATTACKING_CLIENT = config["attacking_client"]
FLIP_TYPE = config["flip_type"]
SHEET_START_CELL = config["sheet_start_cell"]
EHD = config["ehd"]

if os.path.exists(SERVER_LOG):
    os.remove(SERVER_LOG)

# ===================================================================
# METRICS & ALGORITHMS 
# ===================================================================
def compute_ehd(tp, fp, tn, fn, q0, q1):
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

# ===================================================================
# SERVER-SIDE ATTACK DETECTION STRATEGY
# ===================================================================
class FedAvgServerEvalDetection(fl.server.strategy.FedAvg):
    def __init__(self, q0, q1, **kwargs):
        super().__init__(**kwargs)
        self.q0 = q0
        self.q1 = q1
        self.hd_history = {}
        self.suspicion_count = {}
        self.good_model_buffer = {}
        self.MA_WINDOW = 3
        self.RTD_THRESHOLD = 0.05
        self.PROBATION = 2
        self.START_DETECTION_ROUND = 5
        self.round_9_snapshot = {}

    def aggregate_fit(self, rnd, results, failures):
        print(f"\n===== ROUND {rnd} BEFORE AGGREGATION =====")
        results.sort(key=lambda x: x[1].metrics.get("client_id", "NA"))
        mitigated_results = []

        with open(SERVER_LOG, "a") as f:
            f.write(f"\nROUND {rnd}\nBEFORE AGGREGATION\n")
            tot_all = sum(int(res.metrics.get("total_samples", 0)) for _, res in results)
            
            for client, res in results:
                metrics = res.metrics
                cid = metrics.get("client_id", "NA")
                formatted_cid = f"C{int(cid)+1}" if str(cid).isdigit() else str(cid)

                tp, fp, tn, fn = int(metrics.get("tp", 0)), int(metrics.get("fp", 0)), int(metrics.get("tn", 0)), int(metrics.get("fn", 0))
                n_c0 = int(metrics.get("num_class_0", 0))
                n_c1 = int(metrics.get("num_class_1", 0))
                tot = int(metrics.get("total_samples", 0))

                acc, recall, precision, f1 = compute_metrics(tp, fp, tn, fn)
                ehd = compute_ehd(tp, fp, tn, fn, q0=self.q0, q1=self.q1)

                self.hd_history.setdefault(cid, deque(maxlen=self.MA_WINDOW))
                self.suspicion_count.setdefault(cid, 0)
                rtd, status = 0.0, "OK"

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
                            res = fl.common.FitRes(parameters=self.good_model_buffer[cid], num_examples=res.num_examples, metrics=res.metrics, status=res.status)
                        mitigated_results.append((client, res))

                if rnd == ATTACK_ROUND - 1:
                    self.round_9_snapshot[cid] = {"acc": acc, "ehd": ehd, "c0": n_c0, "c1": n_c1, "total": tot}
                elif rnd == ATTACK_ROUND:
                    r9 = self.round_9_snapshot.get(cid, {"acc": 0.0, "ehd": 0.0, "c0": n_c0, "c1": n_c1, "total": tot})
                    attack_pct = (n_c1 / tot_all) * 100 if FLIP_TYPE == "1->0" else (n_c0 / tot_all) * 100

                    delta_acc = r9["acc"] - acc
                    delta_ehd = ehd - r9["ehd"]
                    
                    paper_summary = (
                        f"PAPER_ROW -> Client={formatted_cid} "
                        f"Tot={tot} C0={n_c0} C1={n_c1} AtkPct={attack_pct:.2f} "
                        f"AccBf={r9['acc']:.2f} AccAf={acc:.2f} dAcc={delta_acc:.2f} "
                        f"EhdBf={r9['ehd']:.2f} EhdAf={ehd:.2f} dEhd={delta_ehd:.2f} "
                        f"RecallAf={recall:.2f}"
                    )
                    f.write(paper_summary + "\n")

                msg = f"[{formatted_cid}] Acc={acc:.4f} EHD={ehd:.6f} RTD={rtd:.6f} Suspicion={self.suspicion_count[cid]} Status={status}"
                print(msg)
                f.write(msg + "\n")

        return super().aggregate_fit(rnd, mitigated_results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        return super().aggregate_evaluate(rnd, [], failures)


# ===================================================================
# CENTRALIZED SERVER EVALUATION HOOK (Replicating server.py)
# ===================================================================
def get_server_evaluate_fn(master_csv_path, input_size):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    class NeuralNetwork(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(size, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
        def forward(self, x): return self.net(x).squeeze(1)

    df = pd.read_csv(master_csv_path)
    X = df.drop(columns=["cardio"]).values.astype(np.float32)
    y = df["cardio"].values.astype(np.float32)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SCALER_DIR = os.path.join(BASE_DIR, "shared")
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(SCALER_DIR, "scaler_mean.npy"))
    scaler.scale_ = np.load(os.path.join(SCALER_DIR, "scaler_scale.npy"))
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]

    X_scaled = scaler.transform(X)
    testloader = DataLoader(TensorDataset(torch.tensor(X_scaled), torch.tensor(y)), batch_size=32, shuffle=False)
    
    model = NeuralNetwork(input_size).to(DEVICE)

    def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: dict):
        state_dict = model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        tp = fp = tn = fn = 0
        with torch.no_grad():
            for X_b, y_b in testloader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                pred = (torch.sigmoid(model(X_b)) > 0.5).float()
                tp += int(((pred == 1) & (y_b == 1)).sum())
                fp += int(((pred == 1) & (y_b == 0)).sum())
                tn += int(((pred == 0) & (y_b == 0)).sum())
                fn += int(((pred == 0) & (y_b == 1)).sum())

        acc, recall, precision, f1 = compute_metrics(tp, fp, tn, fn)
        
        msg = f"GLOBAL_EVAL -> Round={server_round} Acc={acc:.4f} Recall={recall:.4f} Precision={precision:.4f} F1={f1:.4f}"
        print(f"\n🌍 {msg}")
        with open(SERVER_LOG, "a") as f:
            f.write(msg + "\n")

        return 1.0 - acc, {"accuracy": acc, "recall": recall, "precision": precision, "f1": f1}
    return evaluate_fn


# ===================================================================
# CLIENT WORKER FACTORY (Updated for Global Baseline Evaluation)
# ===================================================================
def get_client_fn(csv_dir, input_size):
    def client_fn(cid: str) -> fl.client.Client:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        import numpy as np

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        class NeuralNetwork(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(size, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
            def forward(self, x): return self.net(x).squeeze(1)

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
            return tp, fp, tn, fn

        def flip_labels(loader, flip_type):
            X_all, y_all = [], []
            for X, y in loader:
                X_all.append(X)
                y_all.append(y)
            X_all = torch.cat(X_all)
            y_all = torch.cat(y_all)

            if flip_type == "1->0": y_all[y_all == 1] = 0
            if flip_type == "0->1": y_all[y_all == 0] = 1

            return DataLoader(TensorDataset(X_all, y_all), batch_size=32, shuffle=False, drop_last=False)

        class CardioClient(fl.client.NumPyClient):
            def __init__(self, cid, trainloader, testloader, input_size, num_class_0, num_class_1, total_samples):
                self.cid = cid
                self.trainloader = trainloader
                self.testloader = testloader  # Now holds the Global Master testset
                self.model = NeuralNetwork(input_size).to(DEVICE)
                self.loss_fn = nn.BCEWithLogitsLoss()
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                self.current_round = 0
                
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
                if self.current_round >= ATTACK_ROUND and str(self.cid) == str(ATTACKING_CLIENT):
                    print(f"[C{int(self.cid)+1}] ATTACK ACTIVE at round {self.current_round}: flipping all {FLIP_TYPE}")
                    train_loader = flip_labels(self.trainloader, FLIP_TYPE)

                for X, y in train_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(self.model(X), y)
                    loss.backward()
                    self.optimizer.step()

                # 📊 Evaluates local model strictly against the balanced Global Testloader
                tp, fp, tn, fn = evaluate_model(self.model, self.testloader)

                return (
                    self.get_parameters(),
                    len(self.trainloader.dataset),
                    {
                        "client_id": self.cid,
                        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                        "num_class_0": self.num_class_0,
                        "num_class_1": self.num_class_1,
                        "total_samples": self.total_samples
                    }
                )

            def evaluate(self, parameters, config):
                return 0.0, 0, {}

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        SCALER_DIR = os.path.join(BASE_DIR, "shared")
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(SCALER_DIR, "scaler_mean.npy"))
        scaler.scale_ = np.load(os.path.join(SCALER_DIR, "scaler_scale.npy"))
        scaler.var_ = scaler.scale_ ** 2

        # 1. Load Local Training Data (Retains Non-IID Skew)
        partition_map = {"0": "client_1.csv", "1": "client_2.csv", "2": "client_3.csv"}
        csv_target = partition_map.get(cid, f"client_{int(cid)+1}.csv")
        df_train = pd.read_csv(os.path.join(csv_dir, "3_nodes", csv_target))
        
        class_counts = df_train["cardio"].value_counts().to_dict()
        num_class_0 = int(class_counts.get(0.0, 0))
        num_class_1 = int(class_counts.get(1.0, 0))
        total_samples = num_class_0 + num_class_1
        
        X_train = df_train.drop(columns=["cardio"]).values.astype(np.float32)
        y_train = df_train["cardio"].values.astype(np.float32)
        X_train_scaled = scaler.transform(X_train)
        trainloader = DataLoader(TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train)), batch_size=32, shuffle=False)

        # 2. Load Global Master Dataset for Evaluation (Forces 0.5 baseline and 0.6 downward deviation)
        df_test = pd.read_csv(os.path.join(csv_dir, "testset.csv"))
        X_test = df_test.drop(columns=["cardio"]).values.astype(np.float32)
        y_test = df_test["cardio"].values.astype(np.float32)
        X_test_scaled = scaler.transform(X_test)
        testloader = DataLoader(TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test)), batch_size=32, shuffle=False)

        return CardioClient(cid, trainloader, testloader, input_size, num_class_0, num_class_1, total_samples).to_client()
    return client_fn

# ===================================================================
# GOOGLE SHEETS PARSER
# ===================================================================
def export_logs_to_sheets_server_eval(log_file_path, spreadsheet_title, client_start_cell, attacking_client):
    print(f"\n📊 Exporting Server-Side Evaluation Matrix to Sheet '{spreadsheet_title}' at {client_start_cell}...")
    
    paper_rows = []
    g_bf = g_af = g_mit = 0.0

    with open(log_file_path, "r") as file:
        for line in file:
            line = line.strip()
            
            if "GLOBAL_EVAL" in line:
                r_match = re.search(r"Round=(?P<rnd>\d+)\s+Acc=(?P<acc>[\d\.]+)", line)
                if r_match:
                    rnd = int(r_match.group("rnd"))
                    acc = float(r_match.group("acc"))
                    if rnd == ATTACK_ROUND - 1: g_bf = acc
                    elif rnd == ATTACK_ROUND: g_af = acc
                    elif rnd == ATTACK_ROUND + 1: g_mit = acc

            elif "PAPER_ROW" in line:
                attacking_label = f"C{int(attacking_client) + 1}" if str(attacking_client).isdigit() else attacking_client
                if f"Client={attacking_label}" not in line: continue
                
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

    final_upload_matrix = []
    for item in sorted(paper_rows, key=lambda x: x["client"]):
        final_upload_matrix.append([
            f"0 ({item['tot']})", item["client"], item["c0"], item["c1"], item["atkp"], 
            item["abf"], item["aaf"], item["da"], item["ebf"], item["eaf"], item["de"], item["rec"], 
            g_bf, g_af, g_mit
        ])

    if not final_upload_matrix:
        print("⚠️ Warning: No rows matched the data logging syntax. Matrix empty.")
        return

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("sheets_api_credentials.json", scope)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open(spreadsheet_title)
        try:
            client_sheet = spreadsheet.worksheet("server_eval")
        except gspread.exceptions.WorksheetNotFound:
            client_sheet = spreadsheet.add_worksheet(title="server_eval", rows="100", cols="15")
        
        client_sheet.update(range_name=client_start_cell, values=final_upload_matrix)
        print("✅ Server Eval horizontal data matrix updated cleanly!")
    except Exception as e:
        print(f"❌ Error uploading to Google Sheets: {e}")

# ===================================================================
# MAIN EXECUTION THREAD
# ===================================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "custom_splits", "output_final", "server_evaluation", EHD)
    
    MASTER_CSV = os.path.join(DATA_DIR, "testset.csv")
    df_master = pd.read_csv(MASTER_CSV)
    master_counts = df_master["cardio"].value_counts().to_dict()
    total_master = len(df_master)
    global_q0 = master_counts.get(0.0, 0) / total_master
    global_q1 = master_counts.get(1.0, 0) / total_master

    INPUT_SIZE = df_master.shape[1] - 1 
    print(f"🎬 Initializing Server-Side VCE. Global baselines -> q0: {global_q0:.4f} | q1: {global_q1:.4f}")

    server_eval_fn = get_server_evaluate_fn(MASTER_CSV, INPUT_SIZE)

    strategy = FedAvgServerEvalDetection(
        q0=global_q0, q1=global_q1,
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=3, min_evaluate_clients=0, min_available_clients=3,
        evaluate_fn=server_eval_fn,
        on_fit_config_fn=lambda server_round: {"current_round": server_round}
    )

    fl.simulation.start_simulation(
        client_fn=get_client_fn(DATA_DIR, INPUT_SIZE),
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
        client_resources={"num_cpus": 2},
        ray_init_args={"include_dashboard": False, "local_mode": True} 
    )

    export_logs_to_sheets_server_eval(SERVER_LOG, "fl_test", SHEET_START_CELL, ATTACKING_CLIENT)