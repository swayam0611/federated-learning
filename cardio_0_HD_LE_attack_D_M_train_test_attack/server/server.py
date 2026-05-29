import flwr as fl
import numpy as np
import math
import os
import random
from collections import deque

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

SERVER_LOG = "server_metrics.txt"

if os.path.exists(SERVER_LOG):
    os.remove(SERVER_LOG)

# ==========================================
# EHD
# ==========================================

def compute_ehd(tp, fp, tn, fn, q0, q1):
    total = tp + fp + tn + fn
    if total == 0:
        return 0.0
    p0 = (tn + fn) / total
    p1 = (tp + fp) / total
    return (1 / math.sqrt(2)) * math.sqrt(
        (math.sqrt(p0) - math.sqrt(q0)) ** 2 +
        (math.sqrt(p1) - math.sqrt(q1)) ** 2
    )

# ==========================================
# METRICS
# ==========================================

def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    if total == 0:
        return 0, 0, 0, 0
    acc = (tp + tn) / total
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return acc, recall, precision, f1

# ==========================================
# STRATEGY
# ==========================================

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

    # =====================================
    # BEFORE AGGREGATION
    # =====================================

    def aggregate_fit(self, rnd, results, failures):
        print(f"\nROUND {rnd}")
        print("\nBEFORE AGGREGATION")

        mitigated_results = []

        with open(SERVER_LOG, "a") as f:
            f.write(f"\nROUND {rnd}\n")
            f.write("BEFORE AGGREGATION\n")

            for client, res in results:
                metrics = res.metrics
                cid = metrics.get("client_id", "NA")
                tp = int(metrics.get("tp", 0))
                fp = int(metrics.get("fp", 0))
                tn = int(metrics.get("tn", 0))
                fn = int(metrics.get("fn", 0))
                q0 = float(metrics.get("q0", 0))
                q1 = float(metrics.get("q1", 0))

                acc, recall, precision, f1 = compute_metrics(tp, fp, tn, fn)
                ehd = compute_ehd(tp, fp, tn, fn, q0=0.5, q1=0.5)

                self.hd_history.setdefault(cid, deque(maxlen=self.MA_WINDOW))
                self.suspicion_count.setdefault(cid, 0)

                rtd = 0.0
                status = "OK"
                flip_type = "None"

                # ----------------------------------
                # WARMUP
                # ----------------------------------
                if rnd <= self.START_DETECTION_ROUND or len(self.hd_history[cid]) < self.MA_WINDOW:
                    self.hd_history[cid].append(ehd)
                    self.good_model_buffer[cid] = res.parameters
                    mitigated_results.append((client, res))
                    status = "WARMUP"

                else:
                    # ----------------------------------
                    # DETECTION
                    # ----------------------------------
                    ma_ehd = np.mean(list(self.hd_history[cid]))
                    rtd = abs(ehd - ma_ehd)

                    if rtd > self.RTD_THRESHOLD:
                        self.suspicion_count[cid] += 1

                        if self.suspicion_count[cid] >= self.PROBATION:
                            # MITIGATION
                            status = "ATTACK DETECTED - SWAPPED WITH BUFFER"
                            flip_type = "Attack"
                            res = fl.common.FitRes(
                                parameters=self.good_model_buffer[cid],
                                num_examples=res.num_examples,
                                metrics=res.metrics,
                                status=res.status
                            )
                        else:
                            status = "SUSPICIOUS - OBSERVE"

                        mitigated_results.append((client, res))

                    else:
                        # ✅ Only forgive/reset if NOT a confirmed attacker
                        if self.suspicion_count[cid] < self.PROBATION:
                            self.suspicion_count[cid] = 0
                            self.hd_history[cid].append(ehd)
                            self.good_model_buffer[cid] = res.parameters
                            status = "OK"
                        else:
                            # Confirmed attacker looks clean — still swap
                            status = "ATTACK DETECTED - SWAPPED WITH BUFFER"
                            flip_type = "Attack"
                            res = fl.common.FitRes(
                                parameters=self.good_model_buffer[cid],
                                num_examples=res.num_examples,
                                metrics=res.metrics,
                                status=res.status
                            )

                        mitigated_results.append((client, res))

                msg = (
                    f"[{cid}] "
                    f"Acc={acc:.4f} "
                    f"Recall={recall:.4f} "
                    f"Precision={precision:.4f} "
                    f"F1={f1:.4f} "
                    f"EHD={ehd:.6f} "
                    f"RTD={rtd:.6f} "
                    f"Suspicion={self.suspicion_count.get(cid, 0)} "
                    f"Status={status} "
                    f"Flip={flip_type}"
                )
                print(msg)
                f.write(msg + "\n")

        aggregated = super().aggregate_fit(rnd, mitigated_results, failures)
        print("\nFedAvg Aggregation Complete")
        return aggregated

    # =====================================
    # AFTER AGGREGATION
    # =====================================

    def aggregate_evaluate(self, rnd, results, failures):
        print("\nAFTER AGGREGATION")

        global_tp = global_fp = global_tn = global_fn = 0

        for _, res in results:
            metrics = res.metrics
            global_tp += int(metrics.get("tp", 0))
            global_fp += int(metrics.get("fp", 0))
            global_tn += int(metrics.get("tn", 0))
            global_fn += int(metrics.get("fn", 0))

        global_acc, global_recall, global_precision, global_f1 = compute_metrics(
            global_tp, global_fp, global_tn, global_fn
        )

        msg = (
            f"GLOBAL "
            f"TP={global_tp} "
            f"TN={global_tn} "
            f"FP={global_fp} "
            f"FN={global_fn} "
            f"Acc={global_acc:.4f} "
            f"Recall={global_recall:.4f} "
            f"Precision={global_precision:.4f} "
            f"F1={global_f1:.4f}"
        )
        print(msg)

        with open(SERVER_LOG, "a") as f:
            f.write("\nAFTER AGGREGATION\n")
            f.write(msg + "\n")

        return super().aggregate_evaluate(rnd, results, failures)

# ==========================================
# SERVER START
# ==========================================

strategy = FedAvgAttackDetection(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3
)

fl.server.start_server(
    server_address="127.0.0.1:8081",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy
)