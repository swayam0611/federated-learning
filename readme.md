# Federated Learning with Label Flipping Attack Detection

A simulation framework for **federated learning** that models a **label-flipping poisoning attack** on one client and implements an **automated detection and mitigation strategy** using Empirical Hellinger Distance (EHD) tracking. Results are exported directly to Google Sheets for analysis.

---

## Overview

This project simulates a realistic federated learning scenario with 3 clients collaboratively training a binary classifier on cardiovascular disease data. One client is configured as a **malicious attacker** that flips training labels after a specified round, and the server employs a custom strategy to detect and roll back the attack.

The codebase is organized into two evaluation modes:

- **`src_local_eval/`** — model evaluation is performed locally on each client and metrics are reported back to the server
- **`src_server_eval/`** — model evaluation is centralized and performed on the server side after global aggregation

**Key features:**
- Federated learning via the [Flower (flwr)](https://flower.ai/) framework
- Label-flipping poisoning attack simulation (supports `0→1` and `1→0` flips)
- Attack detection using Relative Temporal Deviation (RTD) of Empirical Hellinger Distance (EHD)
- Automatic rollback to a cached good model on attack detection
- Pre-saved metrics output in the `metrics/` directory
- Google Sheets integration for logging and analysis

---

## Project Structure

```
federated-learning/
├── src_local_eval/           # Simulation with client-side (local) evaluation
├── src_server_eval/          # Simulation with server-side (centralized) evaluation
├── metrics/                  # Pre-saved experiment metric outputs
├── data/                     # Dataset partitions for each client
│   └── custom_splits/
│       └── output_final/
│           ├── partition0.csv
│           ├── partition1.csv
│           └── partition2.csv
├── shared/
│   ├── scaler_mean.npy       # Shared StandardScaler mean (fitted on full training data)
│   └── scaler_scale.npy      # Shared StandardScaler scale
├── requirements.txt          # Pinned Python dependencies
└── readme.md                 # This file
```

---

## How It Works

### Federated Setup
- **3 virtual clients** are spawned using Flower's simulation engine (via Ray in local mode).
- Each client trains a small feedforward neural network on its local partition of cardiovascular data.
- The global model is aggregated by the server using a custom `FedAvg`-based strategy (`FedAvgAttackDetection`).

### Label Flipping Attack
- Starting at a configurable `attack_round`, the designated attacking client flips all labels in its training data before local training.
- Supported flip types: `"0->1"` (benign samples poisoned as malicious) or `"1->0"` (malicious samples silenced).

### Detection & Mitigation (`FedAvgAttackDetection`)

1. Each client reports confusion matrix values (`tp, fp, tn, fn`) to the server after every training round.
2. The server computes the **Empirical Hellinger Distance (EHD)** per client — a measure of how far the client's prediction distribution deviates from a uniform baseline.
3. A **moving average (MA)** of EHD over the last 3 rounds is tracked per client.
4. The **Relative Temporal Deviation (RTD)** — the absolute gap between the current EHD and its moving average — triggers suspicion when it exceeds a threshold of `0.05`.
5. A client is confirmed as an attacker after accumulating suspicion over a probation window of 2 consecutive flagged rounds.
6. On confirmed attack, the server **rolls back that client's model** to the last known good snapshot cached in `good_model_buffer`.

### Evaluation Modes

| Mode | Location | How metrics are collected |
|---|---|---|
| Local Eval | `src_local_eval/` | Each client evaluates its own model on its local test set |
| Server Eval | `src_server_eval/` | The server evaluates the aggregated global model centrally |

### Google Sheets Export
After the simulation, an export function parses the server log and uploads a 15-column metrics matrix to a Google Sheet (`Client_Metrics` tab), capturing per-client accuracy, EHD, and delta values before and after the attack, alongside post-mitigation global accuracy.

---

## Model Architecture

A lightweight feedforward neural network for binary classification:

```
Input → Linear(input_size, 8) → ReLU → Linear(8, 4) → ReLU → Linear(4, 1)
```

- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam` (lr = 0.001)
- Batch size: 32

---

## Installation

### Prerequisites
- Python 3.8+
- A Google Cloud service account with Sheets and Drive APIs enabled (for results export)

### Install dependencies

```bash
pip install -r requirements.txt
```

Pinned versions (from `requirements.txt`):

```
flwr==1.30.0
ray==2.51.1
torch==2.12.0
pandas==3.0.3
numpy==2.4.6
scikit-learn==1.9.0
gspread==6.2.1
oauth2client==4.1.3
PyYAML==6.0.3
```

### Google Sheets credentials
Place your service account credentials file as `sheets_api_credentials.json` in the project root. The target Google Sheet must be shared with your service account email.

---

## Running the Simulation

Choose one of the two evaluation modes and run the simulation script inside that folder:

```bash
# Client-side evaluation
cd src_local_eval
python simulation_attack.py

# Server-side evaluation
cd src_server_eval
python simulation_attack.py
```

Or use the shell script to run batch experiments across configurations:

```bash
bash simulation_script.sh
```

The simulation runs for **20 federated rounds**. Attack detection activates after a warmup period (round 5 onwards). Results are logged to a server metrics file and uploaded to Google Sheets at the end.

---

## Metrics Tracked

| Metric | Description |
|---|---|
| Accuracy | Overall prediction accuracy |
| Recall | True positive rate |
| Precision | Positive predictive value |
| F1 Score | Harmonic mean of precision and recall |
| EHD | Empirical Hellinger Distance from a uniform distribution |
| RTD | Relative Temporal Deviation — used to flag suspicious rounds |
| Delta Acc | Change in accuracy from pre-attack to attack round |
| Delta EHD | Change in EHD from pre-attack to attack round |

Pre-saved metric outputs from experiments are stored in the `metrics/` directory for reference and comparison.

---

## Notes

- All experiments are fully deterministic (fixed random seed, `torch.use_deterministic_algorithms(True)`).
- The StandardScaler in `shared/` must be fitted on the full dataset before partitioning and is shared across all clients to ensure consistent feature scaling.
- Ray is configured in `local_mode=True` for single-process sequential execution — ideal for reproducibility and debugging.
- The `src_local_eval/` and `src_server_eval/` variants let you compare how the choice of evaluation location affects detection sensitivity and overall accuracy reporting.