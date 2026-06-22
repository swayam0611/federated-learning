# Federated Learning with Label Flipping Attack Detection

A simulation framework for **federated learning** that models a **label-flipping poisoning attack** on one client and implements an **automated detection and mitigation strategy** using Hellinger Distance (EHD) tracking. Results are exported directly to Google Sheets for analysis.

---

## Overview

This project simulates a realistic federated learning scenario with 3 clients collaboratively training a binary classifier on cardiovascular disease data. One client is configured as a **malicious attacker** that flips training labels after a specified round, and the server employs a custom strategy to detect and roll back the attack.

**Key features:**
- Federated learning via the [Flower (flwr)](https://flower.ai/) framework
- Label-flipping poisoning attack simulation (supports `0→1` and `1→0` flips)
- Attack detection using Relative Temporal Deviation (RTD) of Empirical Hellinger Distance (EHD)
- Automatic rollback to a cached good model on attack detection
- Google Sheets integration for logging and analysis

---

## Project Structure

```
federated-learning/
├── simulation_attack.py      # Main simulation script (clients, server strategy, attack logic)
├── simulation_script.sh      # Shell script to run simulation experiments
├── config.yaml               # Experiment configuration
├── data/                     # Dataset partitions for each client
│   └── custom_splits/
│       └── output_final/
│           ├── partition0.csv
│           ├── partition1.csv
│           └── partition2.csv
└── shared/
    ├── scaler_mean.npy       # Shared StandardScaler mean (fitted on full training data)
    └── scaler_scale.npy      # Shared StandardScaler scale
```

---

## How It Works

### Federated Setup
- **3 virtual clients** are spawned using Flower's simulation engine.
- Each client trains a small feedforward neural network (`8→4→1`) on its local partition of cardiovascular data.
- A global model is aggregated by the server using a custom `FedAvg`-based strategy.

### Label Flipping Attack
- Starting at `attack_round` (set in `config.yaml`), the designated attacking client flips all labels in its training data before local training.
- Supported flip types: `"0->1"` (benign samples poisoned as malicious) or `"1->0"` (malicious samples silenced).

### Detection & Mitigation (`FedAvgAttackDetection`)
1. Each client reports confusion matrix metrics (`tp, fp, tn, fn`) to the server after every local training round.
2. The server computes the **Empirical Hellinger Distance (EHD)** per client — a measure of how far the client's prediction distribution deviates from a uniform baseline.
3. A **moving average (MA)** of EHD over the last 3 rounds is tracked per client.
4. The **Relative Temporal Deviation (RTD)** — the gap between current EHD and the moving average — triggers suspicion if it exceeds a threshold (`0.05`).
5. A client is flagged as an attacker after accumulating suspicion across `PROBATION = 2` consecutive rounds.
6. On confirmed attack, the server **rolls back that client's model** to the last known good snapshot stored in `good_model_buffer`.

### Google Sheets Export
After the simulation, `export_logs_to_sheets()` parses the server log and uploads a 15-column metrics matrix to a Google Sheet (`Client_Metrics` tab), capturing per-client accuracy, EHD, and delta values before and after the attack, alongside post-mitigation global accuracy.

---

## Configuration (`config.yaml`)

```yaml
device: "cpu"           # Compute device ("cpu" or "cuda")
attack_round: 10        # Round at which the attacker begins flipping labels
attacking_client: "2"   # Client ID of the malicious client (0-indexed)
flip_type: "0->1"       # Label flip direction: "0->1" or "1->0"
server_log: "server_metrics_c2_0.txt"   # Path for server-side log file
sheet_start_cell: "A7"  # Starting cell in Google Sheets for data upload
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- A Google Cloud service account with Sheets and Drive API enabled (for results export)

### Install dependencies

```bash
pip install flwr torch pandas numpy scikit-learn gspread oauth2client pyyaml
```

### Google Sheets credentials
Place your service account credentials file as `sheets_api_credentials.json` in the project root. The target spreadsheet must be shared with your service account email.

---

## Running the Simulation

```bash
python simulation_attack.py
```

Or use the provided shell script to run batch experiments:

```bash
bash simulation_script.sh
```

The simulation runs for **20 federated rounds**. Attack detection kicks in from round 5 onwards (warmup period). Results are written to the server log file and uploaded to Google Sheets at the end.

---

## Model Architecture

A lightweight feedforward neural network for binary classification:

```
Input → Linear(input_size, 8) → ReLU → Linear(8, 4) → ReLU → Linear(4, 1)
```

- Optimizer: Adam (lr=0.001)
- Loss: BCEWithLogitsLoss
- Batch size: 32

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

---

## Notes

- All experiments are fully deterministic (fixed `SEED`, `torch.use_deterministic_algorithms(True)`).
- The scaler (`shared/`) must be fitted on the full dataset before partitioning and shared across all clients to ensure consistent feature scaling.
- Ray is configured in `local_mode=True` for single-process sequential execution — useful for reproducibility and debugging.