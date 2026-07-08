# Federated Learning: Label-Flipping Attack and Detection

A small simulation that shows how a federated learning system can detect and recover from a client sending bad (poisoned) updates.

Three clients train a model together on cardiovascular health data, using the [Flower](https://flower.ai/) framework. At a set round, one client flips its labels (a label-flipping attack). The server watches each client's error pattern and rolls back to a client's last known-good model if it looks like an attack.

## How it works

- Each client trains a small neural network on its own share of the data and sends updates to the server.
- The server tracks a distance metric (EHD) for each client's error rate over time.
- If a client's metric jumps past a threshold for a few rounds in a row, the server flags it and rolls that client back to its last trusted update instead of using the bad one.
- Results (accuracy, recall, before/after attack numbers) are logged to a text file and can be pushed to a Google Sheet.

## Files

| File | What it does |
|---|---|
| `simulation_attack.py` | Main script: defines the model, clients, server strategy, attack, and detection logic |
| `simulation_script.sh` | Shell script to run the simulation |
| `config.yaml` | Settings for the run: device, attack round, which client attacks, flip type, log file |
| `data/` | Dataset partitions used by each client |
| `shared/` | Saved scaler values used to normalize data the same way across clients |

## Setup

```bash
pip install flwr torch pandas numpy scikit-learn gspread oauth2client pyyaml
```

Edit `config.yaml` to set:
- `attacking_client` — which client (0, 1, or 2) sends bad updates
- `attack_round` — which round the attack starts
- `flip_type` — `"0->1"` or `"1->0"`, which label gets flipped

## Run

```bash
bash simulation_script.sh
```

or directly:

```bash
python simulation_attack.py
```

Each run uses 3 simulated clients and 20 training rounds. Logs are written to the file named in `server_log`.

## Google Sheets export (optional)

The script can push a summary row to Google Sheets after the run. This needs a service account credentials file named `sheets_api_credentials.json` in the project root. If you don't need this, you can ignore the export step or remove that part of the script.

## Notes

- Training is seeded for repeatable results.
- The dataset used is a cardiovascular disease dataset (target column: `cardio`).
- This is a research/experiment setup, not a production-ready system.