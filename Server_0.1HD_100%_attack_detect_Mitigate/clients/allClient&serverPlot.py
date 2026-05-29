import matplotlib.pyplot as plt
import re
import os

# =====================================================
# PATH CONFIGURATION (UNCHANGED)
# =====================================================
CLIENT_BASE_DIR = (
    "/home/coep/Desktop/HD01_final/HD01_harmful/Server_0.1HD_100%_attack_detect_Mitigate/server"
)

CLIENT_FILES = [
    os.path.join(CLIENT_BASE_DIR, "c1.txt"),
    os.path.join(CLIENT_BASE_DIR, "c2.txt"),
    os.path.join(CLIENT_BASE_DIR, "c3.txt"),
]

CLIENT_NAMES = ["Client 1", "Client 2", "Client 3"]

SERVER_FILE = (
    "/home/coep/Desktop/HD01_final/HD01_harmful/Server_0.1HD_100%_attack_detect_Mitigate/server/server_metrics.txt"
)

SERVER_NAME = "FL Aggregator"

# =====================================================
# 🎯 TARGET ROUNDS
# =====================================================
TARGET_ROUNDS = [9, 10]

# =====================================================
# 🔥 ATTACK CONFIG (UNCHANGED)
# =====================================================
ATTACK_CLIENT = "Client 1"
ATTACK_ROUND = 10
ATTACK_TYPE = "1→0 Flip Attack"

# =====================================================
# REGEX PATTERNS
# =====================================================
ROUND_RE = re.compile(r"ROUND\s+(\d+)")
ACC_RE   = re.compile(r"Acc=([\d.]+)")
REC_RE   = re.compile(r"Recall=([\d.]+)")
HD_RE    = re.compile(r"HD=([\d.]+)")
PREC_RE  = re.compile(r"Precision=([\d.]+)")
F1_RE    = re.compile(r"F1=([\d.]+)")

# =====================================================
# GLOBAL STYLE (UNCHANGED)
# =====================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12
})

# =====================================================
# PARSE FUNCTIONS (UNCHANGED)
# =====================================================
def parse_client_file(filepath):
    rounds, acc, ehd = [], [], []

    with open(filepath, "r") as f:
        for line in f:
            r = ROUND_RE.search(line)
            a = ACC_RE.search(line)
            h = HD_RE.search(line)

            if r and a and h:
                rounds.append(int(r.group(1)))
                acc.append(float(a.group(1)))
                ehd.append(float(h.group(1)))

    return rounds, acc, ehd


def parse_server_file(filepath):
    rounds, acc, precision, recall, f1 = [], [], [], [], []

    with open(filepath, "r") as f:
        for line in f:
            r = ROUND_RE.search(line)
            a = ACC_RE.search(line)
            p = PREC_RE.search(line)
            rc = REC_RE.search(line)
            f1m = F1_RE.search(line)

            if r and a and p and rc and f1m:
                rounds.append(int(r.group(1)))
                acc.append(float(a.group(1)))
                precision.append(float(p.group(1)))
                recall.append(float(rc.group(1)))
                f1.append(float(f1m.group(1)))

    return rounds, acc, precision, recall, f1

# =====================================================
# CLIENT PLOTS
# =====================================================
for file_path, client_name in zip(CLIENT_FILES, CLIENT_NAMES):

    if not os.path.exists(file_path):
        print(f"⚠️ Missing {file_path}, skipping...")
        continue

    rounds, acc, ehd = parse_client_file(file_path)

    plt.figure(figsize=(15, 8))

    if rounds:
        plt.plot(rounds, acc, marker='o', linewidth=2, label="Accuracy")
        plt.plot(rounds, ehd, marker='^', linestyle='--', linewidth=2, label="EHD")

        # 🔥 ATTACK MARKER (UNCHANGED)
        if client_name == ATTACK_CLIENT:
            plt.axvline(x=ATTACK_ROUND, linestyle=':', linewidth=2, label="Attack Point")

            plt.text(
                ATTACK_ROUND + 0.2,
                0.9,
                ATTACK_TYPE,
                rotation=90,
                verticalalignment='center',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )

        # 🔥 SMART NON-OVERLAPPING LABELS (ONLY CHANGE)
        for x, y_acc, y_ehd in zip(rounds, acc, ehd):
            if x in TARGET_ROUNDS:

                if abs(y_acc - y_ehd) < 0.05:
                    acc_offset = -0.05
                    ehd_offset = +0.05
                else:
                    acc_offset = +0.03
                    ehd_offset = -0.05

                plt.text(
                    x, y_acc + acc_offset,
                    f"{y_acc:.2f}",
                    ha='center',
                    fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )

                plt.text(
                    x, y_ehd + ehd_offset,
                    f"{y_ehd:.2f}",
                    ha='center',
                    fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )

    else:
        plt.text(0.5, 0.5, "No Data Available", ha='center', va='center')

    plt.title(client_name)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Metric Value")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig(f"{client_name.replace(' ', '')}_values.png", dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================
# SERVER PLOT (UNCHANGED)
# =====================================================
if os.path.exists(SERVER_FILE):

    rounds, acc, precision, recall, f1 = parse_server_file(SERVER_FILE)

    plt.figure(figsize=(15, 8))

    if rounds:
        plt.plot(rounds, acc, marker='o', linewidth=2, label="Accuracy")
        plt.plot(rounds, precision, marker='^', linewidth=2, label="Precision")
        plt.plot(rounds, recall, marker='s', linewidth=2, label="Recall")
        plt.plot(rounds, f1, marker='d', linewidth=2, label="F1-score")

        for x, y in zip(rounds, acc):
            if x in TARGET_ROUNDS:
                plt.text(x + 0.1, y + 0.03, f"{y:.2f}", ha='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        for x, y in zip(rounds, precision):
            if x in TARGET_ROUNDS:
                plt.text(x + 0.2, y + 0.06, f"{y:.2f}", ha='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        for x, y in zip(rounds, recall):
            if x in TARGET_ROUNDS:
                plt.text(x - 0.1, y - 0.07, f"{y:.2f}", ha='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        for x, y in zip(rounds, f1):
            if x in TARGET_ROUNDS:
                plt.text(x - 0.2, y - 0.10, f"{y:.2f}", ha='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    else:
        plt.text(0.5, 0.5, "No Data Available", ha='center', va='center')

    plt.title(SERVER_NAME)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Metric Value")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig("Server_values.png", dpi=300, bbox_inches='tight')
    plt.close()

else:
    print(f"⚠️ Server file not found: {SERVER_FILE}")