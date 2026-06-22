#!/bin/bash

run_experiment() {
    local attack_round=$1
    local attacking_client=$2
    local flip_type=$3
    local server_log=$4
    local sheet_start_cell=$5

    cat > config.yaml <<EOF
device: "cpu"
attack_round: $attack_round
attacking_client: "$attacking_client"
flip_type: "$flip_type"
server_log: "$server_log"
sheet_start_cell: "$sheet_start_cell"
EOF

    echo "Running: attacking_client=$attacking_client flip_type=$flip_type"
    python simulation_attack.py
}

run_experiment 10 "0" "1->0" "server_metrics_c0_1.txt" "A2"
run_experiment 10 "0" "0->1" "server_metrics_c0_0.txt" "A3"
run_experiment 10 "1" "1->0" "server_metrics_c1_1.txt" "A4"
run_experiment 10 "1" "0->1" "server_metrics_c1_0.txt" "A5"
run_experiment 10 "2" "1->0" "server_metrics_c2_1.txt" "A6"
run_experiment 10 "2" "0->1" "server_metrics_c2_0.txt" "A7"
