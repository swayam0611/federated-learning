#!/bin/bash

run_experiment() {
    local attack_round=$1
    local attacking_client=$2
    local flip_type=$3
    local server_log=$4
    local sheet_start_cell=$5
    local ehd=$6

    cat > config.yaml <<EOF
device: "cpu"
attack_round: $attack_round
attacking_client: "$attacking_client"
flip_type: "$flip_type"
server_log: "$server_log"
sheet_start_cell: "$sheet_start_cell"
ehd: "$ehd"
EOF

    echo "Running: attacking_client=$attacking_client flip_type=$flip_type ehd=$ehd"
    python simulation_attack.py
}

base_dir="../metrics/local_eval/"



run_experiment 10 "0" "1->0" "${base_dir}server_metrics_c0_1.txt" "A2" "0.0HD"
run_experiment 10 "0" "0->1" "${base_dir}server_metrics_c0_0.txt" "A3" "0.0HD"
run_experiment 10 "1" "1->0" "${base_dir}server_metrics_c1_1.txt" "A4" "0.0HD"
run_experiment 10 "1" "0->1" "${base_dir}server_metrics_c1_0.txt" "A5" "0.0HD"
run_experiment 10 "2" "1->0" "${base_dir}server_metrics_c2_1.txt" "A6" "0.0HD"
run_experiment 10 "2" "0->1" "${base_dir}server_metrics_c2_0.txt" "A7" "0.0HD"

run_experiment 10 "0" "1->0" "${base_dir}server_metrics_c0_1.txt" "A9" "0.1HD"
run_experiment 10 "0" "0->1" "${base_dir}server_metrics_c0_0.txt" "A10" "0.1HD"
run_experiment 10 "1" "1->0" "${base_dir}server_metrics_c1_1.txt" "A11" "0.1HD"
run_experiment 10 "1" "0->1" "${base_dir}server_metrics_c1_0.txt" "A12" "0.1HD"
run_experiment 10 "2" "1->0" "${base_dir}server_metrics_c2_1.txt" "A13" "0.1HD"
run_experiment 10 "2" "0->1" "${base_dir}server_metrics_c2_0.txt" "A14" "0.1HD"

run_experiment 10 "0" "1->0" "${base_dir}server_metrics_c0_1.txt" "A16" "0.2HD"
run_experiment 10 "0" "0->1" "${base_dir}server_metrics_c0_0.txt" "A17" "0.2HD"
run_experiment 10 "1" "1->0" "${base_dir}server_metrics_c1_1.txt" "A18" "0.2HD"
run_experiment 10 "1" "0->1" "${base_dir}server_metrics_c1_0.txt" "A19" "0.2HD"
run_experiment 10 "2" "1->0" "${base_dir}server_metrics_c2_1.txt" "A20" "0.2HD"
run_experiment 10 "2" "0->1" "${base_dir}server_metrics_c2_0.txt" "A21" "0.2HD"

run_experiment 10 "0" "1->0" "${base_dir}server_metrics_c0_1.txt" "A23" "0.3HD"
run_experiment 10 "0" "0->1" "${base_dir}server_metrics_c0_0.txt" "A24" "0.3HD"
run_experiment 10 "1" "1->0" "${base_dir}server_metrics_c1_1.txt" "A25" "0.3HD"
run_experiment 10 "1" "0->1" "${base_dir}server_metrics_c1_0.txt" "A26" "0.3HD"
run_experiment 10 "2" "1->0" "${base_dir}server_metrics_c2_1.txt" "A27" "0.3HD"
run_experiment 10 "2" "0->1" "${base_dir}server_metrics_c2_0.txt" "A28" "0.3HD"
