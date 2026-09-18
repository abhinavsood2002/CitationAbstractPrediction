#!/usr/bin/env bash
# Background launcher: one model, all four arms, resumable.
#   bash scripts/launch_generation.sh <gpus> <model> <tp> <out-name> [extra generate.py args]
# e.g. bash scripts/launch_generation.sh 0,1,2,3 meta-llama/Llama-3.3-70B-Instruct 4 llama-3.3-70b
#      bash scripts/launch_generation.sh 0 openai/gpt-oss-20b 1 gpt-oss-20b --max-seeds 200
set -euo pipefail
GPUS=$1; MODEL=$2; TP=$3; NAME=$4; shift 4
PY=${PY:-python}
ARMS=${ARMS:-proposal_t07,proposal_t10,proposal_t13,diverse_list_t10}
mkdir -p results/logs results/gen/$NAME
LOG=results/logs/gen_${NAME}.log
CUDA_VISIBLE_DEVICES=$GPUS nohup $PY scripts/generate.py --input data/acl_a2a.jsonl \
  --model $MODEL --tp $TP --arms $ARMS --output-dir results/gen/$NAME "$@" > $LOG 2>&1 &
echo "pid $! -> $LOG"
