#!/usr/bin/env bash
# Background launcher: one model, one split, all four arms, resumable.
#   bash scripts/launch_generation.sh <gpus> <model> <tp> <split> <out-name> [extra args]
# e.g. bash scripts/launch_generation.sh 0,1,2,3 meta-llama/Llama-3.3-70B-Instruct 4 dev llama-3.3-70b
set -euo pipefail
GPUS=$1; MODEL=$2; TP=$3; SPLIT=$4; NAME=$5; shift 5
PY=${PY:-python}
ARMS=${ARMS:-proposal_t07,proposal_t10,proposal_t13,diverse_list_t10}
mkdir -p results/logs results/gen/$SPLIT/$NAME
LOG=results/logs/gen_${SPLIT}_${NAME}.log
CUDA_VISIBLE_DEVICES=$GPUS nohup $PY scripts/generate.py --input data/acl_a2a.jsonl --split $SPLIT \
  --model $MODEL --tp $TP --arms $ARMS --output-dir results/gen/$SPLIT/$NAME "$@" > $LOG 2>&1 &
echo "pid $! -> $LOG"
