#!/usr/bin/env bash
# The generation run, one model at a time: every GPU runs one tp-1 process of the current
# generator on its own shard of the seeds (all of that generator's conditions), and the next
# generator starts when all of them have exited. Resumable. Runs in the foreground:
#   nohup bash scripts/launch_generation.sh > results/logs/launch_generation.log 2>&1 &
#   bash scripts/launch_generation.sh --input data/<subset>.jsonl --output-dir results/<new-dir>/gen
# Resume is keyed on seed id only: nothing records the prompt or the sampling settings, so
# after changing either, write to a NEW --output-dir (an existing one keeps its old
# generations). For a pilot pass a hash-sampled --input file (test_benchmark/make_subset.py);
# --max-seeds takes the first seeds of the file, which are all early-year seeds.
# Extra arguments go to every scripts/generate.py call. GPUS picks the devices (default 0-7),
# GENERATORS the models (default "gemma gpt-oss"); logs: results/logs/gen_<generator>_shard_<i>$TAG.log.
set -uo pipefail
PY=${PY:-python}
IFS=, read -ra GPU <<< "${GPUS:-0,1,2,3,4,5,6,7}"
N=${#GPU[@]}
mkdir -p results/logs
for g in ${GENERATORS:-gemma gpt-oss}; do
  echo "$(date +%T) $g: $N shards"
  pids=()
  for i in "${!GPU[@]}"; do
    log=results/logs/gen_${g}_shard_${i}${TAG:-}.log
    CUDA_VISIBLE_DEVICES=${GPU[$i]} $PY scripts/generate.py --generator $g --shard $i/$N "$@" > $log 2>&1 &
    pids+=($!)
  done
  failed=0
  for p in "${pids[@]}"; do wait $p || failed=1; done
  if [ $failed -ne 0 ]; then echo "$(date +%T) $g: a shard failed, see results/logs/gen_${g}_shard_*${TAG:-}.log"; exit 1; fi
  echo "$(date +%T) $g: done"
done
