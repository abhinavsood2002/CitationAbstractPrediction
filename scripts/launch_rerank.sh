#!/usr/bin/env bash
# The reranking run: one reranker process per GPU, each scoring its shard of the seeds; exits
# when all of them have. Resumable (finished units are skipped). Runs in the foreground:
#   nohup bash scripts/launch_rerank.sh > results/logs/launch_rerank.log 2>&1 &
#   bash scripts/launch_rerank.sh --data data/<subset>.jsonl --gen-dir results/<dir>/gen --out-dir results/<dir>/rerank
# A unit is skipped when its file exists, whatever generations or instruction produced it:
# delete <out-dir>/<condition>/ whenever that condition is regenerated, and the whole
# <out-dir> when llm.RERANK_INSTRUCTION changes. Start only after every condition has all
# of its seeds (the seed set, the sharding and the shuffled targets depend on it).
# Extra arguments go to every scripts/rerank.py call. GPUS picks the devices (default 0-7);
# logs: results/logs/rerank_shard_<i>$TAG.log.
set -uo pipefail
PY=${PY:-python}
IFS=, read -ra GPU <<< "${GPUS:-0,1,2,3,4,5,6,7}"
N=${#GPU[@]}
mkdir -p results/logs
echo "$(date +%T) rerank: $N shards"
pids=()
for i in "${!GPU[@]}"; do
  log=results/logs/rerank_shard_${i}${TAG:-}.log
  CUDA_VISIBLE_DEVICES=${GPU[$i]} $PY scripts/rerank.py --shard $i/$N "$@" > $log 2>&1 &
  pids+=($!)
done
failed=0
for p in "${pids[@]}"; do wait $p || failed=1; done
if [ $failed -ne 0 ]; then echo "$(date +%T) rerank: a shard failed, see results/logs/rerank_shard_*${TAG:-}.log"; exit 1; fi
echo "$(date +%T) rerank: done"
