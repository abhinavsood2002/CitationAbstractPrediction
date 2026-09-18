#!/usr/bin/env bash
# End-to-end pipeline. Steps 0-1 are one-off (hours, ~400 GB); 2-3 take minutes;
# 4 characterises the data (1 GPU, minutes); 5 generates (GPUs, hours);
# 6-7 score and contrast (1 GPU, minutes).
set -euo pipefail
PY=${PY:-python}
DERIVED=${A2A_DERIVED_ROOT:-/data/derived}
RAW=${A2A_RAW_ROOT:-/data/s2}
DATA=data/acl_a2a.jsonl

step() { echo; echo "== $*"; }

if [[ "${1:-}" == "download" ]]; then
  step "0. download S2 datasets (needs S2_API_KEY)"
  for d in papers abstracts citations; do $PY -m a2a.s2.download $d $RAW/$d --workers 16; done
  step "1. build bucketed parquet"
  $PY -m a2a.s2.build_parquet papers    --source $RAW/papers    --dest $DERIVED/papers
  $PY -m a2a.s2.build_parquet abstracts --source $RAW/abstracts --dest $DERIVED/abstracts
  $PY -m a2a.s2.build_parquet citers    --source $RAW/citations --dest $DERIVED/citers
  for k in papers abstracts citers; do $PY -m a2a.s2.build_parquet compact $k --dest $DERIVED/$k; done
  exit 0
fi

step "2. ACL corpus"
$PY scripts/build_acl_corpus.py all --derived-root $DERIVED
step "3. benchmark"
$PY scripts/make_benchmark.py --corpus-root $DERIVED/acl_corpus --out $DATA
step "4. characterise"
$PY scripts/characterize.py --data $DATA --corpus-root $DERIVED/acl_corpus
step "5. generate (example: one model)"
echo "CUDA_VISIBLE_DEVICES=0,1,2,3 $PY scripts/generate.py --input $DATA \\"
echo "  --model meta-llama/Llama-3.3-70B-Instruct --tp 4 \\"
echo "  --arms proposal_t07,proposal_t10,proposal_t13,diverse_list_t10 \\"
echo "  --output-dir results/gen/llama-3.3-70b"
step "6. score (tau* is calibrated from the random-pool null on the whole benchmark)"
echo "$PY scripts/score.py --data $DATA"
step "7. contrast"
echo "$PY scripts/contrast.py"
