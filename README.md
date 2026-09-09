# abstract-to-abstract

**Task.** Given the abstract of a paper (the *seed*), produce a set of
abstracts of papers that will cite it. The score is *coverage*: the fraction
of the papers that actually came to cite the seed whose abstract is matched
by at least one prediction, at a similarity threshold calibrated against
chance.

**Data.** Seeds are ACL Anthology papers (Semantic Scholar `externalids.ACL`),
targets are their citing papers from *any* venue or field, with abstracts from
the Semantic Scholar Datasets API. The benchmark cut used in the paper has
13,194 seeds (ACL Anthology papers between the 80th and 99th citation-count percentile,
any year) with every qualifying citer, 828 K pairs; a 10% dev split calibrates the
threshold, the rest is scored once.

**Why this task.** A paper's citing literature is a realised, textual record
of where its ideas went. Anticipating it from the abstract alone asks a model
for the *spread* of plausible futures, not a single good continuation. The
analysis scripts show (1) how spread out the real targets are, (2) that
off-the-shelf LLMs do not recreate that spread, and (3) that coverage is not
redundant with reference-free diversity or seed-overlap metrics.

## Layout

```
a2a/
├── s2/            Semantic Scholar downloads (api.py, download.py) and the
│                  hash-bucketed parquet layer (build_parquet.py)
├── corpus.py      ACL-seed corpus (DuckDB) + deterministic benchmark cut
├── generate.py    vLLM generation arms (proposal prompt, temperature sweep,
│                  explicit-diversity list prompt); crash-resilient output
├── embed.py       SciNCL / MiniLM embeddings with an on-disk cache
├── metrics.py     coverage@tau, tau* calibration, Vendi, dispersion,
│                  split-half ceiling, lexical diversity, bootstrap CIs
└── llm.py         chat-template flags and reasoning-channel stripping
scripts/
├── build_acl_corpus.py   derived parquet -> seeds / edges / citers parquet
├── make_benchmark.py     corpus -> data/acl_a2a.jsonl + id manifest + stats
├── characterize.py       section 1: target-set statistics and figures
├── generate.py           section 2: generations per model / arm
├── score.py              section 2: coverage vs nulls and references
├── contrast.py           section 3: correlations, rankings, far-tail coverage
└── run_pipeline.sh       the steps in order
tests/                    pytest (metrics, corpus filters, parsers)
data/                     id manifest + stats committed; text is rebuilt
results/                  summaries and figures committed; generations,
                          embeddings and logs are not
DECISIONS.md              dated rationale for the non-obvious choices
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # analysis
pip install vllm                         # generation only
pytest
```

Set `S2_API_KEY` (free at semanticscholar.org/product/api) for downloads.
`A2A_DERIVED_ROOT` points at the parquet layer (default in `a2a/corpus.py`).

## Pipeline

```bash
# one-off: raw S2 shards (~400 GB) -> bucketed parquet
bash scripts/run_pipeline.sh download

# corpus (minutes) and benchmark cut
python scripts/build_acl_corpus.py all
python scripts/make_benchmark.py --derived-root $A2A_DERIVED_ROOT --out data/acl_a2a.jsonl
python scripts/visualise.py            # distribution of the cut -> images/*.png

# section 1: characterise the targets (1 GPU for embeddings)
python scripts/characterize.py --data data/acl_a2a.jsonl --corpus-root $A2A_DERIVED_ROOT/acl_corpus

# section 2: generate, then score (dev calibrates tau*, test reuses it)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/generate.py --input data/acl_a2a.jsonl --split dev \
    --model meta-llama/Llama-3.3-70B-Instruct --tp 4 \
    --arms proposal_t07,proposal_t10,proposal_t13,diverse_list_t10 \
    --output-dir results/gen/dev/llama-3.3-70b
python scripts/score.py --split dev --calibrate
python scripts/score.py --split test

# section 3: what coverage measures that other metrics do not
python scripts/contrast.py --split dev
```

## Rebuilding the benchmark text

`data/acl_a2a_ids.json` lists every seed and citer id with the split and the
parameters used. `make_benchmark.py` with the same parameters over the same
Semantic Scholar release reproduces `data/acl_a2a.jsonl` exactly; the text
itself is not redistributed.

## Results

See `results/characterize/summary.json`, `results/score/*.json`,
`results/contrast/*.json` and the PNG figures next to them. Headline numbers
are summarised in `RESULTS.md` once a run completes.
