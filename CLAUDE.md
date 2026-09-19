# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for the **abstract-to-abstract (A2A)** task: given a paper's abstract (the *seed*),
generate a set of abstracts anticipating the papers that will *cite* it (influentially). Scored by
**coverage@tau\***: the fraction of real citing abstracts matched (cosine >= tau\*) by at least one
prediction, where tau\* is the 95th percentile of a system-free random-pool null, calibrated on the
whole benchmark (there is no dev/test split; it is evaluation-only).

Seeds are ACL Anthology papers; citers are from any venue/field. `README.md` has the task framing
and full pipeline; `DECISIONS.md` has dated rationale for non-obvious choices (filters, metric,
reference systems, generation arms). **When you make a choice the code cannot explain by itself,
append a dated entry to `DECISIONS.md` (newest at the bottom).**

## Environment and commands

The venv is uv-managed (Python 3.11) and already has vLLM installed. Three packages are pinned by
hand (see DECISIONS.md, 2026-09-09): `torchvision==0.26.0+cu129`, `transformers==5.7.0`,
`huggingface-hub` 1.x. Re-resolving `xgrammar` with uv downgrades transformers to 4.x and breaks
 4; `uv pip check` reporting xgrammar's `transformers<5` pin is expected.

```bash
. .venv/bin/activate
pytest                                   # all tests (~seconds; config in pyproject.toml)
pytest tests/test_metrics.py             # one file
pytest tests/test_corpus.py::test_qualifying_pairs_filters   # one test
ruff check .                             # lint (E/F/W, line length 100, E501 ignored); install with `uv pip install ruff`
```

`pyproject.toml` sets `pythonpath = ["."]` for pytest, so tests import `a2a` directly.
Scripts in `scripts/` insert the repo root onto `sys.path` themselves; run them from the repo root.

Environment variables (nothing loads `.env` automatically; `set -a; . .env; set +a` or export):

- `A2A_DERIVED_ROOT` — root of the hash-bucketed parquet layer. `a2a/corpus.py` and
  `scripts/run_pipeline.sh` have *different* hard-coded fallbacks, so always set it explicitly.
- `S2_API_KEY` — only for `a2a.s2.download` (Semantic Scholar Datasets API).
- `HF_HOME` — Hugging Face cache for encoders and vLLM models. The generation models are cached
  there: `google/gemma-4-E4B-it` (tp 1), `google/gemma-4-26B-A4B-it` (tp 2), `openai/gpt-oss-20b`
  (tp 1), `meta-llama/Llama-4-Scout-17B-16E-Instruct` (tp 4). `python scripts/smoke_test_model.py
  <model> --tp N` checks one loads and generates.

Pipeline entry points (see `scripts/run_pipeline.sh` for the ordered steps):

```bash
bash scripts/run_pipeline.sh download        # one-off: raw S2 shards (~400 GB) -> bucketed parquet
python scripts/build_acl_corpus.py all       # seeds / edges / citers parquet (idempotent, resumable per bucket)
python scripts/make_benchmark.py --derived-root $A2A_DERIVED_ROOT --out data/acl_a2a.jsonl   # resolves per-year influential-citation thresholds
python scripts/characterize.py --data data/acl_a2a.jsonl --corpus-root $A2A_DERIVED_ROOT/acl_corpus
python scripts/visualise.py                      # benchmark distribution figures -> images/*.png
bash scripts/launch_generation.sh 0,1,2,3 meta-llama/Llama-3.3-70B-Instruct 4 llama-3.3-70b   # nohup, logs to results/logs/
python scripts/score.py                           # calibrates tau*, writes tau_star_<enc>.json + <enc>*.csv
python scripts/contrast.py
```

Generation needs CUDA + vLLM. `characterize.py` / `score.py` embed with sentence-transformers
(1 GPU is expected; CPU works but is slow). `build_acl_corpus.py` / `make_benchmark.py` default to
DuckDB `--memory-limit 60GB --threads 16`; lower them on a smaller machine.

## Architecture

`a2a/` is the library; `scripts/` are thin argparse CLIs over it. Data flows strictly left to right:

```
S2 raw .gz shards ──a2a.s2.build_parquet──> derived/{papers,abstracts,citers}/bucket=N/  (64 hash buckets)
                                                     │
                     a2a.corpus.build_{seeds,edges,citers}  (DuckDB over parquet)
                                                     ▼
                        derived/acl_corpus/{seeds.parquet, edges/bucket_NN.parquet, citers.parquet}
                                                     │
                     a2a.corpus.make_benchmark  (filters + BLAKE2b-deterministic sampling)
                                                     ▼
              data/acl_a2a.jsonl  +  acl_a2a_ids.json (manifest)  +  acl_a2a_stats.json
                                                     │
        ┌────────────────────┬───────────────────────┼──────────────────────┐
  characterize.py       generate.py (vLLM)       score.py               contrast.py
  results/characterize  results/gen/<model>/     results/score/         results/contrast/
                        <model>/<arm>.jsonl      (reads gen + emb cache) (reads score CSVs)
```

Key facts that span multiple files:

- **Bucketing.** `papers`/`abstracts` parquet are bucketed on `corpusid % 64`, `citers` on
  `citedcorpusid % 64` (`N_BUCKETS = 64` in both `a2a/s2/build_parquet.py` and `a2a/corpus.py`).
  `build_edges` iterates buckets independently so a killed run resumes at the first missing
  `bucket_NN.parquet`.
- **Determinism.** All sampling goes through `corpus.stable_hash` (BLAKE2b, 8 bytes): seeds
  ordered by `hash("seed", id)`, citers by `hash("citer", seed, citer)`. Changing `stable_hash` or
  `BenchmarkParams` defaults silently changes the benchmark; the committed `data/acl_a2a_ids.json`
  is the reference. Current cut (2026-09-19): 2,470 seeds, 53,034 pairs, no sampling, no split.
  Seeds are in the within-year top 10% but not top 1% of all ACL Anthology papers by S2
  `influentialcitationcount` (per-year floor/cap resolved by `corpus.acl_influential_thresholds`
  and stored in the manifest); citers are `isinfluential` edges whose intent labels are not
  background-only (unlabelled kept) and whose abstract is detected as English (`langdetect`,
  seeded); >= 10 qualifying citers per seed. Rationale in DECISIONS.md.
- **Data policy** (enforced by `.gitignore`): abstract text is never committed (S2 terms). Only
  `data/*_ids.json` and `data/*_stats.json` are tracked; `data/acl_a2a.jsonl` is rebuilt from the
  manifest. `results/gen/`, `results/embeddings/`, `results/logs/`, `*.npz`, `*.parquet` are ignored;
  result summaries/figures are meant to be committed.
- **Benchmark record schema** (`data/acl_a2a.jsonl`, one seed per line): `seed_id`, `acl_id`,
  `seed_title`, `seed_abstract`, `seed_authors`, `seed_year`, `seed_venue`, `seed_citationcount`,
  `seed_influentialcitationcount`, `seed_fields`, `n_citers_qualifying`, `citers: [{id, acl_id, title, abstract, year, venue,
  citationcount, fields, isinfluential, intents, n_contexts}]`. All ids are strings.
- **Generation output contract.** `generate_arm` appends `{"seed_id","arm","model","predictions"}`
  to `results/gen/<model-dir>/<arm>.jsonl` with an `<arm>.jsonl.processed` sidecar of
  finished seed ids, both fsync'd per seed. `load_generations` takes last-write-wins per seed, so a
  resumed run is safe. `score.py` discovers systems as `<model-dir>/<arm>` from this layout.
- **Arms** live in the `ARMS` dict in `a2a/generate.py` (`independent` = one prompt sampled n times;
  `list` = ask for `per_call` proposals as JSON, parsed by `parse_list`). `scripts/generate.py`
  validates `--arms` against that dict, so adding an arm is a one-place change. Prompts forbid invented
  numerical results (fabricated numbers inflate embedding similarity).
- **Reasoning channels.** `a2a/llm.py` turns thinking off per model family
  (`chat_template_kwargs_for`) and strips leaked channels (`strip_reasoning`: gpt-oss Harmony,
  Gemma `<|channel>`, DeepSeek-style `<think>`). New model families go there.
- **Embeddings.** `a2a/embed.py` always returns unit-norm vectors, so `A @ B.T` is cosine everywhere
  in `metrics.py` and the scripts. The encoder is `Qwen/Qwen3-Embedding-0.6B` (`DEFAULT_ENCODER`;
  bfloat16, 1,024 tokens, no instruction prompt so similarity is symmetric); batches of 5,000+ texts
  are spread over every visible GPU. `embed_cached` keeps one `results/embeddings/<enc>/texts.npz`
  keyed by text hash; `characterize.py` and `score.py` share it, so run characterize first to warm it.
- **System naming in score/contrast.** Generation systems are `<model>/<arm>`; baselines are
  `null/seed_copy`, `null/random_pool` (defines tau\*), `ref/retrieval_knn`, `ref/split_half`,
  `ref/target_set` (diversity only). `contrast.py` treats anything not prefixed `null/` or `ref/`
  as a generation system, and reads `score.py`'s `<enc>_per_seed.csv` and
  `<enc>_per_pair.csv.gz` (system columns start at index 8).
- **tau\* discipline.** `score.py` recalibrates tau\* from the random-pool null on every run and
  writes `results/score/tau_star_<enc>.json`; the null involves no system, so there is nothing to
  hold out. `--tau-star` overrides it for sensitivity checks only. The tau-grid
  AUC is reported but deliberately not used for claims (see DECISIONS.md).
- **Atomic writes everywhere.** Parquet, JSONL and `.npz` outputs are written to `*.tmp` and
  renamed; `build_parquet` uses `_done/<shard>.ok` markers. Keep that pattern for new outputs so
  reruns stay idempotent.
