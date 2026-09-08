#!/usr/bin/env python
"""Cut the benchmark from the ACL corpus: deterministic seed + citer sample, dev/test split.

    python scripts/make_benchmark.py --corpus-root /data/derived/acl_corpus \
        --out data/acl_a2a.jsonl [--n-seeds 1000] [--citers-per-seed 100] [--min-citers 20]

Writes ``<out>`` (seed-grouped JSONL with abstracts), ``<out stem>_ids.json``
(id manifest: params + seed/citer ids + split; enough to rebuild the exact
sample from the corpus) and ``<out stem>_stats.json``.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a import corpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-root", type=Path, default=corpus.DERIVED_ROOT / "acl_corpus")
    ap.add_argument("--out", type=Path, default=Path("data/acl_a2a.jsonl"))
    p = corpus.BenchmarkParams()
    for k, v in vars(p).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--memory-limit", default="60GB")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    params = corpus.BenchmarkParams(**{k: getattr(args, k) for k in vars(p)})
    con = corpus.connect(args.memory_limit, args.threads)
    stats = corpus.make_benchmark(
        con, args.corpus_root / "seeds.parquet", args.corpus_root / "edges",
        args.corpus_root / "citers.parquet", args.out,
        args.out.with_name(args.out.stem + "_ids.json"), params)
    with open(args.out.with_name(args.out.stem + "_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
