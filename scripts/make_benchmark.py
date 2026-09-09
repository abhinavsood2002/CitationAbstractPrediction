#!/usr/bin/env python
"""Cut the benchmark from the ACL corpus: eligible seeds, their qualifying citers, dev/test split.

    python scripts/make_benchmark.py --corpus-root /data/derived/acl_corpus \
        --derived-root /data/derived --out data/acl_a2a.jsonl \
        [--seed-citation-top-frac 0.2] [--seed-citation-cap-frac 0.01] [--n-seeds N]

Defaults are the final cut (``a2a.corpus.BenchmarkParams``): every ACL seed between the
80th and 99th percentile of ACL papers by citation count with a non-NULL year, every
qualifying citer, no sampling. The integer floor / cap are resolved from ``--derived-root``
(papers parquet + ACL id table) unless ``--min-seed-citations`` / ``--max-seed-citations``
are given; pass ``none`` to any optional bound (or ``0`` to a fraction) to disable it.

Writes ``<out>`` (seed-grouped JSONL with abstracts), ``<out stem>_ids.json`` (id manifest:
resolved params + seed/citer ids + split; enough to rebuild the exact cut from the corpus)
and ``<out stem>_stats.json``.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a import corpus  # noqa: E402


def _opt_int(v: str):
    return None if v.lower() in ("none", "") else int(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-root", type=Path, default=corpus.DERIVED_ROOT / "acl_corpus")
    ap.add_argument("--derived-root", type=Path, default=corpus.DERIVED_ROOT,
                    help="bucketed S2 parquet; needed to resolve the seed citation floor")
    ap.add_argument("--acl-ids", type=Path, default=None,
                    help="ACL id side table (default: <derived-root>/acl_ids/acl_papers.parquet "
                         "if it exists)")
    ap.add_argument("--out", type=Path, default=Path("data/acl_a2a.jsonl"))
    p = corpus.BenchmarkParams()
    for k, v in vars(p).items():
        typ = float if isinstance(v, float) else _opt_int
        ap.add_argument(f"--{k.replace('_', '-')}", type=typ, default=v,
                        help=f"default: {v}")
    ap.add_argument("--memory-limit", default="60GB")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    params = corpus.BenchmarkParams(**{k: getattr(args, k) for k in vars(p)})
    con = corpus.connect(args.memory_limit, args.threads)
    acl_ids = args.acl_ids
    if acl_ids is None:
        cand = args.derived_root / "acl_ids" / "acl_papers.parquet"
        acl_ids = cand if cand.exists() else None
    if params.min_seed_citations is None and params.seed_citation_top_frac:
        floor = corpus.acl_citation_quantile(con, 1 - params.seed_citation_top_frac,
                                             args.derived_root, acl_ids)
        params = replace(params, min_seed_citations=floor)
        print(f"seed citation floor: top {params.seed_citation_top_frac:.0%} of ACL papers "
              f"-> citationcount >= {floor}", flush=True)
    if params.max_seed_citations is None and params.seed_citation_cap_frac:
        cap = corpus.acl_citation_quantile(con, 1 - params.seed_citation_cap_frac,
                                           args.derived_root, acl_ids)
        params = replace(params, max_seed_citations=cap)
        print(f"seed citation cap: drop top {params.seed_citation_cap_frac:.0%} of ACL papers "
              f"-> citationcount <= {cap}", flush=True)
    stats = corpus.make_benchmark(
        con, args.corpus_root / "seeds.parquet", args.corpus_root / "edges",
        args.corpus_root / "citers.parquet", args.out,
        args.out.with_name(args.out.stem + "_ids.json"), params)
    with open(args.out.with_name(args.out.stem + "_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
