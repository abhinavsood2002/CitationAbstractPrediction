#!/usr/bin/env python
"""Cut the benchmark from the ACL corpus: eligible seeds and their qualifying citers.

    python scripts/make_benchmark.py --corpus-root /data/derived/acl_corpus \
        --derived-root /data/derived --out data/acl_a2a.jsonl \
        [--seed-infl-top-frac 0.1] [--seed-infl-cap-frac 0.01] [--min-citers 10]

Defaults are the final scope (``a2a.corpus.BenchmarkParams``, DECISIONS.md 2026-09-19):
seeds in the within-year top 10% but not top 1% of ACL Anthology papers by S2
influential-citation count, influential citers only, background-only citations dropped,
at least 10 qualifying citers per seed, no sampling, no dev/test split. The per-year
integer thresholds are resolved from ``--derived-root`` (papers parquet + ACL id table)
unless ``--seed-infl-thresholds`` points at a JSON file; pass ``0`` to either fraction
to disable that bound and ``none`` to any optional integer bound.

Writes ``<out>`` (seed-grouped JSONL with abstracts), ``<out stem>_ids.json`` (id
manifest: resolved params incl. per-year thresholds + seed/citer ids; enough to rebuild
the exact cut from the corpus) and ``<out stem>_stats.json``.
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


def _bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-root", type=Path, default=corpus.DERIVED_ROOT / "acl_corpus")
    ap.add_argument("--derived-root", type=Path, default=corpus.DERIVED_ROOT,
                    help="bucketed S2 parquet; needed to resolve the per-year seed thresholds")
    ap.add_argument("--acl-ids", type=Path, default=None,
                    help="ACL id side table (default: <derived-root>/acl_ids/acl_papers.parquet "
                         "if it exists)")
    ap.add_argument("--out", type=Path, default=Path("data/acl_a2a.jsonl"))
    ap.add_argument("--seed-infl-thresholds", type=Path, default=None,
                    help="JSON {year: [floor, cap]} to use instead of resolving them")
    p = corpus.BenchmarkParams()
    for k, v in vars(p).items():
        if k == "seed_infl_thresholds":
            continue
        typ = _bool if isinstance(v, bool) else float if isinstance(v, float) else _opt_int
        ap.add_argument(f"--{k.replace('_', '-')}", type=typ, default=v,
                        help=f"default: {v}")
    ap.add_argument("--memory-limit", default="60GB")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    params = corpus.BenchmarkParams(**{k: getattr(args, k) for k in vars(p)
                                       if k != "seed_infl_thresholds"})
    con = corpus.connect(args.memory_limit, args.threads)
    acl_ids = args.acl_ids
    if acl_ids is None:
        cand = args.derived_root / "acl_ids" / "acl_papers.parquet"
        acl_ids = cand if cand.exists() else None
    if args.seed_infl_thresholds is not None:
        thr = {int(y): v for y, v in json.load(open(args.seed_infl_thresholds)).items()}
        params = replace(params, seed_infl_thresholds=thr)
    elif params.seed_infl_top_frac or params.seed_infl_cap_frac:
        thr = corpus.acl_influential_thresholds(con, params.seed_infl_top_frac,
                                                params.seed_infl_cap_frac,
                                                args.derived_root, acl_ids)
        params = replace(params, seed_infl_thresholds=thr)
        years = sorted(thr)
        print(f"seed thresholds by year (influentialcitationcount floor <= x < cap), "
              f"top {params.seed_infl_top_frac:.0%} minus top {params.seed_infl_cap_frac:.0%} "
              f"of ACL papers of that year, {len(years)} years {years[0]}-{years[-1]}:", flush=True)
        for y in years:
            if y >= 2000:
                print(f"  {y}: {thr[y][0]} <= x < {thr[y][1]}", flush=True)
    stats = corpus.make_benchmark(
        con, args.corpus_root / "seeds.parquet", args.corpus_root / "edges",
        args.corpus_root / "citers.parquet", args.out,
        args.out.with_name(args.out.stem + "_ids.json"), params)
    with open(args.out.with_name(args.out.stem + "_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
