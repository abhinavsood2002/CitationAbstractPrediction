#!/usr/bin/env python
"""Build the ACL-seed corpus (seeds / edges / citers parquet) from the derived S2 layer.

    python scripts/build_acl_corpus.py all --out-root /data/derived/acl_corpus \
        [--derived-root /data/derived] [--acl-ids /data/derived/acl_ids/acl_papers.parquet]

Steps are idempotent: ``seeds`` and ``citers`` are skipped when their output
exists, ``edges`` resumes per bucket. Run a single step by name.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a import corpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["seeds", "edges", "citers", "all"])
    ap.add_argument("--derived-root", type=Path, default=corpus.DERIVED_ROOT)
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default: <derived-root>/acl_corpus")
    ap.add_argument("--acl-ids", type=Path, default=None,
                    help="side table from `build_parquet acl_ids` when papers "
                         "parquet lacks acl_id (default: <derived-root>/acl_ids/"
                         "acl_papers.parquet if it exists)")
    ap.add_argument("--buckets", default=None, help="edges only, e.g. '0,1,5-7'")
    ap.add_argument("--memory-limit", default="60GB")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--temp-dir", default=None)
    args = ap.parse_args()

    out_root = args.out_root or args.derived_root / "acl_corpus"
    out_root.mkdir(parents=True, exist_ok=True)
    acl_ids = args.acl_ids
    if acl_ids is None:
        cand = args.derived_root / "acl_ids" / "acl_papers.parquet"
        acl_ids = cand if cand.exists() else None
    seeds, edges_dir, citers = out_root / "seeds.parquet", out_root / "edges", out_root / "citers.parquet"
    con = corpus.connect(args.memory_limit, args.threads, args.temp_dir)
    steps = ["seeds", "edges", "citers"] if args.step == "all" else [args.step]
    for step in steps:
        t0 = time.time()
        if step == "seeds":
            if seeds.exists():
                print("seeds: exists, skip"); continue
            n = corpus.build_seeds(con, seeds, args.derived_root, acl_ids)
        elif step == "edges":
            buckets = None
            if args.buckets:
                buckets = []
                for part in args.buckets.split(","):
                    a, _, b = part.partition("-")
                    buckets.extend(range(int(a), int(b or a) + 1))
            n = corpus.build_edges(con, seeds, edges_dir, args.derived_root, buckets)
        else:
            if citers.exists():
                print("citers: exists, skip"); continue
            n = corpus.build_citers(con, edges_dir, citers, args.derived_root, acl_ids)
        print(f"{step}: {n:,} rows [{time.time() - t0:.0f}s] -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
