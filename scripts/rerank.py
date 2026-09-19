#!/usr/bin/env python
"""Reranker match probabilities for every (citer, generation) pair: the input of Recall@k.

    CUDA_VISIBLE_DEVICES=0 python scripts/rerank.py --shard 0/8      # one process per GPU
    bash scripts/launch_rerank.sh                                    # all 8 shards

Each process loads the reranker once and walks the (condition, mode) units of its shard,
skipping those already written (see ``a2a.rerank``). ``own`` targets are scored for every
condition; the ``shuffled`` chance floor only for ``--shuffled-for`` (one floor is enough,
and it costs as much as the real scoring).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a.corpus import load_jsonl  # noqa: E402
from a2a.generate import load_generations  # noqa: E402
from a2a.rerank import load_reranker, rerank_unit, shard_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/acl_a2a_noresults.jsonl"))
    ap.add_argument("--gen-dir", type=Path, default=Path("results/gen"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/rerank"))
    ap.add_argument("--conditions", default=None, help="comma-separated; default: all in --gen-dir")
    ap.add_argument("--shuffled-for", default="gemma_t10",
                    help="conditions that also get the shuffled baseline: comma-separated, "
                         "'all' or 'none'")
    ap.add_argument("--shard", default="0/1", help="i/n: score seeds[i::n]")
    args = ap.parse_args()

    shard = tuple(map(int, args.shard.split("/")))
    assert 0 <= shard[0] < shard[1], args.shard
    gens = load_generations(args.gen_dir)
    seeds = load_jsonl(args.data)
    names = args.conditions.split(",") if args.conditions else list(gens)
    assert set(names) <= set(gens), f"no generations for {set(names) - set(gens)}"
    shuffled = {"all": set(names), "none": set()}.get(args.shuffled_for,
                                                      set(args.shuffled_for.split(",")))
    # a pilot run generates for a prefix of the benchmark: keep the seeds every condition has
    have = set.intersection(*(set(gens[c]) for c in names))
    seeds = [s for s in seeds if s["seed_id"] in have]
    units = [(c, m) for c in names for m in ("own", "shuffled") if m == "own" or c in shuffled]
    todo = [(c, m) for c, m in units if not shard_path(args.out_dir, c, m, shard).exists()]
    print(f"{len(seeds)} seeds, shard {args.shard}: {len(todo)}/{len(units)} units to score",
          flush=True)
    if not todo:
        return
    llm = load_reranker()
    for c, m in todo:
        rerank_unit(llm, seeds, gens[c], m, shard, shard_path(args.out_dir, c, m, shard))


if __name__ == "__main__":
    main()
