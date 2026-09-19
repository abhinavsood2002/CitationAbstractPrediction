#!/usr/bin/env python
"""Generate 51 candidate future-citer abstracts per seed with one generator.

    bash scripts/launch_generation.sh                 # the whole run, one model at a time on 8 GPUs
    CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --generator gemma --shard 0/8

One process loads one model (``a2a.llm.GENERATORS``: model id, tensor parallelism, sampling
defaults) and runs every condition of that generator (``a2a.generate.CONDITIONS``), or the
``--conditions`` subset, over ``seeds[i::n]``. A model is run data-parallel: one process per
GPU, each with its own ``--shard``.

Resumable: seeds listed in any ``.processed`` sidecar of a condition are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a.generate import CONDITIONS, N_GENERATIONS, generate_condition  # noqa: E402
from a2a.llm import GENERATORS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", required=True, choices=list(GENERATORS))
    ap.add_argument("--conditions", default=None,
                    help="comma-separated subset of the generator's conditions (default: all)")
    ap.add_argument("--shard", default="0/1", help="i/n: run seeds[i::n]")
    ap.add_argument("--input", type=Path, default=Path("data/acl_a2a_noresults.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/gen"))
    ap.add_argument("--n", type=int, default=N_GENERATIONS)
    ap.add_argument("--max-seeds", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="seeds per llm.chat call (default: the generator's batch_size)")
    ap.add_argument("--tp", type=int, default=None, help="override the generator's tensor parallelism")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    gen = GENERATORS[args.generator]
    own = [c for c, cond in CONDITIONS.items() if cond.generator == args.generator]
    names = args.conditions.split(",") if args.conditions else own
    assert set(names) <= set(own), f"{set(names) - set(own)} are not {args.generator} conditions: {own}"
    shard = tuple(map(int, args.shard.split("/")))
    assert 0 <= shard[0] < shard[1], args.shard

    seeds = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            seeds.append({k: r[k] for k in ("seed_id", "seed_title", "seed_year", "seed_abstract")})
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]
    print(f"{len(seeds)} seeds (shard {args.shard}) x {names} x n={args.n} | {gen.hf_id}", flush=True)

    from vllm import LLM
    llm = LLM(model=gen.hf_id, tensor_parallel_size=args.tp or gen.tp,
              gpu_memory_utilization=args.gpu_mem_util, max_num_seqs=gen.max_num_seqs,
              max_model_len=args.max_model_len, dtype="auto", trust_remote_code=True)
    for name in names:
        generate_condition(llm, name, seeds, args.output_dir, shard=shard, n=args.n,
                           batch_size=args.batch_size)


if __name__ == "__main__":
    main()
