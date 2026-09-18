#!/usr/bin/env python
"""Generate N candidate future-citer abstracts per seed with a vLLM model.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/generate.py \
        --input data/acl_a2a.jsonl \
        --model meta-llama/Llama-3.3-70B-Instruct --tp 4 \
        --arms proposal_t10,proposal_t07,proposal_t13,diverse_list_t10 \
        --output-dir results/gen/llama-3.3-70b

Resumable: re-running skips seeds listed in each arm's ``.processed`` sidecar.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a.generate import ARMS, N_DEFAULT, generate_arm  # noqa: E402
from a2a.llm import chat_template_kwargs_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--arms", default="proposal_t10")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--max-seeds", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    arms = args.arms.split(",")
    unknown = set(arms) - set(ARMS)
    assert not unknown, f"unknown arms: {unknown}; known: {list(ARMS)}"

    seeds = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            seeds.append({k: r[k] for k in ("seed_id", "seed_title", "seed_abstract")})
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]
    print(f"{len(seeds)} seeds x {len(arms)} arms x n={args.n} | {args.model}", flush=True)

    from vllm import LLM
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, dtype="auto", trust_remote_code=True)
    kwargs = chat_template_kwargs_for(args.model)
    for arm in arms:
        generate_arm(llm, arm, seeds, args.output_dir, args.model, n=args.n,
                     batch_size=args.batch_size, chat_template_kwargs=kwargs)


if __name__ == "__main__":
    main()
