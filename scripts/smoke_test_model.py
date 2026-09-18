#!/usr/bin/env python
"""Check that a generation model loads under vLLM and produces a clean abstract.

    CUDA_VISIBLE_DEVICES=0,1 python scripts/smoke_test_model.py google/gemma-4-26B-A4B-it --tp 2

Uses the same ``LLM(...)`` arguments as ``scripts/generate.py`` (shorter context) and the
chat-template kwargs from ``a2a.llm``, then samples two short abstracts and prints them after
``strip_reasoning``. Exit status is non-zero if a sample comes back empty.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a.llm import chat_template_kwargs_for, strip_reasoning  # noqa: E402

PROMPT = ("Write a one-paragraph abstract (about 120 words) for a hypothetical NLP paper that "
          "builds on retrieval-augmented generation for citation recommendation. Do not include "
          "numerical results. Output only the abstract.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
              dtype="auto", trust_remote_code=True)
    print(f"[smoke] loaded {args.model} in {time.time() - t0:.0f}s", flush=True)

    kwargs = chat_template_kwargs_for(args.model)
    sp = SamplingParams(temperature=1.0, max_tokens=400, n=args.n, seed=0)
    t0 = time.time()
    outs = llm.chat([[{"role": "user", "content": PROMPT}]], sp,
                    chat_template_kwargs=kwargs, use_tqdm=False)
    print(f"[smoke] generated in {time.time() - t0:.1f}s; chat_template_kwargs={kwargs}", flush=True)
    ok = True
    for i, o in enumerate(outs[0].outputs):
        clean = strip_reasoning(o.text)
        ok &= bool(clean.strip())
        print(f"--- sample {i}: raw_len={len(o.text)} clean_len={len(clean)} "
              f"finish={o.finish_reason}\n{clean[:900]}")
    print("[smoke] OK" if ok else "[smoke] FAILED: empty sample", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
