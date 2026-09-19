"""Reranker match probabilities between real citing abstracts and generations.

For every (seed, target citer) pair the reranker (``llm.RERANKER``) scores the
citer, as the query, against each of the seed's generations, as documents;
P(yes) > ``llm.MATCH_THRESHOLD`` is a match. Which citers are a seed's targets
depends on the *mode*:

* ``own``: the seed's real citers.
* ``shuffled``: the citers of another seed (the chance floor). Seeds are put on
  a cycle ordered by ``stable_hash("shuffle", seed_id)`` and each takes the
  citers of its successor, so the assignment is deterministic, is the same for
  every condition, never maps a seed to itself and uses every target set once.
  Citers that also cite the receiving seed are left out.

One unit of work is (condition, mode, shard): ``seeds[i::n]`` scored and written
atomically to ``<root>/<condition>/<mode>/shard_<i>_of_<n>.npz`` with one row per
pair: ``seed_id``, ``citer_id``, ``n_gen`` and ``probs`` (n_pairs x 51, float16,
NaN beyond ``n_gen``). An existing file is skipped, so a killed run resumes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .corpus import stable_hash
from .generate import N_GENERATIONS
from .llm import (RERANKER, RERANKER_HF_OVERRIDES, RERANKER_MAX_LEN, rerank_document,
                  rerank_query)

MODES = ("own", "shuffled")


def target_citers(seeds: list[dict], mode: str) -> dict[str, list[dict]]:
    """{seed_id: citer records that are the seed's targets under ``mode``}."""
    assert mode in MODES, mode
    if mode == "own":
        return {s["seed_id"]: s["citers"] for s in seeds}
    assert len(seeds) > 1, "shuffling needs at least two seeds"
    cycle = sorted(seeds, key=lambda s: stable_hash("shuffle", s["seed_id"]))
    out = {}
    for s, donor in zip(cycle, cycle[1:] + cycle[:1]):
        own = {c["id"] for c in s["citers"]}
        out[s["seed_id"]] = [c for c in donor["citers"] if c["id"] not in own]
    return out


def shard_path(root: Path, condition: str, mode: str, shard: tuple[int, int]) -> Path:
    return Path(root) / condition / mode / f"shard_{shard[0]}_of_{shard[1]}.npz"


def load_reranker(gpu_mem_util: float = 0.85):
    from vllm import LLM
    return LLM(model=RERANKER, runner="pooling", hf_overrides=RERANKER_HF_OVERRIDES,
               max_model_len=RERANKER_MAX_LEN, gpu_memory_utilization=gpu_mem_util,
               enable_prefix_caching=True)


def rerank_unit(llm, seeds: list[dict], generations: dict[str, list[str]], mode: str,
                shard: tuple[int, int], out: Path, chunk_pairs: int = 2000,
                max_chars: int = 3000) -> None:
    """Score one (condition, mode, shard) and write ``out``. ``seeds`` is the full list:
    the shuffled assignment must not depend on the sharding."""
    targets = target_citers(seeds, mode)
    rows = [(s["seed_id"], c) for s in seeds[shard[0]::shard[1]] for c in targets[s["seed_id"]]
            if generations.get(s["seed_id"])]
    probs = np.full((len(rows), N_GENERATIONS), np.nan, dtype=np.float16)
    t0 = time.time()
    for lo in range(0, len(rows), chunk_pairs):
        queries, docs, where = [], [], []
        for r, (sid, c) in enumerate(rows[lo:lo + chunk_pairs], start=lo):
            q = rerank_query(c["abstract"][:max_chars])
            for g, text in enumerate(generations[sid][:N_GENERATIONS]):
                queries.append(q)
                docs.append(rerank_document(text[:max_chars]))
                where.append((r, g))
        outs = llm.score(queries, docs, use_tqdm=False)
        for (r, g), o in zip(where, outs):
            probs[r, g] = o.outputs.score
        done = min(lo + chunk_pairs, len(rows))
        rate = done * N_GENERATIONS / (time.time() - t0 + 1e-9)
        print(f"[{out.parent.parent.name}/{mode} {shard[0]}/{shard[1]}] {done}/{len(rows)} "
              f"citers ({rate:.0f} pairs/s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:  # file handle: np.savez must not append ".npz"
        np.savez(f, seed_id=np.array([sid for sid, _ in rows], dtype=str),
                 citer_id=np.array([c["id"] for _, c in rows], dtype=str),
                 n_gen=np.array([min(len(generations[sid]), N_GENERATIONS) for sid, _ in rows]),
                 probs=probs)
    tmp.replace(out)


def load_probs(root: Path, condition: str, mode: str) -> dict[tuple[str, str], np.ndarray]:
    """{(seed_id, citer_id): probabilities over that seed's generations} from every shard."""
    out = {}
    for p in sorted((Path(root) / condition / mode).glob("shard_*.npz")):
        z = np.load(p)
        for sid, cid, n, row in zip(z["seed_id"], z["citer_id"], z["n_gen"], z["probs"]):
            out[(str(sid), str(cid))] = row[:n].astype(np.float32)
    return out
