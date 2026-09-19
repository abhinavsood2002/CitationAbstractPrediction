#!/usr/bin/env python
"""Score every generation condition against the real citing abstracts.

    python scripts/score.py                       # results/gen + results/rerank -> results/score
    python scripts/score.py --gen-dir results/pilot/gen --rerank-dir results/pilot/rerank \
        --out-dir results/pilot/score

Per condition and target mode (``own`` = the seed's citers, ``shuffled`` = another seed's
citers, the chance floor; see ``a2a.rerank``):

  Recall@k         k = 10, 51 and the K_GRID curve; needs scripts/rerank.py output, and is
                   skipped for a (condition, mode) that has none
  cosine coverage  mean over citers of the best cosine to any of the 51 generations
  Vendi@51         per seed, cosine kernel of its generation embeddings (own mode only)

Means are over citers (Vendi: over seeds) with a 95% bootstrap CI that resamples seeds.
Robustness: the same numbers for citers dated from CUTOFF_YEAR on (after the generators'
training data ends) against earlier ones, and by citer year, for every condition;
``best_gemma`` names the Gemma condition with the highest Recall@51.

Writes <enc>_summary.json, <enc>_per_seed.csv and <enc>_per_pair.csv.gz (long format: one
row per condition x mode x pair with n_gen, m and best cosine).
"""
import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a import metrics as M  # noqa: E402
from a2a.corpus import load_jsonl  # noqa: E402
from a2a.embed import DEFAULT_ENCODER, embed_cached, short_name  # noqa: E402
from a2a.generate import CONDITIONS, load_generations  # noqa: E402
from a2a.llm import MATCH_THRESHOLD  # noqa: E402
from a2a.rerank import MODES, load_probs, target_citers  # noqa: E402

CUTOFF_YEAR = 2025   # citers from this year on post-date every generator's training data
FIRST_YEAR = 2015    # earlier citer years are pooled in the by-year table


def year_label(y):
    return None if not y else (f"<={FIRST_YEAR - 1}" if y < FIRST_YEAR else str(y))


def summarise(rows, n_boot, seed):
    """Metrics over a list of pair rows (dicts with seed_id, n_gen, m, best_cos)."""
    groups = [r["seed_id"] for r in rows]
    out = {"n_pairs": len(rows), "n_seeds": len(set(groups))}
    if not rows:
        return out
    mean, ci = M.bootstrap_pooled_ci([r["best_cos"] for r in rows], groups, n_boot, seed)
    out["cosine_coverage"], out["cosine_coverage_ci95"] = mean, list(ci)
    if rows and rows[0]["m"] is not None:
        m, n = np.array([r["m"] for r in rows]), np.array([r["n_gen"] for r in rows])
        for k in M.K_HEADLINE:
            mean, ci = M.bootstrap_pooled_ci(M.recall_at_k(m, n, k), groups, n_boot, seed)
            out[f"recall@{k}"], out[f"recall@{k}_ci95"] = mean, list(ci)
        out["recall_curve"] = M.recall_curve(m, n)
        out["mean_matches_per_citer"] = float(m.mean())
    return out


def write_atomic(path: Path, write, opener=open, **kw):
    tmp = path.with_name(path.name + ".tmp")
    with opener(tmp, "wt", **kw) as f:
        write(f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/acl_a2a_noresults.jsonl"))
    ap.add_argument("--gen-dir", type=Path, default=Path("results/gen"))
    ap.add_argument("--rerank-dir", type=Path, default=Path("results/rerank"))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--emb-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/score"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    enc = short_name(args.encoder)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gens = load_generations(args.gen_dir)
    assert gens, f"no generations under {args.gen_dir}"
    have = set.intersection(*(set(g) for g in gens.values()))
    seeds = [s for s in load_jsonl(args.data) if s["seed_id"] in have]
    print(f"{len(seeds)} seeds, conditions: {list(gens)}", flush=True)

    # ---- embeddings: citers share the benchmark cache, each condition keeps its own
    emb_dir = args.emb_dir / enc
    citers = {c["id"]: c for s in seeds for c in s["citers"]}
    ids = list(citers)
    E = embed_cached([citers[i]["abstract"] for i in ids], args.encoder, emb_dir / "texts.npz")
    E_cit = dict(zip(ids, E))
    targets = {mode: target_citers(seeds, mode) for mode in MODES}

    pair_rows, seed_rows, summary = [], [], {}
    for name, recs in gens.items():
        texts = [t for s in seeds for t in recs[s["seed_id"]]]
        print(f"[{name}] embedding {len(texts)} generations", flush=True)
        G = embed_cached(texts, args.encoder, emb_dir / f"gen_{name}.npz")
        E_gen, k = {}, 0
        for s in seeds:
            n = len(recs[s["seed_id"]])
            E_gen[s["seed_id"]] = G[k:k + n]
            k += n

        vendi = {sid: M.vendi_score(P @ P.T) for sid, P in E_gen.items() if len(P)}
        n_gen = {sid: len(P) for sid, P in E_gen.items()}
        seed_rows += [{"condition": name, "seed_id": sid, "n_gen": n_gen[sid], "vendi": v}
                      for sid, v in vendi.items()]
        v_mean, v_ci = M.bootstrap_mean_ci(list(vendi.values()), args.n_boot, args.seed)
        summary[name] = {"model": CONDITIONS[name].generator if name in CONDITIONS else None,
                         "n_seeds": len(vendi), "mean_n_gen": float(np.mean(list(n_gen.values()))),
                         "seeds_short_of_51": int(sum(n < 51 for n in n_gen.values())),
                         "vendi@51": v_mean, "vendi@51_ci95": list(v_ci)}

        for mode in MODES:
            probs = load_probs(args.rerank_dir, name, mode)
            rows = []
            for s in seeds:
                sid, P = s["seed_id"], E_gen[s["seed_id"]]
                if not len(P):
                    continue
                for c in targets[mode][sid]:
                    p = probs.get((sid, c["id"]))
                    if p is not None:
                        assert len(p) == len(P), f"{name}/{mode} {sid}: rerank is stale"
                    rows.append({"condition": name, "mode": mode, "seed_id": sid,
                                 "citer_id": c["id"], "citer_year": c["year"], "n_gen": len(P),
                                 "m": None if p is None else int((p > MATCH_THRESHOLD).sum()),
                                 "best_cos": float((P @ E_cit[c["id"]]).max())})
            if probs and any(r["m"] is None for r in rows):
                print(f"[{name}/{mode}] rerank incomplete: Recall@k skipped", flush=True)
                for r in rows:
                    r["m"] = None
            res = summarise(rows, args.n_boot, args.seed)
            post = [r for r in rows if (r["citer_year"] or 0) >= CUTOFF_YEAR]
            pre = [r for r in rows if 0 < (r["citer_year"] or 0) < CUTOFF_YEAR]
            res["post_cutoff"] = summarise(post, args.n_boot, args.seed)
            res["pre_cutoff"] = summarise(pre, args.n_boot, args.seed)
            res["by_citer_year"] = {
                y: summarise([r for r in rows if year_label(r["citer_year"]) == y],
                             args.n_boot, args.seed)
                for y in sorted({year_label(r["citer_year"]) for r in rows} - {None})}
            summary[name][mode] = res
            pair_rows += rows

    gemma = {n: r["own"].get("recall@51") for n, r in summary.items()
             if r["model"] == "gemma" and r["own"].get("recall@51") is not None}
    out = {"encoder": args.encoder, "match_threshold": MATCH_THRESHOLD, "cutoff_year": CUTOFF_YEAR,
           "n_seeds": len(seeds), "best_gemma": max(gemma, key=gemma.get) if gemma else None,
           "conditions": summary}
    write_atomic(args.out_dir / f"{enc}_summary.json", lambda f: json.dump(out, f, indent=1))

    def write_csv(rows):
        def write(f):
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return write
    write_atomic(args.out_dir / f"{enc}_per_seed.csv", write_csv(seed_rows), newline="")
    write_atomic(args.out_dir / f"{enc}_per_pair.csv.gz", write_csv(pair_rows), gzip.open,
                 newline="")

    def fmt(r, key):
        return f"{r[key]:>8.3f}" if key in r else f"{'-':>8}"
    print(f"\n{'condition':<14} {'targets':<9} {'R@10':>8} {'R@51':>8} {'cos-cov':>8} "
          f"{'R@51 post':>9} {'cos post':>8} {'vendi@51':>8}")
    for name, r in summary.items():
        for mode in MODES:
            d = r[mode]
            print(f"{name:<14} {mode:<9} {fmt(d, 'recall@10')} {fmt(d, 'recall@51')} "
                  f"{fmt(d, 'cosine_coverage')} {fmt(d['post_cutoff'], 'recall@51'):>9} "
                  f"{fmt(d['post_cutoff'], 'cosine_coverage')} "
                  f"{r['vendi@51'] if mode == 'own' else float('nan'):>8.2f}")
    print(f"best Gemma condition by Recall@51: {out['best_gemma']}")
    print(f"-> {args.out_dir}/{enc}_summary.json")


if __name__ == "__main__":
    main()
