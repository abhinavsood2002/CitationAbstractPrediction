#!/usr/bin/env python
"""Does coverage of the real citing literature measure something that
reference-free diversity and seed-overlap metrics do not?

    python scripts/contrast.py --encoder Qwen/Qwen3-Embedding-0.6B \
        --gen-root results/gen --score-dir results/score --out-dir results/contrast

Three analyses, all on the per-seed / per-pair tables written by score.py:

A. Per-seed rank correlation (Spearman) between coverage@tau* and, for the
   same prediction set: Vendi, mean pairwise similarity, centroid dispersion,
   distinct-2, self-BLEU, mean similarity to the seed, ROUGE-L to the seed,
   mean length. Reported per system and pooled over generation systems.
B. System ranking: order the generation arms by coverage and by each
   alternative metric; temperature raises every reference-free diversity
   metric monotonically, coverage does not follow.
C. Coverage by seed-similarity decile of the *target*: which citers are
   covered by generations, by retrieval of real text, and by the seed's own
   other citers. The far tail is what only this benchmark scores.
"""
import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a import metrics as M  # noqa: E402
from a2a.corpus import load_jsonl  # noqa: E402
from a2a.embed import DEFAULT_ENCODER, short_name  # noqa: E402
from a2a.generate import load_generations  # noqa: E402

ALT = ["vendi", "mps", "disp", "distinct2", "self_bleu", "seed_sim", "rougeL_seed", "mean_len"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/acl_a2a.jsonl"))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--gen-root", type=Path, default=Path("results/gen"))
    ap.add_argument("--score-dir", type=Path, default=Path("results/score"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/contrast"))
    ap.add_argument("--max-seeds-lexical", type=int, default=300,
                    help="self-BLEU / ROUGE are O(n^2) per seed; cap seeds for them")
    ap.add_argument("--lexical-items", type=int, default=15)
    args = ap.parse_args()
    enc = short_name(args.encoder)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = {s["seed_id"]: s for s in load_jsonl(args.data)}
    per_seed = defaultdict(dict)
    with open(args.score_dir / f"{enc}_per_seed.csv") as f:
        for r in csv.DictReader(f):
            per_seed[r["system"]][r["seed_id"]] = {k: float(v) if v not in ("", None) else np.nan
                                                   for k, v in r.items() if k not in ("system", "seed_id")}
    gen_systems = [s for s in per_seed if not s.startswith(("null/", "ref/"))]
    print(f"{len(gen_systems)} generation systems, {len(seeds)} seeds", flush=True)

    # ---- lexical metrics on the generations (capped)
    lex_sids = sorted(seeds, key=lambda s: int(s))[:args.max_seeds_lexical]
    for name in gen_systems:
        model, arm = name.split("/", 1)
        recs = load_generations(args.gen_root / model).get(arm, {})
        for sid in lex_sids:
            texts = recs.get(sid, [])
            if not texts or sid not in per_seed[name]:
                continue
            sub = texts[:args.lexical_items]
            per_seed[name][sid].update({
                "distinct2": M.distinct_n(texts, 2),
                "self_bleu": M.self_bleu(sub, max_items=args.lexical_items),
                "rougeL_seed": M.rouge_l_max(sub, [seeds[sid]["seed_abstract"]]),
                "mean_len": float(np.mean([len(t.split()) for t in texts])),
            })
        print(f"lexical metrics done for {name}", flush=True)

    # ---- A: per-seed correlations
    corr = {}
    pooled = defaultdict(lambda: ([], []))
    for name in gen_systems:
        corr[name] = {}
        for alt in ALT:
            x = [r.get(alt, np.nan) for r in per_seed[name].values()]
            y = [r["coverage"] for r in per_seed[name].values()]
            x, y = np.array(x), np.array(y)
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() >= 10:
                rho, p = spearmanr(x[ok], y[ok])
                corr[name][alt] = {"rho": float(rho), "p": float(p), "n": int(ok.sum())}
                pooled[alt][0].extend(x[ok].tolist()); pooled[alt][1].extend(y[ok].tolist())
    corr["pooled"] = {}
    for alt, (x, y) in pooled.items():
        if len(x) >= 10:
            rho, p = spearmanr(x, y)
            corr["pooled"][alt] = {"rho": float(rho), "p": float(p), "n": len(x)}

    # ---- B: system-level table + rankings
    table = {}
    for name in list(per_seed):
        rows = list(per_seed[name].values())
        table[name] = {k: float(np.nanmean([r.get(k, np.nan) for r in rows]))
                       for k in ["coverage"] + ALT}
    rankings = {}
    by_model = defaultdict(list)
    for name in gen_systems:
        by_model[name.split("/")[0]].append(name)
    for model, names in by_model.items():
        rankings[model] = {k: sorted(names, key=lambda n: -table[n][k] if k not in ("mps", "self_bleu", "seed_sim", "rougeL_seed") else table[n][k])
                           for k in ["coverage"] + ALT}

    # ---- C: coverage by target seed-similarity decile
    tau = json.load(open(args.score_dir / f"tau_star_{enc}.json"))["tau_star"]
    with gzip.open(args.score_dir / f"{enc}_per_pair.csv.gz", "rt") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [r for r in rd]
    sys_cols = header[8:]
    seed_sim = np.array([float(r[2]) for r in rows])
    best = {n: np.array([float(r[8 + j]) for r in rows]) for j, n in enumerate(sys_cols)}
    edges = np.percentile(seed_sim, np.linspace(0, 100, 11))
    dec = np.clip(np.searchsorted(edges, seed_sim, side="right") - 1, 0, 9)
    by_decile = {n: [float(np.nanmean(best[n][dec == d] >= tau)) for d in range(10)] for n in sys_cols}
    decile_edges = [float(e) for e in edges]

    out = {"encoder": args.encoder, "tau_star": tau,
           "A_spearman_coverage_vs_alt": corr, "B_system_table": table, "B_rankings": rankings,
           "C_coverage_by_target_seed_sim_decile": {"decile_edges": decile_edges, "systems": by_decile}}
    with open(args.out_dir / f"{enc}.json", "w") as f:
        json.dump(out, f, indent=1)
    make_figures(args.out_dir, enc, corr, by_decile, gen_systems)

    print(f"\nA. pooled Spearman rho(coverage, metric) over {len(gen_systems)} systems")
    for alt, r in corr["pooled"].items():
        print(f"  {alt:<12} rho={r['rho']:+.3f}  p={r['p']:.1e}  n={r['n']}")
    print("\nB. system table")
    print(f"  {'system':<40} " + " ".join(f"{k:>9}" for k in ['coverage'] + ALT))
    for name, t in table.items():
        print(f"  {name:<40} " + " ".join(f"{t[k]:>9.3f}" for k in ['coverage'] + ALT))
    print(f"-> {args.out_dir}/{enc}.json")


def make_figures(out_dir, enc, corr, by_decile, gen_systems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    names = [n for n in gen_systems if n in corr] + ["pooled"]
    mat = np.array([[corr[n].get(a, {}).get("rho", np.nan) for a in ALT] for n in names])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ALT))); ax.set_xticklabels(ALT, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("A: Spearman rho(coverage@tau*, metric) per seed")
    ax = axes[1]
    for n, v in by_decile.items():
        style = "--" if n.startswith(("null/", "ref/")) else "-"
        ax.plot(range(1, 11), v, style, marker="o", ms=3, label=n, lw=1)
    ax.set_xlabel("target's similarity-to-seed decile (1 = farthest)")
    ax.set_ylabel("coverage@tau*"); ax.set_title("C: which citers get covered?")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"contrast_{enc}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
