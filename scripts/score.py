#!/usr/bin/env python
"""Score generation arms against the real citing abstracts, next to null and
reference systems, and record the diversity of every prediction set.

    python scripts/score.py --data data/acl_a2a.jsonl --gen-root results/gen --out-dir results/score

tau* is the 95th percentile of the random-pool null's best similarity, computed on the
whole benchmark every run and written to <out-dir>/tau_star_<encoder>.json. The null
draws real citers of *other* seeds and involves no system, so there is nothing to hold
out (DECISIONS.md 2026-09-19); --tau-star overrides it for sensitivity checks.

Systems scored per seed (prediction set -> its real citers):
  <model>/<arm>     generations found under <gen-root>/<model>/<arm>.jsonl
  null/seed_copy    the seed abstract as the single prediction
  null/random_pool  N random citers of OTHER seeds (chance level; defines tau*)
  ref/retrieval_knn N citers of other seeds nearest to the seed (what topical
                    proximity alone buys, using real text)
  ref/split_half    half of the seed's own citers predicting the other half
                    (what N *real* futures of the same paper cover)

Per system: pooled and per-seed coverage@tau* (bootstrap CI over seeds), by
stratum (field match, ACL vs non-ACL citer, horizon, post-cutoff citers,
influential flag), coverage@N, the tau-grid curve, and prediction-set
diversity (Vendi, mean pairwise similarity, centroid dispersion, similarity
to the seed). Writes <encoder>.json, <encoder>_per_seed.csv and
<encoder>_per_pair.csv.gz (best similarity per pair per system, consumed by
scripts/contrast.py).
"""
import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a import metrics as M  # noqa: E402
from a2a.corpus import load_jsonl  # noqa: E402
from a2a.embed import DEFAULT_ENCODER, embed_cached, embed_texts, load_encoder, short_name  # noqa: E402
from a2a.generate import load_generations  # noqa: E402

N_GRID = (1, 5, 10, 20, 50)
CUTOFF_YEAR = 2025  # citers from this year on post-date every generator's training data
HORIZONS = [(0, 1, "0-1y"), (2, 3, "2-3y"), (4, 6, "4-6y"), (7, 99, "7y+")]


def hbucket(h):
    for lo, hi, n in HORIZONS:
        if lo <= h <= hi:
            return n
    return "0-1y" if h < 0 else "7y+"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/acl_a2a.jsonl"))
    ap.add_argument("--gen-root", type=Path, default=Path("results/gen"))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--emb-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/score"))
    ap.add_argument("--tau-star", type=float, default=None,
                    help="override the calibrated tau* (sensitivity checks only)")
    ap.add_argument("--n", type=int, default=50, help="prediction budget for nulls/refs")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    enc = short_name(args.encoder)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    seeds = load_jsonl(args.data)
    sid_index = {s["seed_id"]: i for i, s in enumerate(seeds)}
    print(f"{len(seeds)} seeds", flush=True)

    # ---- generations: <gen-root>/<model>/<arm>.jsonl
    systems = {}  # name -> {seed_id: [texts]}
    for mdir in sorted(p for p in args.gen_root.glob("*") if p.is_dir()) if args.gen_root.exists() else []:
        for arm, recs in load_generations(mdir).items():
            systems[f"{mdir.name}/{arm}"] = recs
    print(f"{len(systems)} generation systems: {list(systems)}", flush=True)

    # ---- embeddings
    cache = args.emb_dir / enc / "texts.npz"
    E_seed = embed_cached([s["seed_abstract"] for s in seeds], args.encoder, cache)
    citer_texts = [c["abstract"] for s in seeds for c in s["citers"]]
    E_cit = embed_cached(citer_texts, args.encoder, cache)
    offsets, k = [], 0
    for s in seeds:
        offsets.append((k, k + len(s["citers"])))
        k += len(s["citers"])
    model = load_encoder(args.encoder)
    gen_emb = {}
    for name, recs in systems.items():
        texts = [t for s in seeds for t in recs.get(s["seed_id"], [])]
        print(f"embedding {len(texts)} generations for {name}", flush=True)
        E = embed_texts(texts, args.encoder, model=model, show_progress=False)
        gen_emb[name], k = {}, 0
        for s in seeds:
            n = len(recs.get(s["seed_id"], []))
            gen_emb[name][s["seed_id"]] = E[k:k + n]
            k += n

    # ---- pair covariates
    pairs = []
    for i, s in enumerate(seeds):
        sf = set(s["seed_fields"])
        for j, c in enumerate(s["citers"]):
            h = (c["year"] or s["seed_year"]) - s["seed_year"]
            pairs.append({"seed_id": s["seed_id"], "citer_id": c["id"],
                          "seed_sim": float(E_cit[offsets[i][0] + j] @ E_seed[i]),
                          "same_field": bool(sf & set(c["fields"])),
                          "citer_acl": c["acl_id"] is not None,
                          "horizon": hbucket(h),
                          "post_cutoff": (c["year"] or 0) >= CUTOFF_YEAR,
                          "influential": bool(c["isinfluential"])})
    strata = {"all": np.ones(len(pairs), bool)}
    for key in ("same_field", "citer_acl", "post_cutoff", "influential"):
        v = np.array([p[key] for p in pairs])
        strata[f"{key}=1"], strata[f"{key}=0"] = v, ~v
    for _, _, name in HORIZONS:
        strata[f"horizon={name}"] = np.array([p["horizon"] == name for p in pairs])

    # ---- prediction embeddings per system per seed (nulls/refs built here)
    all_idx = np.arange(len(citer_texts))

    def other_citers(i):
        lo, hi = offsets[i]
        return np.concatenate([all_idx[:lo], all_idx[hi:]])

    def preds_for(name, i):
        s = seeds[i]
        if name == "null/seed_copy":
            return E_seed[i:i + 1]
        if name == "null/random_pool":
            return E_cit[rng.choice(other_citers(i), size=args.n, replace=False)]
        if name == "ref/retrieval_knn":
            oc = other_citers(i)
            top = np.argpartition(-(E_cit[oc] @ E_seed[i]), args.n)[:args.n]
            return E_cit[oc[top]]
        return gen_emb[name][s["seed_id"]]

    names = list(systems) + ["null/seed_copy", "null/random_pool", "ref/retrieval_knn"]
    best = {n: np.full(len(pairs), np.nan) for n in names}
    per_seed = defaultdict(dict)  # name -> seed_id -> metrics
    for i, s in enumerate(seeds):
        lo, hi = offsets[i]
        T = E_cit[lo:hi]
        for name in names:
            P = preds_for(name, i)
            if P.shape[0] == 0:
                continue
            sim = P @ T.T
            best[name][lo:hi] = sim.max(axis=0)
            rec = {"n_pred": int(P.shape[0]),
                   "vendi": M.vendi_score(P @ P.T) if P.shape[0] > 1 else 1.0,
                   "mps": M.mean_pairwise_sim(P), "disp": M.centroid_dispersion(P),
                   "seed_sim": float((P @ E_seed[i]).mean())}
            per_seed[name][s["seed_id"]] = rec
            per_seed[name][s["seed_id"]]["_sim"] = sim
        # real target set at matched budget (diversity reference)
        idx = rng.choice(T.shape[0], size=min(args.n, T.shape[0]), replace=False)
        Ts = T[idx]
        per_seed["ref/target_set"][s["seed_id"]] = {
            "n_pred": int(Ts.shape[0]), "vendi": M.vendi_score(Ts @ Ts.T),
            "mps": M.mean_pairwise_sim(Ts), "disp": M.centroid_dispersion(Ts),
            "seed_sim": float((Ts @ E_seed[i]).mean())}

    # ---- tau*
    tau_file = args.out_dir / f"tau_star_{enc}.json"
    if args.tau_star is not None:
        tau = args.tau_star
    else:
        tau = M.calibrate_tau(best["null/random_pool"], 95)
        json.dump({"tau_star": tau, "encoder": args.encoder, "n_seeds": len(seeds),
                   "rule": "95th percentile of random-pool best similarity", "n": args.n},
                  open(tau_file, "w"), indent=1)
    print(f"tau* = {tau:.4f} ({enc})", flush=True)

    # ---- coverage per system
    results = {}
    for name in names:
        b = best[name]
        by_stratum = {st: {"n": int(mask.sum()), "coverage": float(np.nanmean(b[mask] >= tau))}
                      for st, mask in strata.items() if mask.sum()}
        ps = []
        for i, s in enumerate(seeds):
            r = per_seed[name].get(s["seed_id"])
            if r is None:
                continue
            sim = r.pop("_sim")
            r["coverage"] = M.coverage_at(sim, tau)
            r.update({f"cov@{n}": v for n, v in M.coverage_at_n(sim, tau, N_GRID).items()})
            ps.append(r["coverage"])
        mean, ci = M.bootstrap_mean_ci(ps, args.n_boot, args.seed)
        results[name] = {
            "pooled": by_stratum,
            "per_seed_coverage_mean": mean, "per_seed_coverage_ci95": list(ci),
            "coverage_curve": M.coverage_curve(b[~np.isnan(b)]),
            "coverage_at_n": {str(n): float(np.nanmean([r.get(f"cov@{n}", np.nan)
                                                        for r in per_seed[name].values()]))
                              for n in N_GRID},
            "diversity": {k: float(np.nanmean([r[k] for r in per_seed[name].values()]))
                          for k in ("n_pred", "vendi", "mps", "disp", "seed_sim")},
        }
    # split-half reference (per-seed only) + target-set diversity
    sh = [M.split_half_coverage(E_cit[lo:hi], tau, args.n, rng) for lo, hi in offsets]
    mean, ci = M.bootstrap_mean_ci(sh, args.n_boot, args.seed)
    results["ref/split_half"] = {"per_seed_coverage_mean": mean, "per_seed_coverage_ci95": list(ci)}
    results["ref/target_set"] = {"diversity": {
        k: float(np.nanmean([r[k] for r in per_seed["ref/target_set"].values()]))
        for k in ("n_pred", "vendi", "mps", "disp", "seed_sim")}}
    for i, s in enumerate(seeds):
        per_seed["ref/split_half"][s["seed_id"]] = {"coverage": sh[i]}

    out = {"encoder": args.encoder, "tau_star": tau, "n": args.n,
           "n_seeds": len(seeds), "n_pairs": len(pairs),
           "strata_sizes": {k: int(v.sum()) for k, v in strata.items()}, "systems": results}
    with open(args.out_dir / f"{enc}.json", "w") as f:
        json.dump(out, f, indent=1)

    cols = ["system", "seed_id", "n_pred", "coverage", "vendi", "mps", "disp", "seed_sim"] + \
        [f"cov@{n}" for n in N_GRID]
    with open(args.out_dir / f"{enc}_per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for name, recs in per_seed.items():
            for sid, r in recs.items():
                w.writerow({"system": name, "seed_id": sid, **r})
    with gzip.open(args.out_dir / f"{enc}_per_pair.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed_id", "citer_id", "seed_sim", "same_field", "citer_acl", "horizon",
                    "post_cutoff", "influential"] + names)
        for j, p in enumerate(pairs):
            w.writerow([p["seed_id"], p["citer_id"], f"{p['seed_sim']:.4f}", int(p["same_field"]),
                        int(p["citer_acl"]), p["horizon"], int(p["post_cutoff"]),
                        int(p["influential"])] + [f"{best[n][j]:.4f}" for n in names])

    print(f"\n{'system':<40} {'cov@tau*':>9} {'CI95':>16} {'cross':>6} {'acl=0':>6} "
          f"{'post25':>6} {'vendi':>6} {'mps':>6}")
    for name, r in results.items():
        p = r.get("pooled", {})
        d = r.get("diversity", {})
        ci = r.get("per_seed_coverage_ci95", [np.nan, np.nan])
        print(f"{name:<40} {r.get('per_seed_coverage_mean', np.nan):>9.3f} "
              f"[{ci[0]:.3f},{ci[1]:.3f}] "
              f"{p.get('same_field=0', {}).get('coverage', np.nan):>6.3f} "
              f"{p.get('citer_acl=0', {}).get('coverage', np.nan):>6.3f} "
              f"{p.get('post_cutoff=1', {}).get('coverage', np.nan):>6.3f} "
              f"{d.get('vendi', np.nan):>6.2f} {d.get('mps', np.nan):>6.3f}")
    print(f"-> {args.out_dir}/{enc}.json")


if __name__ == "__main__":
    main()
