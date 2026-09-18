#!/usr/bin/env python
"""Characterise the target sets of the ACL-seed benchmark: how many futures a paper
has, how far they sit from the seed, and how spread out they are.

One question per section, a fixed handful of metrics, no model in the loop.

    python scripts/characterize.py --data data/acl_a2a.jsonl \
        [--corpus-root /data/derived/acl_corpus] [--encoders malteos/scincl,...] \
        --out-dir results/characterize

Q1 (corpus-level, needs --corpus-root): how many citers with abstracts do ACL
   papers have, and who cites them (ACL vs non-ACL venues, fields)?
Q2 (benchmark): seed -> citer cosine similarity, by field match, venue type,
   horizon and citation intent.
Q3 (benchmark): per-seed spread of the target set (Vendi, mean pairwise
   similarity, centroid dispersion) against a size-matched random set of
   other seeds' citers.
Q4 (benchmark): horizon (years after the seed) distribution.

Outputs ``summary.json`` and PNG figures under --out-dir. Embeddings are
cached under results/embeddings/<encoder>/ and reused by scripts/score.py.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a import metrics as M  # noqa: E402
from a2a.corpus import load_jsonl  # noqa: E402
from a2a.embed import DEFAULT_ENCODER, embed_cached, short_name  # noqa: E402

HORIZON_BUCKETS = [(0, 1, "0-1y"), (2, 3, "2-3y"), (4, 6, "4-6y"), (7, 99, "7y+")]


def q(v, ps=(10, 25, 50, 75, 90)):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {}
    return {"n": int(v.size), "mean": float(v.mean()),
            **{f"p{p}": float(np.percentile(v, p)) for p in ps}}


def horizon_bucket(h):
    for lo, hi, name in HORIZON_BUCKETS:
        if lo <= h <= hi:
            return name
    return "neg" if h < 0 else "7y+"


def corpus_level(corpus_root: Path, year_min: int, year_max: int) -> dict:
    """Q1 over ALL ACL seeds in the year window, from the corpus parquet."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET threads=16")
    seeds, edges, citers = (corpus_root / "seeds.parquet", corpus_root / "edges" / "bucket_*.parquet",
                            corpus_root / "citers.parquet")
    base = f"""
        FROM read_parquet('{edges}') e
        JOIN read_parquet('{seeds}') s ON s.corpusid = e.citedcorpusid
        JOIN read_parquet('{citers}') c ON c.corpusid = e.citingcorpusid
        WHERE s.year BETWEEN {year_min} AND {year_max}
    """
    out = {}
    out["n_seeds_in_window"] = con.execute(
        f"SELECT count(*) FROM read_parquet('{seeds}') WHERE year BETWEEN {year_min} AND {year_max}"
    ).fetchone()[0]
    out["n_edges"], out["n_edges_with_abstract"] = con.execute(
        f"SELECT count(*), count(c.abstract) {base}").fetchone()
    per_seed = con.execute(
        f"SELECT s.corpusid, count(c.abstract) n {base} GROUP BY 1").fetchall()
    n_c = np.array([n for _, n in per_seed])
    bins = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 499), (500, 10**9)]
    n_all = out["n_seeds_in_window"]
    out["citers_with_abstract_per_seed"] = {
        f"{lo}-{hi if hi < 10**9 else 'inf'}": int(((n_c >= lo) & (n_c <= hi)).sum())
        for lo, hi in bins}
    out["citers_with_abstract_per_seed"]["0-0"] += int(n_all - len(per_seed))
    out["citers_with_abstract_per_seed_quantiles"] = q(np.concatenate([n_c, np.zeros(n_all - len(per_seed))]))
    out["citer_is_acl_frac"] = con.execute(
        f"SELECT avg(CASE WHEN c.acl_id IS NULL THEN 0 ELSE 1 END) {base} AND c.abstract IS NOT NULL"
    ).fetchone()[0]
    out["citer_top_fields"] = con.execute(f"""
        SELECT f, count(*) n FROM (SELECT unnest(c.s2fieldsofstudy) f {base}
        AND c.abstract IS NOT NULL) GROUP BY 1 ORDER BY n DESC LIMIT 15""").fetchall()
    out["citer_top_nonacl_venues"] = con.execute(f"""
        SELECT c.venue, count(*) n {base} AND c.abstract IS NOT NULL AND c.acl_id IS NULL
        AND c.venue IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT 20""").fetchall()
    out["citer_same_field_frac"] = con.execute(f"""
        SELECT avg(CASE WHEN list_has_any(c.s2fieldsofstudy, s.s2fieldsofstudy) THEN 1 ELSE 0 END)
        {base} AND c.abstract IS NOT NULL""").fetchone()[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/acl_a2a.jsonl"))
    ap.add_argument("--corpus-root", type=Path, default=None)
    ap.add_argument("--encoders", default=f"{DEFAULT_ENCODER},sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out-dir", type=Path, default=Path("results/characterize"))
    ap.add_argument("--emb-dir", type=Path, default=Path("results/embeddings"))
    ap.add_argument("--n-random-repeats", type=int, default=3)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_jsonl(args.data)
    summary = {"n_seeds": len(seeds), "n_pairs": sum(len(s["citers"]) for s in seeds)}
    years = [s["seed_year"] for s in seeds]
    summary["seed_year_counts"] = dict(sorted(Counter(years).items()))
    summary["n_citers_qualifying"] = q([s["n_citers_qualifying"] for s in seeds])
    summary["citers_sampled_per_seed"] = q([len(s["citers"]) for s in seeds])
    if args.corpus_root:
        yrs = (min(years), max(years))
        summary["corpus"] = corpus_level(args.corpus_root, *yrs)

    # ---- per-pair covariates
    pairs = []
    for s in seeds:
        sf = set(s["seed_fields"])
        for c in s["citers"]:
            pairs.append({
                "seed_id": s["seed_id"], "citer_id": c["id"],
                "same_field": bool(sf & set(c["fields"])),
                "citer_acl": c["acl_id"] is not None,
                "horizon": (c["year"] or s["seed_year"]) - s["seed_year"],
                "influential": c["isinfluential"],
                "intent": "/".join(sorted(c["intents"])) or "unknown",
                "citer_citations": c["citationcount"] or 0,
            })
    summary["pair_covariates"] = {
        "same_field_frac": float(np.mean([p["same_field"] for p in pairs])),
        "citer_acl_frac": float(np.mean([p["citer_acl"] for p in pairs])),
        "influential_frac": float(np.mean([p["influential"] for p in pairs])),
        "horizon": q([p["horizon"] for p in pairs]),
        "horizon_buckets": dict(Counter(horizon_bucket(p["horizon"]) for p in pairs)),
        "intent_counts": dict(Counter(p["intent"] for p in pairs).most_common(12)),
        "citer_citations": q([p["citer_citations"] for p in pairs]),
    }

    rng = np.random.default_rng(0)
    for encoder in args.encoders.split(","):
        enc = short_name(encoder)
        cache = args.emb_dir / enc / "texts.npz"
        seed_texts = [s["seed_abstract"] for s in seeds]
        citer_texts = [c["abstract"] for s in seeds for c in s["citers"]]
        print(f"[{enc}] embedding {len(seed_texts)} seeds + {len(citer_texts)} citers", flush=True)
        E_seed = embed_cached(seed_texts, encoder, cache)
        E_cit = embed_cached(citer_texts, encoder, cache)

        # Q2: seed -> citer similarity by stratum
        sims, k = [], 0
        for i, s in enumerate(seeds):
            n = len(s["citers"])
            sims.extend((E_cit[k:k + n] @ E_seed[i]).tolist())
            k += n
        sims = np.array(sims)
        q2 = {"all": q(sims)}
        for key, val in [("same_field", True), ("same_field", False), ("citer_acl", True),
                         ("citer_acl", False), ("influential", True), ("influential", False)]:
            mask = np.array([p[key] == val for p in pairs])
            q2[f"{key}={val}"] = q(sims[mask])
        for _, _, name in HORIZON_BUCKETS:
            mask = np.array([horizon_bucket(p["horizon"]) == name for p in pairs])
            q2[f"horizon={name}"] = q(sims[mask])
        for intent, _ in Counter(p["intent"] for p in pairs).most_common(6):
            mask = np.array([p["intent"] == intent for p in pairs])
            q2[f"intent={intent}"] = q(sims[mask])

        # Q3: spread of the target set vs size-matched random sets
        per_seed, k = [], 0
        for i, s in enumerate(seeds):
            n = len(s["citers"])
            e = E_cit[k:k + n]
            rec = {"seed_id": s["seed_id"], "n": n,
                   "vendi": M.vendi_score(e @ e.T), "mps": M.mean_pairwise_sim(e),
                   "disp": M.centroid_dispersion(e),
                   "seed_sim_mean": float((e @ E_seed[i]).mean())}
            rv, rm, rd = [], [], []
            for _ in range(args.n_random_repeats):
                others = np.setdiff1d(np.arange(len(citer_texts)), np.arange(k, k + n))
                idx = rng.choice(others, size=n, replace=False)
                er = E_cit[idx]
                rv.append(M.vendi_score(er @ er.T)); rm.append(M.mean_pairwise_sim(er))
                rd.append(M.centroid_dispersion(er))
            rec.update({"vendi_random": float(np.mean(rv)), "mps_random": float(np.mean(rm)),
                        "disp_random": float(np.mean(rd))})
            per_seed.append(rec)
            k += n
        q3 = {key: q([r[key] for r in per_seed]) for key in
              ("vendi", "vendi_random", "mps", "mps_random", "disp", "disp_random", "seed_sim_mean")}
        q3["vendi_ratio_real_over_random"] = q([r["vendi"] / r["vendi_random"] for r in per_seed])
        summary[enc] = {"q2_seed_citer_similarity": q2, "q3_target_spread": q3}

        with open(args.out_dir / f"per_seed_{enc}.json", "w") as f:
            json.dump(per_seed, f)
        np.save(args.out_dir / f"pair_seed_sim_{enc}.npy", sims)
        make_figures(args.out_dir, enc, sims, pairs, per_seed)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=1))
    print(f"-> {args.out_dir}/summary.json")


def make_figures(out_dir: Path, enc: str, sims, pairs, per_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    same = np.array([p["same_field"] for p in pairs])
    ax.hist(sims[same], bins=40, alpha=0.6, density=True, label="same field")
    ax.hist(sims[~same], bins=40, alpha=0.6, density=True, label="cross field")
    ax.set_xlabel(f"seed→citer cosine ({enc})"); ax.set_ylabel("density"); ax.legend()
    ax.set_title("Q2: how far do citers sit from the seed?")
    ax = axes[1]
    hs = np.array([p["horizon"] for p in pairs])
    order = [n for _, _, n in HORIZON_BUCKETS]
    data = [sims[np.array([horizon_bucket(h) == n for h in hs])] for n in order]
    ax.boxplot([d for d in data if d.size], labels=[n for n, d in zip(order, data) if d.size],
               showfliers=False)
    ax.set_xlabel("years after seed"); ax.set_ylabel("seed→citer cosine")
    ax.set_title("Q4: similarity by horizon")
    ax = axes[2]
    ax.scatter([r["vendi_random"] for r in per_seed], [r["vendi"] for r in per_seed], s=8, alpha=0.5)
    lim = max(max(r["vendi_random"] for r in per_seed), max(r["vendi"] for r in per_seed))
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("Vendi of size-matched random citers"); ax.set_ylabel("Vendi of the seed's citers")
    ax.set_title("Q3: spread of the target set")
    fig.tight_layout()
    fig.savefig(out_dir / f"characterize_{enc}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
