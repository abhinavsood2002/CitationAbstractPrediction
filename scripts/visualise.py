#!/usr/bin/env python
"""Plot the distribution of the benchmark cut (``data/acl_a2a.jsonl``) as PNGs.

    python scripts/visualise.py [--data data/acl_a2a.jsonl] [--out-dir images]

One pass over the JSONL extracts per-seed and per-pair arrays (cached as an ``.npz``
under ``results/visualise/`` so re-plotting is instant). Figures, light surface,
one hue per entity (seeds blue, citers / pairs orange):

    seeds_per_year.png       seeds by publication year
    seed_citations.png       S2 citation count of seeds (log bins)
    citers_per_seed.png      qualifying citers per seed (log bins)
    pair_concentration.png   cumulative share of pairs vs seeds ranked by citer count
    citer_years.png          citing papers by year
    horizon.png              citer year minus seed year
    abstract_lengths.png     seed vs citer abstract length
    citer_fields.png         top citer fields of study (share of pairs)
    benchmark_overview.png   the six headline panels on one page
    intent_similarity.png    citer-seed cosine by citation intent: quantile rows against a
                             random-other-seed null, and the share of pairs above its 95th pct
    intent_similarity_hist.png  the same distributions as small-multiple histograms

The two intent figures embed every abstract with ``--encoder`` (default Qwen3-Embedding) through the
``results/embeddings/<enc>/texts.npz`` cache shared with characterize.py / score.py, so the
first run needs a GPU (or ``--no-similarity``); per-pair cosines are then cached next to the
array cache and re-plotting is instant again.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a.embed import DEFAULT_ENCODER, embed_cached, short_name  # noqa: E402

# ---- palette (dataviz reference instance, light mode) -----------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEED, CITER = "#2a78d6", "#eb6834"   # categorical slots 1 and 2; validated adjacent pair

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 14,
})

LOG_EDGES = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10_000, 10**9]

# S2 citation intents, stored per pair as a bitmask so the array cache stays numeric.
INTENT_BITS = {"background": 1, "methodology": 2, "result": 4}
INTENT_NAMES = {0: "no label", 1: "background", 2: "methodology", 3: "background + methodology",
                4: "result", 5: "background + result", 6: "methodology + result",
                7: "background + methodology + result"}
ENCODER_LABEL = {"qwen3-embedding-0.6b": "Qwen3-Embedding", "all-minilm-l6-v2": "MiniLM"}


# ---- data ---------------------------------------------------------------------------

def extract(data: Path) -> dict:
    """One streaming pass over the JSONL -> flat numpy arrays + a citer-field counter."""
    seed_year, seed_cc, seed_n, seed_len = [], [], [], []
    citer_year, citer_len, citer_acl, citer_seed = [], [], [], []
    citer_intent, citer_infl = [], []
    fields: Counter = Counter()
    with open(data) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            seed_year.append(r["seed_year"])
            seed_cc.append(r["seed_citationcount"])
            seed_n.append(len(r["citers"]))
            seed_len.append(len(r["seed_abstract"]))
            for c in r["citers"]:
                citer_year.append(c["year"])
                citer_len.append(len(c["abstract"]))
                citer_acl.append(c["acl_id"] is not None)
                citer_seed.append(i)
                citer_intent.append(sum(INTENT_BITS.get(x, 0) for x in c["intents"]))
                citer_infl.append(bool(c["isinfluential"]))
                fields.update(set(c["fields"]))
    names, counts = zip(*fields.most_common()) if fields else ((), ())
    return {
        "seed_year": np.array(seed_year), "seed_cc": np.array(seed_cc),
        "seed_n": np.array(seed_n), "seed_len": np.array(seed_len),
        "citer_year": np.array(citer_year),
        "citer_len": np.array(citer_len), "citer_acl": np.array(citer_acl),
        "citer_seed": np.array(citer_seed, dtype=np.int32),
        "citer_intent": np.array(citer_intent, dtype=np.uint8),
        "citer_infl": np.array(citer_infl),
        "field_names": np.array(names), "field_counts": np.array(counts),
    }


def load(data: Path, cache: Path) -> dict:
    if cache.exists() and cache.stat().st_mtime >= data.stat().st_mtime:
        with np.load(cache, allow_pickle=False) as z:
            if "citer_intent" in z.files and "seed_dev" not in z.files:
                return {k: z[k] for k in z.files}
        print("array cache predates the current schema; re-extracting", flush=True)
    print(f"extracting arrays from {data} (one pass, a couple of minutes)", flush=True)
    d = extract(data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:   # file handle: np.savez would otherwise append .npz
        np.savez(fh, **d)
    tmp.rename(cache)
    return d


def pair_similarity(data: Path, encoder: str, emb_dir: Path, cache: Path) -> dict:
    """Per pair (file order, same as ``extract``): cosine(seed, citer) and a null
    cosine(random *other* seed, citer). Embeds through the cache shared with
    characterize.py / score.py; the per-pair cosines are cached in ``cache``."""
    if cache.exists() and cache.stat().st_mtime >= data.stat().st_mtime:
        with np.load(cache, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    seed_texts, citer_texts, seed_idx, citer_idx = [], {}, [], []
    with open(data) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            seed_texts.append(r["seed_abstract"])
            for c in r["citers"]:
                citer_idx.append(citer_texts.setdefault(c["abstract"], len(citer_texts)))
                seed_idx.append(i)
    citer_texts = list(citer_texts)
    emb_cache = emb_dir / short_name(encoder) / "texts.npz"
    print(f"[{short_name(encoder)}] embedding {len(seed_texts):,} seeds + "
          f"{len(citer_texts):,} distinct citer abstracts (cache: {emb_cache})", flush=True)
    E_seed = embed_cached(seed_texts, encoder, emb_cache)
    E_cit = embed_cached(citer_texts, encoder, emb_cache)
    seed_idx = np.array(seed_idx, dtype=np.int32)
    citer_idx = np.array(citer_idx, dtype=np.int32)
    n_seeds = len(seed_texts)
    rng = np.random.default_rng(0)
    null_idx = (seed_idx + rng.integers(1, n_seeds, size=len(seed_idx))) % n_seeds  # never own seed
    sim = np.empty(len(seed_idx), dtype=np.float32)
    null_sim = np.empty_like(sim)
    for a in range(0, len(seed_idx), 100_000):
        b = slice(a, a + 100_000)
        C = E_cit[citer_idx[b]]
        sim[b] = np.einsum("ij,ij->i", C, E_seed[seed_idx[b]])
        null_sim[b] = np.einsum("ij,ij->i", C, E_seed[null_idx[b]])
    out = {"sim": sim, "null_sim": null_sim}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh, **out)
    tmp.rename(cache)
    return out


# ---- helpers ------------------------------------------------------------------------

def title(ax, main: str, sub: str | None = None):
    ax.set_title(main)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")


def log_bin_labels(edges=LOG_EDGES):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo == 1:
            out.append(f"{lo:,}")
        elif hi >= 10**9:
            out.append(f"{lo:,}+")
        else:
            out.append(f"{lo:,}-{hi - 1:,}")
    return out


def log_hist(ax, values, color, xlabel, sub=None, main="", mark_floor=None):
    counts, _ = np.histogram(values, bins=LOG_EDGES)
    labels = log_bin_labels()
    nz = np.flatnonzero(counts)
    lo, hi = int(nz[0]), int(nz[-1]) + 1      # trim empty bins at both ends
    counts, labels = counts[lo:hi], labels[lo:hi]
    x = np.arange(len(counts))
    ax.bar(x, counts, width=0.62, color=color, linewidth=0)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("seeds")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    med = float(np.median(values))
    i_max = int(np.argmax(counts))
    ax.text(x[i_max], counts[i_max], f" {counts[i_max]:,}", ha="center", va="bottom",
            color=INK2, fontsize=8)
    note = f"median {med:,.0f}"
    if mark_floor is not None:
        note += f" · floor {mark_floor:,}"
    title(ax, main, (sub + " · " if sub else "") + note)


def year_bars(ax, years, color, ylabel, main, sub=None, start=1990):
    years = np.asarray(years)
    pre = int((years < start).sum())
    yr = np.arange(start, years.max() + 1)
    counts = np.array([(years == y).sum() for y in yr])
    labels = [f"<{start}"] + [str(y) for y in yr]
    vals = np.concatenate([[pre], counts])
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.62, color=color, linewidth=0)
    ticks = [0] + [i for i, lab in enumerate(labels)
                   if lab.isdigit() and int(lab) % 5 == 0 and int(lab) > start]
    ax.set_xticks(ticks, [labels[i] for i in ticks], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    i_max = int(np.argmax(vals))
    ax.text(x[i_max], vals[i_max], f" {labels[i_max]}: {vals[i_max]:,}", ha="center",
            va="bottom", color=INK2, fontsize=8)
    title(ax, main, sub)


def concentration(ax, seed_n, main="Pairs are concentrated in a few seeds"):
    n = np.sort(seed_n)[::-1]
    cum = np.cumsum(n) / n.sum() * 100
    frac = np.arange(1, len(n) + 1) / len(n) * 100
    ax.plot(frac, cum, color=SEED, linewidth=2, solid_joinstyle="round")
    ax.plot([0, 100], [0, 100], color=AXIS, linewidth=0.8)
    for pct in (1, 10):
        k = max(1, int(round(len(n) * pct / 100)))
        ax.plot(frac[k - 1], cum[k - 1], "o", ms=8, color=SEED, markeredgecolor=SURFACE,
                markeredgewidth=2)
        above = cum[k - 1] < 15                     # keep the label off the baseline
        ax.text(frac[k - 1] + 2, cum[k - 1] + (3 if above else -3),
                f"top {pct}% of seeds hold {cum[k - 1]:.0f}% of pairs",
                color=INK2, fontsize=8, va="bottom" if above else "top")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("seeds, ranked by number of citers (%)")
    ax.set_ylabel("cumulative share of pairs (%)")
    title(ax, main, f"{len(n):,} seeds, {int(n.sum()):,} pairs; largest seed {int(n[0]):,} citers")


def horizon_bars(ax, h, main="Citation horizon", fold=16):
    h = np.asarray(h)
    top = fold
    counts = np.array([(h == k).sum() for k in range(top)] + [(h >= top).sum()])
    labels = [str(k) for k in range(top)] + [f"{top}+"]
    x = np.arange(len(counts))
    ax.bar(x, counts, width=0.62, color=CITER, linewidth=0)
    shown = [i for i in x if i % 2 == 0 and i < top] + [len(x) - 1]
    ax.set_xticks(shown, [labels[i] for i in shown], fontsize=8)
    ax.set_xlabel("citer year minus seed year")
    ax.set_ylabel("pairs")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    i_max = int(np.argmax(counts))
    ax.text(x[i_max], counts[i_max], f" {counts[i_max]:,}", ha="center", va="bottom",
            color=INK2, fontsize=8)
    title(ax, main, f"median {np.median(h):.0f} years · 90th percentile {np.percentile(h, 90):.0f}")


def length_panels(axes, seed_len, citer_len, lo=200, hi=5000):
    bins = np.arange(lo, hi + 100, 100)
    for ax, vals, color, name in [(axes[0], seed_len, SEED, "Seed abstracts"),
                                  (axes[1], citer_len, CITER, "Citer abstracts")]:
        counts, _ = np.histogram(vals, bins=bins)
        ax.bar(bins[:-1], counts, width=88, align="edge", color=color, linewidth=0)
        ax.set_ylabel("count")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if v < 10_000 else f"{v / 1000:,.0f}k"))
        title(ax, name, f"n = {len(vals):,} · median {np.median(vals):,.0f} characters")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[1].set_xlabel(f"abstract length (characters; cut kept {lo:,}-{hi:,})")


def field_bars(ax, names, counts, n_pairs, acl_frac, top=10):
    names, counts = list(names[:top]), list(counts[:top])
    share = np.array(counts) / n_pairs * 100
    y = np.arange(len(names))[::-1]
    ax.barh(y, share, height=0.62, color=CITER, linewidth=0)
    ax.set_yticks(y, names, fontsize=8.5)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("share of pairs whose citer carries the field (%)")
    ax.text(share[0], y[0], f" {share[0]:.0f}%", va="center", color=INK2, fontsize=8)
    title(ax, "Where the citers come from",
          f"multi-label, so shares overlap · {acl_frac * 100:.0f}% of citers are ACL Anthology papers")


def intent_groups(sim, code, null_sim):
    """Quantile summary per intent combination, ordered by median (high -> low), plus the
    random-other-seed reference and its 95th percentile."""
    p95 = float(np.percentile(null_sim, 95))
    rows = []
    for k, name in INTENT_NAMES.items():
        v = sim[code == k]
        if v.size == 0:
            continue
        rows.append({"code": k, "name": name, "n": int(v.size),
                     "q": np.percentile(v, [10, 25, 50, 75, 90]),
                     "above": float((v > p95).mean() * 100)})
    rows.sort(key=lambda r: -r["q"][2])
    ref = {"code": -1, "name": "citer vs a random other seed", "n": int(null_sim.size),
           "q": np.percentile(null_sim, [10, 25, 50, 75, 90]), "above": 5.0}
    return rows, ref, p95


def _row_ys(rows):
    return np.arange(len(rows))[::-1] + 1.5      # groups top to bottom; the reference sits at 0


def quantile_rows(ax, rows, ref, p95, n_pairs, enc_label):
    ys = _row_ys(rows)
    for r, y, color in [(r, y, CITER) for r, y in zip(rows, ys)] + [(ref, 0, MUTED)]:
        p10, p25, p50, p75, p90 = r["q"]
        ax.plot([p10, p90], [y, y], color=color, alpha=0.4, lw=1.5, solid_capstyle="round")
        ax.plot([p25, p75], [y, y], color=color, lw=7, solid_capstyle="butt")
        ax.plot(p50, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
    ax.axvline(p95, color=AXIS, lw=0.8, zorder=0)
    ax.text(p95, ys[0] + 0.75, " 95th pct of the random pairs", color=MUTED, fontsize=7.5,
            va="bottom")
    ax.set_yticks(list(ys) + [0], [r["name"] for r in rows] + [ref["name"]], fontsize=8.5)
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for r, y in zip(rows, ys):
        ax.text(-0.012, y - 0.42, f"{r['n']:,} pairs · {r['n'] / n_pairs:.1%}", transform=tr,
                ha="right", va="center", color=MUTED, fontsize=7)
    lo = min(min(r["q"][0] for r in rows), ref["q"][0])
    hi = max(max(r["q"][4] for r in rows), ref["q"][4])
    pad = (hi - lo) * 0.06
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.7, ys[0] + 1.3)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel(f"{enc_label} cosine between citer and seed abstract")
    title(ax, "Citer-seed similarity by citation intent",
          f"{n_pairs:,} pairs · bar p25-p75 · line p10-p90 · dot median")


def exceedance_bars(ax, rows, ref):
    ys = _row_ys(rows)
    vals = [r["above"] for r in rows]
    ax.barh(ys, vals, height=0.62, color=CITER, linewidth=0)
    ax.barh([0], [ref["above"]], height=0.62, color=MUTED, linewidth=0)
    for y, v in list(zip(ys, vals)) + [(0, ref["above"])]:
        ax.text(v + 1.2, y, f"{v:.0f}%", va="center", color=INK2, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, ys[0] + 1.3)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("share of pairs above that 95th percentile (%)")
    title(ax, "Pairs the seed alone would flag", "by construction 5% of random pairs do")


def intent_hists(fig, axes, sim, code, null_sim, rows, enc_label):
    lo, hi = float(np.percentile(null_sim, 0.5)), float(max(sim.max(), null_sim.max()))
    bins = np.linspace(lo, hi, 61)
    w = bins[1] - bins[0]
    ref_d, _ = np.histogram(null_sim, bins=bins, density=True)
    for ax, r in zip(axes.flat, rows):
        v = sim[code == r["code"]]
        d, _ = np.histogram(v, bins=bins, density=True)
        ax.bar(bins[:-1], d, width=w * 0.88, align="edge", color=CITER, linewidth=0)
        ax.stairs(ref_d, bins, color=MUTED, lw=1.5, baseline=None)
        ax.set_yticks([])
        ax.grid(False)
        title(ax, r["name"], f"{r['n']:,} pairs · median {r['q'][2]:.2f}")
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"{enc_label} cosine")
    handles = [matplotlib.patches.Patch(color=CITER, label="citer vs its seed"),
               matplotlib.lines.Line2D([], [], color=MUTED, lw=1.5,
                                       label="citer vs a random other seed")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.99, 0.995))
    fig.text(0.01, 0.985, "Citer-seed similarity by citation intent", color=INK, fontsize=11,
             weight="bold", va="top")
    fig.text(0.01, 0.955, "density, shared axes; the gray outline is the same random reference"
             " in every panel", color=INK2, fontsize=8.5, va="top")


# ---- main ---------------------------------------------------------------------------

def save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/acl_a2a.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("images"))
    ap.add_argument("--cache", type=Path, default=Path("results/visualise/benchmark_arrays.npz"))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help="sentence encoder for the intent-similarity figures")
    ap.add_argument("--emb-dir", type=Path, default=Path("results/embeddings"),
                    help="embedding cache shared with characterize.py / score.py")
    ap.add_argument("--no-similarity", action="store_true",
                    help="skip the intent-similarity figures (no encoder / GPU needed)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = load(args.data, args.cache)
    n_seeds, n_pairs = len(d["seed_n"]), int(d["seed_n"].sum())
    horizon = d["citer_year"] - d["seed_year"][d["citer_seed"]]
    stamp = f"{n_seeds:,} seeds · {n_pairs:,} seed-citer pairs"

    fig, ax = plt.subplots(figsize=(9, 3.6))
    year_bars(ax, d["seed_year"], SEED, "seeds", "Seeds by publication year", stamp)
    save(fig, args.out_dir / "seeds_per_year.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    log_hist(ax, d["seed_cc"], SEED, "S2 citation count", main="Seed citation counts",
             sub="seeds: within-year top 10% minus top 1% by influential citations")
    save(fig, args.out_dir / "seed_citations.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    log_hist(ax, d["seed_n"], SEED, "qualifying citers", main="Qualifying citers per seed",
             sub="every citer with a usable abstract, later than the seed, no shared author")
    save(fig, args.out_dir / "citers_per_seed.png")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    concentration(ax, d["seed_n"])
    save(fig, args.out_dir / "pair_concentration.png")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    year_bars(ax, d["citer_year"], CITER, "pairs", "Citing papers by year",
              "one bar per seed-citer pair; the citer needs an S2 abstract, which skews recent")
    save(fig, args.out_dir / "citer_years.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    horizon_bars(ax, horizon)
    save(fig, args.out_dir / "horizon.png")

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
    length_panels(axes, d["seed_len"], d["citer_len"])
    fig.subplots_adjust(hspace=0.55)
    save(fig, args.out_dir / "abstract_lengths.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    field_bars(ax, d["field_names"], d["field_counts"].astype(int), n_pairs,
               float(d["citer_acl"].mean()))
    save(fig, args.out_dir / "citer_fields.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
    year_bars(axes[0, 0], d["seed_year"], SEED, "seeds", "Seeds by publication year", stamp)
    log_hist(axes[0, 1], d["seed_cc"], SEED, "S2 citation count", main="Seed citation counts")
    log_hist(axes[0, 2], d["seed_n"], SEED, "qualifying citers", main="Qualifying citers per seed")
    concentration(axes[1, 0], d["seed_n"])
    year_bars(axes[1, 1], d["citer_year"], CITER, "pairs", "Citing papers by year")
    horizon_bars(axes[1, 2], horizon)
    fig.subplots_adjust(hspace=0.55, wspace=0.28)
    save(fig, args.out_dir / "benchmark_overview.png")

    if args.no_similarity:
        return
    enc = short_name(args.encoder)
    enc_label = ENCODER_LABEL.get(enc, enc)
    s = pair_similarity(args.data, args.encoder, args.emb_dir,
                        args.cache.with_name(f"pair_sims_{enc}.npz"))
    assert len(s["sim"]) == n_pairs, "pair order drifted between extract() and pair_similarity()"
    rows, ref, p95 = intent_groups(s["sim"], d["citer_intent"], s["null_sim"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [2.3, 1]})
    quantile_rows(axes[0], rows, ref, p95, n_pairs, enc_label)
    exceedance_bars(axes[1], rows, ref)
    fig.subplots_adjust(wspace=0.08)
    save(fig, args.out_dir / "intent_similarity.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
    intent_hists(fig, axes, s["sim"], d["citer_intent"], s["null_sim"], rows, enc_label)
    fig.subplots_adjust(top=0.82, hspace=0.6, wspace=0.22)
    save(fig, args.out_dir / "intent_similarity_hist.png")


if __name__ == "__main__":
    main()
