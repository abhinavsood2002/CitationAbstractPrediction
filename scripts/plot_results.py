#!/usr/bin/env python
"""Plot the scored run (``scripts/score.py`` output) as PNGs.

    python scripts/plot_results.py [--score-dir results/score] [--out-dir images]

    recall_at_k.png   Recall@k for every k = 1..51, one line per condition, with the shuffled
                      chance floors; and the same curves minus Recall@1, i.e. the share of
                      citers that only the extra samples recover
    <paper-dir>/recall_at_k.pdf   the Recall@k panel alone, single-column width, for the paper

Recall@k is recomputed for every k from the per-pair match counts in
``<enc>_per_pair.csv.gz`` (``metrics.recall_at_k``), so the curve is exact at every k and
not only on ``metrics.K_GRID``.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a import metrics as M  # noqa: E402
from a2a.embed import DEFAULT_ENCODER, short_name  # noqa: E402
from a2a.generate import N_GENERATIONS  # noqa: E402

# ---- palette (same surface and inks as visualise.py) --------------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

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

# One hue per model and sampling scheme: Gemma's temperature sweep is a blue ramp (darker =
# hotter), Verbalized Sampling is green, gpt-oss is an orange ramp (darker = more reasoning).
STYLE = {
    "gemma_t05": ("Gemma T=0.5", "#9cc3ee"),
    "gemma_t10": ("Gemma T=1.0", "#2a78d6"),
    "gemma_t15": ("Gemma T=1.5", "#143f78"),
    "gemma_t10_vs": ("Gemma T=1.0 + VS", "#2e9d6a"),
    "gptoss": ("gpt-oss low", "#f0955f"),
    "gptoss_medium": ("gpt-oss medium", "#b8431a"),
}


def title(ax, main: str, sub: str | None = None):
    ax.set_title(main)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")


def save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def recall_curves(per_pair: Path) -> dict[tuple[str, str], np.ndarray]:
    """{(condition, mode): Recall@k for k = 1..51}, for every unit that was reranked."""
    df = pd.read_csv(per_pair, usecols=["condition", "mode", "n_gen", "m"]).dropna(subset=["m"])
    ks = range(1, N_GENERATIONS + 1)
    return {key: np.array([M.recall_at_k(g["m"].to_numpy(), g["n_gen"].to_numpy(), k).mean()
                           for k in ks])
            for key, g in df.groupby(["condition", "mode"])}


def end_labels(ax, ends: list[tuple[float, str, str]], x: float, min_gap: float):
    """Label each line at its right end, pushed apart so that no two labels overlap."""
    ends = sorted(ends)
    ys = [y for y, _, _ in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for y, (_, label, color) in zip(ys, ends):
        ax.text(x, y, label, color=color, fontsize=8.5, va="center", ha="left", fontweight="bold")


def recall_at_k_figure(curves, out: Path):
    ks = np.arange(1, N_GENERATIONS + 1)
    names = [c for c in STYLE if (c, "own") in curves]
    # the gap between the panels holds the left panel's end-of-line labels
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.62})

    ends = []
    for c in names:
        label, color = STYLE[c]
        r = curves[c, "own"]
        a.plot(ks, r, color=color, lw=2, solid_capstyle="round")
        ends.append((r[-1], f"{label}  {r[-1]:.2f}", color))
        if (c, "shuffled") in curves:
            a.plot(ks, curves[c, "shuffled"], color=color, lw=1.4, ls=(0, (3, 2)))
    end_labels(a, ends, N_GENERATIONS + 1, min_gap=0.043)
    floors = [c for c in names if (c, "shuffled") in curves]
    if floors:
        top = max(curves[c, "shuffled"][-1] for c in floors)
        a.text(N_GENERATIONS + 1, top + 0.005, "shuffled floors (dashed)", color=MUTED,
               fontsize=8, va="bottom", ha="left")
    a.set_ylim(0, 0.9)
    a.set_ylabel("Recall@k (share of citers matched)")
    title(a, "More samples recover more citers, with diminishing returns",
          "k generations drawn from the 51; a match is reranker P(yes) > 0.5")

    ends = []
    for c in names:
        label, color = STYLE[c]
        gain = curves[c, "own"] - curves[c, "own"][0]
        b.plot(ks, gain, color=color, lw=2, solid_capstyle="round")
        ends.append((gain[-1], f"{label}  +{gain[-1]:.2f}", color))
    end_labels(b, ends, N_GENERATIONS + 1, min_gap=0.024)
    b.set_ylim(0, 0.5)
    b.set_ylabel("Recall@k - Recall@1")
    title(b, "What the extra samples add over a single one",
          "citers missed by one generation but matched by some of k")

    for ax in (a, b):
        ax.set_xlim(1, N_GENERATIONS)
        ax.set_xticks([1, 10, 20, 30, 40, N_GENERATIONS])
        ax.set_xlabel("k (generations per seed)")
    save(fig, out)


# Paper figure: Okabe-Ito colours (colour-blind safe) plus a marker and a line style per
# condition, so the lines stay distinguishable in greyscale print.
PAPER_STYLE = {
    "gptoss_medium": ("gpt-oss medium", "#D55E00", "s", "-"),
    "gptoss": ("gpt-oss low", "#E69F00", "o", "-"),
    "gemma_t10_vs": ("Gemma $T$=1.0 + VS", "#009E73", "^", "-"),
    "gemma_t15": ("Gemma $T$=1.5", "#0072B2", "D", "--"),
    "gemma_t10": ("Gemma $T$=1.0", "#000000", "v", "-"),
    "gemma_t05": ("Gemma $T$=0.5", "#56B4E9", "x", ":"),
}


def recall_at_k_paper(curves, out: Path):
    """Recall@k alone for the paper: ACL column width (7.7 cm), vector PDF with embedded
    TrueType fonts, Times-like serif at 8 pt to match the body text, readable in greyscale."""
    ks = np.arange(1, N_GENERATIONS + 1)
    rc = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
          "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
          "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
          "ytick.labelsize": 7.5, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
          "axes.edgecolor": "black", "axes.labelcolor": "black", "axes.linewidth": 0.6,
          "xtick.color": "black", "ytick.color": "black", "xtick.labelcolor": "black",
          "ytick.labelcolor": "black", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6, "grid.color": "#d9d9d9",
          "grid.linewidth": 0.5}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(3.03, 2.75))
        for c, (label, color, marker, ls) in PAPER_STYLE.items():
            if (c, "own") not in curves:
                continue
            ax.plot(ks, curves[c, "own"], color=color, ls=ls, lw=1.2, marker=marker, ms=3.2,
                    markevery=[0, 4, 9, 19, 29, 39, 50], mfc="white", mew=0.9, label=label)
            if (c, "shuffled") in curves:
                ax.plot(ks, curves[c, "shuffled"], color=color, lw=0.9, ls=(0, (1, 1.5)))
        ax.plot([], [], color="black", lw=0.9, ls=(0, (1, 1.5)), label="chance floor")
        ax.set_xlim(0, N_GENERATIONS + 1)
        ax.set_ylim(0, 0.9)
        ax.set_xticks([1, 10, 20, 30, 40, N_GENERATIONS])
        ax.set_yticks(np.arange(0, 0.91, 0.1))
        ax.set_xlabel("$k$ (generations per seed)")
        ax.set_ylabel("Recall@$k$")
        ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.08),
                  handlelength=2.4, columnspacing=1.0, labelspacing=0.3, borderaxespad=0.2)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-dir", type=Path, default=Path("results/score"))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--out-dir", type=Path, default=Path("images"))
    ap.add_argument("--paper-dir", type=Path, default=Path("paper/figures"),
                    help="where the single-column PDF for the paper goes")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    curves = recall_curves(args.score_dir / f"{short_name(args.encoder)}_per_pair.csv.gz")
    recall_at_k_figure(curves, args.out_dir / "recall_at_k.png")
    recall_at_k_paper(curves, args.paper_dir / "recall_at_k.pdf")


if __name__ == "__main__":
    main()
