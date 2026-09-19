"""Metrics for the abstract-to-abstract task.

A *generation set* is the 51 abstracts produced for one seed; the *targets* are
that seed's real citing abstracts (result sentences removed). Embeddings are
unit-norm, so every similarity below is a cosine.

Reported metrics
----------------
``recall_at_k``: a generation *matches* a citer when the reranker's probability
for the pair exceeds ``llm.MATCH_THRESHOLD``. With ``m`` of a seed's ``n``
generations matching a citer, Recall@k is the probability that a uniformly
drawn size-k subset of the generations contains a match,
``1 - C(n - m, k) / C(n, k)``, averaged over citers. It is the expectation of
"any match in the first k" over generation orderings, so it does not depend on
the order the sampler happened to emit. Headline k = 10 and 51; ``K_GRID`` is
the curve.

``cosine_coverage``: for each citer, the highest cosine to any generation;
averaged over citers. Threshold-free and reranker-free.

``vendi_score``: exponentiated entropy of the eigenvalues of the normalised
cosine kernel of a generation set (Friedman & Dieng 2023) = effective number
of distinct generations. Vendi@51 is over all 51.

All three are means over citers (pairs) or seeds; ``bootstrap_pooled_ci``
resamples seeds, since citers of one seed are not independent.

``mean_pairwise_sim`` / ``centroid_dispersion`` (used by characterize.py) and the
lexical metrics (``self_bleu``, ``distinct_n``, ``rouge_l_max``) are descriptive
extras.
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb

K_HEADLINE = (10, 51)
K_GRID = (1,) + tuple(range(5, 51, 5)) + (51,)


# ------------------------------------------------------------------- recall

def recall_at_k(m, n, k: int) -> np.ndarray:
    """Per-citer Recall@k. ``m``: matching generations per citer; ``n``: generations
    available for that citer's seed (scalar or array). A seed short of k generations
    is scored at k = n."""
    m, n = np.broadcast_arrays(np.asarray(m, dtype=float), np.asarray(n, dtype=float))
    kk = np.minimum(k, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 1.0 - comb(n - m, kk) / comb(n, kk)
    return np.where(n > 0, r, 0.0)


def recall_curve(m, n, ks=K_GRID) -> dict[int, float]:
    return {int(k): float(np.mean(recall_at_k(m, n, k))) for k in ks}


# ---------------------------------------------------------- cosine coverage

def best_similarity(sim: np.ndarray) -> np.ndarray:
    """Per-target max over generations. ``sim``: (n_generations, n_targets)."""
    if sim.ndim != 2 or sim.shape[0] == 0:
        return np.full(sim.shape[-1] if sim.ndim == 2 else 0, np.nan)
    return sim.max(axis=0)


def cosine_coverage(sim: np.ndarray) -> float:
    if sim.ndim != 2 or 0 in sim.shape:
        return float("nan")
    return float(best_similarity(sim).mean())


# ----------------------------------------------------------------- diversity

def vendi_score(kernel: np.ndarray) -> float:
    """Vendi score of an (n, n) similarity kernel with unit diagonal."""
    n = kernel.shape[0]
    if n == 0:
        return 0.0
    ev = np.linalg.eigvalsh(kernel / n)
    ev = ev[ev > 1e-12]
    return float(np.exp(-(ev * np.log(ev)).sum()))


def mean_pairwise_sim(emb: np.ndarray) -> float:
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    k = emb @ emb.T
    return float((k.sum() - np.trace(k)) / (n * (n - 1)))


def centroid_dispersion(emb: np.ndarray) -> float:
    """Mean cosine distance of items to their (normalised) centroid."""
    if emb.shape[0] < 2:
        return float("nan")
    c = emb.mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    return float(np.mean(1.0 - emb @ c))


# ------------------------------------------------------------------ lexical

def _tokens(text: str) -> list[str]:
    return text.lower().split()


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Fraction of unique n-grams over all n-grams in the set (Li et al. 2016)."""
    grams, total = set(), 0
    for t in texts:
        tok = _tokens(t)
        for i in range(len(tok) - n + 1):
            grams.add(tuple(tok[i:i + n]))
            total += 1
    return float(len(grams) / total) if total else float("nan")


def self_bleu(texts: list[str], max_items: int = 50) -> float:
    """Mean sentence-BLEU of each text against the others (Zhu et al. 2018).

    Higher = more mutually similar = less diverse. 0-100 scale (sacrebleu).
    """
    import sacrebleu

    texts = [t for t in texts if t.strip()][:max_items]
    if len(texts) < 2:
        return float("nan")
    scores = []
    for i, hyp in enumerate(texts):
        refs = texts[:i] + texts[i + 1:]
        scores.append(sacrebleu.sentence_bleu(hyp, refs).score)
    return float(np.mean(scores))


def rouge_l_max(predictions: list[str], references: list[str]) -> float:
    """Mean over references of the best ROUGE-L F1 against any prediction."""
    from rouge_score import rouge_scorer

    if not predictions or not references:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best = []
    for ref in references:
        best.append(max(scorer.score(ref, p)["rougeL"].fmeasure for p in predictions))
    return float(np.mean(best))


# ----------------------------------------------------------------- bootstrap

def bootstrap_mean_ci(values, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean over an array of per-seed values."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), (float(lo), float(hi))


def bootstrap_pooled_ci(values, groups, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Mean of per-citer ``values`` with a percentile CI that resamples ``groups`` (seeds)."""
    values = np.asarray(values, dtype=float)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    ok = ~np.isnan(values)
    sums = np.bincount(inv[ok], weights=values[ok], minlength=inv.max() + 1)
    cnts = np.bincount(inv[ok], minlength=inv.max() + 1).astype(float)
    if cnts.sum() == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(axis=1) / np.maximum(cnts[idx].sum(axis=1), 1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(sums.sum() / cnts.sum()), (float(lo), float(hi))
