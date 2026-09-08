"""Metrics for the abstract-to-abstract task.

All similarity inputs are cosine similarities between unit-norm embeddings.
A *prediction set* is the N abstracts produced for one seed; the *target set*
is that seed's real citing-paper abstracts.

Primary metric
--------------
``coverage_at(sim, tau)``: fraction of targets whose best similarity over the
prediction set is at least ``tau``. ``sim`` is (n_predictions, n_targets).
The operating threshold ``tau*`` is not chosen by hand: it is the 95th
percentile of the best-similarity distribution obtained when the prediction
set is a random pool of other seeds' citers (``calibrate_tau``), so
coverage@tau* reads as "targets matched better than 95% of chance matches".

Diversity of a set
------------------
``vendi_score``: exponentiated entropy of the eigenvalues of the normalised
similarity kernel (Friedman & Dieng 2023) = effective number of distinct
items. ``mean_pairwise_sim``: mean off-diagonal cosine. These describe a set
on its own, with no reference, which is exactly what the contrast analysis
needs: a reference-free diversity can be high while coverage of the real
futures stays low.

Lexical diversity (``self_bleu``, ``distinct_n``) and reference overlap
(``rouge_l_max``) are the conventional metrics we contrast against.
"""

from __future__ import annotations

import numpy as np

TAUS = tuple(round(t, 2) for t in np.arange(0.30, 0.96, 0.05))


# ----------------------------------------------------------------- coverage

def best_similarity(sim: np.ndarray) -> np.ndarray:
    """Per-target max over predictions. ``sim``: (n_pred, n_targets)."""
    if sim.ndim != 2 or sim.shape[0] == 0:
        return np.full(sim.shape[-1] if sim.ndim == 2 else 0, np.nan)
    return sim.max(axis=0)


def coverage_at(sim: np.ndarray, tau: float) -> float:
    if sim.ndim != 2 or 0 in sim.shape:
        return float("nan")
    return float(np.mean(best_similarity(sim) >= tau))


def coverage_curve(best: np.ndarray, taus=TAUS) -> dict[float, float]:
    best = np.asarray(best)
    if best.size == 0:
        return {float(t): float("nan") for t in taus}
    return {float(t): float(np.mean(best >= t)) for t in taus}


def coverage_at_n(sim: np.ndarray, tau: float, ns) -> dict[int, float]:
    """Coverage@tau using only the first n predictions, for each n in ``ns``."""
    return {int(n): coverage_at(sim[: int(n)], tau) for n in ns if n <= sim.shape[0]}


def calibrate_tau(null_best: np.ndarray, percentile: float = 95.0) -> float:
    """tau* = given percentile of the null (random-pool) best-similarity."""
    return float(np.percentile(np.asarray(null_best), percentile))


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


def split_half_coverage(emb: np.ndarray, tau: float, n_pred: int, rng: np.random.Generator,
                        repeats: int = 5) -> float:
    """Reference ceiling: real targets predicting held-out real targets.

    Randomly split the target set; up to ``n_pred`` items of one half act as
    the prediction set for the other half. Averaged over ``repeats``.
    Answers "at this budget, how much of the realised future does a set of
    *other real futures of the same paper* cover?"
    """
    n = emb.shape[0]
    if n < 4:
        return float("nan")
    vals = []
    for _ in range(repeats):
        perm = rng.permutation(n)
        half = n // 2
        pred, tgt = perm[:half][:n_pred], perm[half:]
        vals.append(coverage_at(emb[pred] @ emb[tgt].T, tau))
    return float(np.mean(vals))


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
