import numpy as np
import pytest

from a2a import metrics as M


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_recall_at_k_is_chance_a_random_k_subset_holds_a_match():
    # 51 generations, one matches: a size-k subset contains it with probability k/51
    assert M.recall_at_k([1], 51, 10)[0] == pytest.approx(10 / 51)
    assert M.recall_at_k([1], 51, 51)[0] == pytest.approx(1.0)
    # no match is never recalled; all matching is always recalled
    assert M.recall_at_k([0, 51], 51, 1).tolist() == [0.0, 1.0]
    # m = 2 of n = 4, k = 2: 1 - C(2,2)/C(4,2)
    assert M.recall_at_k([2], 4, 2)[0] == pytest.approx(1 - 1 / 6)


def test_recall_at_k_clips_k_to_the_generations_available():
    assert M.recall_at_k([1, 0], [40, 40], 51).tolist() == [1.0, 0.0]
    assert M.recall_at_k([3], [0], 10).tolist() == [0.0]


def test_recall_curve_is_monotone_and_ends_at_any_match_rate():
    m = np.array([0, 1, 5, 0])
    curve = M.recall_curve(m, 51)
    vals = [curve[k] for k in M.K_GRID]
    assert M.K_GRID[0] == 1 and M.K_GRID[-1] == 51
    assert vals == sorted(vals)
    assert curve[51] == pytest.approx(0.5)


def test_cosine_coverage_is_mean_best_similarity():
    sim = np.array([[0.9, 0.2, 0.5],
                    [0.1, 0.8, 0.4]])  # 2 generations x 3 citers
    assert M.cosine_coverage(sim) == pytest.approx((0.9 + 0.8 + 0.5) / 3)
    assert np.isnan(M.cosine_coverage(np.zeros((0, 3))))


def test_vendi_counts_effective_distinct_items():
    e = _unit(np.eye(4))
    assert M.vendi_score(e @ e.T) == pytest.approx(4.0)
    e = _unit(np.ones((4, 3)))
    assert M.vendi_score(e @ e.T) == pytest.approx(1.0)


def test_pairwise_and_centroid_dispersion_move_together():
    tight = _unit(np.array([[1, 0.01], [1, -0.01], [1, 0.02]]))
    loose = _unit(np.array([[1, 0], [0, 1], [-1, 0]]))
    assert M.mean_pairwise_sim(tight) > M.mean_pairwise_sim(loose)
    assert M.centroid_dispersion(tight) < M.centroid_dispersion(loose)


def test_distinct_n_and_self_bleu_direction():
    same = ["the cat sat on the mat"] * 4
    diff = ["the cat sat on the mat", "a dog ran in the park",
            "quantum chips beat noise", "rivers carve deep canyons"]
    assert M.distinct_n(diff, 2) > M.distinct_n(same, 2)
    assert M.self_bleu(same) > M.self_bleu(diff)


def test_bootstrap_ci_brackets_mean():
    mean, (lo, hi) = M.bootstrap_mean_ci([0.2, 0.4, 0.6, 0.8], n_boot=200)
    assert lo <= mean <= hi
    assert mean == pytest.approx(0.5)


def test_pooled_bootstrap_resamples_groups():
    vals, groups = [1, 1, 1, 0, 0, 0], ["a", "a", "a", "b", "b", "b"]
    mean, (lo, hi) = M.bootstrap_pooled_ci(vals, groups, n_boot=200)
    assert mean == pytest.approx(0.5)
    assert (lo, hi) == (0.0, 1.0)      # whole seeds are drawn, so both extremes occur
