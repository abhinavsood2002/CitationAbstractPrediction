import numpy as np
import pytest

from a2a import metrics as M


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_coverage_is_fraction_of_targets_with_a_close_prediction():
    sim = np.array([[0.9, 0.2, 0.5],
                    [0.1, 0.95, 0.4]])  # 2 predictions x 3 targets
    assert M.coverage_at(sim, 0.8) == pytest.approx(2 / 3)
    assert M.coverage_at(sim, 0.45) == pytest.approx(1.0)
    assert M.coverage_at(sim, 0.99) == 0.0


def test_coverage_with_no_predictions_is_nan():
    assert np.isnan(M.coverage_at(np.zeros((0, 3)), 0.5))


def test_coverage_at_n_uses_prefix_of_predictions():
    sim = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]])
    assert M.coverage_at_n(sim, 0.8, [1, 2, 3]) == {1: 0.0, 2: 0.5, 3: 1.0}


def test_calibrate_tau_is_percentile_of_null():
    null = np.linspace(0, 1, 101)
    assert M.calibrate_tau(null, 95) == pytest.approx(0.95)


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


def test_split_half_coverage_is_one_for_duplicate_targets():
    e = _unit(np.ones((8, 5)))
    rng = np.random.default_rng(0)
    assert M.split_half_coverage(e, 0.99, n_pred=4, rng=rng) == pytest.approx(1.0)


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
