import numpy as np

from auditembodied.vlm_token_readout import answer_scores, release_margin, risk_coverage


def test_conditional_answer_probability_and_vocabulary_mass_are_distinct():
    logp = np.log([[.2, .3], [.08, .02]])
    support, mass = answer_scores(logp)
    np.testing.assert_allclose(support, [.4, .8])
    np.testing.assert_allclose(mass, [.5, .1])
    shifted_support, shifted_mass = answer_scores(logp - 10)
    np.testing.assert_allclose(shifted_support, support)
    np.testing.assert_allclose(shifted_mass, mass * np.exp(-10))


def test_risk_curve_keeps_ties_and_zero_coverage_has_no_defined_risk():
    curve = risk_coverage([1, 0, 0], [.8, .8, .2])
    assert [p["accepted"] for p in curve] == [0, 2, 3]
    assert curve[0]["risk"] is None
    assert curve[1]["coverage"] == 2 / 3
    assert curve[1]["risk"] == .5
    assert curve[1]["false_positive_rate"] == .5


def test_release_curve_matches_giver_boundary_and_receiver_only_completion():
    probabilities = np.array([[.5, .8, .1], [.49, .8, .1], [.7, .8, .8], [.1, .1, .9]])
    margin = release_margin(probabilities, [.5, .5, .5])
    np.testing.assert_array_equal(margin >= 0, [False, True, True, False])
    np.testing.assert_array_equal(margin >= .35, [False, False, False, False])
