import numpy as np

from auditembodied.vlm_diagnostics import head_replacements, threshold_curve


def test_tied_support_scores_cannot_be_repaired_by_thresholds():
    diagnostic = threshold_curve([1, 1, 0, 0], [.05, .05, .05, .05])
    assert diagnostic["auc"] == .5
    assert diagnostic["best_recall_at_fpr_02"]["recall"] == 0
    assert diagnostic["minimum_fpr_for_recall_90"]["false_positive_rate"] == 1


def test_perfect_ordering_has_full_recall_with_no_false_release():
    diagnostic = threshold_curve([1, 0, 1, 0], [.8, .2, .9, .1])
    assert diagnostic["auc"] == 1
    assert diagnostic["best_recall_at_fpr_02"]["recall"] == 1
    assert diagnostic["best_recall_at_fpr_02"]["false_positive_rate"] == 0


def test_head_repair_exposes_both_coordination_and_role_errors():
    labels = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0]])
    predictions = np.array([[.95, .05, 0], [.95, .05, 0], [.95, .05, 0]])
    rows = head_replacements(labels, predictions, [.5, .5, .5])
    by_heads = {tuple(r["replaced_heads"]): r for r in rows}
    assert by_heads[()]["continue"] == 0
    assert by_heads[("receiver_supports",)]["continue"] == 0
    assert by_heads[("receiver_supports", "coordinated_support")]["continue"] == 1
    assert rows[-1]["agreement"] == 1
    assert rows[-1]["unsafe_continue"] == 0
