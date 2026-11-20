"""Regression tests for wide-range weighted mining.

Two failures are pinned down with directly computable small baskets:

1. weights that are individually finite but whose raw sums leave the float
   range (``2 ** 1023``) must mine exactly what a unit change of the same
   weights would mine, in both weight-only and cohort modes;
2. a positive support so small that it displays as zero must still produce a
   finite, defined reference confidence, whereas genuinely zero reference
   weights keep that confidence undefined (NaN).

The tests also cover: true zero versus displayed zero, a mining result whose
rows were copied and permuted, and the weight-only (no cohorts) entry point.
"""

import numpy as np
import pandas as pd
import pytest

from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.frequent_patterns import fpcommon as fpc


def _comparison_frames():
    """Observed: two {a, b} baskets and one empty basket.

    Reference: {a, b}, {a}, empty.  Returns the stacked boolean frame plus
    its cohort labels.
    """
    observed = pd.DataFrame({"a": [1, 1, 0], "b": [1, 1, 0]})
    reference = pd.DataFrame({"a": [1, 1, 0], "b": [1, 0, 0]})
    frame = pd.concat([observed, reference], ignore_index=True)
    cohorts = ["obs", "obs", "obs", "ref", "ref", "ref"]
    return frame, cohorts


# ---------------------------------------------------------------------------
# exact power-of-two rescaling
# ---------------------------------------------------------------------------


def test_rescale_is_identity_for_ordinary_weights():
    weights = np.array([1.0, 2.5, 0.5, 0.0])
    scaled, exponent, total = fpc.rescale_weights(weights)
    assert exponent == 0
    np.testing.assert_array_equal(scaled, weights)
    assert total == pytest.approx(4.0)


def test_rescale_keeps_ratios_and_positivity():
    raw = np.array([2.0 ** 1023, 2.0 ** 1023, 0.5 * 2.0 ** 1023, 0.0])
    scaled, exponent, total = fpc.rescale_weights(raw)
    assert exponent < 0
    assert np.isfinite(scaled).all()
    assert np.isfinite(total)
    # zero rows stay zero, positive rows stay positive
    assert scaled[-1] == 0.0
    assert (scaled[:-1] > 0.0).all()
    # every weighted ratio is preserved by the common power of two.  The raw
    # total overflows, so compare against the exact rational ratios.
    np.testing.assert_allclose(
        scaled[:-1] / total, [0.4, 0.4, 0.2], rtol=1e-15, atol=0.0
    )
    assert scaled[-1] / total == 0.0


def test_rescale_per_cohort_uses_group_specific_units():
    # One group's raw total overflows and needs a smaller unit; the other
    # group holds tiny weights that are already exactly representable and so
    # is left untouched.  A single global factor could not satisfy the first
    # group without collapsing the second, which is precisely why the unit
    # change is applied per cohort.
    weights = np.array(
        [2.0 ** 1023, 2.0 ** 1023, 2.0 ** -1073, 2.0 ** -1073]
    )
    codes = np.array([0, 0, 1, 1])
    scaled, exponents, totals = fpc.rescale_weights(
        weights, cohort_codes=codes, n_cohorts=2
    )
    assert exponents[0] < 0
    assert exponents[1] == 0
    assert np.isfinite(scaled).all()
    assert (scaled > 0.0).all()
    assert np.isfinite(totals).all()
    # each group is internally just a unit change, so the within-group
    # weighted fractions are preserved (the raw sums would overflow)
    np.testing.assert_allclose(
        scaled[codes == 0] / totals[0], [0.5, 0.5], rtol=1e-15
    )
    np.testing.assert_allclose(
        scaled[codes == 1] / totals[1], [0.5, 0.5], rtol=1e-15
    )


def test_rescale_rejects_impossible_span():
    # 2 ** -1074 and 2 ** 1023 in one group: no power of two keeps both.
    weights = np.array([2.0 ** -1074, 2.0 ** 1023])
    with pytest.raises(ValueError, match="range too wide"):
        fpc.rescale_weights(weights)


# ---------------------------------------------------------------------------
# failure 1: overflowing total weight, both entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cohorts", [None, ["g", "g", "g"]])
def test_huge_weights_mine_the_same_itemsets_as_unit_weights(cohorts):
    df = pd.DataFrame({"a": [1, 1, 0], "b": [1, 0, 1]})
    huge = 2.0 ** 1023
    heavy = [huge, huge, huge]
    plain = [1.0, 1.0, 1.0]

    if cohorts is None:
        baseline = fpgrowth(df, min_support=0.5, sample_weight=plain)
        result = fpgrowth(df, min_support=0.5, sample_weight=heavy)
        assert result.columns.tolist() == ["support", "itemsets"]
    else:
        baseline = fpgrowth(
            df, min_support=0.5, cohorts=cohorts, sample_weight=plain,
            use_colnames=True,
        )
        result = fpgrowth(
            df, min_support=0.5, cohorts=cohorts, sample_weight=heavy,
            use_colnames=True,
        )
        assert result.columns.tolist() == ["cohort", "support", "itemsets"]

    # The overflowing batch must not come back empty, and must match the
    # unit-weight run itemset for itemset.
    assert not result.empty
    if cohorts is None:
        base_index = dict(zip(baseline["itemsets"], baseline["support"]))
        result_index = dict(zip(result["itemsets"], result["support"]))
        assert set(base_index) == set(result_index)
        for itemset, support in base_index.items():
            assert result_index[itemset] == pytest.approx(support)
    else:
        base_index = {
            frozenset(row.itemsets): row.support
            for row in baseline.itertuples(index=False)
        }
        result_index = {
            frozenset(row.itemsets): row.support
            for row in result.itertuples(index=False)
        }
        assert set(base_index) == set(result_index)
        for itemset in base_index:
            assert result_index[itemset] == pytest.approx(base_index[itemset])


def test_huge_weights_weight_only_support_values_are_ratios():
    df = pd.DataFrame({"a": [1, 1, 0, 1], "b": [1, 0, 1, 0]})
    huge = 2.0 ** 1023
    result = fpgrowth(
        df, min_support=0.5, sample_weight=[huge, huge, huge, huge]
    )
    index = dict(zip(result["itemsets"], result["support"]))
    # three of four baskets carry a, two carry b
    assert index[frozenset({0})] == pytest.approx(0.75)
    assert index[frozenset({1})] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# failure 2: tiny positive reference evidence versus true zeros
# ---------------------------------------------------------------------------


def test_tiny_positive_reference_confidence_is_defined():
    frame, cohorts = _comparison_frames()
    tiny, huge = 2.0 ** -600, 2.0 ** 600
    weights = [1.0, 1.0, 1.0, tiny, tiny, huge]

    itemsets = fpgrowth(
        frame,
        min_support=0.5,
        cohorts=cohorts,
        sample_weight=weights,
        use_colnames=True,
    )
    # The reference supports display as zero, but evidence is retained.
    ref_rows = itemsets[itemsets["cohort"] == "ref"]
    assert (ref_rows["support"] == 0.0).all()
    evidence = fpc.get_weighted_evidence(itemsets)
    assert evidence is not None
    assert evidence[fpc._TINY_EVIDENCE_KEY]

    rules = association_rules(
        itemsets,
        min_threshold=0.0,
        reference_cohort="ref",
        min_confidence_delta=0.4,
    )
    rule = rules[
        (rules["cohort"] == "obs")
        & (rules["antecedents"] == frozenset({"a"}))
        & (rules["consequents"] == frozenset({"b"}))
    ]
    assert len(rule) == 1
    row = rule.iloc[0]
    # Half of the reference a-weight also carries b.
    assert row["reference confidence"] == pytest.approx(0.5)
    assert row["confidence"] == pytest.approx(1.0)
    assert row["confidence delta"] == pytest.approx(0.5)
    # The displayed reference support column keeps its float precision.
    assert row["reference support"] == 0.0


def test_true_zero_reference_weights_leave_confidence_undefined():
    frame, cohorts = _comparison_frames()
    huge = 2.0 ** 600
    weights = [1.0, 1.0, 1.0, 0.0, 0.0, huge]

    itemsets = fpgrowth(
        frame,
        min_support=0.5,
        cohorts=cohorts,
        sample_weight=weights,
        use_colnames=True,
    )
    evidence = fpc.get_weighted_evidence(itemsets)
    # Real zeros: nothing positive is hidden behind the zero display.
    assert evidence[fpc._TINY_EVIDENCE_KEY] == {}

    rules = association_rules(
        itemsets,
        min_threshold=0.0,
        reference_cohort="ref",
        min_confidence_delta=0.4,
    )
    rule = rules[
        (rules["cohort"] == "obs")
        & (rules["antecedents"] == frozenset({"a"}))
        & (rules["consequents"] == frozenset({"b"}))
    ]
    # An undefined reference confidence produces a NaN delta that cannot
    # pass the lower-bound filter.
    assert len(rule) == 0

    unfiltered = association_rules(
        itemsets, min_threshold=0.0, reference_cohort="ref"
    )
    row = unfiltered[
        (unfiltered["cohort"] == "obs")
        & (unfiltered["antecedents"] == frozenset({"a"}))
        & (unfiltered["consequents"] == frozenset({"b"}))
    ].iloc[0]
    assert np.isnan(row["reference confidence"])
    assert np.isnan(row["confidence delta"])


def test_tiny_positive_evidence_survives_row_reordering_and_copy():
    frame, cohorts = _comparison_frames()
    tiny, huge = 2.0 ** -600, 2.0 ** 600
    weights = [1.0, 1.0, 1.0, tiny, tiny, huge]

    itemsets = fpgrowth(
        frame,
        min_support=0.5,
        cohorts=cohorts,
        sample_weight=weights,
        use_colnames=True,
    )
    # Full copy with a deliberately different row order.
    permuted = itemsets.sample(frac=1.0, random_state=7).reset_index(drop=True)
    copied = permuted.copy()
    assert fpc.get_weighted_evidence(copied) is not None

    rules = association_rules(
        copied,
        min_threshold=0.0,
        reference_cohort="ref",
        min_confidence_delta=0.4,
    )
    rule = rules[
        (rules["cohort"] == "obs")
        & (rules["antecedents"] == frozenset({"a"}))
        & (rules["consequents"] == frozenset({"b"}))
    ]
    assert len(rule) == 1
    assert rule.iloc[0]["reference confidence"] == pytest.approx(0.5)


def test_weight_only_tiny_evidence_sidecar_is_attached():
    # Weight-only mode with a huge empty basket dominating the total: the
    # populated baskets carry only tiny weight.  Nothing reaches a positive
    # min_support, but the empty result must still carry a valid sidecar and
    # keep the classic columns.
    df = pd.DataFrame(
        {
            "a": [1, 1, 0],
            "b": [1, 0, 0],
        }
    )
    tiny, huge = 2.0 ** -600, 2.0 ** 600
    weights = [tiny, tiny, huge]
    itemsets = fpgrowth(df, min_support=0.5, sample_weight=weights)
    assert itemsets.columns.tolist() == ["support", "itemsets"]
    assert itemsets.empty
    evidence = fpc.get_weighted_evidence(itemsets)
    assert evidence is not None
    assert evidence[fpc._TOTAL_EVIDENCE_KEY][None] > 0.0


def test_missing_itemset_is_still_a_loud_error():
    frame, cohorts = _comparison_frames()
    itemsets = fpgrowth(
        frame,
        min_support=0.5,
        cohorts=cohorts,
        sample_weight=[1.0] * 6,
        use_colnames=True,
    )
    # Drop a singleton needed to evaluate the a -> b rule while keeping the
    # combined itemset, so the universe stays identical but a subset lookup
    # fails.
    incomplete = itemsets[itemsets["itemsets"] != frozenset({"b"})]
    with pytest.raises(ValueError, match="no support entry|complete"):
        association_rules(incomplete, min_threshold=0.0, reference_cohort="ref")


def test_three_evidence_states_remain_distinguishable_in_current_cohort():
    # One cohort holds tiny positive evidence (displayed support zero), a
    # second holds genuine zero weights for the same itemset, and a third
    # strong cohort makes the itemset clear the shared threshold so the
    # result carries rows for all three.  The tiny cohort's rule must
    # survive with a finite confidence, while the true-zero cohort's version
    # is simply not emitted.
    df = pd.DataFrame(
        {
            "a": [1, 1, 0, 1, 1, 0, 1, 1, 1],
            "b": [1, 0, 0, 1, 0, 0, 1, 1, 1],
        }
    )
    tiny, huge, one = 2.0 ** -600, 2.0 ** 600, 1.0
    cohorts = (
        ["tiny"] * 3 + ["zero"] * 3 + ["strong"] * 3
    )
    weights = (
        [tiny, tiny, huge] + [0.0, 0.0, huge] + [one, one, one]
    )

    itemsets = fpgrowth(
        df,
        min_support=0.5,
        cohorts=cohorts,
        sample_weight=weights,
        use_colnames=True,
    )
    evidence = fpc.get_weighted_evidence(itemsets)
    # The tiny cohort hides positive counts behind zero displays; the
    # true-zero cohort has no such entries.
    tiny_labels = {
        cohort
        for (_itemset, cohort) in evidence[fpc._TINY_EVIDENCE_KEY]
    }
    assert "tiny" in tiny_labels
    assert "zero" not in tiny_labels

    rules = association_rules(itemsets, min_threshold=0.0)
    tiny_rules = rules[rules["cohort"] == "tiny"]
    zero_rules = rules[rules["cohort"] == "zero"]
    # a -> b in the tiny cohort: one of the two a-weighted baskets has b
    ab = tiny_rules[
        (tiny_rules["antecedents"] == frozenset({"a"}))
        & (tiny_rules["consequents"] == frozenset({"b"}))
    ]
    assert len(ab) == 1
    assert ab.iloc[0]["confidence"] == pytest.approx(0.5)
    # The zero cohort never produces a rule (no positive antecedent).
    assert len(zero_rules) == 0


def test_plain_support_table_without_evidence_keeps_legacy_behaviour():
    # A hand-built cohort support table with no sidecar must keep working
    # and treat an explicit zero as a genuine zero (NaN confidence).
    itemsets = pd.DataFrame(
        {
            "cohort": ["A", "A", "A", "B", "B", "B"],
            "support": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "itemsets": [
                frozenset({"a"}),
                frozenset({"b"}),
                frozenset({"a", "b"}),
                frozenset({"a"}),
                frozenset({"b"}),
                frozenset({"a", "b"}),
            ],
        }
    )
    assert fpc.get_weighted_evidence(itemsets) is None
    rules = association_rules(
        itemsets, min_threshold=0.0, reference_cohort="B"
    )
    row = rules.iloc[0]
    assert np.isnan(row["reference confidence"])
    assert np.isnan(row["confidence delta"])
