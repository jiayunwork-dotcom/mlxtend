"""Tests for weighted and cohort-aware frequent pattern mining.

The expected values are computed directly from the small input tables (see
``_direct_supports`` and ``_direct_rules``) instead of being copied from the
implementation.
"""

import itertools
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from mlxtend.frequent_patterns import association_rules, fpgrowth, fpmax
from mlxtend.frequent_patterns import fpcommon as fpc

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bools(df):
    return df.to_numpy().astype(bool)


def _direct_supports(df, cohorts, weights, min_support):
    """Return ``{itemset: {cohort: support}}`` by direct counting."""
    rows = _bools(df)
    items = list(df.columns)
    labels = list(dict.fromkeys(cohorts))
    totals = {label: 0.0 for label in labels}
    for label, weight in zip(cohorts, weights):
        totals[label] += weight

    result = {}
    for size in range(1, len(items) + 1):
        for combo in itertools.combinations(range(len(items)), size):
            counts = {label: 0.0 for label in labels}
            for row_idx in range(rows.shape[0]):
                if all(rows[row_idx, col] for col in combo):
                    counts[cohorts[row_idx]] += weights[row_idx]
            supports = {
                label: counts[label] / totals[label] if totals[label] else 0.0
                for label in labels
            }
            if any(value >= min_support - 1e-12 for value in supports.values()):
                result[frozenset(items[col] for col in combo)] = supports
    return labels, result


def _assert_support_index_equal(left, right):
    assert set(left) == set(right)
    for itemset in left:
        assert set(left[itemset]) == set(right[itemset])
        for label in left[itemset]:
            assert left[itemset][label] == pytest.approx(right[itemset][label])


def _index(itemsets_df):
    index = {}
    for row in itemsets_df.itertuples(index=False):
        index.setdefault(frozenset(row.itemsets), {})[row.cohort] = row.support
    return index


def _direct_rule_evidence(index, cohort, antecedent, consequent):
    support = index[antecedent | consequent][cohort]
    antecedent_support = index[antecedent][cohort]
    consequent_support = index[consequent][cohort]
    if antecedent_support <= 0:
        return support, None
    confidence = support / antecedent_support
    lift = confidence / consequent_support if consequent_support > 0 else np.nan
    return support, (confidence, lift)


def _small_baskets():
    baskets = [
        ["bread", "milk"],
        ["bread", "milk"],
        ["bread"],
        ["milk", "eggs"],
        ["bread", "milk", "eggs"],
    ]
    items = ["bread", "milk", "eggs"]
    df = pd.DataFrame(
        [[1 if item in basket else 0 for item in items] for basket in baskets],
        columns=items,
    )
    return df, ["A", "A", "A", "B", "B"], [1.0, 1.0, 1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# itemset mining
# ---------------------------------------------------------------------------


def test_itemsets_match_direct_statistics():
    df, cohorts, weights = _small_baskets()
    weights = [1.0, 2.0, 1.0, 0.5, 3.0]
    labels, expected = _direct_supports(df, cohorts, weights, 0.4)

    result = fpgrowth(
        df,
        min_support=0.4,
        cohorts=cohorts,
        sample_weight=weights,
        use_colnames=True,
    )
    assert result.columns.tolist() == ["cohort", "support", "itemsets"]
    index = _index(result)

    assert set(index) == set(expected)
    for itemset, supports in expected.items():
        assert set(index[itemset]) == set(labels)
        for label in labels:
            assert index[itemset][label] == pytest.approx(supports[label])

    # no duplicate (cohort, itemset) rows
    pairs = list(zip(result["cohort"], result["itemsets"]))
    assert len(pairs) == len(set(pairs))


def test_threshold_in_any_cohort_emits_every_cohort():
    # ``eggs`` reaches the threshold only in cohort B, yet cohort A gets a row
    # with support 0.0.
    df, cohorts, weights = _small_baskets()
    result = fpgrowth(
        df,
        min_support=0.5,
        cohorts=cohorts,
        sample_weight=weights,
        use_colnames=True,
    )
    eggs = result[result["itemsets"] == frozenset({"eggs"})]
    assert eggs.set_index("cohort")["support"].to_dict() == {"A": 0.0, "B": 1.0}

    # bread+milk+eggs occurs in every B basket, so it does clear 0.5.
    assert (result["itemsets"] == frozenset({"bread", "milk", "eggs"})).any()


def test_itemset_below_threshold_in_all_cohorts_is_absent():
    # 8 rows per cohort: "a" reaches 0.25 in P and 0 in Q, "b" reaches 0.25 in
    # both, and the pair a+b reaches 0.25 / 0.0.  None reaches 0.5 anywhere.
    df = pd.DataFrame(
        {
            "a": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "b": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    cohorts = ["P"] * 8 + ["Q"] * 8
    result = fpgrowth(df, 0.5, cohorts=cohorts, use_colnames=True)
    assert result.empty
    result_low = fpgrowth(df, 0.25, cohorts=cohorts, use_colnames=True)
    index = _index(result_low)
    assert index[frozenset({"a"})] == {"P": 0.25, "Q": 0.0}
    # "b" and "a+b" are below 0.25 everywhere, so only "a" survives at 0.25;
    # "a+b" is the case we care about: it must not be reported for either
    # cohort even though it exists in P.
    assert frozenset({"b"}) not in index
    assert frozenset({"a", "b"}) not in index


def test_low_support_cohorts_are_zero_filled_not_dropped():
    df = pd.DataFrame(
        {
            "a": [1, 1, 0, 0, 0, 0],
            "b": [1, 1, 0, 0, 0, 0],
            "c": [0, 0, 1, 1, 0, 0],
        }
    )
    result = fpgrowth(
        df,
        min_support=0.5,
        cohorts=["P", "P", "Q", "Q", "R", "R"],
        use_colnames=True,
    )
    index = _index(result)
    assert index[frozenset({"c"})] == {"P": 0.0, "Q": 1.0, "R": 0.0}
    # "a" is frequent only in P, but appears for all three cohorts.
    assert sorted(index[frozenset({"a"})].items()) == [("P", 1.0), ("Q", 0.0), ("R", 0.0)]
    # pairs a+b frequent only in P; still reported for all cohorts
    assert index[frozenset({"a", "b"})] == {"P": 1.0, "Q": 0.0, "R": 0.0}


def test_empty_baskets_count_in_denominator():
    df = pd.DataFrame([[1, 0], [0, 0], [0, 1], [1, 1]], columns=["a", "b"])
    result = fpgrowth(df, min_support=0.5, cohorts=["A", "A", "B", "B"], use_colnames=True)
    index = _index(result)
    # cohort A has 2 baskets, one empty: a -> 1/2, b -> 0/2
    assert index[frozenset({"a"})] == {"A": 0.5, "B": 0.5}
    assert index[frozenset({"b"})] == {"A": 0.0, "B": 1.0}
    assert index[frozenset({"a", "b"})] == {"A": 0.0, "B": 0.5}


def test_fractional_and_zero_weights():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 0, 0],
            "b": [1, 0, 0, 1, 0],
        }
    )
    # cohort P: weights 0.5, 2.5, 0.0 -> total 3; a is in all three rows but
    # the zero-weight row contributes nothing: 3.0 / 3 = 1.0.
    # cohort Q: weights 0.0, 1.0 -> total 1; a is absent, b is in the
    # zero-weight row only, so b's support is 0.0 in both cohorts and it never
    # reaches the threshold.
    result = fpgrowth(
        df,
        min_support=0.5,
        cohorts=["P", "P", "P", "Q", "Q"],
        sample_weight=[0.5, 2.5, 0.0, 0.0, 1.0],
        use_colnames=True,
    )
    index = _index(result)
    assert index[frozenset({"a"})]["P"] == pytest.approx(1.0)
    assert index[frozenset({"a"})]["Q"] == pytest.approx(0.0)
    # b: P = 0.5/3, Q = 0.0 -> below 0.5 everywhere
    assert frozenset({"b"}) not in index
    # zero-weight rows never contribute
    assert frozenset({"a", "b"}) not in index


def test_zero_weight_row_does_not_contribute_weighted_only():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 0, 1]})
    result = fpgrowth(df, min_support=0.5, sample_weight=[1.0, 1.0, 0.0])
    index = dict(zip(result["itemsets"], result["support"]))
    assert index[frozenset({0})] == pytest.approx(1.0)
    assert index[frozenset({1})] == pytest.approx(0.5)
    assert index[frozenset({0, 1})] == pytest.approx(0.5)
    assert frozenset({0, 1}) in index


def test_weighted_only_keeps_classic_columns():
    df = pd.DataFrame({"a": [1, 1, 0], "b": [1, 0, 1]})
    plain = fpgrowth(df, min_support=0.5)
    weighted = fpgrowth(df, min_support=0.5, sample_weight=[1, 1, 1])
    assert weighted.columns.tolist() == ["support", "itemsets"]
    assert plain.columns.tolist() == ["support", "itemsets"]
    assert sorted(map(tuple, plain["itemsets"])) == sorted(
        map(tuple, weighted["itemsets"])
    )
    for row_plain, row_weighted in zip(
        plain.sort_values("support").itertuples(index=False),
        weighted.sort_values("support").itertuples(index=False),
    ):
        assert row_plain.itemsets == row_weighted.itemsets
        assert row_plain.support == pytest.approx(row_weighted.support)


# ---------------------------------------------------------------------------
# positional alignment / input layouts
# ---------------------------------------------------------------------------


def test_duplicate_index_is_matched_positionally():
    df = pd.DataFrame(
        {"a": [1, 0, 0, 1], "b": [0, 1, 0, 1]},
        index=["dup", "dup", "dup", "dup"],
    )
    result = fpgrowth(
        df,
        min_support=0.5,
        cohorts=["P", "Q", "Q", "P"],
        sample_weight=[1.0, 1.0, 1.0, 1.0],
        use_colnames=True,
    )
    index = _index(result)
    assert index[frozenset({"a"})] == {"P": 1.0, "Q": 0.0}
    assert index[frozenset({"b"})] == {"P": 0.5, "Q": 0.5}
    assert index[frozenset({"a", "b"})] == {"P": 0.5, "Q": 0.0}


def test_row_column_and_cohort_reordering():
    df, cohorts, weights = _small_baskets()
    weights = [1.0, 1.5, 0.5, 2.0, 1.0]
    base = fpgrowth(
        df, min_support=0.3, cohorts=cohorts, sample_weight=weights, use_colnames=True
    )
    base_index = _index(base)

    permutation = [4, 1, 3, 0, 2]
    reordered = fpgrowth(
        df.iloc[permutation],
        min_support=0.3,
        cohorts=[cohorts[i] for i in permutation],
        sample_weight=[weights[i] for i in permutation],
        use_colnames=True,
    )
    _assert_support_index_equal(_index(reordered), base_index)

    column_reordered = fpgrowth(
        df[["eggs", "bread", "milk"]],
        min_support=0.3,
        cohorts=cohorts,
        sample_weight=weights,
        use_colnames=True,
    )
    _assert_support_index_equal(_index(column_reordered), base_index)

    # relabel: cohort labels are arbitrary strings
    mapping = {"A": "z", "B": "y"}
    relabelled = fpgrowth(
        df,
        min_support=0.3,
        cohorts=[mapping[label] for label in cohorts],
        sample_weight=weights,
        use_colnames=True,
    )
    inverse = {value: key for key, value in mapping.items()}
    relabelled_index = {
        itemset: {inverse[label]: value for label, value in supports.items()}
        for itemset, supports in _index(relabelled).items()
    }
    _assert_support_index_equal(relabelled_index, base_index)


def test_sparse_input_matches_dense():
    df, cohorts, weights = _small_baskets()
    weights = [1.0, 2.0, 1.0, 0.5, 3.0]
    dense = fpgrowth(
        df, min_support=0.3, cohorts=cohorts, sample_weight=weights, use_colnames=True
    )
    sparse_int = df.astype(pd.SparseDtype("int", 0))
    sparse_bool = df.astype(bool).astype(pd.SparseDtype("bool", False))
    for frame in (sparse_int, sparse_bool):
        result = fpgrowth(
            frame,
            min_support=0.3,
            cohorts=cohorts,
            sample_weight=weights,
            use_colnames=True,
        )
        _assert_support_index_equal(_index(result), _index(dense))


def test_sparse_input_is_never_densified(monkeypatch):
    df, cohorts, weights = _small_baskets()
    sparse = df.astype(pd.SparseDtype("int", 0))
    seen = []
    original = fpc._weighted_matrix

    def spy(frame, *args, **kwargs):
        matrix, is_sparse = original(frame, *args, **kwargs)
        seen.append(is_sparse)
        if not is_sparse and hasattr(frame, "sparse"):
            raise AssertionError("sparse input was densified")
        return matrix, is_sparse

    monkeypatch.setattr(fpc, "_weighted_matrix", spy)
    result = fpgrowth(
        sparse, min_support=0.3, cohorts=cohorts, sample_weight=weights
    )
    # the single sparse conversion is reused for counting and insertion
    assert seen == [True]
    assert len(result) > 0

    # Dense input must not be converted into a sparse matrix either; it is
    # scanned in chunks.
    seen.clear()
    fpgrowth(df, min_support=0.3, cohorts=cohorts, sample_weight=weights)
    assert seen == [False]


def test_max_len_and_use_colnames():
    df = pd.DataFrame(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
        ],
        columns=list("abcd"),
    )
    cohorts = ["P", "P", "P", "Q", "Q"]

    unlimited = fpgrowth(df, 0.4, cohorts=cohorts, use_colnames=True)
    assert max(len(itemset) for itemset in unlimited["itemsets"]) >= 3

    limited = fpgrowth(df, 0.4, cohorts=cohorts, max_len=2, use_colnames=True)
    assert max(len(itemset) for itemset in limited["itemsets"]) == 2

    limited_one = fpgrowth(df, 0.4, cohorts=cohorts, max_len=1, use_colnames=True)
    assert max(len(itemset) for itemset in limited_one["itemsets"]) == 1

    positional = fpgrowth(df, 0.4, cohorts=cohorts)
    assert all(isinstance(itemset, frozenset) for itemset in positional["itemsets"])
    assert all(
        all(isinstance(item, (int, np.integer)) for item in itemset)
        for itemset in positional["itemsets"]
    )


def test_empty_result_keeps_schema():
    df = pd.DataFrame({"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]})
    result = fpgrowth(df, 0.75, cohorts=["P", "P", "Q", "Q"], use_colnames=True)
    assert result.columns.tolist() == ["cohort", "support", "itemsets"]
    assert result.empty

    weighted = fpgrowth(df, 0.75, sample_weight=[1, 1, 1, 1])
    assert weighted.columns.tolist() == ["support", "itemsets"]
    assert weighted.empty


# ---------------------------------------------------------------------------
# invalid inputs
# ---------------------------------------------------------------------------


def test_zero_column_input_with_new_modes():
    df = pd.DataFrame([[]])
    assert fpgrowth(df, cohorts=["A"]).columns.tolist() == [
        "cohort",
        "support",
        "itemsets",
    ]
    assert fpgrowth(df, sample_weight=[1.0]).columns.tolist() == [
        "support",
        "itemsets",
    ]
    empty_columns = pd.DataFrame({"a": [], "b": []})
    assert fpgrowth(empty_columns, cohorts=[]).empty
    assert fpgrowth(empty_columns, sample_weight=[]).empty


def test_invalid_cohorts_and_weights():
    df = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})

    with pytest.raises(ValueError, match="length equal to the number of rows"):
        fpgrowth(df, cohorts=["P", "Q"])
    with pytest.raises(ValueError, match="length equal to the number of rows"):
        fpgrowth(df, sample_weight=[1.0, 1.0])

    with pytest.raises(ValueError, match="non-empty strings"):
        fpgrowth(df, cohorts=["P", None, "Q"])
    with pytest.raises(ValueError, match="non-empty strings"):
        fpgrowth(df, cohorts=["P", "", "Q"])
    with pytest.raises(ValueError, match="non-empty strings"):
        fpgrowth(df, cohorts=[1, 1, 2])

    with pytest.raises(ValueError, match="non-negative"):
        fpgrowth(df, cohorts=["P", "P", "Q"], sample_weight=[1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        fpgrowth(df, cohorts=["P", "P", "Q"], sample_weight=[1.0, np.inf, 1.0])
    with pytest.raises(ValueError, match="finite"):
        fpgrowth(df, sample_weight=[1.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="positive total weight"):
        fpgrowth(
            df, cohorts=["P", "P", "Q"], sample_weight=[1.0, 1.0, 0.0]
        )

    with pytest.raises(ValueError, match="NaN"):
        fpgrowth(
            pd.DataFrame({"a": [1, np.nan, 0], "b": [0, 1, 1]}),
            cohorts=["P", "Q", "Q"],
        )
    with pytest.raises(ValueError, match="NaN"):
        fpgrowth(
            pd.DataFrame({"a": [1, np.nan, 0], "b": [0, 1, 1]}),
            sample_weight=[1.0, 1.0, 1.0],
        )

    duplicated = pd.DataFrame([[1, 0, 1], [0, 1, 1]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="Duplicate item column names"):
        fpgrowth(duplicated, cohorts=["P", "Q"])
    with pytest.raises(ValueError, match="Duplicate item column names"):
        fpgrowth(duplicated, sample_weight=[1.0, 1.0])

    with pytest.raises(ValueError, match="null_values"):
        fpgrowth(df, cohorts=["P", "P", "Q"], null_values=True)

    # ordinary mode keeps its own validation and columns
    assert fpgrowth(df, 0.5).columns.tolist() == ["support", "itemsets"]


def test_non_1d_cohort_sequences_rejected():
    df = pd.DataFrame({"a": [1, 0], "b": [0, 1]})
    with pytest.raises(ValueError, match="one-dimensional"):
        fpgrowth(df, cohorts=np.array([["P", "Q"]]))
    with pytest.raises(ValueError, match="one-dimensional"):
        fpgrowth(df, cohorts=pd.DataFrame({"c": ["P", "Q"], "d": ["P", "Q"]}))
    # a one-column Series/DataFrame is a positional sequence
    ok = fpgrowth(
        df, cohorts=pd.DataFrame({"c": ["P", "Q"]}), sample_weight=[1, 1]
    )
    assert set(ok["cohort"]) == {"P", "Q"}


# ---------------------------------------------------------------------------
# association rules
# ---------------------------------------------------------------------------


def test_rules_match_direct_statistics():
    df, cohorts, _ = _small_baskets()
    weights = [1.0, 2.0, 1.0, 0.5, 3.0]
    itemsets = fpgrowth(
        df,
        min_support=0.3,
        cohorts=cohorts,
        sample_weight=weights,
        use_colnames=True,
    )
    index = _index(itemsets)

    reference = "B"
    rules = association_rules(
        itemsets,
        min_threshold=0.0,
        reference_cohort=reference,
    )
    assert set(rules["cohort"]) == {"A"}
    assert set(rules["reference_cohort"]) == {reference}

    seen = set()
    for row in rules.to_dict("records"):
        antecedent, consequent = row["antecedents"], row["consequents"]
        cohort = row["cohort"]
        expected_support, expected_confidence_lift = _direct_rule_evidence(
            index, cohort, antecedent, consequent
        )
        seen.add((cohort, antecedent, consequent))
        assert row["support"] == pytest.approx(expected_support)
        assert row["confidence"] == pytest.approx(expected_confidence_lift[0])
        if np.isnan(expected_confidence_lift[1]):
            assert np.isnan(row["lift"])
        else:
            assert row["lift"] == pytest.approx(expected_confidence_lift[1])

        ref_support, ref_confidence_lift = _direct_rule_evidence(
            index, reference, antecedent, consequent
        )
        assert row["reference support"] == pytest.approx(ref_support)
        if ref_confidence_lift is None:
            assert np.isnan(row["reference confidence"])
            assert np.isnan(row["reference lift"])
            assert np.isnan(row["confidence delta"])
        else:
            assert row["reference confidence"] == pytest.approx(
                ref_confidence_lift[0]
            )
            assert row["confidence delta"] == pytest.approx(
                expected_confidence_lift[0] - ref_confidence_lift[0]
            )
        assert row["support delta"] == pytest.approx(expected_support - ref_support)

    # every rule with a non-zero current antecedent support is present
    expected = set()
    for itemset, supports in index.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent_items in itertools.combinations(itemset, size):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                if index[antecedent]["A"] > 0:
                    expected.add(("A", antecedent, consequent))
    assert seen == expected


def test_rule_skips_zero_current_antecedent_support():
    # cohort A never contains "eggs" or "b", so A->B rules on those items have
    # zero current antecedent support and must be absent for cohort A.
    df = pd.DataFrame({"a": [1, 1, 0, 0], "b": [0, 0, 1, 1]})
    itemsets = fpgrowth(df, 0.5, cohorts=["A", "A", "B", "B"], use_colnames=True)
    rules = association_rules(itemsets, min_threshold=0.0, reference_cohort="B")
    assert rules.empty or all(
        row.antecedent_support > 0 for row in rules.itertuples(index=False)
    )
    # explicit check for the rule b -> a in cohort A: support(A) = 0
    assert not (
        (rules["cohort"] == "A")
        & (rules["antecedents"] == frozenset({"b"}))
    ).any()


def test_zero_reference_antecedent_gives_nan_delta_and_fails_filter():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 0, 0, 0],
            "b": [1, 1, 0, 0, 1, 1],
            "c": [1, 1, 0, 0, 0, 0],
        }
    )
    cohorts = ["A", "A", "A", "B", "B", "B"]
    itemsets = fpgrowth(df, 0.5, cohorts=cohorts, use_colnames=True)
    index = _index(itemsets)
    assert index[frozenset({"a"})]["B"] == 0.0

    rules = association_rules(itemsets, min_threshold=0.0, reference_cohort="B")
    # a is never in B: any A rule with antecedent containing a has zero
    # reference antecedent support -> NaN reference confidence/delta.
    affected = rules[rules["antecedents"].apply(lambda s: "a" in s)]
    assert not affected.empty
    assert affected["reference confidence"].isna().all()
    assert affected["confidence delta"].isna().all()

    filtered = association_rules(
        itemsets,
        min_threshold=0.0,
        reference_cohort="B",
        min_confidence_delta=-1e9,
    )
    # NaN deltas can never pass a lower bound
    assert filtered["confidence delta"].notna().all()
    assert not (
        filtered["antecedents"].apply(lambda s: "a" in s)
    ).any()


def test_zero_consequent_support_lift_is_nan_per_cohort():
    # Cohort Q contains a+b twice and c twice, so Q produces the rule a -> b.
    # Cohort P contains c only, hence the consequent b has zero support in P
    # and the P version of a -> b (mined because the itemset a+b survived in
    # Q) must have a NaN lift.
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 1],
            "b": [0, 0, 1, 1],
        }
    )
    cohorts = ["P", "P", "Q", "Q"]
    itemsets = fpgrowth(df, 0.5, cohorts=cohorts, use_colnames=True)
    rules = association_rules(itemsets, min_threshold=0.0)
    p_rules = rules[
        (rules["cohort"] == "P")
        & (rules["antecedents"] == frozenset({"a"}))
        & (rules["consequents"] == frozenset({"b"}))
    ]
    assert len(p_rules) == 1
    assert p_rules.iloc[0]["consequent support"] == 0.0
    assert np.isnan(p_rules.iloc[0]["lift"])


def test_rules_without_reference_per_cohort():
    df, cohorts, _ = _small_baskets()
    itemsets = fpgrowth(df, 0.3, cohorts=cohorts, use_colnames=True)

    rules = association_rules(itemsets, min_threshold=0.5)
    assert "cohort" in rules.columns
    assert "reference_cohort" not in rules.columns
    assert not any(column.startswith("reference ") for column in rules.columns)
    assert set(rules["cohort"]) == {"A", "B"}

    # each cohort's rules must satisfy the within-cohort threshold
    assert (rules["confidence"] >= 0.5).all()

    with pytest.raises(ValueError, match="min_confidence_delta"):
        association_rules(itemsets, min_threshold=0.5, min_confidence_delta=0.1)


def test_metric_and_threshold_filtering_per_cohort():
    df, cohorts, _ = _small_baskets()
    itemsets = fpgrowth(df, 0.3, cohorts=cohorts, use_colnames=True)
    rules = association_rules(itemsets, metric="lift", min_threshold=1.0)
    assert (rules["lift"] >= 1.0).all()

    with pytest.raises(ValueError, match="Metric must be"):
        association_rules(itemsets, metric="not-a-metric")


def test_cohort_mode_rejects_null_values_and_support_only():
    df, cohorts, _ = _small_baskets()
    itemsets = fpgrowth(df, 0.3, cohorts=cohorts, use_colnames=True)
    with pytest.raises(ValueError, match="support_only"):
        association_rules(itemsets, support_only=True)
    with pytest.raises(ValueError, match="null_values"):
        association_rules(itemsets, null_values=True)
    with pytest.raises(ValueError, match="null_values"):
        association_rules(itemsets, df_orig=df)


def test_rule_input_validation():
    itemsets = pd.DataFrame(
        {
            "cohort": ["A", "A", "B", "B"],
            "support": [0.5, 0.25, 0.5, 0.25],
            "itemsets": [
                frozenset({"x"}),
                frozenset({"x", "y"}),
                frozenset({"x"}),
                frozenset({"x", "y"}),
            ],
        }
    )

    # duplicate (cohort, itemset)
    duplicated = pd.concat([itemsets, itemsets.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate \\(cohort, itemset\\)"):
        association_rules(duplicated, min_threshold=0.0)

    # incomplete cohort: A has {x}, {x,y}; B has {x} twice (duplicate)
    duplicated_in_b = itemsets.copy()
    duplicated_in_b.loc[3, "itemsets"] = frozenset({"x"})
    with pytest.raises(ValueError, match="Duplicate \\(cohort, itemset\\)"):
        association_rules(duplicated_in_b, min_threshold=0.0)

    # incomplete cohort: B simply lacks the {x,y} row
    incomplete = itemsets.iloc[[0, 1, 2]].copy()
    with pytest.raises(ValueError, match="missing|complete"):
        association_rules(incomplete, min_threshold=0.0)

    # Extra row in one cohort but same universe in the other -> also rejected.
    extra = itemsets.copy()
    extra.loc[2, "itemsets"] = frozenset({"z"})
    with pytest.raises(ValueError, match="missing|complete"):
        association_rules(extra, min_threshold=0.0)

    # missing subset support for a rule: y is absent everywhere while x+y
    # survives, so the antecedent/consequent lookup must fail loudly.
    missing_subset = itemsets.copy()
    missing_subset["itemsets"] = [
        frozenset({"x"}),
        frozenset({"x", "y"}),
        frozenset({"x"}),
        frozenset({"x", "y"}),
    ]
    with pytest.raises(ValueError, match="no support entry|complete"):
        association_rules(missing_subset, min_threshold=0.0)

    with pytest.raises(ValueError, match="reference_cohort"):
        association_rules(itemsets, reference_cohort="missing", min_threshold=0.0)


def test_reference_parameters_rejected_for_plain_input():
    df, _, _ = _small_baskets()
    plain = fpgrowth(df, 0.5, use_colnames=True)
    with pytest.raises(ValueError, match="reference_cohort"):
        association_rules(plain, reference_cohort="A")
    with pytest.raises(ValueError, match="min_confidence_delta"):
        association_rules(plain, min_confidence_delta=0.1)


# ---------------------------------------------------------------------------
# ordinary mode is unchanged
# ---------------------------------------------------------------------------


def test_plain_fpgrowth_fpmax_and_rules_regression():
    df = pd.DataFrame(
        [
            [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        ],
        columns=[
            "Apple",
            "Corn",
            "Dill",
            "Eggs",
            "Ice cream",
            "Kidney Beans",
            "Milk",
            "Nutmeg",
            "Onion",
            "Unicorn",
            "Yogurt",
        ],
    )
    res = fpgrowth(df, 0.6)
    assert res.columns.tolist() == ["support", "itemsets"]
    assert res[res["support"] == 1.0]["itemsets"].tolist() == [frozenset({5})]

    maxed = fpmax(df, 0.6)
    assert len(maxed) == 3
    assert max(len(itemset) for itemset in maxed["itemsets"]) == 3

    rules = association_rules(res, len(df))
    assert len(rules) == 9
    assert "cohort" not in rules.columns
    weighted = fpgrowth(df, 0.6, sample_weight=[1.0] * len(df))
    assert np.allclose(
        sorted(res["support"]), sorted(weighted["support"])
    )


def test_weighted_matches_row_duplication():
    rng = np.random.default_rng(2)
    array = rng.random((30, 5)) > 0.55
    df = pd.DataFrame(array, columns=list("abcde"))
    weights = rng.integers(1, 4, size=30)

    duplicated = pd.concat(
        [df.iloc[[row]].astype(int) for row in df.index for _ in range(weights[row])],
        ignore_index=True,
    )
    weighted = fpgrowth(df, 0.25, sample_weight=weights, use_colnames=True)
    expanded = fpgrowth(duplicated, 0.25, use_colnames=True)

    weighted = weighted.sort_values("itemsets").reset_index(drop=True)
    expanded = expanded.sort_values("itemsets").reset_index(drop=True)
    assert weighted["itemsets"].tolist() == expanded["itemsets"].tolist()
    assert np.allclose(weighted["support"], expanded["support"])


# ---------------------------------------------------------------------------
# shared search / workload
# ---------------------------------------------------------------------------


def _shared_prefix_data(n_baskets=800, n_cohorts=8, seed=0):
    """Baskets made of a long shared prefix plus one cohort-specific item.

    This is the regime where a naive "split the data and mine each cohort"
    strategy duplicates almost the whole tree: the 10-item prefix would be
    rebuilt once per cohort instead of once in total.
    """
    rng = np.random.default_rng(seed)
    common = list(range(10))
    n_items = 10 + n_cohorts
    rows = np.zeros((n_baskets, n_items), dtype=np.int8)
    cohorts = []
    for row in range(n_baskets):
        cohort = int(rng.integers(n_cohorts))
        cohorts.append(f"c{cohort}")
        rows[row, common] = 1
        rows[row, 10 + cohort] = 1
    columns = [f"i{i}" for i in range(n_items)]
    return pd.DataFrame(rows, columns=columns), cohorts


def _tree_node_count(tree):
    nodes = 0
    stack = [tree.root]
    while stack:
        node = stack.pop()
        nodes += 1
        stack.extend(node.children.values())
    return nodes


def test_shared_tree_does_not_mine_each_cohort_separately():
    df, cohorts = _shared_prefix_data()
    codes, labels, totals, weights = fpc.prepare_cohorts(df, cohorts, None)

    shared_tree, _, _ = fpc.setup_fptree(
        df,
        0.05,
        sample_weight=weights,
        cohort_codes=codes,
        cohort_totals=totals,
    )
    shared_nodes = _tree_node_count(shared_tree)

    codes_array = np.asarray(codes)
    independent_nodes = 0
    for code in range(len(labels)):
        mask = codes_array == code
        sub = df.iloc[np.nonzero(mask)[0]]
        sub_weights = weights[mask]
        sub_tree, _, _ = fpc.setup_fptree(
            sub,
            0.05,
            sample_weight=sub_weights,
            cohort_codes=np.zeros(len(sub), dtype=np.int64),
            cohort_totals=[float(sub_weights.sum())],
        )
        independent_nodes += _tree_node_count(sub_tree)

    # The shared tree stores the common 10-item prefix once; independent
    # cohort trees store it once per cohort.
    assert shared_nodes < independent_nodes * 0.5

    # Same evidence is exposed by the shared tree: every cohort's counter on
    # each common-prefix item equals the weight of that cohort's baskets (all
    # baskets contain the common prefix).  A scalar shared tree could not
    # express these distinct per-cohort numbers at all.
    for code in range(len(labels)):
        mask = codes_array == code
        expected = float(weights[mask].sum())
        assert shared_tree.root.count[code] == pytest.approx(expected)
        for item in range(10):
            node = shared_tree.nodes[item][0]
            assert node.count[code] == pytest.approx(expected)


def test_shared_tree_memory_is_below_independent_trees():
    import gc

    df, cohorts = _shared_prefix_data(n_baskets=2000, n_cohorts=8, seed=1)
    codes, labels, totals, weights = fpc.prepare_cohorts(df, cohorts, None)
    codes_array = np.asarray(codes)

    gc.collect()
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    shared_tree, _, _ = fpc.setup_fptree(
        df,
        0.01,
        sample_weight=weights,
        cohort_codes=codes,
        cohort_totals=totals,
    )
    shared_retained = tracemalloc.get_traced_memory()[0] - baseline
    tracemalloc.stop()
    del shared_tree

    gc.collect()
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    independent_trees = []
    for code in range(len(labels)):
        mask = codes_array == code
        sub = df.iloc[np.nonzero(mask)[0]]
        sub_weights = weights[mask]
        sub_tree, _, _ = fpc.setup_fptree(
            sub,
            0.01,
            sample_weight=sub_weights,
            cohort_codes=np.zeros(len(sub), dtype=np.int64),
            cohort_totals=[float(sub_weights.sum())],
        )
        independent_trees.append(sub_tree)
    independent_retained = tracemalloc.get_traced_memory()[0] - baseline
    tracemalloc.stop()
    del independent_trees

    assert shared_retained > 0 and independent_retained > 0
    assert shared_retained < independent_retained * 0.5


def test_cohort_work_is_linear_in_rows_not_cohorts():
    """Insertion work counts each basket once, regardless of cohort count."""
    df, cohorts = _shared_prefix_data(n_baskets=600, n_cohorts=16, seed=2)
    codes, labels, totals, weights = fpc.prepare_cohorts(df, cohorts, None)

    original_insert = fpc.FPTree.insert_itemset
    calls = {"count": 0}

    def counting_insert(self, itemset, count=1):
        calls["count"] += 1
        return original_insert(self, itemset, count)

    fpc.FPTree.insert_itemset = counting_insert
    try:
        fpc.setup_fptree(
            df,
            0.01,
            sample_weight=weights,
            cohort_codes=codes,
            cohort_totals=totals,
        )
    finally:
        fpc.FPTree.insert_itemset = original_insert

    # Exactly one insertion per non-empty basket, *not* one per basket/cohort.
    non_empty = int((df.to_numpy().sum(axis=1) > 0).sum())
    assert calls["count"] == non_empty
