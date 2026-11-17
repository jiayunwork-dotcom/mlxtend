# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Function for generating association rules
#
# Author: Joshua Goerner <https://github.com/JoshuaGoerner>
#         Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from ..frequent_patterns import fpcommon as fpc

_metrics = [
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "representativity",
    "leverage",
    "conviction",
    "zhangs_metric",
    "jaccard",
    "certainty",
    "kulczynski",
]


def association_rules(
    df: pd.DataFrame,
    num_itemsets: Optional[int] = 1,
    df_orig: Optional[pd.DataFrame] = None,
    null_values=False,
    metric="confidence",
    min_threshold=0.8,
    support_only=False,
    return_metrics: list = _metrics,
    reference_cohort=None,
    min_confidence_delta=None,
) -> pd.DataFrame:
    """Generates a DataFrame of association rules including the
    metrics 'score', 'confidence', and 'lift'

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame of frequent itemsets
      with columns ['support', 'itemsets'], or the cohort-aware layout
      ['cohort', 'support', 'itemsets'] produced by
      ``fpgrowth(..., cohorts=...)``.

    df_orig : pandas DataFrame (default: None)
      DataFrame with original input data. Only provided when null_values exist

    num_itemsets : int (default: 1)
      Number of transactions in original input data (df_orig)

    null_values : bool (default: False)
      In case there are null values as NaNs in the original input data.
      Not supported for cohort input.

    metric : string (default: 'confidence')
      Metric to evaluate if a rule is of interest.
      **Automatically set to 'support' if `support_only=True`.**
      Otherwise, supported metrics are 'support', 'confidence', 'lift',
      'leverage', 'conviction' and 'zhangs_metric'
      These metrics are computed as follows:

      - support(A->C) = support(A+C) [aka 'support'], range: [0, 1]\n
      - confidence(A->C) = support(A+C) / support(A), range: [0, 1]\n
      - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]\n
      - leverage(A->C) = support(A->C) - support(A)*support(C),
        range: [-1, 1]\n
      - conviction = [1 - support(C)] / [1 - confidence(A->C)],
        range: [0, inf]\n
      - zhangs_metric(A->C) =
        leverage(A->C) / max(support(A->C)*(1-support(A)), support(A)*(support(C)-support(A->C)))
        range: [-1,1]\n

    min_threshold : float (default: 0.8)
      Minimal threshold for the evaluation metric,
      via the `metric` parameter,
      to decide whether a candidate rule is of interest.  For cohort input
      this threshold is applied to the rule's metric *within its own cohort*.

    support_only : bool (default: False)
      Only computes the rule support and fills the other
      metric columns with NaNs. This is useful if:

      a) the input DataFrame is incomplete, e.g., does
      not contain support values for all rule antecedents
      and consequents

      b) you simply want to speed up the computation because
      you don't need the other metrics.

      Not supported for cohort input.

    reference_cohort : str (default: None)
      Cohort used as the comparison baseline.  When provided (cohort input
      only), rules are generated for every *other* cohort and augmented
      with ``reference_cohort``, ``reference support``,
      ``reference confidence``, ``reference lift``, ``support delta`` and
      ``confidence delta``.  Both deltas are the current cohort's value minus
      the reference cohort's value.  Without this argument, ordinary (non
      comparative) rules are produced for each cohort separately and no
      reference columns are added.

    min_confidence_delta : float (default: None)
      Optional lower bound on ``confidence delta``.  Requires
      `reference_cohort`; rules whose delta is NaN (because the reference
      cohort has zero antecedent support) can never pass it.

    Returns
    ----------
    pandas DataFrame with columns "antecedents" and "consequents"
      that store itemsets, plus the scoring metric columns:
      "antecedent support", "consequent support",
      "support", "confidence", "lift",
      "leverage", "conviction"
      of all rules for which
      metric(rule) >= min_threshold.
      For cohort input a "cohort" column is present, and with
      `reference_cohort` the reference/delta columns described above are
      appended.
      Each entry in the "antecedents" and "consequents" columns are
      of type `frozenset`, which is a Python built-in type that
      behaves similarly to sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

    """
    is_cohort_input = "cohort" in df.columns

    if is_cohort_input:
        return _cohort_association_rules(
            df,
            num_itemsets=num_itemsets,
            metric=metric,
            min_threshold=min_threshold,
            support_only=support_only,
            null_values=null_values,
            df_orig=df_orig,
            return_metrics=return_metrics,
            reference_cohort=reference_cohort,
            min_confidence_delta=min_confidence_delta,
        )

    if reference_cohort is not None:
        raise ValueError(
            "`reference_cohort` is only supported for cohort-aware input "
            "with a 'cohort' column, as produced by "
            "`fpgrowth(..., cohorts=...)`."
        )
    if min_confidence_delta is not None:
        raise ValueError(
            "`min_confidence_delta` requires cohort-aware input with a "
            "'cohort' column together with `reference_cohort`."
        )

    # if null values exist, df_orig must be provided
    if null_values and df_orig is None:
        raise TypeError("If null values exist, df_orig must be provided.")

    # if null values exist, num_itemsets must be provided
    if null_values and num_itemsets == 1:
        raise TypeError("If null values exist, num_itemsets must be provided.")

    # check for valid input
    fpc.valid_input_check(df_orig, null_values)

    if not df.shape[0]:
        raise ValueError(
            "The input DataFrame `df` containing " "the frequent itemsets is empty."
        )

    # check for mandatory columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError(
            "Dataframe needs to contain the\
                         columns 'support' and 'itemsets'"
        )

    def kulczynski_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        conf_AC = sAC * (num_itemsets - disAC) / (sA * (num_itemsets - disA) - dis_int)
        conf_CA = sAC * (num_itemsets - disAC) / (sC * (num_itemsets - disC) - dis_int_)
        kulczynski = (conf_AC + conf_CA) / 2
        return kulczynski

    def conviction_helper(conf, sC):
        conviction = np.empty(conf.shape, dtype=float)
        if not len(conviction.shape):
            conviction = conviction[np.newaxis]
            conf = conf[np.newaxis]
            sC = sC[np.newaxis]
        conviction[:] = np.inf
        conviction[conf < 1.0] = (1.0 - sC[conf < 1.0]) / (1.0 - conf[conf < 1.0])

        return conviction

    def zhangs_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
        numerator = metric_dict["leverage"](
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            # ignoring the divide by 0 warning since it is addressed in the below np.where
            zhangs_metric = np.where(denominator == 0, 0, numerator / denominator)

        return zhangs_metric

    def jaccard_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        numerator = metric_dict["support"](
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        )
        denominator = sA + sC - numerator

        jaccard_metric = numerator / denominator
        return jaccard_metric

    def certainty_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        certainty_num = (
            metric_dict["confidence"](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
            - sC
        )
        certainty_denom = 1 - sC

        cert_metric = np.where(certainty_denom == 0, 0, certainty_num / certainty_denom)
        return cert_metric

    # metrics for association rules
    metric_dict = {
        "antecedent support": lambda _, sA, ___, ____, _____, ______, _______, ________: sA,
        "consequent support": lambda _, __, sC, ____, _____, ______, _______, ________: sC,
        "support": lambda sAC, _, __, ___, ____, _____, ______, _______: sAC,
        "confidence": lambda sAC, sA, _, disAC, disA, __, dis_int, ___: (
            sAC * (num_itemsets - disAC)
        )
        / (sA * (num_itemsets - disA) - dis_int),
        "lift": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: metric_dict[
            "confidence"
        ](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
        / sC,
        "representativity": lambda _, __, ___, disAC, ____, ______, _______, ________: (
            num_itemsets - disAC
        )
        / num_itemsets,
        "leverage": lambda sAC, sA, sC, _, __, ____, _____, ______: metric_dict[
            "support"
        ](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
        - sA * sC,
        "conviction": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: conviction_helper(
            metric_dict["confidence"](
                sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
            ),
            sC,
        ),
        "zhangs_metric": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: zhangs_metric_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
        "jaccard": lambda sAC, sA, sC, _, __, ____, _____, ______: jaccard_metric_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
        "certainty": lambda sAC, sA, sC, _, __, ____, _____, ______: certainty_metric_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
        "kulczynski": lambda sAC, sA, sC, _, __, ____, _____, ______: kulczynski_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
    }

    # check for metric compliance
    if support_only:
        metric = "support"
    else:
        if metric not in metric_dict.keys():
            raise ValueError(
                "Metric must be 'confidence' or 'lift', got '{}'".format(metric)
            )

    # get dict of {frequent itemset} -> support
    keys = df["itemsets"].values
    values = df["support"].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    # prepare buckets to collect frequent rules
    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    # Define the disabled df, assign columns from original df to be the same on the disabled.
    if null_values:
        disabled = df_orig.copy()
        disabled = np.where(pd.isna(disabled), 1, np.nan) + np.where(
            (disabled == 0) | (disabled == 1), np.nan, 0
        )
        disabled = pd.DataFrame(disabled)
        if all(isinstance(key, str) for key in list(frequent_items_dict.keys())[0]):
            disabled.columns = df_orig.columns

        if all(
            isinstance(key, np.int64) for key in list(frequent_items_dict.keys())[0]
        ):
            cols = np.arange(0, len(df_orig.columns), 1)
            disabled.columns = cols
            df_orig.columns = cols

    # iterate over all frequent itemsets
    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]
        # to find all possible combinations
        for idx in range(len(k) - 1, 0, -1):
            # of antecedent and consequent
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)

                if support_only:
                    # support doesn't need these,
                    # hence, placeholders should suffice
                    sA = None
                    sC = None
                    disAC, disA, disC, dis_int, dis_int_ = 0, 0, 0, 0, 0

                else:
                    try:
                        sA = frequent_items_dict[antecedent]
                        sC = frequent_items_dict[consequent]

                        # if the input dataframe is complete
                        if not null_values:
                            disAC, disA, disC, dis_int, dis_int_ = 0, 0, 0, 0, 0

                        else:
                            an = list(antecedent)
                            con = list(consequent)
                            an.extend(con)

                            # select data of antecedent, consequent and combined from disabled
                            dec = disabled.loc[:, an]
                            _dec = disabled.loc[:, list(antecedent)]
                            __dec = disabled.loc[:, list(consequent)]

                            # select data of antecedent and consequent from original
                            dec_ = df_orig.loc[:, list(antecedent)]
                            dec__ = df_orig.loc[:, list(consequent)]

                            # disabled counts
                            disAC, disA, disC, dis_int, dis_int_ = 0, 0, 0, 0, 0
                            for i in range(len(dec.index)):
                                # select the i-th iset from the disabled dataset
                                item_comb = list(dec.iloc[i, :])
                                item_dis_an = list(_dec.iloc[i, :])
                                item_dis_con = list(__dec.iloc[i, :])

                                # select the i-th iset from the original dataset
                                item_or_an = list(dec_.iloc[i, :])
                                item_or_con = list(dec__.iloc[i, :])

                                # check and keep count if there is a null value in combined, antecedent, consequent
                                if 1 in set(item_comb):
                                    disAC += 1
                                if 1 in set(item_dis_an):
                                    disA += 1
                                if 1 in item_dis_con:
                                    disC += 1

                                # check and keep count if there is a null value in consequent AND all items are present in antecedent
                                if (1 in item_dis_con) and all(
                                    j == 1 for j in item_or_an
                                ):
                                    dis_int += 1

                                # check and keep count if there is a null value in antecedent AND all items are present in consequent
                                if (1 in item_dis_an) and all(
                                    j == 1 for j in item_or_con
                                ):
                                    dis_int_ += 1

                    except KeyError as e:
                        s = (
                            str(e) + "You are likely getting this error"
                            " because the DataFrame is missing "
                            " antecedent and/or consequent "
                            " information."
                            " You can try using the "
                            " `support_only=True` option"
                        )
                        raise KeyError(s)
                    # check for the threshold

                score = metric_dict[metric](
                    sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
                )
                if score >= min_threshold:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append(
                        [sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_]
                    )

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)

    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["antecedents", "consequents"],
        )

        if support_only:
            sAC = rule_supports[0]
            for m in return_metrics:
                df_res[m] = np.nan
            df_res["support"] = sAC

        else:
            sAC = rule_supports[0]
            sA = rule_supports[1]
            sC = rule_supports[2]
            disAC = rule_supports[3]
            disA = rule_supports[4]
            disC = rule_supports[5]
            dis_int = rule_supports[6]
            dis_int_ = rule_supports[7]

            for m in return_metrics:
                df_res[m] = metric_dict[m](
                    sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
                )

        return df_res


_COHORT_REFERENCE_COLUMNS = [
    "reference_cohort",
    "reference support",
    "reference confidence",
    "reference lift",
    "support delta",
    "confidence delta",
]


def _cohort_rule_metrics(sAC, sA, sC):
    """Compute rule metrics from *within-cohort* supports.

    All three supports come from the same cohort.  Nothing is re-aligned
    across cohorts, and a below-threshold itemset is not silently turned into
    zero support: the caller validates the index first.

    NaN semantics:
    * ``confidence``/``lift`` are NaN when the antecedent support is zero.
      Callers skip those rules entirely.
    * ``lift`` is NaN when the consequent support is zero in this cohort.
    * ``representativity`` is 1 because cohort layouts carry no null-value
      bookkeeping (a cohort's baskets are fully attributable to it).
    """
    nan = float("nan")
    confidence = sAC / sA if sA > 0 else nan
    lift = confidence / sC if (sC > 0 and confidence == confidence) else nan
    leverage = sAC - sA * sC
    if confidence < 1.0:
        conviction = (1.0 - sC) / (1.0 - confidence)
    else:
        conviction = float("inf")
    denominator = max(sAC * (1 - sA), sA * (sC - sAC))
    zhangs = leverage / denominator if denominator != 0 else 0.0
    jaccard_denom = sA + sC - sAC
    jaccard = sAC / jaccard_denom if jaccard_denom != 0 else nan
    certainty_denom = 1.0 - sC
    certainty = (confidence - sC) / certainty_denom if certainty_denom != 0 else 0.0
    confidence_ca = sAC / sC if sC > 0 else nan
    kulczynski = (
        (confidence + confidence_ca) / 2.0 if confidence == confidence else nan
    )
    return {
        "antecedent support": sA,
        "consequent support": sC,
        "support": sAC,
        "confidence": confidence,
        "lift": lift,
        "representativity": 1.0,
        "leverage": leverage,
        "conviction": conviction,
        "zhangs_metric": zhangs,
        "jaccard": jaccard,
        "certainty": certainty,
        "kulczynski": kulczynski,
    }


def _cohort_association_rules(
    df,
    num_itemsets,
    metric,
    min_threshold,
    support_only,
    null_values,
    df_orig,
    return_metrics,
    reference_cohort,
    min_confidence_delta,
):
    if support_only:
        raise ValueError(
            "`support_only=True` is not supported for cohort-aware input."
        )
    if null_values or df_orig is not None:
        raise ValueError(
            "`null_values`/`df_orig` are not supported for cohort-aware input."
        )
    if reference_cohort is None and min_confidence_delta is not None:
        raise ValueError(
            "`min_confidence_delta` requires `reference_cohort` to be set."
        )
    if min_confidence_delta is not None:
        try:
            min_confidence_delta = float(min_confidence_delta)
        except (TypeError, ValueError):
            raise ValueError("`min_confidence_delta` must be a number.")
        if not np.isfinite(min_confidence_delta):
            raise ValueError("`min_confidence_delta` must be finite.")

    if not all(col in df.columns for col in ["cohort", "support", "itemsets"]):
        raise ValueError(
            "Cohort-aware input needs the columns 'cohort', 'support' and "
            "'itemsets'."
        )
    if not df.shape[0]:
        raise ValueError(
            "The input DataFrame `df` containing the frequent itemsets is empty."
        )

    # Validate metrics against the ordinary metric vocabulary.
    if metric not in _metrics:
        raise ValueError(
            "Metric must be 'confidence' or 'lift', got '{}'".format(metric)
        )

    valid_metric_names = set(_metrics)
    missing_return = [m for m in return_metrics if m not in valid_metric_names]
    if missing_return:
        raise ValueError(
            "Unknown return metric(s): %s." % ", ".join(map(str, missing_return))
        )

    # --- Build the per-cohort support index -------------------------------
    # Rows are read positionally.  No groupby/dict comprehension that would
    # collapse *all* rows of an itemset into a single scalar is used: each
    # (cohort, itemset) pair must appear at most once, and completeness of
    # every cohort's itemset set is required for antecedent/consequent lookup.
    cohort_order = []
    per_cohort = {}
    duplicate_pairs = []
    for cohort, support, itemset in zip(
        df["cohort"].to_numpy(), df["support"].to_numpy(), df["itemsets"].to_numpy()
    ):
        if not isinstance(cohort, str) or cohort == "":
            raise ValueError(
                "The 'cohort' column must contain non-empty strings; got %r."
                % (cohort,)
            )
        try:
            support = float(support)
        except (TypeError, ValueError):
            raise ValueError(
                "The 'support' column must contain numeric values; got %r."
                % (support,)
            )
        if not np.isfinite(support) or support < 0 or support > 1:
            raise ValueError(
                "The 'support' column must contain finite values in [0, 1]; "
                "got %r." % (support,)
            )
        key = frozenset(itemset)
        if cohort not in per_cohort:
            per_cohort[cohort] = {}
            cohort_order.append(cohort)
        if key in per_cohort[cohort]:
            duplicate_pairs.append((cohort, key))
            continue
        per_cohort[cohort][key] = support

    if duplicate_pairs:
        cohort, key = duplicate_pairs[0]
        raise ValueError(
            "Duplicate (cohort, itemset) rows are not allowed; found %r in "
            "cohort %r." % (sorted(key, key=repr), cohort)
        )

    # Every cohort must expose the same itemset universe.  A missing itemset
    # is *not* interpreted as zero support: that would silently turn an
    # unfinished mining result into a confidence of zero.
    universe = set(per_cohort[cohort_order[0]])
    for cohort in cohort_order[1:]:
        if set(per_cohort[cohort]) != universe:
            missing = sorted(
                set(per_cohort[cohort_order[0]]) - set(per_cohort[cohort]),
                key=lambda s: sorted(map(repr, s)),
            )
            if missing:
                raise ValueError(
                    "Cohort %r is missing %d itemset(s) present in the other "
                    "cohorts (e.g. %r).  The cohort input from fpgrowth must "
                    "be complete; missing supports are not filled with zero."
                    % (cohort, len(missing), sorted(missing[0], key=repr))
                )
            raise ValueError(
                "Cohort %r contains itemset(s) missing from cohort %r."
                % (cohort, cohort_order[0])
            )

    if reference_cohort is not None:
        if reference_cohort not in per_cohort:
            raise ValueError(
                "`reference_cohort` %r is not one of the cohorts in the input "
                "(%s)." % (reference_cohort, ", ".join(map(repr, cohort_order)))
            )
        if len(cohort_order) < 2:
            raise ValueError(
                "`reference_cohort` requires at least two cohorts to compare."
            )

    # --- Generate rules per cohort, reusing the same evidence -------------
    rows = []
    for cohort in cohort_order:
        if reference_cohort is not None and cohort == reference_cohort:
            continue
        supports = per_cohort[cohort]
        for itemset in supports:
            if len(itemset) < 2:
                continue
            sAC = supports[itemset]
            for idx in range(len(itemset) - 1, 0, -1):
                for antecedent_items in combinations(itemset, r=idx):
                    antecedent = frozenset(antecedent_items)
                    consequent = itemset.difference(antecedent)
                    sA = _cohort_support(supports, antecedent, cohort)
                    sC = _cohort_support(supports, consequent, cohort)
                    if sA <= 0:
                        # A rule with no antecedent evidence is not a rule.
                        continue
                    computed = _cohort_rule_metrics(
                        np.float64(sAC), np.float64(sA), np.float64(sC)
                    )
                    score = computed[metric]
                    if not _score_passes(score, min_threshold):
                        continue
                    if reference_cohort is not None:
                        reference = _reference_evidence(
                            per_cohort[reference_cohort],
                            antecedent,
                            consequent,
                            reference_cohort,
                        )
                        confidence_delta = computed["confidence"] - reference["confidence"]
                        if min_confidence_delta is not None and not (
                            confidence_delta >= min_confidence_delta
                        ):
                            continue
                        support_delta = computed["support"] - reference["support"]
                        row = {
                            "antecedents": antecedent,
                            "consequents": consequent,
                            "cohort": cohort,
                            "reference_cohort": reference_cohort,
                        }
                        row.update(
                            {
                                name: computed[name]
                                for name in return_metrics
                            }
                        )
                        row["reference support"] = reference["support"]
                        row["reference confidence"] = reference["confidence"]
                        row["reference lift"] = reference["lift"]
                        row["support delta"] = support_delta
                        row["confidence delta"] = confidence_delta
                        rows.append(row)
                    else:
                        row = {
                            "antecedents": antecedent,
                            "consequents": consequent,
                            "cohort": cohort,
                        }
                        row.update(
                            {name: computed[name] for name in return_metrics}
                        )
                        rows.append(row)

    if reference_cohort is not None:
        columns = (
            ["antecedents", "consequents"]
            + list(return_metrics)
            + ["cohort", "reference_cohort"]
            + _COHORT_REFERENCE_COLUMNS[1:]
        )
    else:
        columns = ["antecedents", "consequents"] + list(return_metrics) + ["cohort"]

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _score_passes(score, threshold):
    if isinstance(score, np.ndarray):
        score = score.item() if score.size else np.nan
    if score is None:
        return False
    try:
        return bool(score >= threshold)
    except TypeError:
        return False


def _cohort_support(supports, itemset, cohort):
    """Fetch support, turning an incomplete index into a loud error."""
    try:
        return supports[itemset]
    except KeyError:
        raise ValueError(
            "Cohort %r has no support entry for itemset %r, which is needed to "
            "evaluate a rule.  The cohort input must be complete; missing "
            "supports are not filled with zero." % (cohort, sorted(itemset, key=repr))
        )


def _reference_evidence(reference_supports, antecedent, consequent, cohort):
    """Look up the *same* rule in the reference cohort.

    Supports are those of itemsets that survived mining in at least one
    cohort, not just in the reference one.  Hence a reference support of zero
    is a genuine zero, while a rule whose reference antecedent support is
    zero yields NaN confidence and delta (never a fabricated value).
    """
    sAC = _cohort_support(reference_supports, antecedent.union(consequent), cohort)
    sA = _cohort_support(reference_supports, antecedent, cohort)
    sC = _cohort_support(reference_supports, consequent, cohort)
    with np.errstate(divide="ignore", invalid="ignore"):
        confidence = np.nan if sA <= 0 else sAC / sA
        lift = np.nan if (sA <= 0 or sC <= 0) else confidence / sC
    return {"support": sAC, "confidence": confidence, "lift": lift}
