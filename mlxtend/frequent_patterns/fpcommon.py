import collections
import math
import warnings

import numpy as np
import pandas as pd
from pandas import __version__ as pandas_version

warnings.simplefilter("always", DeprecationWarning)

# Relative slack used when comparing weighted supports against a threshold.
# Weighted counts are sums of floats, so an exact-boundary support such as
# 0.6 can be represented as 0.5999999999999999.
_SUPPORT_EPS = 1e-12

# Names used to attach the tiny-but-positive evidence that a plain float
# ``support`` column cannot itself carry.  The sidecar lives in the result
# frame's ``attrs`` mapping, so copying the frame or permuting its rows keeps
# the evidence attached without any dependence on row order.
EVIDENCE_ATTR = "weighted_evidence"
_TINY_EVIDENCE_KEY = "tiny_counts"
_TOTAL_EVIDENCE_KEY = "cohort_totals"


def build_weighted_evidence(tiny_counts, cohort_totals=None):
    """Assemble the evidence sidecar attached to a mining result frame.

    ``tiny_counts`` maps ``(frozenset itemset, cohort label)`` (the label is
    ``None`` in weight-only mode) to the positive scaled count of that
    itemset.  Only itemsets whose *displayed* support underflowed to zero
    need an entry; the positive counts of every other itemset are recoverable
    from the support column and the cohort totals.  Cohort labels, rather
    than first-appearance positions, key the sidecar so the evidence stays
    correct after the result rows are copied or permuted.
    ``cohort_totals`` maps the same labels to their scaled total weights.
    """
    return {
        _TINY_EVIDENCE_KEY: dict(tiny_counts),
        _TOTAL_EVIDENCE_KEY: dict(cohort_totals or {}),
    }


def get_weighted_evidence(df):
    """Return the evidence sidecar of a frame, or ``None``."""
    if not hasattr(df, "attrs"):
        return None
    evidence = df.attrs.get(EVIDENCE_ATTR)
    if not isinstance(evidence, dict):
        return None
    if _TINY_EVIDENCE_KEY not in evidence:
        return None
    return evidence


def evidence_count(evidence, itemset, cohort_label, displayed_support, total):
    """Resolve the positive weighted count behind a support entry.

    The three cases the caller must keep apart are:

    * **true zero** - no tiny entry and the displayed support is zero: the
      count really is zero;
    * **tiny positive** - a tiny entry exists even though the displayed
      support underflowed to zero: return that stored positive count;
    * **ordinary** - a positive, representable support: the count is the
      displayed support times the (scaled) cohort total.
    """
    if evidence is not None:
        tiny = evidence[_TINY_EVIDENCE_KEY].get((itemset, cohort_label))
        if tiny is not None:
            return float(tiny)
        if displayed_support <= 0.0:
            return 0.0
        return float(displayed_support) * float(total)
    # No sidecar (plain hand-built support table): only the displayed value
    # exists, so there is no hidden positive count to recover.  Returning the
    # displayed value itself keeps the evidence-free code path on its
    # original support-based arithmetic.
    return float(displayed_support)


_FLOAT_MAX_LOG = math.log(np.finfo(np.float64).max)
_FLOAT_MIN_LOG = math.log(
    np.finfo(np.float64).tiny * np.spacing(np.float64(1.0))
)


def safe_fraction(numerator_terms, denominator_terms):
    """Product of ``numerator_terms`` over product of ``denominator_terms``.

    Rule metrics such as confidence and lift are ratios of weighted counts
    whose magnitudes can span the whole floating-point range.  A direct
    product can overflow to ``inf`` or underflow to zero even when the
    fraction itself is representable.  The ordinary floating-point product
    is therefore tried first, which preserves the exact rounding behaviour
    (e.g. the exact 0.5 of two equal counts); only when that result is
    non-finite or zero while a positive numerator exists does the function
    fall back to log-space accumulation, mapping back to ``inf``/``0.0``
    only when the true value really crosses the float bounds.

    A zero denominator term yields NaN (the metric is undefined, matching
    the support-based code path); a zero numerator term with strictly
    positive denominators yields ``0.0``.
    """
    numerator_terms = list(numerator_terms)
    denominator_terms = list(denominator_terms)
    if any(term <= 0.0 or not math.isfinite(term) for term in denominator_terms):
        return float("nan")
    if any(term <= 0.0 or not math.isfinite(term) for term in numerator_terms):
        return 0.0

    numerator = 1.0
    for term in numerator_terms:
        numerator *= term
    denominator = 1.0
    for term in denominator_terms:
        denominator *= term
    if denominator > 0.0:
        direct = numerator / denominator
        if math.isfinite(direct) and direct > 0.0:
            return direct

    # Log-space fallback: math.log of a positive finite count is always
    # finite, so no intermediate product can overflow or underflow.
    log_value = math.fsum(math.log(term) for term in numerator_terms) - math.fsum(
        math.log(term) for term in denominator_terms
    )
    if log_value >= _FLOAT_MAX_LOG:
        return float("inf")
    if log_value <= _FLOAT_MIN_LOG:
        return 0.0
    return math.exp(log_value)


def _ge_threshold(value, threshold):
    """Return ``value >= threshold`` with a small relative tolerance."""
    return value >= threshold - _SUPPORT_EPS * max(1.0, abs(threshold))


# Exponent of the largest finite binary64 float (2 ** 1023).  Rescaled group
# totals are kept a safety margin below it: cross-cohort sums used only for
# the item-ordering heuristic can then combine a realistic number of cohorts
# without overflowing.
_MAX_EXPONENT = 1023
_SCALING_SAFETY_MARGIN = 32


def _group_scaling_exponent(max_weight, min_positive, n_positive):
    """Pick a power-of-two unit change ``2**k`` for one weight group.

    The weights of a single cohort (or of the whole batch in weight-only
    mode) are rescaled by a common power of two.  Multiplying a binary
    floating-point number by a power of two only moves the exponent, so the
    rescaling is exact whenever the result stays inside the floating-point
    range, and every weighted ratio is unchanged because the numerator and
    denominator of a support are multiplied by the same factor.

    Two bounds must hold simultaneously:

    * the *largest* accumulated counter stays finite: a sum of
      ``n_positive`` weights, ``n_positive * max_weight * 2**k``, must not
      exceed ``2**1023`` (half of the finite range);
    * the *smallest positive* weight stays strictly positive:
      ``min_positive * 2**k`` must not leave the representable range at the
      subnormal end (``2**-1074``).

    ``frexp(x) = (m, e)`` with ``0.5 <= m < 1`` gives ``x = m * 2**e``, so
    the exponent of ``x`` is ``e - 1``.  No rescaling is applied when plain
    weights already satisfy both bounds, keeping ordinary inputs on their
    original code path; otherwise the largest exponent that still prevents
    the overflow is chosen, which keeps rescaled counters as large (and
    supports as precise) as possible.
    """
    # Lower bound: frexp(min_positive) = (m, e) with 0.5 <= m < 1 means the
    # value's binary exponent is e - 1.  Require e - 1 + k >= -1074 so the
    # smallest weight never leaves the (subnormal) representable range.
    _, min_exp = math.frexp(min_positive)
    exponent_min = -1073 - min_exp

    # Upper bound: use frexp's exponent additivity instead of forming
    # n_positive * max_weight (which may already overflow to inf).  The
    # binary exponent of a product is (e_max - 1) + floor(log2(n)); require
    # it plus k to stay below the finite limit minus the safety margin, so
    # later cross-cohort ordering sums stay finite as well.  ``max_weight``
    # is the largest *positive* weight even when the group also has zero rows.
    _, max_exp = math.frexp(max_weight)
    exponent_max = (
        _MAX_EXPONENT
        - _SCALING_SAFETY_MARGIN
        - (max_exp - 1)
        - int(math.log2(n_positive))
    )
    if exponent_max < exponent_min:
        raise ValueError(
            "The weights span a range too wide for floating-point "
            "arithmetic: no common power-of-two unit keeps the largest sums "
            "finite while keeping the smallest positive weight (%r) "
            "non-zero." % min_positive
        )
    if exponent_max >= 0:
        # Ordinary case: raw weights already fit, leave them untouched.
        return 0
    return exponent_max


def rescale_weights(weights, cohort_codes=None, n_cohorts=None):
    """Rescale weights group-wise by exact power-of-two unit changes.

    Returns ``(scaled_weights, exponents, scaled_totals)``.  With cohort
    codes, ``exponents`` and ``scaled_totals`` are per-cohort arrays; without
    them a single integer exponent and one total describe the whole batch.
    Every positive input weight stays positive after rescaling and every
    zero stays zero, so the positions of zero and positive evidence do not
    change and all weighted ratios (selected itemsets, supports, rules) are
    identical.  Groups whose weights already fit are returned unchanged
    (exponent zero), so ordinary inputs pay nothing beyond the bounds check.
    """
    weights = np.asarray(weights, dtype=float)
    if cohort_codes is None:
        positive = weights > 0.0
        n_positive = int(np.count_nonzero(positive))
        if n_positive == 0:
            # An empty batch (or an all-zero one) has nothing to rescale;
            # callers validate positivity separately for non-empty inputs.
            return weights.copy(), 0, 0.0
        exponent = _group_scaling_exponent(
            float(weights[positive].max()),
            float(weights[positive].min()),
            n_positive,
        )
        scaled = weights if exponent == 0 else weights * (2.0 ** exponent)
        return scaled, exponent, float(scaled.sum())

    exponents = np.zeros(n_cohorts, dtype=np.int64)
    scaled = np.empty_like(weights)
    codes = np.asarray(cohort_codes)
    for code in range(n_cohorts):
        group = weights[codes == code]
        positive = group > 0.0
        n_positive = int(np.count_nonzero(positive))
        if n_positive == 0:
            raise ValueError(
                "Cohort code %d has no positive weight; every cohort must "
                "have a positive total weight." % code
            )
        exponent = _group_scaling_exponent(
            float(group[positive].max()),
            float(group[positive].min()),
            n_positive,
        )
        exponents[code] = exponent
        scaled[codes == code] = (
            group if exponent == 0 else group * (2.0 ** exponent)
        )

    totals = np.zeros(n_cohorts, dtype=float)
    with np.errstate(over="ignore"):
        np.add.at(totals, codes, scaled)
    return scaled, exponents, totals


def _add_count(current, increment):
    """Add two tree-node counters.

    Counters are either scalars (the historical behaviour) or dictionaries
    mapping a cohort position to a weighted count.  Dictionaries let the
    multi-cohort miner keep one shared FP tree while still exposing each
    cohort's evidence at every prefix.
    """
    if isinstance(increment, dict):
        if not isinstance(current, dict):
            current = {}
        for cohort, value in increment.items():
            current[cohort] = current.get(cohort, 0.0) + value
        return current
    return current + increment


def _copy_count(count):
    if isinstance(count, dict):
        return dict(count)
    return count


def _sum_counts(values):
    """Sum a list of scalar or cohort-vector counters."""
    values = list(values)
    if not values:
        return 0
    if isinstance(values[0], dict):
        total = {}
        for value in values:
            for cohort, count in value.items():
                total[cohort] = total.get(cohort, 0.0) + count
        return total
    return sum(values)


def _has_support(count, minsup):
    """Return whether a counter reaches the mining threshold.

    ``minsup`` is either a scalar count or, for cohort mining, a dict of
    per-cohort absolute thresholds.  A node is retained if *either* its
    scalar count reaches the threshold or any cohort reaches its own
    threshold.  Reaching a single cohort is enough because the output
    contains every itemset frequent in at least one cohort.
    """
    if isinstance(minsup, dict):
        if isinstance(count, dict):
            for cohort, threshold in minsup.items():
                if _ge_threshold(count.get(cohort, 0.0), threshold):
                    return True
            return False
        # A scalar counter can still be compared with the largest cohort
        # threshold; this happens only for legacy scalar trees.
        return _ge_threshold(count, max(minsup.values()))
    if isinstance(count, dict):
        return _ge_threshold(_rank_key(count), minsup)
    return _ge_threshold(count, minsup)


def _rank_key(count):
    """Ordering key for an item counter (most frequent last).

    Cohort counters may each hold a rescaled weight close to the finite
    float limit, so adding them across cohorts can overflow even though the
    ordering only needs their combined *magnitude*.  When a plain sum is not
    finite, fall back to a magnitude comparison built from the largest
    summand and the number of contributors, which stays finite and preserves
    the ranking.
    """
    if isinstance(count, dict):
        if not count:
            return 0.0
        total = sum(count.values())
        if math.isfinite(total):
            return total
        largest = max(count.values())
        return largest * (1.0 + math.log1p(len(count)))
    return count


def _supported_path_items(counts, minsup):
    """Return the path items that can still reach a threshold.

    ``counts`` are the node counters in FP-tree path order (most frequent
    first), so each counter is >= the next one in every cohort.  A node
    failing all thresholds can therefore only be followed by nodes that fail
    too, and every itemset containing such a node fails.  Only the supported
    prefix is returned; enumerating the remaining suffixes would produce
    itemsets no cohort supports.  In the classic single-cohort tree every path
    node already passed the threshold, so this is a no-op there.
    """
    if not counts:
        return []
    for stop in range(len(counts), 0, -1):
        if _has_support(counts[stop - 1], minsup):
            return list(range(stop))
    return []


def _min_counts(values):
    """Element-wise minimum of scalar or cohort-vector counters.

    A cohort missing from a counter has contributed nothing there, hence its
    count is zero.  For a single FP-tree path this minimum is the support of
    an itemset made of several path items: the group of transactions
    containing every selected item is the group containing the deepest one.
    Zero entries are dropped to keep the evidence sparse.
    """
    values = list(values)
    if not values:
        return 0
    if not isinstance(values[0], dict):
        return min(values)
    result = {}
    keys = set()
    for value in values:
        keys.update(value)
    for key in keys:
        smallest = min(value.get(key, 0.0) for value in values)
        if smallest:
            result[key] = smallest
    return result


def _weighted_matrix(df):
    """Return ``(matrix, is_sparse)`` without densifying a sparse input."""
    if hasattr(df, "sparse") and df.size != 0:
        matrix = df.sparse.to_coo().tocsr()
        matrix.eliminate_zeros()
        return matrix, True
    return df.values, False


def _iter_row_items(matrix, is_sparse, chunk_rows=16384):
    """Yield ``(row_indices, column_indices)`` blocks of non-zero cells.

    Only the coordinates of actually present goods are materialised: O(nnz)
    of the current block, never a dense ``rows x items`` copy.  Sparse input
    is walked through its CSR structure; dense input is scanned in row chunks
    so the transient boolean mask is bounded by ``chunk_rows x items``.
    """
    n_rows = matrix.shape[0]
    if is_sparse:
        indptr = matrix.indptr
        indices = matrix.indices
        counts = np.diff(indptr)
        # Block by rows so a 100k-row CSR does not create a 1.2M-entry
        # temporary at once.
        start = 0
        while start < n_rows:
            stop = min(start + chunk_rows, n_rows)
            rows = np.repeat(np.arange(start, stop), counts[start:stop])
            cols = indices[indptr[start] : indptr[stop]]
            yield rows, cols
            start = stop
        if n_rows == 0:
            return
    else:
        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            rows, cols = np.nonzero(matrix[start:stop, :])
            yield rows + start, cols


def _item_counts(
    matrix, is_sparse, weights, cohort_codes=None, n_cohorts=None, n_items=None
):
    """Count weighted item occurrences (optionally per cohort).

    Returns ``(counts, order_weights)`` where ``counts`` is either a 1-D
    array of item weights or a ``cohorts x items`` array.  The latter costs
    ``64 x 500`` floats even for million-row inputs, i.e. it grows with the
    number of cohorts and goods, not with the number of baskets.
    """
    if cohort_codes is not None:
        counts = np.zeros((n_cohorts, n_items), dtype=float)
        for rows, cols in _iter_row_items(matrix, is_sparse):
            if rows.size == 0:
                continue
            block_weights = weights[rows]
            keys = cohort_codes[rows].astype(np.int64) * n_items + cols
            block = np.bincount(
                keys, weights=block_weights, minlength=n_cohorts * n_items
            )
            counts += block.reshape(n_cohorts, n_items)
        # The cross-cohort sum only drives the item ordering heuristic.  A
        # group-wise rescale can put each cohort counter close to the float
        # limit, so the sum may overflow; rank the affected ties by the
        # largest finite summand instead of an +inf value.  Columns without
        # any positive counter simply get order weight zero.
        finite_sum = counts.sum(axis=0)
        # Columns with no positive counter make the nanmax input all-NaN;
        # that is an expected "order weight zero" case, not an error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            safe_max = np.nanmax(
                np.where(counts > 0.0, counts, np.nan), axis=0
            )
        order_weights = np.where(
            np.isfinite(finite_sum),
            finite_sum,
            np.nan_to_num(safe_max, nan=0.0),
        )
        return counts, order_weights

    counts = np.zeros(n_items, dtype=float)
    for rows, cols in _iter_row_items(matrix, is_sparse):
        if rows.size == 0:
            continue
        counts += np.bincount(cols, weights=weights[rows], minlength=n_items)
    return counts, counts


def _setup_weighted_fptree(
    df, min_support, sample_weight=None, cohort_codes=None, cohort_totals=None
):
    """Build an FP tree whose node counters are row weights or cohort vectors.

    ``cohort_codes is None`` keeps scalar counters (weighted, single cohort).
    With cohort codes each node stores ``{cohort_code: weighted_count}`` only
    for cohorts that actually contributed to that prefix, so memory follows
    the observed prefix/cohort activity rather than a dense
    ``cohorts x rows x items`` layout.
    """
    n_rows = len(df.index)
    n_items = len(df.columns)
    weights = np.ones(n_rows, dtype=float) if sample_weight is None else sample_weight
    use_cohorts = cohort_codes is not None

    if df.size == 0:
        # No items to count; build an empty tree that still propagates rank.
        return FPTree({}), None, {}

    matrix, is_sparse = _weighted_matrix(df)

    if use_cohorts:
        n_cohorts = len(cohort_totals)
        thresholds = np.asarray(cohort_totals, dtype=float) * min_support
        slack = _SUPPORT_EPS * np.maximum(1.0, np.abs(thresholds))
        counts, order_weights = _item_counts(
            matrix,
            is_sparse,
            weights,
            cohort_codes=cohort_codes,
            n_cohorts=n_cohorts,
            n_items=n_items,
        )
        keep = np.any(counts >= (thresholds - slack)[:, np.newaxis], axis=0)
    else:
        counts, order_weights = _item_counts(
            matrix, is_sparse, weights, n_items=n_items
        )
        # A tiny tolerance keeps exact-boundary weighted supports stable.
        threshold = min_support * float(weights.sum())
        slack = _SUPPORT_EPS * max(1.0, abs(threshold))
        keep = counts >= threshold - slack

    items = np.nonzero(keep)[0]
    order = order_weights[items].argsort()
    rank = {item: i for i, item in enumerate(items[order])}

    tree = FPTree(rank)
    indptr = matrix.indptr if is_sparse else None
    for i in range(n_rows):
        weight = weights[i]
        if weight == 0:
            # A zero-weight basket contributes no support to any counter.
            continue
        if is_sparse:
            nonnull = matrix.indices[indptr[i] : indptr[i + 1]]
        else:
            nonnull = np.nonzero(matrix[i, :])[0]
        itemset = [item for item in nonnull if item in rank]
        if not itemset:
            continue
        itemset.sort(key=rank.get, reverse=True)
        if use_cohorts:
            count = {int(cohort_codes[i]): float(weight)}
        else:
            count = float(weight)
        tree.insert_itemset(itemset, count)

    return tree, None, rank


def setup_fptree(
    df, min_support, sample_weight=None, cohort_codes=None, cohort_totals=None
):
    if sample_weight is not None or cohort_codes is not None:
        return _setup_weighted_fptree(
            df, min_support, sample_weight, cohort_codes, cohort_totals
        )
    num_itemsets = len(df.index)  # number of itemsets in the database

    is_sparse = False
    if hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            itemsets = df.values
        else:
            itemsets = df.sparse.to_coo().tocsr()
            is_sparse = True
    else:
        # dense DataFrame
        itemsets = df.values

    # support of each individual item
    # if itemsets is sparse, np.sum returns an np.matrix of shape (1, N)
    disabled = df.copy()
    disabled = np.where(pd.isna(disabled), 1, np.nan) + np.where(
        (disabled == 0) | (disabled == 1), np.nan, 0
    )

    item_support = np.array(
        np.nansum(df.values, axis=0)
        / (float(num_itemsets) - np.nansum(disabled, axis=0))
    )
    item_support = item_support.reshape(-1)
    items = np.nonzero(item_support >= min_support)[0]

    # Define ordering on items for inserting into FPTree
    indices = item_support[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}

    if is_sparse:
        # Ensure that there are no zeros in sparse DataFrame
        itemsets.eliminate_zeros()

    # Building tree by inserting itemsets in sorted order
    # Heuristic for reducing tree size is inserting in order
    #   of most frequent to least frequent
    tree = FPTree(rank)
    for i in range(num_itemsets):
        if is_sparse:
            # itemsets has been converted to CSR format to speed-up the line
            # below.  It has 3 attributes:
            #  - itemsets.data contains non null values, shape(#nnz,)
            #  - itemsets.indices contains the column number of non null
            #    elements, shape(#nnz,)
            #  - itemsets.indptr[i] contains the offset in itemset.indices of
            #    the first non null element in row i, shape(1+#nrows,)
            nonnull = itemsets.indices[itemsets.indptr[i] : itemsets.indptr[i + 1]]
        else:
            nonnull = np.where(itemsets[i, :])[0]
        itemset = [item for item in nonnull if item in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset)

    return tree, disabled, rank


def generate_itemsets(
    generator,
    df,
    disabled,
    min_support,
    num_itemsets,
    colname_map,
    sample_weight=None,
    cohort_codes=None,
    cohort_labels=None,
    cohort_totals=None,
    weighted_total=None,
):
    if disabled is None and cohort_codes is not None:
        return _generate_cohort_itemsets(
            generator,
            min_support,
            colname_map,
            cohort_labels,
            cohort_totals,
        )

    if disabled is None:
        # Weighted single-cohort mining: support is weighted count over the
        # total row weight, and zero-weight rows were never inserted.
        # ``weighted_total`` is the (possibly rescaled) total the tree was
        # built with; rescaling multiplies every count and the total by the
        # same power of two, so the displayed support is unchanged.
        if weighted_total is not None:
            total_weight = float(weighted_total)
        else:
            total_weight = (
                float(len(df.index))
                if sample_weight is None
                else float(sample_weight.sum())
            )
        itemsets = []
        supports = []
        tiny_counts = {}
        threshold_count = min_support * total_weight
        for count, iset in generator:
            itemset = frozenset(iset)
            # Selection runs on the actual (rescaled) counts.  This is what
            # keeps a positive count whose support displays as zero from
            # being confused with a true zero.
            if not _ge_threshold(float(count), threshold_count):
                continue
            support = count / total_weight if total_weight else 0.0
            translated = (
                frozenset(colname_map[i] for i in itemset)
                if colname_map is not None
                else itemset
            )
            itemsets.append(translated)
            supports.append(support)
            if support <= 0.0 and count > 0.0:
                # Genuine positive evidence that the float support column
                # cannot show: hand it to the rule stage through the sidecar.
                tiny_counts[(translated, None)] = float(count)
        res_df = pd.DataFrame({"support": supports, "itemsets": itemsets})
        if len(res_df.index) == 0:
            res_df = pd.DataFrame(
                {"support": pd.Series(dtype=float), "itemsets": pd.Series(dtype=object)}
            )
        res_df.attrs[EVIDENCE_ATTR] = build_weighted_evidence(
            tiny_counts, {None: total_weight}
        )
        return res_df

    itemsets = []
    supports = []
    for sup, iset in generator:
        itemsets.append(frozenset(iset))
        # select data of iset from disabled dataset
        dec = disabled[:, iset]
        # select data of iset from original dataset
        _dec = df.values[:, iset]

        # case if iset only has one element
        if len(iset) == 1:
            supports.append((sup - np.nansum(dec)) / (num_itemsets - np.nansum(dec)))

        # case if iset has multiple elements
        elif len(iset) > 1:
            denom = 0
            num = 0
            for i in range(dec.shape[0]):
                # select the i-th iset from disabled dataset
                item_dsbl = list(dec[i, :])
                # select the i-th iset from original dataset
                item_orig = list(_dec[i, :])

                # check and keep count if there is a null value in iset of disabled
                if 1 in set(item_dsbl):
                    denom += 1

                    # check and keep count if item doesn't exist OR all values are null in iset of original
                    if (0 not in set(item_orig)) or (
                        all(np.isnan(x) for x in item_orig)
                    ):
                        num -= 1

            if num_itemsets - denom == 0:
                supports.append(0)
            else:
                supports.append((sup + num) / (num_itemsets - denom))

    res_df = pd.DataFrame({"support": supports, "itemsets": itemsets})
    res_df = res_df[res_df["support"] >= min_support]

    if colname_map is not None:
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([colname_map[i] for i in x])
        )

    return res_df


def _generate_cohort_itemsets(
    generator, min_support, colname_map, cohort_labels, cohort_totals
):
    """Expand per-itemset cohort counters into one row per cohort.

    Every retained itemset is reported for *all* input cohorts, including
    cohorts whose support is below the threshold or zero.  The support of an
    itemset within a cohort is its weighted count in that cohort divided by
    the cohort's total row weight (empty baskets included in the
    denominator).

    Selection uses the raw weighted counts against each cohort's absolute
    count threshold rather than the computed support, so a positive count
    whose support underflows to zero is never mistaken for a true zero.
    Positively supported itemsets whose displayed support is zero are
    recorded in the evidence sidecar for the rule stage.
    """
    hits = []
    counters_by_itemset = []
    translated_itemsets = []
    tiny_counts = {}
    seen = set()
    n_cohorts = len(cohort_labels)
    thresholds = np.asarray(cohort_totals, dtype=float) * min_support

    for count, iset in generator:
        itemset = frozenset(iset)
        if itemset in seen:
            continue
        if isinstance(count, dict):
            counters = [float(count.get(code, 0.0)) for code in range(n_cohorts)]
        else:
            # Defensive: a scalar counter means total weight with no cohort
            # breakdown, which cannot be attributed to individual cohorts.
            counters = [float(count)] + [0.0] * (n_cohorts - 1)
        if not any(
            _ge_threshold(counters[code], thresholds[code])
            for code in range(n_cohorts)
        ):
            continue
        seen.add(itemset)
        translated = (
            frozenset(colname_map[i] for i in itemset)
            if colname_map is not None
            else itemset
        )
        hits.append(itemset)
        translated_itemsets.append(translated)
        counters_by_itemset.append(counters)

        for code in range(n_cohorts):
            positive_count = counters[code]
            if positive_count > 0.0:
                support = (
                    positive_count / cohort_totals[code]
                    if cohort_totals[code]
                    else 0.0
                )
                if support <= 0.0:
                    # Positive evidence hidden by float underflow; remember
                    # the count keyed by the stable cohort *label* (not its
                    # first-appearance position, which changes when rows are
                    # permuted) so association_rules can tell it apart from a
                    # genuinely absent itemset.
                    tiny_counts[(translated, cohort_labels[code])] = positive_count

    if not hits:
        empty_df = pd.DataFrame(columns=["cohort", "support", "itemsets"])
        empty_df.attrs[EVIDENCE_ATTR] = build_weighted_evidence(
            {},
            {
                cohort_labels[code]: float(total)
                for code, total in enumerate(cohort_totals)
            },
        )
        return empty_df

    # Expand itemset-major: each retained itemset gets one row per cohort.
    # Building the flattened columns directly (rather than one Python tuple
    # per output row) keeps the transient cost close to the final frame even
    # when the output has millions of rows.
    counters_matrix = np.asarray(counters_by_itemset, dtype=float)
    totals = np.asarray(cohort_totals, dtype=float)
    # Every validated cohort has a positive total, but guard the division
    # anyway: explicit underflow zeros are fine because their positive counts
    # survive in the sidecar, and true zeros stay exactly zero.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        supports_matrix = counters_matrix / totals[np.newaxis, :]
    supports_matrix = np.nan_to_num(
        supports_matrix, nan=0.0, posinf=0.0, neginf=0.0
    )
    flat_supports = supports_matrix.reshape(-1)
    cohort_indices = np.tile(np.arange(n_cohorts), len(hits))
    cohort_column = np.asarray(cohort_labels, dtype=object)[cohort_indices]
    itemset_column = [
        itemset for itemset in translated_itemsets for _ in range(n_cohorts)
    ]

    res_df = pd.DataFrame(
        {
            "cohort": cohort_column,
            "support": flat_supports,
            "itemsets": itemset_column,
        }
    )
    res_df.attrs[EVIDENCE_ATTR] = build_weighted_evidence(
        tiny_counts,
        {
            cohort_labels[code]: float(total)
            for code, total in enumerate(cohort_totals)
        },
    )
    return res_df


def _as_positional_sequence(values, name, n_rows):
    """Convert a user-supplied per-row sequence to a positional NumPy array.

    The values are *always* consumed positionally.  Label-based alignment is
    deliberately avoided: a DataFrame index with duplicate labels must not
    make pandas reorder or broadcast the values.
    """
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(
                "`%s` must be one-dimensional; got a DataFrame with %d columns."
                % (name, values.shape[1])
            )
        values = values.iloc[:, 0]
    elif isinstance(values, pd.Series):
        values = values.to_numpy()
    elif isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(
                "`%s` must be one-dimensional; got an array with %d dimensions."
                % (name, values.ndim)
            )
    else:
        if isinstance(values, (str, bytes)):
            raise ValueError("`%s` must be a sequence, not a string." % name)
        try:
            values = list(values)
        except TypeError:
            raise ValueError("`%s` must be a per-row sequence." % name)
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            "`%s` must be one-dimensional; got shape %s." % (name, array.shape)
        )
    if len(array) != n_rows:
        raise ValueError(
            "`%s` must have length equal to the number of rows (%d); got %d."
            % (name, n_rows, len(array))
        )
    return array


def prepare_cohorts(df, cohorts, sample_weight):
    """Validate cohorts/weights and return positional cohort metadata.

    Returns ``(cohort_codes, cohort_labels, cohort_totals, weights)`` where
    codes index into ``cohort_labels`` in first-appearance order.

    Validation is intentionally strict because silently repairing any of
    these inputs would change the denominators of every reported support.
    """
    n_rows = len(df.index)
    cohort_array = _as_positional_sequence(cohorts, "cohorts", n_rows)

    labels = []
    label_to_code = {}
    codes = np.empty(n_rows, dtype=np.int64)
    for i, label in enumerate(cohort_array):
        if not isinstance(label, str) or label == "":
            raise ValueError(
                "`cohorts` must contain non-empty strings; got %r at row %d."
                % (label, i)
            )
        code = label_to_code.get(label)
        if code is None:
            code = len(labels)
            label_to_code[label] = code
            labels.append(label)
        codes[i] = code

    if sample_weight is None:
        weights = np.ones(n_rows, dtype=float)
    else:
        raw = _as_positional_sequence(sample_weight, "sample_weight", n_rows)
        try:
            weights = raw.astype(float)
        except (TypeError, ValueError):
            raise ValueError("`sample_weight` must contain numeric values.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("`sample_weight` must contain only finite values.")
        if np.any(weights < 0):
            raise ValueError("`sample_weight` must be non-negative.")

    # Positivity is established from the individual weights instead of the
    # summed totals: a legitimate cohort total may overflow to +inf when the
    # weights are huge, but it is still a positive total.  The raw totals are
    # kept for callers on ordinary-scale inputs; weighted mining rescales
    # them before any absolute threshold is derived.
    positive_rows = np.bincount(
        codes, weights=(weights > 0.0).astype(np.int64), minlength=len(labels)
    )
    if len(labels) and np.any(positive_rows == 0):
        empty = [labels[i] for i in np.nonzero(positive_rows == 0)[0]]
        raise ValueError(
            "Every cohort must have a positive total weight; "
            "cohort(s) %s have total weight <= 0." % ", ".join(map(repr, empty))
        )
    totals = np.zeros(len(labels), dtype=float)
    # The raw totals may legitimately overflow for huge weights; weighted
    # mining rescales such groups before using them.  The overflow is not an
    # error here, only a signal that the absolute-total path is unusable.
    with np.errstate(over="ignore"):
        np.add.at(totals, codes, weights)
    return codes, labels, totals, weights


def prepare_sample_weight(df, sample_weight):
    """Validate a weight-only input (no cohorts) positionally."""
    n_rows = len(df.index)
    raw = _as_positional_sequence(sample_weight, "sample_weight", n_rows)
    try:
        weights = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("`sample_weight` must contain numeric values.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("`sample_weight` must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("`sample_weight` must be non-negative.")
    # Check positivity directly: the summed total may legitimately overflow
    # for huge weights and is rescaled later.
    if n_rows and not np.any(weights > 0.0):
        raise ValueError("`sample_weight` must have a positive total weight.")
    return weights


def valid_new_mode_input_check(df, null_values, cohorts, sample_weight):
    """Extra checks for the weighted/cohort entry points."""
    if (cohorts is not None or sample_weight is not None) and null_values:
        raise ValueError(
            "`null_values=True` is not supported in weighted or cohort mode."
        )
    if sample_weight is None and cohorts is None:
        return
    if df.columns.duplicated().any():
        dupes = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
        raise ValueError(
            "Duplicate item column names are not supported in weighted or "
            "cohort mode; found %s." % ", ".join(dupes)
        )
    if df.size != 0 and pd.isna(df).any().any():
        raise ValueError(
            "NaN values are not permitted in weighted or cohort mode."
        )


def valid_input_check(df, null_values=False):
    # Return early if df is None
    if df is None:
        return

    if f"{type(df)}" == "<class 'pandas.core.frame.SparseDataFrame'>":
        msg = (
            "SparseDataFrame support has been deprecated in pandas 1.0,"
            " and is no longer supported in mlxtend. "
            " Please"
            " see the pandas migration guide at"
            " https://pandas.pydata.org/pandas-docs/"
            "stable/user_guide/sparse.html#sparse-data-structures"
            " for supporting sparse data in DataFrames."
        )
        raise TypeError(msg)

    if df.size == 0:
        return
    if hasattr(df, "sparse"):
        if not isinstance(df.columns[0], str) and df.columns[0] != 0:
            raise ValueError(
                "Due to current limitations in Pandas, "
                "if the sparse format has integer column names,"
                "names, please make sure they either start "
                "with `0` or cast them as string column names: "
                "`df.columns = [str(i) for i in df.columns`]."
            )

    # Fast path: if all columns are boolean, there is nothing to checks
    if null_values:
        all_bools = (
            df.apply(lambda col: col.apply(lambda x: pd.isna(x) or isinstance(x, bool)))
            .all()
            .all()
        )
    else:
        all_bools = df.dtypes.apply(pd.api.types.is_bool_dtype).all()

    if not all_bools:
        warnings.warn(
            "DataFrames with non-bool types result in worse computational"
            "performance and their support might be discontinued in the future."
            "Please use a DataFrame with bool type",
            DeprecationWarning,
        )

        # If null_values is True but no NaNs are found, raise an error
        has_nans = pd.isna(df).any().any()
        if null_values and not has_nans:
            warnings.warn(
                "null_values=True is inefficient when there are no NaN values in the DataFrame."
                "Set null_values=False for faster output."
            )
        # If null_values is False but NaNs are found, raise an error
        if not null_values and has_nans:
            raise ValueError(
                "NaN values are not permitted in the DataFrame when null_values=False."
            )

        # Pandas is much slower than numpy, so use np.where on Numpy arrays
        if hasattr(df, "sparse"):
            if df.size == 0:
                values = df.values
            else:
                values = df.sparse.to_coo().tocoo().data
        else:
            values = df.values

        # Ignore NaNs if null_values is True
        if null_values:
            idxs = np.where((values != 1) & (values != 0) & (~np.isnan(values)))
        else:
            idxs = np.where((values != 1) & (values != 0))

        if len(idxs[0]) > 0:
            # idxs has 1 dimension with sparse data and 2 with dense data
            val = values[tuple(loc[0] for loc in idxs)]
            s = (
                "The allowed values for a DataFrame"
                " are True, False, 0, 1. Found value %s" % (val)
            )

            if null_values:
                s = (
                    "The allowed values for a DataFrame"
                    " are True, False, 0, 1, NaN. Found value %s" % (val)
                )
            raise ValueError(s)


class FPTree(object):
    def __init__(self, rank=None):
        self.root = FPNode(None)
        self.nodes = collections.defaultdict(list)
        self.cond_items = []
        self.rank = rank

    def conditional_tree(self, cond_item, minsup):
        """
        Creates and returns the subtree of self conditioned on cond_item.

        Parameters
        ----------
        cond_item : int | str
            Item that the tree (self) will be conditioned on.
        minsup : int | dict
            Minimum support threshold.  When it is a dict it maps a cohort
            position to that cohort's absolute threshold; an item is kept
            when any cohort reaches its threshold.

        Returns
        -------
        cond_tree : FPtree
        """
        # Find all path from root node to nodes for item
        branches = []
        count = collections.defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] = _add_count(count[item], node.count)

        # Define new ordering or deep trees may have combinatorially explosion
        items = [item for item in count if _has_support(count[item], minsup)]
        items.sort(key=lambda item: _rank_key(count[item]))
        rank = {item: i for i, item in enumerate(items)}

        # Create conditional tree
        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted(
                [i for i in branch if i in rank], key=rank.get, reverse=True
            )
            cond_tree.insert_itemset(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    def insert_itemset(self, itemset, count=1):
        """
        Inserts a list of items into the tree.

        Parameters
        ----------
        itemset : list
            Items that will be inserted into the tree.
        count : int | dict
            The number of occurrences of the itemset.  A dict of
            cohort -> weighted count keeps per-cohort evidence.
        """
        self.root.count = _add_count(self.root.count, _copy_count(count))

        if len(itemset) == 0:
            return

        # Follow existing path in tree as long as possible
        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count = _add_count(child.count, _copy_count(count))
                node = child
                index += 1
            else:
                break

        # Insert any remaining items
        for item in itemset[index:]:
            child_node = FPNode(item, _copy_count(count), node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True

    def print_status(self, count, colnames):
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        print(
            "\r%d itemset(s) from tree conditioned on items (%s)" % (count, cond_items),
            end="\n",
        )


class FPNode(object):
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = collections.defaultdict(FPNode)

        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        """Returns the top-down sequence of items from self to
        (but not including) the root node."""
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path
