import collections
import warnings

import numpy as np
import pandas as pd
from pandas import __version__ as pandas_version

warnings.simplefilter("always", DeprecationWarning)

# Relative slack used when comparing weighted supports against a threshold.
# Weighted counts are sums of floats, so an exact-boundary support such as
# 0.6 can be represented as 0.5999999999999999.
_SUPPORT_EPS = 1e-12


def _ge_threshold(value, threshold):
    """Return ``value >= threshold`` with a small relative tolerance."""
    return value >= threshold - _SUPPORT_EPS * max(1.0, abs(threshold))


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
        return _ge_threshold(sum(count.values()), minsup)
    return _ge_threshold(count, minsup)


def _rank_key(count):
    """Ordering key for an item counter (most frequent last)."""
    if isinstance(count, dict):
        return sum(count.values())
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
        return counts, counts.sum(axis=0)

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
        total_weight = (
            float(len(df.index))
            if sample_weight is None
            else float(sample_weight.sum())
        )
        itemsets = []
        supports = []
        for sup, iset in generator:
            support = sup / total_weight if total_weight else 0.0
            slack = _SUPPORT_EPS * max(1.0, abs(min_support))
            if support >= min_support - slack:
                itemsets.append(frozenset(iset))
                supports.append(support)
        res_df = pd.DataFrame({"support": supports, "itemsets": itemsets})
        if len(res_df.index) == 0:
            res_df = pd.DataFrame(
                {"support": pd.Series(dtype=float), "itemsets": pd.Series(dtype=object)}
            )
        if colname_map is not None:
            res_df["itemsets"] = res_df["itemsets"].apply(
                lambda x: frozenset([colname_map[i] for i in x])
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
    """
    hits = []
    counters_by_itemset = []
    seen = set()
    slack = _SUPPORT_EPS * max(1.0, abs(min_support))
    n_cohorts = len(cohort_labels)
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
        supports = [
            counters[code] / cohort_totals[code] if cohort_totals[code] else 0.0
            for code in range(n_cohorts)
        ]
        if not any(support >= min_support - slack for support in supports):
            continue
        seen.add(itemset)
        hits.append(itemset)
        counters_by_itemset.append(supports)

    if not hits:
        return pd.DataFrame(columns=["cohort", "support", "itemsets"])

    # Expand itemset-major: each retained itemset gets one row per cohort.
    # Building the flattened columns directly (rather than one Python tuple
    # per output row) keeps the transient cost close to the final frame even
    # when the output has millions of rows.
    supports_matrix = np.asarray(counters_by_itemset, dtype=float)
    flat_supports = supports_matrix.reshape(-1)
    cohort_indices = np.tile(np.arange(n_cohorts), len(hits))
    cohort_column = np.asarray(cohort_labels, dtype=object)[cohort_indices]
    itemset_column = [itemset for itemset in hits for _ in range(n_cohorts)]

    res_df = pd.DataFrame(
        {
            "cohort": cohort_column,
            "support": flat_supports,
            "itemsets": itemset_column,
        }
    )
    if colname_map is not None:
        # Translate each distinct itemset once and reuse the object for all
        # of its cohort rows; a per-row ``apply`` would allocate one frozen
        # set per output row and double the transient footprint.
        translated = {
            itemset: frozenset([colname_map[i] for i in itemset]) for itemset in hits
        }
        res_df["itemsets"] = [
            translated[itemset] for itemset in hits for _ in range(n_cohorts)
        ]
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

    totals = np.zeros(len(labels), dtype=float)
    np.add.at(totals, codes, weights)
    if len(labels) and np.any(totals <= 0):
        empty = [labels[i] for i in np.nonzero(totals <= 0)[0]]
        raise ValueError(
            "Every cohort must have a positive total weight; "
            "cohort(s) %s have total weight <= 0." % ", ".join(map(repr, empty))
        )
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
    if n_rows and float(weights.sum()) <= 0:
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
