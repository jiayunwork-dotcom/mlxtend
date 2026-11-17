# mlxtend Machine Learning Library Extensions
# Author: Steve Harenberg <harenbergsd@gmail.com>
#
# License: BSD 3 clause

import itertools
import math

from ..frequent_patterns import fpcommon as fpc


def fpgrowth(
    df,
    min_support=0.5,
    null_values=False,
    use_colnames=False,
    max_len=None,
    verbose=0,
    cohorts=None,
    sample_weight=None,
):
    """Get frequent itemsets from a one-hot DataFrame

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame the encoded format. Also supports
      DataFrames with sparse data; for more info, please
      see https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html#sparse-data-structures.

      Please note that the old pandas SparseDataFrame format
      is no longer supported in mlxtend >= 0.17.2.

      The allowed values are either 0/1 or True/False.
      For example,

    ```
           Apple  Bananas   Beer  Chicken   Milk   Rice
        0   True    False   True     True  False   True
        1   True    False   True    False  False   True
        2   True    False   True    False  False  False
        3   True     True  False    False  False  False
        4  False    False   True     True   True   True
        5  False    False   True    False   True   True
        6  False    False   True    False   True  False
        7   True     True  False    False  False  False
    ```

    min_support : float (default: 0.5)
      A float between 0 and 1 for minimum support of the itemsets returned.
      The support is computed as the fraction
      transactions_where_item(s)_occur / total_transactions.

      With `cohorts`, the threshold is applied *within each cohort*: an
      itemset is kept when at least one cohort reaches `min_support`.

    null_values : bool (default: False)
      In case there are null values as NaNs in the original input data.
      Not supported together with `cohorts`.

    use_colnames : bool (default: False)
      If true, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths are evaluated.

    verbose : int (default: 0)
      Shows the stages of conditional tree generation.

    cohorts : sequence of str (default: None)
      Optional group label per row.  Values are matched to rows by position
      (never by the DataFrame index, which may contain duplicates), so the
      length must equal the number of rows.  Every label must be a non-empty
      string and every row belongs to exactly one cohort.  When supplied,
      the result has columns ``['cohort', 'support', 'itemsets']``: each
      itemset frequent in at least one cohort is emitted once for *every*
      cohort.  ``support`` is the weighted fraction of the cohort's baskets
      (including empty ones in the denominator) that contain the itemset.

    sample_weight : sequence of float (default: None)
      Optional non-negative, finite row weight matched by position.  When
      omitted every row counts as 1.  Zero-weight rows do not contribute
      support; every cohort must still have a positive total weight.  Can be
      given without `cohorts`, in which case the output keeps the original
      ``['support', 'itemsets']`` columns and support is the weighted
      fraction of all row weight.

    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that are >= `min_support` and < than `max_len`
      (if `max_len` is not None); or with columns
      ['cohort', 'support', 'itemsets'] when `cohorts` is provided.
      Each itemset in the 'itemsets' column is of type `frozenset`,
      which is a Python built-in type that behaves similarly to
      sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    ----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/

    """
    fpc.valid_input_check(df, null_values)
    fpc.valid_new_mode_input_check(df, null_values, cohorts, sample_weight)

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )

    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    if cohorts is not None:
        codes, labels, totals, weights = fpc.prepare_cohorts(
            df, cohorts, sample_weight
        )
        minsup = {code: min_support * totals[code] for code in range(len(labels))}
        tree, _, _ = fpc.setup_fptree(
            df,
            min_support,
            sample_weight=weights,
            cohort_codes=codes,
            cohort_totals=totals,
        )
        generator = fpg_step(tree, minsup, colname_map, max_len, verbose)
        return fpc.generate_itemsets(
            generator,
            df,
            None,
            min_support,
            len(df.index),
            colname_map,
            sample_weight=weights,
            cohort_codes=codes,
            cohort_labels=labels,
            cohort_totals=totals,
        )

    if sample_weight is not None:
        weights = fpc.prepare_sample_weight(df, sample_weight)
        threshold = min_support * float(weights.sum())
        tree, _, _ = fpc.setup_fptree(df, min_support, sample_weight=weights)
        generator = fpg_step(tree, threshold, colname_map, max_len, verbose)
        return fpc.generate_itemsets(
            generator,
            df,
            None,
            min_support,
            len(df.index),
            colname_map,
            sample_weight=weights,
        )

    tree, disabled, _ = fpc.setup_fptree(df, min_support)
    minsup = math.ceil(min_support * len(df.index))  # min support as count
    generator = fpg_step(tree, minsup, colname_map, max_len, verbose)

    return fpc.generate_itemsets(
        generator, df, disabled, min_support, len(df.index), colname_map
    )


def fpg_step(tree, minsup, colnames, max_len, verbose):
    """
    Performs a recursive step of the fpgrowth algorithm.

    Parameters
    ----------
    tree : FPTree
    minsup : int | float | dict
        Absolute support threshold.  For cohort mining this is a dict of
        cohort position -> absolute weighted threshold; node counters are
        then dicts too and the yielded support is a cohort-vector counter.

    Yields
    ------
    support counter, list of items
        The counter is a scalar for the classic/weighted modes and a
        ``{cohort: weighted count}`` dict in cohort mode.
    """
    count = 0
    items = list(tree.nodes.keys())
    if tree.is_path():
        # If the tree is a path, we can combinatorally generate all
        # remaining itemsets without generating additional conditional trees.
        #
        # Counts along a path are nested (a child never exceeds its parent in
        # any cohort), so ``_has_support`` is downward closed along the path:
        # once one node fails every cohort's threshold, every deeper node and
        # every combination containing it fails too.  Dropping those nodes
        # keeps the enumeration proportional to the (already large) output
        # instead of enumerating subsets that no cohort supports.
        path_counts = [tree.nodes[item][0].count for item in items]
        supported = fpc._supported_path_items(path_counts, minsup)
        candidate_items = [items[position] for position in supported]
        size_remain = len(candidate_items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(candidate_items, i):
                count += 1
                support = fpc._min_counts([tree.nodes[j][0].count for j in itemset])
                yield support, tree.cond_items + list(itemset)
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = fpc._sum_counts([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    if verbose:
        tree.print_status(count, colnames)

    # Generate conditional trees to generate frequent itemsets one item larger
    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for sup, iset in fpg_step(cond_tree, minsup, colnames, max_len, verbose):
                yield sup, iset
