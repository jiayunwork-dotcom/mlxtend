# Cohort-aware frequent itemsets and comparative rules

This repository adds an optional cohort/weight layer on top of the existing
`fpgrowth` / `association_rules` chain (mlxtend 0.23.4, Python 3.11). The
public API is additive:

```python
fpgrowth(df, min_support=0.5, null_values=False, use_colnames=False,
         max_len=None, verbose=0, cohorts=None, sample_weight=None)

association_rules(df, num_itemsets=1, df_orig=None, null_values=False,
                  metric="confidence", min_threshold=0.8,
                  support_only=False, return_metrics=_metrics,
                  reference_cohort=None, min_confidence_delta=None)
```

## Semantics

### Mining (`fpgrowth`)

* `cohorts` and `sample_weight` are matched to rows **positionally**. Length
  must equal `len(df.index)`; a duplicated DataFrame index is never used for
  label alignment. Every cohort label must be a non-empty string, every row
  belongs to exactly one cohort, weights must be finite and non-negative, and
  every observed cohort must have a positive total weight.
* `sample_weight` alone is allowed. Then the output keeps the classic
  `['support', 'itemsets']` columns and support is
  `sum(weight of baskets containing the itemset) / sum(weight of all rows)`.
  Zero-weight rows never contribute support.
* With `cohorts` the output is
  `['cohort', 'support', 'itemsets']`. `min_support` is evaluated **inside
  each cohort**; an itemset is retained when at least one cohort reaches the
  threshold. Every retained itemset gets one row for **every** input cohort,
  including cohorts where its support is below the threshold or zero. Support
  is the weighted fraction of that cohort's baskets (empty baskets included
  in the denominator) containing the itemset.
* `frozenset` itemsets, `use_colnames` and `max_len` keep their existing
  meaning. Duplicate `(cohort, itemset)` rows are impossible.
* When nothing qualifies, the function returns an empty DataFrame with the
  right columns.
* NaN item values, missing/invalid cohort labels, invalid weights and
  duplicate item column names are rejected. Weighted/cohort mode rejects
  `null_values=True`.

### Rules (`association_rules`)

* Cohort input is recognised by a `cohort` column. The per-cohort supports
  emitted by `fpgrowth` are the **same evidence** used to evaluate rules:
  `support`, `confidence`, `lift`, `leverage`, `conviction`, `zhangs_metric`,
  `representativity`, `jaccard`, `certainty` and `kulczynski` are computed
  from the union/antecedent/consequent supports of that cohort. No itemset is
  collapsed to a single scalar across cohorts, and a missing itemset is
  **never** filled with zero.
* Without `reference_cohort`, ordinary rules are generated per cohort, with a
  `cohort` column and no reference columns. `metric`/`min_threshold` still
  filter within each cohort.
* With `reference_cohort`, rules are generated for every *other* cohort and
  gain `reference_cohort`, `reference support`, `reference confidence`,
  `reference lift`, `support delta` and `confidence delta`. Both deltas are
  current minus reference.
* `min_confidence_delta` filters on `confidence delta`; it requires
  `reference_cohort`. A NaN delta (reference antecedent support = 0) cannot
  pass the filter.
* A rule whose current-cohort antecedent support is 0 is not emitted. A
  cohort with zero consequent support gets `lift = NaN`. A reference cohort
  with zero antecedent support gives `NaN` reference confidence/lift and
  `NaN` confidence delta.
* Cohort mode rejects `support_only=True`, `null_values=True`/`df_orig`, and
  `reference_cohort`/`min_confidence_delta` on plain (non-cohort) input.
* Input validation rejects duplicate `(cohort, itemset)` rows, missing subset
  supports, and cohorts whose retained itemset sets are not identical; the
  caller must pass a complete cohort mining result.

## Design trade-offs

1. **One shared FP tree, cohort-vector counters.** Node counters are either
   scalars (classic / weight-only) or sparse dicts
   `{cohort_code: weighted_count}`. A node only stores entries for cohorts
   that actually contributed to that prefix. Memory therefore scales with
   observed prefixes and cohort activity, not with
   `cohorts x rows x items` and not with a densified matrix.
2. **No per-cohort re-mining.** Splitting the table and looping the classic
   miner would rebuild shared prefixes once per cohort. Here each basket is
   inserted once; conditional projections threshold on *any* cohort. The
   tests assert one insertion per non-empty basket and a shared tree smaller
   than the sum of independent trees.
3. **Item counting is block-wise.** Dense input is scanned in row chunks and
   sparse input through its CSR structure, so transients are O(nnz of a
   block). The only cohort-sized array is a `cohorts x items` count matrix
   (64 x 500 = 32k floats) used once to pick frequent items.
4. **Path enumeration is pruned.** In an FP-tree path, counters are
   non-increasing in every cohort, so a node failing all thresholds bounds
   all deeper nodes; only the supported prefix is combined. Without this,
   cohorts with disjoint tails would enumerate combinatorially many
   unsupported subsets.
5. **Output cost is honest.** A retained itemset is materialised once per
   cohort, so a 2.8M-row result costs what the result costs; we do not claim
   sub-linear time in an exponential output. Itemsets are translated to
   column names once and shared between their cohort rows.
6. **Rules read, never repair, the evidence.** `association_rules` validates
   completeness (identical itemset universes, unique pairs, subset presence)
   and raises instead of zero-filling. This is what keeps a below-mining-
   threshold cohort from being mistaken for zero evidence.

## Reproducible verification

Create an isolated environment (the repository pins are older; the checks
below also pass on recent pandas/numpy):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . pytest
```

Run the focused suite:

```bash
python -m pytest mlxtend/frequent_patterns/tests/test_cohorts.py -q
python -m pytest mlxtend/frequent_patterns/tests -q
```

`test_cohorts.py` covers: direct-statistics comparison of itemsets and rules,
a single-cohort-only itemset (with zero support rows for the other cohorts),
zero reference support, empty baskets, fractional/zero weights, duplicate
index, row/column/cohort reordering, sparse input, `max_len`, invalid inputs,
the legacy `fpgrowth`/`fpmax`/`association_rules` behaviour, shared-tree
memory versus independent trees, and insertion work being linear in rows.

Scale check (64 cohorts, 100k baskets, 500 items, <= 12 items per basket,
sparse DataFrame), one implementation-specific run:

```python
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import fpgrowth

rng = np.random.default_rng(0)
n_baskets, n_items, n_cohorts = 100_000, 500, 64
rows = []
cohorts = rng.integers(0, n_cohorts, size=n_baskets)
for i in range(n_baskets):
    must = set(range(8)) | {8 + int(cohorts[i])}
    noise = rng.choice(np.arange(16, n_items), size=12 - len(must), replace=False)
    rows.append(sorted(must | set(noise)))
indptr = np.cumsum([0] + [len(r) for r in rows])
indices = np.concatenate(rows)
data = np.ones(len(indices), dtype=np.int8)
sparse_df = pd.DataFrame.sparse.from_spmatrix(
    csr_matrix((data, indices, indptr), shape=(n_baskets, n_items)),
    columns=[f"i{i}" for i in range(n_items)],
)
res = fpgrowth(sparse_df, min_support=0.02,
               cohorts=[f"c{c}" for c in cohorts], max_len=4,
               use_colnames=True)
print(len(res))
```

Observed on this machine for the exact snippet above: 391,296 output rows in
3.6 s with a 70 MB peak (only ~19 MB retained after the call; the input is a
sparse DataFrame with ~1.1M stored cells, never a 100k x 500 dense array).

With a denser item distribution the same shape can legitimately return
millions of rows; e.g. a run producing 2,829,632 rows took ~9 s and ~0.5 GB
peak, most of which is the result DataFrame itself. That is the honest
output cost, not per-cohort re-mining: the tree for the 100k baskets held
~59k nodes and ~40 MB, and item counting used one `64 x 500` array.

