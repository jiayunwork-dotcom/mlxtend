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

## Wide weight ranges: scaling and retained evidence

Weighted mining must keep three situations apart at every stage (FP-tree
counts and conditional projections, `fpgrowth` pruning and the result table,
rule evidence lookup and filtering):

* **no positive weight** for an itemset in a cohort (a genuine zero);
* **positive support so small that the float `support` column displays it as
  zero** (underflow, but real evidence);
* **an itemset missing from the result table altogether** (incomplete input,
  which must raise rather than be treated as zero).

### Exact per-group power-of-two rescaling

A batch whose individual weights are finite can still have totals outside
the float range (e.g. `3 * 2**1023` overflows). Before the tree is built,
each cohort's weights (or the whole batch in weight-only mode) are multiplied
by one common power of two, `2**k`. Multiplying a binary float by a power of
two only changes its exponent, so the operation is exact while the result
stays in range, and every support ratio is unchanged because the numerator
and denominator of a cohort use the same factor. The exponent is chosen from
the frexp exponents of the group's smallest positive weight and its largest
possible accumulated count, without ever forming an overflowing product.
Groups that already fit keep exponent zero, so ordinary inputs follow their
original arithmetic and pay only an O(rows) bounds scan. When no power of
two can keep the largest sum finite and the smallest weight positive, the
caller gets a clear `ValueError` instead of silently corrupted counts.

Cost: one extra pass over the weight array per group and one rescaled copy
of it (O(rows) memory, the same order as the existing weight vector). Tree
counters, thresholds and pruning are otherwise untouched.

### What mining keeps, and what is deferred to the rules stage

The result table keeps its existing float `support` column and display
precision, and an empty basket still counts in the cohort denominator. A
positive count whose support underflows to zero cannot be reconstructed from
that column, so the miner additionally attaches a small **evidence sidecar**
to the result frame's `attrs`:

* `tiny_counts`: only the `(itemset, cohort label) -> positive scaled count`
  entries whose displayed support is zero — typically a handful, never the
  whole result;
* `cohort_totals`: the scaled total weight per cohort (one float each).

Everything else is deferred to rule time: ordinary positive supports are
turned back into counts as `support * cohort_total` on demand, and rule
metrics are computed from those counts. Ratios such as confidence and lift
go through a safe fraction helper that uses ordinary division when it is
finite and exact, and only falls back to log-space accumulation when a
product would overflow or underflow; `inf`/`0.0`/`NaN` appear only when the
true metric crosses those bounds. The sidecar is keyed by stable cohort
labels (not first-appearance positions) and rides on `DataFrame.attrs`, so
copying the whole result or permuting its rows keeps the evidence attached
and the rule output independent of row order. A hand-built support table
without the sidecar keeps the legacy behaviour: an explicit zero is a real
zero and yields the historical NaN confidence.

Resident overhead is therefore proportional to the actual underflowing
prefix/candidate/result entries plus one float per cohort, not to the number
of rows or items; sparse inputs still share the single mined tree, and the
rule entry point needs nothing beyond the result table.

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
python -m pytest mlxtend/frequent_patterns/tests/test_weighted_scaling.py -q
python -m pytest mlxtend/frequent_patterns/tests -q
```

`test_cohorts.py` covers: direct-statistics comparison of itemsets and rules,
a single-cohort-only itemset (with zero support rows for the other cohorts),
zero reference support, empty baskets, fractional/zero weights, duplicate
index, row/column/cohort reordering, sparse input, `max_len`, invalid inputs,
the legacy `fpgrowth`/`fpmax`/`association_rules` behaviour, shared-tree
memory versus independent trees, and insertion work being linear in rows.

`test_weighted_scaling.py` pins the two wide-range failures with small
baskets: overflowing total weights (`2**1023`) in both entry points, tiny
positive reference evidence (`2**-600` / `2**600`) that displays as zero but
keeps a finite 0.5 reference confidence, genuinely zero reference weights
that keep that confidence undefined (NaN and filtered by a delta bound),
true-zero versus displayed-zero handling, a copied and row-permuted mining
result, and the weight-only call. It also property-checks the rescaling
exponent bounds and the impossible-span error.

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

