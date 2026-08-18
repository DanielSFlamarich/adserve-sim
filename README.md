# adserve-sim

A simulator for mobile display ad serving: replays real CTR logs through a
ranking and auction path, and measures how calibration error propagates into
revenue.

The question it exists to answer is narrow and concrete: **if a click-prediction
model is well-ranked but badly calibrated, what does that cost at auction?**
AUC is invariant to monotone transformations of the score, so a model can rank
perfectly and still price every impression wrong. In an ad system where the bid
is `value x pCTR`, that error lands directly on revenue.

Not affiliated with, endorsed by, or derived from any ad platform operator.

## Why a simulator

Real ad-serving data is not public. What *is* public is a set of research
datasets containing real requests, real clicks, and real auction clearing
prices. This project treats those logs as the environment and supplies its own
agent: the ranking policy, the bidder, the pacing controller.

That division is deliberate. Everything about the traffic — arrival rates,
hourly seasonality, feature distributions, base click rates — is real and
untouched. Everything about the decision-making is mine, and therefore something
I can vary and measure. Nothing is synthetic except the parts that *should* be
synthetic.

## Design commitments

Three choices shape most of the code.

**Time is not shuffled.** Ad serving is forecasting: a model fitted today is
scored on traffic that has not happened yet. Splits cut on whole-day boundaries,
and the leakage check runs in `TemporalSplit.__post_init__` — an invalid split
cannot be constructed, rather than being detectable after the fact.

**Policies cannot see labels.** The replay stream yields `AdRequest` (features,
no outcome) and `Outcome` as separate objects. A policy handed a request has no
access to the click. This is enforced by the type, not by convention, and tested
against the dataclass fields so that adding a label-bearing field later fails
the suite.

**Failures are loud.** Unparseable timestamps raise instead of coercing to
`NaT`; unsorted input to replay raises instead of being silently re-sorted;
missing columns raise instead of producing a quietly different feature space.
Every one of these would otherwise surface much later as an unexplained metric
shift.

## Data

| Dataset | Role | Licence |
|---|---|---|
| [Avazu CTR](https://www.kaggle.com/competitions/avazu-ctr-prediction) | Request stream, features, click labels | Competition terms; Kaggle account required |
| [iPinYou](https://contest.ipinyou.com/) | Competing-bid distribution (`payprice`) | Research use |
| [Criteo Attribution](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/) | Conversion delay distribution *(planned)* | CC-BY-NC |
| [Open Bandit](https://research.zozo.com/data.html) | Logged propensities for off-policy evaluation *(planned)* | CC-BY |

No dataset is committed to this repository. `download.py` fetches Avazu via the
Kaggle CLI and prints manual instructions if credentials are absent — the
competition requires interactively accepting its rules, so the fetch cannot be
made fully unattended.

## Layout

```
src/adserve_sim/
  data/
    schema.py      column contract, timestamp parsing
    download.py    Kaggle fetch, per-hour stratified sample
    split.py       day-boundary train/validation/test
  sim/
    replay.py      chronological request stream
tests/             one file per module
reports/           written findings and figures
references/        dataset licences, SDK contract notes
```

## Running it

```bash
make install       # uv sync + pre-commit hooks
make check         # ruff, format, mypy --strict, pytest
make data          # fetch and prepare the Avazu sample
```

`make check` is the gate. It runs everything CI runs, in the same order.

## Plan

Five steps to a working v0. Each lands as its own commit with tests, and each
has a check that fails if the step is wrong rather than merely incomplete.

### 1. Data acquisition — done

`schema.py`, `download.py`. Pins the 24-column contract; samples stratified by
hour so the daily volume curve survives sampling.

*Verified:* max hourly-share drift under 0.01 between source and sample;
sampling is deterministic under a fixed seed.

### 2. Temporal split and replay — done

`split.py`, `replay.py`. Day-boundary partitions with leakage enforced at
construction; a chronological stream that withholds outcomes from policies.

*Verified:* partitions are strictly ordered and share no impression ids;
`AdRequest` exposes exactly three fields, none of them the label.

### 3. Features and pCTR model — next

CatBoost on the split, tracked in MLflow. Avazu's categoricals are
high-cardinality (`device_ip`, `device_id` run to millions of distinct values),
so the encoding choice is the substantive decision: CatBoost's native ordered
target statistics as the baseline, with an explicit out-of-fold target encoder
as a comparison. The comparison is the point — it shows the leakage mechanism
rather than delegating it to a library.

*Check:* test-set AUC above a frequency-only baseline; encoder comparison
recorded as two MLflow runs.

### 4. Calibration

Isotonic against Platt scaling, fitted on validation and evaluated on test.
Reliability diagrams and expected calibration error.

*Check:* ECE falls materially while ranking AUC is unchanged. If AUC moves,
the calibrator is doing something it should not.

### 5. Second-price auction with reserve

Clearing against a competing-bid distribution, then a reserve-price sweep
plotting the revenue/fill-rate frontier. The bid distribution comes from
iPinYou's observed clearing prices, or a parametric log-normal with the
assumption stated explicitly.

*Check:* revenue is non-monotonic in the reserve price. A monotone curve means
the simulation is wrong — the whole point of a reserve is the trade-off between
price and fill.

### Then

Deliberately deferred until v0 works end to end: contextual bandits for creative
selection, delayed-feedback modelling for conversion attribution, budget pacing
as a control problem, and off-policy evaluation with IPS and doubly-robust
estimators.

## Status

Steps 1–2 complete, 36 tests, CI green. Step 3 in progress.
