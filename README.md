# adserve-sim

[![CI](https://github.com/DanielSFlamarich/adserve-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielSFlamarich/adserve-sim/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/types-mypy%20strict-2A6DB2)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A simulator for mobile display ad serving. It replays real ad-request logs
through a prediction model and an auction, and measures what happens to revenue
when the model's predicted click probabilities are wrong in a specific way.

Not affiliated with, endorsed by, or derived from any ad platform operator.

## Concepts

**Impression**: one ad shown to one person, once. The basic unit of everything
here: one row of data, one decision, one outcome.

**Ad request**: the moment a phone app has a slot to fill and asks a server
what to put in it. The server has some tens of milliseconds and knows only what
the app tells it: roughly where the slot is, what kind of app, what kind of
device. It does not know who the person is or what they will do.

**CTR (click-through rate)**: clicks divided by impressions.
**pCTR** is a model's *predicted* probability that a
specific impression will be clicked.

**eCPM (effective cost per mille)**: expected revenue per thousand
impressions. It is how otherwise incomparable ads get compared: a cheap ad that
is often clicked can be worth more than an expensive one that never is.

**Auction**: several advertisers want the same slot, so the server ranks them
and picks one. In a **second-price** auction the winner pays what the
*runner-up* bid, not their own bid, which under standard assumptions, removes
the incentive to shade bids downward and makes bidding the true value the
sensible strategy.

**Reserve price**: a floor the publisher sets. Bids below it lose, and the slot
goes unfilled. Raising the reserve extracts more from the auctions you still
win, but wins fewer of them.

**Fill rate**: the share of ad requests that end up showing an ad. A reserve
price too high leaves slots empty.

**Calibration**: whether predicted probabilities match observed frequencies.
Across all the impressions a model scored at 2%, did about 2% get clicked? A
model can be badly calibrated and still be useful for some purposes, which is
exactly the problem this project is about.

**Ranking quality (AUC)**: whether the model puts likelier clicks above less
likely ones. Note that this is a question about *order*, not about the numbers
themselves.

## The question

Having worked out roughly what an ad request looks like, the obvious next
question was what to do with one. This is the one I got interested in:

**If a model ranks impressions correctly but predicts the wrong probabilities,
what does that cost?**

The two properties come apart, and the reason is worth stating precisely.
Ranking quality only cares about order. Multiply every prediction by 1.5 and the
order is untouched, AUC does not move at all. The model is exactly as good at
telling likely clicks from unlikely ones as it was before.

But an ad server does not only rank. It prices. A bid is roughly

```
bid = value_of_a_click x predicted_click_probability
```

and if the second term is 50% too high across the board, every bid is 50% too
high. The advertiser overpays, or the publisher sets a reserve price against
inflated bids and misjudges the floor, or budgets exhaust by lunchtime. The
ranking was perfect throughout. The money was still wrong.

This is a common and consequential failure, for a mundane reason: the standard
metric for click models is AUC, and AUC cannot see it. A team can watch AUC
improve release over release while calibration quietly drifts, and nothing in
the offline evaluation says so. It shows up in spend.

Measuring the cost requires an auction. On a static dataset you can compute a
calibration error and report it as a number, but you cannot say what it was
worth, because "worth" only exists once bids compete for a slot. That is what
this simulator supplies.

## Why a simulator, and where the data comes from

Real ad-serving data is not public, and neither is any production ad server. But
two things *are* available, so combining them is the approach here.

**Real traffic**, from public research datasets: genuine ad requests with
genuine features and genuine click outcomes, including the hourly rhythm of real
usage. Everything about the environment is real and untouched.


**A request shape** worked out from a real SDK. I came across the publicly documented
[RUNA mobile SDK](https://rakuten-ads.github.io) and wanted to see how much of an
ad platform could be inferred from the client-side half alone.

That is roughly the position anyone integrating against a third-party ad system
is in: you see what your app sends and what comes back, and you reason about
the rest.
Two things came out of it. The first is that there are three ad formats; banner,
carousel and interstitial which are three different kinds of inventory, and probably
should not share one reserve price. The second is more consequential: the SDK ships
an IAB Open Measurement adapter, so viewability is a measured signal rather than
an assumption. That changes the objective. What a publisher maximises is viewable
eCPM, so the ranking function here is

``
bid x pCTR x p(viewable)
``

rather than the textbook bid x pCTR. I would not have built it that way without reading
the contract. Everything server-side stays invisible, which is exactly as it should be
for a publisher SDK: no auction logic, no pricing, no ranking, no reporting API, and
not a single impression record. So the traffic comes from public research datasets and
the SDK contributes only structure. Nothing here calls the SDK or contacts any ad server.

What is simulated is only the decision-making: the ranking policy, the bidder,
the auction. Those are mine, and therefore things I can vary and measure.

| Dataset | Role | Licence |
|---|---|---|
| [Avazu CTR](https://www.kaggle.com/competitions/avazu-ctr-prediction) | Request stream, features, click labels | Competition terms; Kaggle account required |
| [iPinYou](https://contest.ipinyou.com/) | Competing-bid distribution | Research use |
| [Criteo Attribution](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/) | Conversion delay distribution *(planned)* | CC-BY-NC |
| [Open Bandit](https://research.zozo.com/data.html) | Logged propensities for off-policy evaluation *(planned)* | CC-BY |

No dataset is committed here. `download.py` fetches Avazu through the Kaggle
CLI and prints manual instructions if credentials are missing, since the
competition requires accepting its rules interactively.

## How the code is built

Three principles.

### Never train on the future

A model deployed on Monday is scored on Tuesday's traffic. So evaluation has to
imitate that: fit on earlier days, test on later ones.

This project splits on day boundaries: the last two days are the test set, the
day before that is validation, everything earlier is training. The check that
these three windows do not overlap runs **automatically whenever a split object
is created**. There is no separate "remember to validate" step that someone can
forget, because a split that overlaps cannot be built in the first place, the
constructor will refuse.

### Never let the model see the answer

When the simulator replays a logged impression, it produces two separate
objects: the **request** (what an ad server would know at decision time (device
type, slot position, app category) and the **outcome** (whether the click
happened).

A prediction model is handed the request only. The outcome is used afterwards,
to score what the model decided. This sounds obvious, but leakage bugs are
usually accidental: a stray column, a feature computed after the fact, and the
separation makes the mistake structurally difficult rather than merely
discouraged. A test asserts that the request object contains exactly three
things, none of which is the click, so any future change that smuggles the
answer in fails immediately.

### Fail loudly, not quietly

Where a shortcut would let bad data pass, the code stops instead:

- A timestamp that will not parse raises an error, rather than becoming a null
  that silently disappears from a later filter.
- Data arriving out of chronological order raises, rather than being quietly
  re-sorted because if it arrived unsorted, something upstream is broken, and
  fixing the symptom hides it.
- A missing column raises, rather than producing a model trained on a different
  feature set than intended.
Each of these would otherwise show up weeks later as a metric that moved for no
apparent reason.

## What I could not work out

The point of reading a public SDK is that you hit the edge of it. These are the
questions the client-side contract raised and could not answer as they need
either server-side knowledge or production logs, and the simulator's assumptions
stand in for them:

- **Is the reserve price set per format, per placement, or globally?** Three
  inventory types with different attention profiles suggest per-format, but the
  contract says nothing either way. The reserve sweep here assumes global,
  which is the simpler and probably wrong answer.
- **Does the viewability prediction feed the ranker, or is it only
  measurement?** Open Measurement proves viewability is *measured*. Whether it
  is also *predicted* and used at decision time is a different question, and it
  determines whether the third term in the ranking function above is real.
- **What does the attribution window look like?** Beacon timing implies
  conversions arrive well after the impression, which is the delayed-label
  problem. How long the window is changes how a model should be trained.
- **How much does traffic seasonality distort pacing?** Retail inventory is
  far spikier than steady app inventory. Avazu's ten days give a daily rhythm
  but say nothing about campaign-scale or seasonal effects.

Each of these is currently a stated assumption in the code rather than a
measured value. That is the honest state of a simulator built from outside.

## Layout

```
src/adserve_sim/
  data/
    schema.py      column contract, timestamp parsing
    download.py    dataset fetch, per-hour stratified sample
    split.py       day-boundary train/validation/test
  sim/
    replay.py      chronological request stream
tests/             one file per module
reports/           written findings and figures
references/        dataset licences, SDK contract notes
```

## Requirements

| | Tool | Purpose |
|---|---|---|
| **Python 3.12** | Pinned in `.python-version`; `uv` installs it if absent |
| **uv** | Dependency resolution and virtual environment |
| **CatBoost** | Gradient-boosted trees with native categorical handling |
| **pandas / NumPy / PyArrow** | Data handling and Parquet I/O |
| **scikit-learn** | Calibration and evaluation metrics |
| **MLflow** | Experiment tracking |
| **Ruff** | Linting and formatting |
| **mypy** | Static typing, strict mode |
| **pytest** | Test suite |
| **pre-commit** | Git hooks |
| **Kaggle account** | Required to download Avazu (see Data) |

Install everything with `make install`, `uv` reads `uv.lock`, so the
environment resolves identically here and in CI.

## Running it

```bash
make install       # dependencies and git hooks
make check         # lint, format, types, tests
make data          # fetch and prepare the sample
```

`make check` is the gate, it runs exactly what CI runs, in the same order.



### 1. Data acquisition

Pin the column contract; sample stratified by hour so the daily traffic curve
survives sampling.

*Check:* hourly distribution of the sample matches the source within 1%;
sampling is reproducible from a seed.

### 2. Temporal split and replay

Day-boundary partitions with the overlap check enforced at construction; a
chronological request stream that withholds outcomes.

*Check:* windows are strictly ordered and share no impressions; the request
object exposes three fields, none of them the label.

### 3. Features and click prediction

Gradient-boosted trees on the split, tracked with MLflow.

Avazu's categorical fields are enormous: `device_ip` and `device_id` run to
millions of distinct values, so how they are encoded is the real decision.
CatBoost handles this natively using ordered target statistics, an approach
designed specifically to avoid the leakage that naive target encoding causes.
The baseline uses it; a hand-written out-of-fold encoder runs alongside as a
comparison, because implementing the mechanism demonstrates understanding it in
a way that calling a library does not.

*Check:* test AUC beats a frequency-only baseline; both encoders recorded as
separate MLflow runs.

### 4. Calibration

Fit isotonic regression and Platt scaling on the validation window, evaluate on
test. Reliability diagrams and expected calibration error.

*Check:* calibration error falls substantially while AUC is unchanged. If AUC
moves, the calibrator is altering the ranking, which it should not.

### 5. Auction and reserve price

Clear a second-price auction against a competing-bid distribution, then sweep
the reserve price and plot revenue against fill rate. The competing bids come
from iPinYou's observed clearing prices, or a fitted log-normal with the
assumption stated openly.

Then the payoff: run the calibrated and uncalibrated models through the same
auction and compare revenue. That number is the answer to the question at the
top.

*Check:* revenue is non-monotonic in the reserve price. A curve that only rises
means the simulation is broken; a reserve trades price against fill by
construction, so the trade-off must appear.

### Deferred

Held back deliberately until the above works end to end: contextual bandits for
creative selection, delayed-feedback modelling for conversions, budget pacing as
a control problem, and off-policy evaluation with inverse-propensity and
doubly-robust estimators.
