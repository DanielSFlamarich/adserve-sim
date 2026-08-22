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
through a click model and an auction, and measures what miscalibrated
probabilities cost once bids compete.

Not affiliated with, endorsed by, or derived from any ad platform operator.

## The question

**If a model ranks impressions correctly but predicts the wrong probabilities,
what does that cost?**

Ranking quality only cares about order. Multiply every prediction by 1.5 and
the order is untouched and AUC does not move at all.

But an ad server does not only rank, it prices. A bid is roughly
`value_of_a_click × predicted_click_probability`, so if the second term is 50%
too high across the board, every bid is 50% too high. The advertiser overpays,
the publisher misjudges its floor, budgets exhaust by lunchtime. The ranking was
perfect throughout. The money was still wrong.

This is a common failure for a mundane reason: the standard metric for click
models is AUC, and AUC cannot see it. Calibration drifts, offline evaluation
says nothing, and it shows up in spend.

Measuring the cost needs an auction. On a static dataset you can compute a
calibration error and report it as a number, but not what it was worth;
"worth" only exists once bids compete for a slot.

## Data

Real traffic, real clicks, real hourly rhythm from a public research dataset.
**What is simulated is only the decision-making: the ranking policy, the bidder,
the auction.**

| Dataset | Role | Licence |
|---|---|---|
| [Avazu CTR](https://www.kaggle.com/competitions/avazu-ctr-prediction) | Request stream, features, click labels | Competition terms; Kaggle account required |

No dataset is committed here. `download.py` fetches Avazu through the Kaggle
CLI and prints manual instructions if credentials are missing, since the
competition requires accepting its rules interactively.

Everything the auction needs beyond that is invented, and deliberately so.
Competing bids are drawn from a fitted log-normal, viewability from a prior on
slot position. Both are assumptions with sensitivity analyses around them,
not attempts to look like measurements. This is an exercise in how the pieces
fit together, and a made-up bid distribution answers that as well as a real one would.

## Concepts

**Impression**: one ad shown once. One row, one decision, one outcome.

**Ad request**: a phone app has a slot to fill and asks a server what to put
in it. The server has tens of milliseconds and knows only what the app tells
it: roughly where the slot is, what kind of app, what kind of device.

**CTR**: clicks divided by impressions. **pCTR** is a model's *predicted*
probability that a given impression will be clicked.

**eCPM**: expected revenue per thousand impressions. How otherwise
incomparable ads get compared.

**Auction**: several advertisers want the same slot; the server ranks them and
picks one.

**Pricing rule**: what the winner pays. Under *first-price*, their own bid;
under *second-price*, the runner-up's. Second-price makes honest bidding
optimal, since your bid decides only whether you win, not what you pay. Display
advertising ran second-price historically and moved to first-price around 2019,
after *layered auctions*, *hidden floors* and *last-look* had already eroded the
guarantee. This **project starts with second-price precisely because it is the
clean case: bids equal valuations, so the cost of miscalibration is isolated
from bidding strategy**.

**Reserve price**: a floor the publisher sets. Raising it extracts more from
the auctions you still win, but wins fewer of them.

**Fill rate**: the share of requests that end up showing an ad.

**Calibration**: whether predicted probabilities match observed frequencies.
Across all impressions scored at 2%, did about 2% get clicked?

**AUC**: whether likelier clicks are ranked above less likely ones. A question
about *order*, not about the numbers themselves.

## What the SDK contributed, and what it didn't

I came across the publicly documented [RUNA mobile SDK](https://rakuten-ads.github.io)
and wanted to see how much of an ad platform could be inferred from the
client-side half alone. That is roughly the position anyone integrating with a
third-party ad system is in: you see what your app sends and what comes back,
and you reason about the rest.

Two things came out of it:

- **Three ad formats**: banner, carousel, interstitial. Three kinds of
  inventory with different attention profiles, which probably should not share
  one reserve price.
- **An IAB Open Measurement adapter**, so viewability is measured rather than
  assumed. The docs establish that it is *measured*; the step to "therefore
  viewable eCPM is the objective" is my inference. If it holds, the ranking
  function is `bid × pCTR × p(viewable)` rather than the textbook
  `bid × pCTR`.

Everything server-side stays invisible, exactly as it should for a publisher
SDK: no auction logic, no pricing, no ranking, no reporting API, not one
impression record. Nothing here calls the SDK or contacts any ad server.

**Both inferences turn out to be untestable as stated**, because the public data
lacks the fields. Avazu has no format column and no viewability labels. So the
simulator bounds them instead of claiming them: the reserve sweep segments by
slot position, the nearest structural analogue to format, and viewability enters
as a prior with a sensitivity analysis around it. Neither is a measurement.
Both show how much the question matters.

## Why CatBoost
First, latency. A real ad server has tens of milliseconds for the whole
auction, which rules out models that are marginally more accurate but slower to
score. This is why logistic regression and GBDTs still dominate CTR prediction
in production over deep networks, the constraint is inference time, not
capacity.

Second, the categoricals. Avazu's fields are anonymised hashes with no ordinal
meaning and enormous cardinality, so one-hot encoding is impossible and the
encoding scheme *is* the modelling decision. CatBoost's ordered target
statistics address exactly that, which makes it the natural choice here and the
natural thing to compare a hand-written encoder against.

Logistic regression is the honest alternative and is better calibrated out of
the box, which, given what this project is about, is worth noting rather than
hiding. It would need the feature crosses built by hand, which is the work
CatBoost does for free.

## What is being optimised

The model's target is **`click`**, a 0/1 column. CatBoost fits it with log
loss and outputs a probability.

Avazu carries no conversion, purchase or revenue data, so `value_of_a_click` is
a parameter, not a measurement. In reality it comes from the advertiser's own
conversion model. Revenue figures here are therefore *simulated publisher
revenue under an assumed click value*. The calibrated-versus-uncalibrated
comparison holds for any constant value, which is the point; the absolute
currency figure is not one anyone should believe.

## How the code is built

**Never train on the future.** A model deployed on Monday is scored on
Tuesday's traffic, so evaluation imitates that: fit on earlier days, test on
later ones. Splits cut on day boundaries, and the overlap check runs
automatically whenever a split is created, an overlapping split cannot be
built, the constructor refuses.

**Never let the model see the answer.** Replay produces two separate objects:
the request (what a server knows at decision time) and the outcome (whether the
click happened). The model is handed the request only. A test asserts the
request carries exactly three fields, none of them the click, so any future
change that smuggles the answer in fails immediately.

**Fail loudly.** An unparseable timestamp raises rather than becoming a null
that vanishes from a later filter. Out-of-order data raises rather than being
quietly re-sorted, because arriving unsorted means something upstream is broken.
A missing column raises rather than producing a model trained on a different
feature set than intended.

**Name assumptions, don't bury them.** Competing bids, viewability... none are
in the public data. Each is a stated assumption with a
sensitivity analysis, not a number chosen quietly to make results look
reasonable. The point is that when real logs replace an assumption, you know
which result moves.

## Plan

Each step lands as its own commit with tests, and each has a check that fails if
the step is *wrong*, not merely unfinished.

**1. Data acquisition**: pin the column contract; sample stratified by hour so
the daily traffic curve survives sampling.
*Check:* hourly distribution matches the source within 1%; sampling reproducible
from a seed.

**2. Temporal split and replay**: day-boundary partitions with the overlap
check enforced at construction.
*Check:* windows strictly ordered and sharing no impressions; the request object
exposes three fields, none of them the label.

**3. Features and click prediction**: gradient-boosted trees, MLflow-tracked.
Avazu's categoricals are enormous (`device_ip`, `device_id` run to millions of
values), so encoding is the real decision. CatBoost handles it natively with
ordered target statistics; a hand-written out-of-fold encoder runs alongside,
because implementing the mechanism demonstrates understanding it in a way that
calling a library does not.
*Check:* test AUC beats a frequency-only baseline; both encoders recorded as
separate runs.

**4. Calibration**: isotonic regression and Platt scaling fitted on validation,
evaluated on test. Reliability diagrams and expected calibration error.
*Check:* calibration error falls substantially while AUC is unchanged. If AUC
moves, the calibrator is altering the ranking, which it should not.

**5. Auction and reserve price**: clear a second-price auction against a synthetic
competing-bid distribution, sweep the reserve, plot revenue against fill rate.
Then the payoff: run the calibrated and uncalibrated models through the same
auction and compare.
*Check:* revenue is non-monotonic in the reserve price. A curve that only rises
means the simulation is broken. A reserve trades price against fill by
construction, so the trade-off must appear.

```
0. Read the RUNA SDK          public docs, no code used
                                └── findings feed step 5 only
          ↓
1. Data acquisition           schema.py, download.py
          ↓
2. Split and replay           split.py, replay.py
          ↓
3. Click model                build.py, train.py
          ↓
4. Calibration                calibrate.py, reliability.py
          ↓
5. Auction and reserve        ranking.py, second_price.py
```

Steps 1–4 are pure Avazu work and would be unchanged if the SDK did not exist.
The findings only bite at step 5, which is honest rather than disappointing: a
publisher SDK was never going to inform how you split data or encode
categoricals. What it constrains is the decision layer: which is the layer a
simulator exists to explore.

## What I could not work out

The point of reading a public SDK is that you hit the edge of it. These need
server-side knowledge or production logs; the simulator's assumptions stand in
for them.

- **Is the reserve price set per format, per placement, or globally?** Three
  inventory types with different attention profiles suggest per-format. The
  sweep here assumes global — the simpler and probably wrong answer.
- **Does viewability feed the ranker, or is it only measurement?** Open
  Measurement proves it is measured. Whether it is also *predicted* at decision
  time determines whether the third term in the ranking function is real.
- **What does the attribution window look like?** Beacon timing implies
  conversions arrive well after the impression — the delayed-label problem.
  Window length changes how a model should be trained.
- **How much does traffic seasonality distort pacing?** Avazu's ten days give a
  daily rhythm but say nothing about campaign-scale or seasonal effects.


**Deferred** until the above works end to end: contextual bandits for creative
selection, delayed-feedback modelling, budget pacing as a control problem,
off-policy evaluation with IPS and doubly-robust estimators.

## Layout

```
src/adserve_sim/
  data/      schema.py, download.py, split.py
  sim/       replay.py
  features/  build.py
  models/    train.py
tests/       one file per module
reports/     written findings and figures
references/  dataset licences, SDK contract notes
```

## Running it

```bash
make install       # dependencies and git hooks
make check         # lint, format, types, tests
make data          # fetch and prepare the sample
```

`make check` is the gate and it runs exactly what CI runs, in the same order.

You'll need Python 3.12. uv installs it if you don't have it and everything else comes
from uv.lock.
The Avazu download needs a Kaggle account, an API token at ~/.kaggle/access_token,
and acceptance of the competition rules at the rules page. The last one is easy to
miss: without it, listing the competition's files still works but downloading returns a generic error.
