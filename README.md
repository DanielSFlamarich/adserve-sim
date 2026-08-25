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
**What is simulated is the decision layer and the market around it: the bid,
the rivals it competes against, and the auction that clears them.**

| Dataset | Role | Licence |
|---|---|---|
| [Avazu CTR](https://www.kaggle.com/competitions/avazu-ctr-prediction) | Request stream, features, click labels | Competition terms; Kaggle account required |

No dataset is committed here. `download.py` fetches Avazu through the Kaggle
CLI and prints manual instructions if credentials are missing, since the
competition requires accepting its rules interactively.

Everything the auction needs beyond that is invented, and deliberately so.
Competing bids are drawn from a log-normal centred on the median honest
bid; `sample_competing_bids` in `auction/second_price.py`, with `DEFAULT_N_COMPETITORS = 5`
and `DEFAULT_BID_SIGMA = 0.6`. Viewability is a hardcoded prior on slot position:
`VIEWABILITY_PRIOR` in `auction/ranking.py`, seven values encoding only the assumption
that position 0 is seen more than position 7.

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

**Drift**: Calibration can degrade (drift) over time. Cheap to monitor, since it
needs only
predictions and outcomes that are already logged, and it usually
moves before online KPIs do. Often a symptom of *data drift* (the input mix
changed) or *concept drift* (the same inputs now imply a different click rate),
so it works as a leading indicator for both.

**Selection bias**: a different reason probabilities can be wrong: the training
data itself is unrepresentative, because you only observe outcomes for
impressions you won. Post-hoc calibration cannot fix this. A model can be
perfectly calibrated on what it won and systematically blind to what it stopped
bidding on. The asymmetry matters as over-prediction is self-correcting (win
more, see the bad outcomes, retrain pulls it down) while under-prediction is
self-perpetuating (stop winning, get no data, nothing contradicts it).

**AUC**: whether likelier clicks are ranked above less likely ones. A question
about *order*, not about the numbers themselves.

## What the SDK contributed, and what it didn't

I came across the publicly documented [RUNA mobile SDK](https://rakuten-ads.github.io)
and wanted to see how much of an ad platform could be inferred from the
client-side half alone.

Two things came out of it:

- **Three ad formats**: banner, carousel, interstitial. Three kinds of
  inventory with different attention profiles, which probably should not share
  one reserve price.
- **An IAB Open Measurement adapter**, so viewability is measured rather than
  assumed. The docs establish that viewability is measured; the step to "therefore
  it belongs in the ranking function" is my inference. If it holds, the serving
  decision maximises `bid × pCTR × p(viewable)` rather than the textbook `bid × pCTR`.
  Note: this is what the server maximises when choosing an ad, not what any model is
  trained on. The click model's target stays click.

Everything server-side stays invisible, exactly as it should for a publisher
SDK: no auction logic, no pricing, no ranking, no reporting API, not one
impression record. Nothing here calls the SDK or contacts any ad server.

**Both inferences are untestable as stated**, because the public data lacks the
fields. Avazu has no format column and no viewability labels. So the simulator
bounds them instead of claiming them, and each bound is marked in the code at
the point where the real answer would go:

| Finding | With server access | Here |
|---|---|---|
| Open Measurement -> viewable eCPM | Fit `p(viewable)` on measured viewability; ask whether it feeds the ranker or only reporting | `VIEWABILITY_PRIOR` in `auction/ranking.py`; seven hardcoded values by `banner_pos`. The function takes an override, but this iteration doesn't use it; the sensitivity analysis is unrun. |
| Three formats -> per-format reserve | Sweep the reserve per format and placement | Sweep per `banner_pos`, the nearest structural analogue `auction/second_price.py` |
| Request contract | Observe the wire format | Not done. `AdRequest` carries Avazu's schema `sim/replay.py` |

The third is a scoping decision rather than a data limitation: running the
sample app behind a proxy would have shown the wire format, and it was left out.

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

Avazu carries no conversion, purchase or revenue data, so what a click is worth
is a parameter, not a measurement: `DEFAULT_VALUE_PER_CLICK = 1.0` in `auction/ranking.py`,
which makes every currency figure below a multiple of "one click's worth".
In reality it comes from the advertiser's own
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
in the public data. Each is stated where it is used rather than chosen quietly
to make results look reasonable. Sensitivity analysis around them is the obvious
next step and is not done yet. The point is that when real logs replace an
assumption, you know which result moves.

## Every invented number, and where it lives

| Assumption | Value | Where |
|---|---|---|
| Value of a click | 1.0 | `DEFAULT_VALUE_PER_CLICK`, `auction/ranking.py` |
| Viewability by slot | 0.75 → 0.25 across `banner_pos` | `VIEWABILITY_PRIOR`, `auction/ranking.py` |
| Rivals per auction | 5 (swept over 1, 2, 5) | `DEFAULT_N_COMPETITORS`, `auction/second_price.py` |
| Competing-bid spread | log-normal, σ = 0.6, centred on median honest bid | `sample_competing_bids`, `auction/second_price.py` |
| Reserve price | swept 0 → 0.5 | notebook `03-auction.ipynb` |
| Train/validation/test | 7 / 1 / 2 days | `split_by_day` defaults, `data/split.py` |
| Encoder smoothing | 20 pseudo-observations | `DEFAULT_SMOOTHING`, `features/build.py` |
| Distortion magnitudes | shift ±0.4, sharpness 1.5 / 0.6 | `STANDARD_SCENARIOS`, `eval/distortion.py` |

None of these is fitted. Each is chosen to be plausible and stated where it is used.
One caveat on how far that goes: the calibration comparison is robust to the click
value, since a constant multiplies every bid equally and cancels. The auction results
are not equally robust. The reserve finding depends on the competing-bid spread, and
which scenario wins depends on the budget level. Those are conclusions about
conditions, not about numbers, which is why the budget sweep reports several levels
rather than one.

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

**4. Calibration**: the honest version of this step is that a public dataset can't
deliver the failure it was meant to demonstrate. Avazu is a competition file, and
Kaggle's own description states clicks and non-clicks were subsampled at different
rates before release. The $16.9%$ click rate is real for this file and inflated
relative to the traffic it came from.

Two consequences. A model fitted here is calibrated to a distribution that does not
exist (deployed against real traffic it would over-predict by roughly two orders of magnitude),
which is precisely the failure this project studies, except the correction is a log-odds
offset of an unpublished constant and so cannot be applied. And a $16.9%$ base rate is a
comfortable estimation problem; real display `CTR` near $0.2%$ is where probability
estimates actually break, and that regime isn't in the file.

On top of that, the fitting was done the way that makes calibration most likely: Logloss is
a proper scoring rule, penalising wrong probabilities directly rather than only wrong ordering,
and early stopping fired at iteration 408 before the model could grow confident enough to distort.
Measured gap: +0.0026 on a 16.4% base rate. See `notebooks/01-avazu-eda.ipynb`.

So the miscalibration has to be manufactured. Distort a good model by a known amount, affine in
$log-odds — a·logit(p) + b$, which stays in range and separates the two failure modes: b shifts
the base rate, a changes sharpness.

**5. Auction and reserve price**: clear a second-price auction against a synthetic
competing-bid distribution, sweep the reserve, plot revenue against fill rate.
Then the payoff: run the calibrated and uncalibrated models through the same
auction and compare.
*Check*: revenue is non-monotonic in the reserve price. It is, but weakly (with five rivals the
peak sits barely above the no-reserve level, because the runner-up's bid already captures most of
the surplus). The effect only becomes material in thin auctions: with one rival the optimum
is worth about 8% over no reserve at all. Whether a floor is worth setting turns out to depend
on how many bidders show up.

**6. Budget**: cap spending and re-run every scenario. Without a cap, over-predicting
is *more* profitable than honest bidding, in a second-price auction you pay the
runner-up's bid, so bidding above your value wins extra auctions at prices someone
else sets, and while ROI collapses, volume more than compensates. Unlimited money
was an unstated assumption doing a great deal of work.

Once spend is capped the ranking inverts, and the interesting part is that it inverts
differently at different budgets. At a cap equal to what honest bidding naturally
spends, over-predicting loses about 38%. At a quarter of that, the *timid* models win
— under-prediction becomes an accidental virtue when money rather than opportunity is
the binding constraint.

*Check:* over-bidding exhausts the budget strictly earlier than honest bidding.

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
4. Calibration                distortion.py, reliability.py
          ↓
5. Auction and reserve        ranking.py, second_price.py
          ↓
6. Budget                     budget.py
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


*Deferred*: contextual bandits for creative selection, delayed-feedback modelling,
off-policy evaluation with IPS and doubly-robust estimators, and a real pacing
controller, `pacing_rate` in `auction/budget.py` is a fixed participation rate,
where a production pacer would use feedback on realised spend.

## Layout

```
src/adserve_sim/
  data/      schema.py, download.py, split.py
  sim/       replay.py
  features/  build.py
  models/    train.py
  eval/      distortion.py, reliability.py
  auction/   ranking.py, second_price.py, budget.py
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
