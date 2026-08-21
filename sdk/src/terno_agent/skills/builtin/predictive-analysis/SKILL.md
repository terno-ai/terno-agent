---
name: predictive-analysis
description: Forecast business time series — sales, order quantity, volume, demand, revenue, spend, headcount — from SQL, spreadsheets, or CSV. Use for any request to predict, project, extrapolate, or plan a future period ("next month by region", "run rate", "will we hit target?", "demand plan"), including quick one-number answers. Covers scoping, auditing history for partial periods and structural breaks, method selection, backtesting, and reporting a range.
---

# Predictive Analysis

A forecast is a claim about the future backed by a stated method and an
honest error bar. Most bad business forecasts fail not because the model
was too simple, but because the history fed to it was not what the
analyst assumed, or because a point estimate got reported without a
range and then treated as a promise.

Weight the effort accordingly. With the 24-60 monthly observations most
business series offer, simple methods beat complex ones reliably, so
framing and auditing deserve more of your time than modelling.

Everything you need is in this file. Aggregate the history in SQL, then
do the arithmetic in `run_python` — pandas, numpy, scipy and
scikit-learn are available there. **statsmodels and Prophet are usually
not installed**, so treat ETS/ARIMA/Prophet as unavailable unless you
import them successfully; the methods below are all a few lines of
pandas and don't need them.

## 1. Frame it first

Settle these before writing SQL — each changes the query, not just the
presentation. Ask the user only where the answer materially changes the
work; infer the rest and state your inference.

- **Metric** — value or quantity, gross or net of returns and tax, and
  which column is canonical. Warehouses usually carry several
  near-synonyms and only one matches the dashboards.
- **Horizon** — error grows fast with it. One period ahead and twelve
  ahead are different exercises admitting different methods.
- **Grain** — total, or split by channel/region/product. Splits are
  usually what's wanted and are where accuracy collapses, since each
  slice carries less signal.
- **Calendar** — fiscal or calendar year. Seasonality, targets, and
  year-end push behaviour follow the fiscal year where one exists.
- **Mandatory scope filters** — entity, row-level security, business
  unit, active vs closed, domestic vs export. Warehouses often have a
  required scoping join that isn't visible from the fact table alone;
  check saved memory, project docs, or an existing dashboard definition
  before assuming the raw table is the population.
- **Purpose** — a purchasing decision needs the downside of the range; a
  board slide needs the drivers.

If the user expects roughly X and you produce half of X, one of you is
wrong about the scope. Surface that gap early, not at the end.

## 2. Audit the history before trusting it

Pull at least three seasonal cycles (36 monthly points), at daily grain
if the source has it — several checks need within-period detail. Then:

- **Partial current period.** The most common error by far: a
  month-to-date figure looks like a collapse. Find the true max date and
  check whether the last day or two are thin against their weekday norm,
  since warehouses often load partially. Drop it or gross it up (§3).
- **Reporting lag.** Sources that backfill make you forecast a downtrend
  that isn't there.
- **Negatives and reversals.** Credit notes, returns, cancellations.
  Decide gross vs net explicitly, and check whether reversals sit in the
  same table under a separate document or entity code — excluding them
  by accident overstates every number.
- **Sentinel keys.** `-1`, `0`, `'N/A'`, `'Not_Defined'`, `9999-12-31`
  create fake categories or silently drop rows on an inner join.
- **Grain of the split dimension.** `SELECT DISTINCT` before trusting a
  breakdown. Categories are often combined in one column and separated
  in another, so a filter that looks right returns zero rows.
- **Which link of the chain.** Sell-in (billing to distributors),
  sell-through, and sell-out are different series. Sell-in is lumpy,
  back-loaded to period end, and reflects trade loading, not demand — if
  the question is demand, no model fixes the wrong series.
- **Structural breaks.** A new channel, an acquisition, a migration, a
  re-based data model. Pre-break history is not evidence about the
  post-break level: start the series at the break or model it, never
  average across it.
- **Zeros vs missing.** Identical after a `GROUP BY`. Join to a calendar
  spine so gaps become explicit.
- **Outliers with a known cause** — bulk orders, strikes, lockdowns.
  Flag them, damp them for fitting, never delete them silently.
- **Unit mismatches** — actuals in units vs targets in thousands.
- **Duplicate sources** — exclude `_bkp`, `_test`, `_copy`, `dump`
  variants.

Write down what you find. These notes become the caveats in your report
and are usually its most valuable part.

## 3. Grossing up a partial period

```
full_estimate = period_to_date / share_of_period_completed_by_day_D
```

Estimate the share from the **same calendar period in prior years, per
slice**. Within-period profiles are rarely flat — month-end billing
pushes, retail weekends and payroll cycles concentrate volume — so
`PTD / days_elapsed * days_in_period` is systematically wrong unless you
have verified the profile actually is flat.

Check stability before relying on it. A day-D share reading 0.475,
0.469, 0.480 across three years is dependable. One reading 0.56, 0.49,
0.41 is drifting: weight recent years more, widen the range, and say so.
Report the estimate as a range spanned by the best and worst
reference-year shares, not as a point.

## 4. Diagnose the series

Look at the data before choosing a model. Check usable length (periods
since the last break, not total rows), trend, noise, share of zero
periods, and whether the recent 3-6 periods differ from the same periods
last year.

Two diagnostics settle most method arguments:

- **Seasonal consistency** — is the period-of-year pattern repeatable in
  sign and rough size across years? A consistent +7%, +8%, +7% is real
  seasonality worth using; −3%, +17%, −2% is noise in a costume.
- **Growth stability** — compute YoY growth for each of the last 6-12
  periods. Steady growth means a ratio method will be accurate and
  trivially explainable. Accelerating or sign-flipping growth means a
  trend or break problem, and a naive ratio will overshoot.

## 5. Choose a method

Start at the top and move down only when a diagnostic justifies it.

| Situation | Method |
| --- | --- |
| < 12 periods | Last value or mean of last 3 — label it a placeholder |
| Seasonal, ≥ 24 periods, flat trend | Seasonal naive |
| Seasonal + steady trend | Seasonal naive × pooled recent YoY growth |
| Need the seasonal shape auditable | Ratio-to-moving-average decomposition |
| Seasonal + trend, want one fitted model | Holt-Winters — hand-rolled in numpy, or skip it |
| Known *future* driver values | Regression on drivers, as an adjustment layer |
| > 30% zero periods | Croston / SBA: smooth demand size and interval separately |
| Future largely already booked | Bottom-up from the order book |

ARIMA and Prophet are deliberately absent: they need libraries that are
usually missing here, and on 30-60 monthly points they rarely beat the
top three rows anyway. If you believe one is needed, say why rather than
reaching for it by reflex.

Three rules matter more than the table:

1. **Always benchmark against seasonal naive.** If a fancier method
   can't beat it in a backtest, ship the simple one and say so — that's
   a real finding about the series, not a failure of effort.
2. **Blend when two defensible methods disagree.** Combination reliably
   reduces error, and the spread between components is a well-founded
   uncertainty range to report.
3. **Prefer methods explainable in one sentence.** A forecast the
   business can't interrogate won't be trusted or corrected.

## 6. Compute it

Aggregate to one row per period per slice in SQL, then work in pandas.
The two workhorse methods and the backtest are short enough to write
directly:

```python
import numpy as np

# s: pd.Series indexed by period end, one slice, oldest first, no gaps.
# m = periods per cycle (12 for monthly). Each fn needs len(s) > m.
def seasonal_naive(s, m=12, h=1):
    return float(s.iloc[-m + (h - 1)])

def yoy_ratio(s, m=12, h=1, pool=3):
    # Pool several periods: a single-month ratio inherits that month's
    # noise and any shipment that slipped across a period boundary.
    base = s.iloc[-pool - m:-m].sum()
    growth = (s.iloc[-pool:].sum() / base) if base > 0 else 1.0
    return float(s.iloc[-m + (h - 1)] * min(max(growth, 0.3), 3.0))

def seasonal_ratio(s, m=12, h=1, years=3):
    # Next period vs current period, averaged over as many prior years
    # as the history actually covers (iloc would raise, not pad).
    years = min(years, (len(s) - 1) // m)
    r = [s.iloc[-m * k + (h - 1)] / s.iloc[-m * k - 1]
         for k in range(1, years + 1) if s.iloc[-m * k - 1] > 0]
    return float(s.iloc[-1] * np.mean(r)) if r else float(s.iloc[-1])

def rolling_origin(s, fn, origins=12, m=12, h=1):
    # Forecast period t from data up to t-1, step forward, repeat.
    # Cap origins so every training slice still spans a full cycle.
    origins = min(origins, len(s) - h - m)
    out = [(s.iloc[t + h - 1], fn(s.iloc[:t], m, h))
           for t in range(len(s) - origins - h + 1, len(s) - h + 1)]
    a, p = np.array([x[0] for x in out]), np.array([x[1] for x in out])
    return dict(wape=np.abs(a - p).sum() / np.abs(a).sum(),
                bias=(p - a).sum() / np.abs(a).sum(), n=len(a))
```

Adapt rather than copy blindly — the slicing assumes a complete monthly
index with no gaps, which is exactly what §2 told you to verify.

## 7. Backtest, then attach a range

Never report a forecast whose method you haven't tested on held-out
history. Use rolling origins (6-12), not one split — a single cut point
on a short series tells you almost nothing.

- **WAPE** (Σ|error| ÷ Σactual) — the default for value forecasts:
  robust, interpretable, doesn't explode near zero.
- **MASE** (error ÷ seasonal-naive error) — below 1 means you beat the
  benchmark. The honest answer to "did the complexity buy anything?"
- **Bias** (mean signed error) — track separately. A model 10% low every
  period is a different and more fixable problem than one randomly off
  by 10%.

Derive the range from the spread of backtest residuals, or from the
disagreement between blended methods. State what it excludes: it
captures historical variability, not shocks the history has never seen.
Don't dress up eight backtest points as a 95% interval.

## 8. Reconcile and overlay

If you forecast slices, check they sum to a total you'd defend —
forecast the total directly as a cross-check, and roll the long tail
into "other" rather than modelling tiny erratic slices. If the two
disagree by more than a few percent, find out which slice is
misbehaving before shipping.

Statistical output is an input to a forecast, not the forecast. Adjust
for what the history can't know — a confirmed price change, a lost
customer, a promo calendar that differs from last year — but **show the
baseline and the adjustment as separate lines**. "Baseline 68, less 5
for the delisting, = 63" is auditable; a single adjusted number hides
the judgement and can't be learned from.

## 9. Report it

```markdown
## Forecast — [metric], [period], [grain]

**Scope:** [source, filters, entity scope, what's in and out]. [Units].
**Data through:** [date]; [partial period and how it was handled].

### Recent trend
[Last 6-12 periods at the forecast grain; flag any estimated period]

### Forecast
| Slice | Same period last year | Forecast | Range | YoY |
| --- | --- | --- | --- | --- |
| **Total** | | | | |

**Method:** [one or two sentences — what you did and why it fits]
**Backtest:** [WAPE/MASE over N rolling origins vs seasonal naive]

### Steps followed
1. [Framing — metric, horizon, grain, calendar, scope filters settled]
2. [History pulled — source, date range, aggregation]
3. [Audit findings — partial period, breaks, outliers, and what you did about each]
4. [Diagnostics — seasonal consistency, growth stability]
5. [Method chosen, and the benchmark it was compared against]
6. [Backtest setup — origins, horizon, metrics]
7. [Adjustments overlaid on the baseline, if any]

### What could move this
- [Biggest source of error, named specifically]
- [Data caveats from the audit]
- [Business events not in the model]
```

Round to the precision the error supports — 68.34291 from a method with
12% backtest error is noise dressed as precision.

Always include the steps-followed trail: a forecast is only auditable if
the reader can see which decisions produced it. Keep it to one line per
step naming the choice and its reason — "dropped Aug, only 6 days
loaded", not "cleaned the data". Where a step changed the number
materially, say by how much. Omit a step only if it genuinely didn't
apply, and say so rather than dropping the line silently.

## 10. Pitfalls

Treating a partial period as complete. Averaging across a structural
break. Extrapolating growth off a depressed base ("+62% YoY" against a
bad year is not a rate — sanity-check the absolute level). Forecasting
the wrong link of the chain. Ignoring the fiscal calendar. Over-fitting
short history, which only backtesting catches. Silent scope drift —
forecasting a filtered subset and reporting it as the total. A point
estimate with no range, which becomes a commitment in someone's plan.
Deleting outliers without saying so, when a real bulk order may recur.
