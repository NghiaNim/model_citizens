# 2026 Senate Forecast

A pipeline that turns polls, election history, fundraising, forecaster ratings and prediction markets into win probabilities for the 35 Senate races on November 3, 2026. Data is as of November 2. The output is `predictions.csv`.

**Result:** Democrats win an expected 52.2 seats (80% range 49–55) and have about an 80% chance of a majority. Markets price that majority at 74.5% (Kalshi) and 78.5% (Polymarket). Democrats need 51, because the Vice President breaks a 50–50 tie.

## Pipeline

```
Polls ──────► poll average ─┐
Results, money ─► fundamentals ─┴► model ─► calibration ─┐
Forecaster ratings ─► calibrated ratings ───────────────┼─► log-odds average ─► prediction
Kalshi, Polymarket ─► market price ─────────────────────┘
```

Each step below states what it does and the evidence for it. Every test is out of sample unless noted.

### 1. Data

| Source | Use |
|---|---|
| Wikipedia race pages | 603 polls from 2026, nominees, 14 forecasters' ratings |
| [FiveThirtyEight polls](https://github.com/fivethirtyeight/data/tree/master/pollster-ratings), Wikipedia for 2024 | 5,915 Senate polls and 333 governor races, 1998–2024 |
| MIT Election Lab ([Senate](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PEJ5QU), [President](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/42MVDX)) | Results and state lean |
| [FEC bulk data](https://www.fec.gov/data/browse-data/?tab=bulk-data) | Fundraising and incumbency |
| [FiveThirtyEight forecasts](https://github.com/fivethirtyeight/checking-our-work-data) | Benchmark |
| Polymarket and Kalshi public APIs | 2026 prices, and 2024 prices for testing |

### 2. Poll average

$$w \propto e^{-\text{age}/14}\;\sqrt{N/600}\;\cdot\tfrac12^{\text{sponsored}}\;\cdot(\text{polls by that pollster})^{-1/2},\quad \text{age}\le 60\text{ days}$$

Sponsored polls are also corrected by their measured lean: D-sponsored polls run 2.1 points toward Democrats and R-sponsored polls 2.2 toward Republicans, relative to the race median. Ranked-choice states use the final-round match-up. Undecided voters move the margin toward the state's lean, with a fitted coefficient of 0.167.

I set these values by hand, then tested them against results in competitive races. The RMSE changes little across the range, so no setting drives the forecast:

| Setting | Used | RMSE | Alternatives |
|---|---|---:|---|
| Decay (days) | 14 | 5.23 | 7: 5.21; 28: 5.35; 60: 5.44 |
| Window (days) | 60 | 5.23 | 21: 5.11; 90: 5.21 |
| Sponsored-poll weight | 0.5 | 5.23 | 1: 5.28; 0.25: 5.17 |
| Sponsor correction | on | 5.23 | off: 5.25 |
| Sample-size exponent | 0.5 | 5.23 | 0: 5.29; 1: 5.23 |
| Repeat-pollster exponent | 0.5 | 5.23 | 0: 5.24; 1: 5.19 |

A 21-day window scores best, but I kept 60 so that thinly polled races still have polls to use.

### 3. Fundamentals

A within-cycle regression on 452 races gives 0.51 per point of state lean, 8.4 for incumbency, and 5.4 per unit of log fundraising ratio. Lean is the presidential margin relative to the nation, weighted 75/25 across the last two elections, which had the best fit of the four weightings tested (r = 0.725). The fundraising effect reflects candidate strength as well as money, so it isn't a causal estimate.

The national environment is estimated from the race polls at +3.9. As a cross-check, a D+8 generic ballot implies +4.1.

### 4. Combining polls and fundamentals

Precision weighting, with variances fitted by maximum likelihood:

$$V_p = 3.15^2 + \frac{3.62^2}{n_\text{eff}} + (0.15\,p)^2,\qquad V_f = 19.3^2$$

Errors within a cycle share national (SD 2.43), state-lean (0.60) and regional (0.13) components. The data decide the weights: polls get 96–98% in well-polled races, and fundamentals drive only races with no polls. Combining beats polls alone on margin error (RMSE 5.82 vs 6.17), though win-probability accuracy is the same (Brier 0.038 for both).

### 5. Independent candidates

Independents are shifted by −4.85 points with a shared SD of 4.5, because comparable candidates ran behind their polls: Orman −12, McMullin −3, Osborn −4, Walker +1. This affects NE, MT, ID and SD. Four races is thin evidence, so the wide SD is deliberate.

### 6. Calibration

$$P = \text{logit}^{-1}\big(k\cdot\text{logit}\,\Phi(\text{margin}/\text{SD})\big),\qquad k = 1.05$$

The value of k is pooled across Senate and governor races, because they disagree: Senate alone gives 1.50 and governors alone 0.80. Independent races use k = 1, since no backtest supports adjusting them.

Two features were tested and dropped when they failed on 333 governor races:
- **Leads growing beyond the polls:** margins ran 1.10–1.15 times the poll lead in Senate races, but 0.97–1.05 in governor races.
- **Sharper Senate-only calibration:** it scored significantly worse on governors (log-loss difference +7.96, 90% CI +3.03 to +13.18).

| Poll lead | Senate | Governor | Pooled |
|---|---:|---:|---:|
| 0–2 | 21/36 | 20/33 | 59% |
| 2–4 | 24/32 | 24/40 | 67% |
| 4–6 | 26/27 | 23/28 | 89% |
| 6–8 | 20/20 | 31/31 | 100% |
| 8–12 | 48/48 | 41/45 | 96% |

### 7. Forecaster ratings

Fourteen forecasters' ratings are averaged on a −4 to +4 scale and mapped to probabilities with a logistic fit to 267 Senate races from 2010–24 (slope 1.64). They're included because they beat my model on the 273 shared races (Brier 0.040 vs 0.046). All 218 races with an average rating of Lean or stronger went to the favorite.

### 8. Market prices

The market input is the average of Kalshi's bid-ask midpoint and Polymarket's price. Markets are included because they can reflect information the polls miss, such as late news and early voting.

I tested this on 32 Senate races in 2024, using Polymarket prices from the morning of the election:

| Forecast | Brier | Log-loss |
|---|---:|---:|
| Polymarket | 0.0322 | 0.122 |
| Model | 0.0315 | 0.113 |
| Ratings | 0.0300 | 0.091 |
| Model + ratings | 0.0289 | 0.094 |
| **Model + ratings + Polymarket** | **0.0287** | 0.097 |

Polymarket alone was the weakest input, and adding it barely changed the score, so 2024 shows that including it does no harm but not that it helps. Its favorites at 90% or higher won 24 of 24. That supports treating high prices as strong evidence, and suggests they may understate certainty. Long-shot contracts are known to trade rich (Snowberg & Wolfers, 2010).

This test covers one cycle. It also uses the final model calibration, which was partly fitted on 2024.

### 9. Final probability

The prediction is the equal-weight average of model, ratings and market on the log-odds scale, rounded to a whole percent. Averages of 99.5 or more round to 100. Equal weights are used because one cycle of market data can't support fitted weights, and simple averages are hard to beat in forecast combination (Clemen, 1989). All three sources read the same polls, so the average doesn't count as stronger evidence than its parts.

### 10. Seats

A simulation of 400,000 elections with t-distributed errors (12 degrees of freedom by race, 8 national) gives the seat count, matched to the final race probabilities. The residuals fit about 15 degrees of freedom, so 12 is slightly conservative. The count starts from 34 Democratic seats not up this year, and independent winners aren't counted as Democrats.

## Predictions

Probabilities are for the predicted winner, in percent. Polls are the weighted D−R or I−R margin. Lean is the state's margin relative to the nation. A dash means no data.

| Race | Winner | Final | Model | Ratings | Kalshi | Polymarket | Polls | Lean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IA | Ashley Hinson (R) | 58 | 50.5 | 71.9 | 49.0 | 50.5 | +0.8 | −12 |
| KS | Adam Hamilton (D) | 64 | 74.1 | 55.8 | 62.5 | 61.5 | +3.4 | −16 |
| AK | Mary Sattler Peltola (D) | 69 | 62.2 | 71.9 | — | 72.5 | +2.0 | −12 |
| OH | Sherrod Brown (D) | 74 | 70.0 | 78.4 | 71.5 | 72.5 | +2.9 | −10 |
| TX | James Talarico (D) | 78 | 77.6 | 82.1 | 73.5 | 75.5 | +3.4 | −12 |
| ME | Troy Dale Jackson (D) | 82 | 86.0 | 85.3 | 72.5 | 73.5 | +4.3 | +7 |
| SC | Darline Graham (R) | 83 | 78.0 | 93.4 | 67.5 | 68.5 | −1.6 | −16 |
| MI | Abdul El-Sayed (D) | 87 | 88.5 | 92.2 | 78.3 | 78.5 | +4.9 | 0 |
| NE | Pete Ricketts (R) | 89 | 89.6 | 96.4 | 69.5 | 71.5 | −1.8 | −20 |
| LA | Julia Letlow (R) | 95 | 80.7 | 99.6 | — | 86.5 | −2.3 | −21 |
| MS | Cindy Hyde-Smith (R) | 95 | 90.2 | 99.5 | 84.7 | 85.5 | −5.1 | −21 |
| MN | Peggy Flanagan (D) | 96 | 94.6 | 98.4 | 92.2 | 93.5 | +6.5 | +5 |
| FL | Ashley Moody (R) | 97 | 96.4 | 99.3 | 85.5 | 87.5 | −6.0 | −11 |
| KY | Andy Barr (R) | 98 | 87.2 | 99.8 | 95.8 | 96.5 | — | −29 |
| NH | Chris Pappas (D) | 98 | 98.5 | 98.7 | 93.9 | 95.5 | +9.3 | +4 |
| GA | Jon Ossoff (D) | 99 | 99.1 | 99.4 | 97.0 | 97.5 | +9.9 | −2 |
| NC | Roy Cooper (D) | 99 | 99.1 | 99.2 | 97.2 | 97.7 | +10.4 | −3 |
| MT | Kurt Alme (R) | 99 | 99.1 | 99.7 | 91.1 | 91.2 | −7.1 | −19 |
| ID | Jim Risch (R) | 99 | 98.5 | 99.9 | 95.5 | 96.5 | −6.3 | −35 |
| SD | Mike Rounds (R) | 99 | 99.8 | 99.8 | 94.5 | 94.8 | −10.8 | −28 |
| AR | Tom Cotton (R) | 99 | 97.8 | 99.9 | 95.5 | 96.2 | −9.1 | −30 |
| TN | Bill Hagerty (R) | 99 | 96.9 | 99.9 | 98.2 | 98.5 | — | −28 |
| OK | Kevin Hern (R) | 99 | 94.4 | 99.9 | 98.1 | 98.2 | — | −34 |
| WV | Shelley Moore Capito (R) | 99 | 98.7 | 99.9 | 98.5 | 98.7 | — | −41 |
| WY | Harriet Hageman (R) | 99 | 97.0 | 99.9 | 98.6 | 99.0 | — | −45 |
| VA | Mark Warner (D) | 99 | 95.4 | 99.9 | 99.2 | 99.0 | — | +7 |
| OR | Jeff Merkley (D) | 99 | 97.1 | 99.9 | 98.6 | 99.0 | — | +15 |
| NJ | Cory Booker (D) | 99 | 96.1 | 99.9 | 98.1 | 99.0 | — | +8 |
| DE | Chris Coons (D) | 99 | 97.5 | 99.9 | 98.8 | 99.0 | — | +16 |
| AL | Barry Moore (R) | 100 | 100.0 | 99.9 | 98.3 | 98.9 | −15.4 | −29 |
| IL | Juliana Stratton (D) | 100 | 99.7 | 99.9 | 99.3 | 99.5 | +13.5 | +12 |
| NM | Ben Ray Luján (D) | 100 | 99.9 | 99.8 | 99.0 | 99.0 | +16.5 | +7 |
| CO | John Hickenlooper (D) | 100 | 99.9 | 99.9 | 99.4 | 99.2 | +17.4 | +12 |
| RI | John Francis Reed (D) | 100 | 100.0 | 99.9 | 99.4 | 99.0 | +29.9 | +16 |
| MA | Ed Markey (D) | 100 | 100.0 | 99.9 | 98.8 | 99.4 | +23.8 | +27 |

### Contested races

- **IA, Hinson 58:** A near coin flip. Polls are +0.8 for Turek, and undecided voters breaking toward the state's lean erase that lead. Model and markets sit at about 50, so the ratings (Sabato's final call is Leans R) decide the pick. This is the least secure pick here.
- **KS, Hamilton 64:** His lead has grown (−0.1, +1.4, +4.6 across time windows), with NYT/Siena at +5. Ratings are nearly even, the state runs 16 points redder than the nation, and Marshall is an incumbent.
- **AK, Peltola 69:** She led the August primary 49.5–41.4 and is up about 2 in final-round polls. Alaska polls missed against Democrats in all five cycles on record, by a mean of 4.4 points, which argues for caution.
- **OH, Brown 74:** He leads by 2.9 across 20 polls, though the lead has drifted down. All four sources fall between 70 and 78, the tightest agreement of any close race.
- **TX, Talarico 78:** He leads by 3.4 in 32 recent polls, the deepest polling in the country, and has raised $146M to Paxton's $22M. Texas polls missed against Democrats by 4 points in both 2020 and 2024.
- **ME, Jackson 82:** His lead has widened to 5.2 in the last 10 days. This may be too high: Collins beat her polls by about 13 in 2020, and challengers leading incumbents won only 18 of 26 pooled races. Polymarket is at 73.5.
- **SC, Graham 83:** Ratings favor her at 93, and narrow Republican leaders in red states won 94% of the time (n=31). This is the largest gap with the markets (about 68). NYT/Siena has it tied, and the race has tightened 3.6 points in two months.
- **MI, El-Sayed 87:** He leads by 4.9, and 5.6 in the last 10 days, including R-sponsored Rasmussen at +5. Leads of 4–6 points held 89% of the time.
- **NE, Ricketts 89:** This depends on the independent adjustment. Polls at face value give about 69; Polymarket has 71.5. The adjustment rests on four races, and Osborn lost by 6.7 in 2024, a much worse national environment for him.
- **LA, Letlow 95:** There are only three polls, all D-sponsored, with the latest tied. The sources span 81 (model) to 99.6 (ratings), so this is uncertain for a 95. Polymarket is at 86.5.
- **MS, Hyde-Smith 95:** All four polls are D-sponsored, and all show her ahead. Mississippi polls have missed against Democrats by a mean of 4.5 points.
- **MN, Flanagan 96:** She leads by 6.5, and Cook still rates it Lean D. A 5.3-point polling miss would cut the model to 61%, making this the weakest race in this group of favorites.
- **FL, Moody 97:** She leads by about 6 across 19 polls, stable over time. Every 6–8 point lead in the data held (51 of 51).

### Clear races

These follow from the state's partisanship, the candidates and the polls, and the Polymarket price agrees with each one. Traders see the same polls, so the price is agreement, not independent proof. Every race here has an average rating of Lean or stronger for the favorite, and all 218 such Senate races from 2010–24 went to the favorite.

| Race | Rationale | Polymarket |
|---|---|---:|
| KY | State 29 points redder than the nation; last two polls R+9 and R+11 | 96.5 |
| NH | Pappas +9 across 7 recent polls, against a former senator | 95.5 |
| GA | Ossoff +10, $116M raised to $11M; about a 2% chance of a margin close enough for a runoff | 97.5 |
| NC | Cooper +10 across 11 recent polls | 97.7 |
| MT | Bankhead (D) polls at 10–23%, splitting the anti-Alme vote; the unstable polls explain the lower price | 91.2 |
| ID | State 35 points redder than the nation; polls vary widely but Risch leads on average | 96.5 |
| SD | Rounds +11 in a state 28 points redder than the nation | 94.8 |
| AR | Cotton +9; state 30 points redder than the nation; every forecaster rates it Safe | 96.2 |
| TN | State 28 points redder than the nation; opponent raised $35K to Hagerty's $8.4M | 98.5 |
| OK | State 34 points redder than the nation; opponent raised $116K to Hern's $10.9M | 98.2 |
| WV | State 41 points redder than the nation; incumbent Capito | 98.7 |
| WY | State 45 points redder than the nation, the most lopsided race here | 99.0 |
| AL | Moore +15 in a state 29 points redder than the nation | 98.9 |
| VA | Incumbent Warner; every forecaster rates it Safe | 99.0 |
| OR | State 15 points bluer than the nation; incumbent Merkley | 99.0 |
| NJ | Incumbent Booker; opponent raised $68K to his $21M | 99.0 |
| DE | State 16 points bluer than the nation; incumbent Coons | 99.0 |
| IL | Stratton +13.5 in an open blue seat | 99.5 |
| NM | Luján +16.5 | 99.0 |
| CO | Hickenlooper +17 | 99.2 |
| RI | Reed +30 | 99.0 |
| MA | Markey +24 | 99.4 |

## Validation

Each Senate cycle from 1998 to 2024 was forecast from a model fitted on the other cycles.

- **Accuracy:** 93.8% of winners called across 452 races, with a Brier score of 0.040 (0.25 is a coin flip).
- **Benchmark:** tied FiveThirtyEight's final forecasts on the same 95 races (0.051 each).
- **Most recent cycle:** Brier 0.032 in 2024, missing only Pennsylvania.
- **Calibration:** in the 50–90% band it predicted 71.7% and 73.1% happened (n=104). In the 90–99% band it predicted 95.8% and 100% happened (n=80), though governor races argue against sharpening that.

## Limitations

- **A shared polling miss is the main risk.** If polls overstate Democrats by 5.3 points, as they did in 2020, the model alone drops Talarico to about 28%, El-Sayed to 44% and Brown to 20%. Actual cycle-to-cycle bias (SD 2.83) is above the fitted 2.43, so seat outcomes may be more correlated than simulated and the 80% majority figure may be a little high.
- **Calibration is a compromise.** If Senate races really are more predictable than governor races, these numbers are too cautious.
- **Some numbers rest on thin evidence.** That applies most to IA (pick decided by ratings), LA (three polls), NE (four analogs), SC and ME.
- **Choices followed the data.** Settings were hand-set first and tested afterward, and I made about 15 historical comparisons while developing this. Only the governor races and the 2024 market prices were not used in building the core model.
- **Some inputs are approximate.**
  - 2026 polls come from Wikipedia tables.
  - D+8 comes from October trackers; I found no final generic-ballot average.
  - House margins in the environment cross-check were entered from memory.
  - The FEC coverage date didn't parse.
  - Governor incumbency was inferred from past winners.
- **Market gaps.** Louisiana has no Kalshi price in the feed. Alaska's Kalshi market (71–72) wasn't merged because it is listed by party; this changes the result by under 0.1 point.
- **Not modeled:** early-vote data, pollster quality scores, and news after the data was collected.

## Reproduce

Run from the repo root with `numpy`, `scipy`, `pandas`, `statsmodels`, `lxml`, `beautifulsoup4` and `html5lib`, after downloading the sources into `data/raw/`:

```
01 → 02 → 02c → 03 → 04 → 05 → 06 → 07 → 08 → 12 → 14 → 15 → 16 → 17 → 18 → 19 → 20 → 21
```

Delete `output/final_comparison_senate_only.csv` before a full rerun. The downloads aren't scripted, some Kalshi prices are hard-coded in `08_forecast.py`, and market prices will have moved. Scripts 08, 12, 16, 17 and 20 read an earlier predictions file for the race list and candidate labels, so a rerun needs `output/predictions_original_backup.csv` copied to `predictions.csv` first. No final probability depends on the values in that file.

| Output | Contents |
|---|---|
| `predictions.csv` | Final predictions (identical to `output/predictions_pipeline.csv`) |
| `output/report.html` | Full report with charts |
| `output/justification_by_race.csv` | Inputs, scenarios and basis for each race |
| `output/final_comparison.csv` | Every input and scenario for each race |
| `output/market_test_2024.csv` | Market backtest |
| `output/reference_classes.csv` | Win rates in comparable past races |
| `output/seat_distribution.csv` | Seat simulation |
