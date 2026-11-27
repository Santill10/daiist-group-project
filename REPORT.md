# Group Assignment Report

- **Team** (Blackboard group name):
- **Section**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]
- **App URL** (on Render):
- **Final commit** (tag `final`):

*At most about 3,000 words, tables excluded. Members and who did what go in CONTRIBUTIONS.md.*

## Business case

Product. Loan Screening Copilot: a credit-risk model scores a consumer-loan application, and Gemini turns the score and its main drivers into a short credit memo with a recommendation (approve, review or decline).

User and decision. A credit analyst at a consumer lender or fintech. For each incoming application they decide: approve quickly, send to full review, or decline.

Baseline today. The analyst reads the application and writes the memo by hand, about 12 minutes per case. As a second baseline we use a simple cutoff rule: approve if grade is A to C and DTI is below 25; decline if grade is F or G, or DTI is 35 or more; otherwise manual review. Both are compared with the product on the same evaluation cases.

Success metric. The share of evaluation cases where the final decision matches the correct one (the real outcome of the loan, or "review" for grey-zone cases), against the cutoff rule; supported by AUC and PR-AUC of the model, the memo's faithfulness, time per case and cost per request. Results are in the Evaluation section.

Value in euros (assumptions, not measurements). Volume 200 applications a day, loaded analyst cost 35 euros an hour.

Baseline: 200 x 12 min = 40 h = 1,400 euros a day.
With the product, a case sent to a person takes about 6 minutes (memo and drivers are ready) and an automatic case gets a 1-minute spot check. The LLM costs about 0.0004 euros per request, 0.08 euros a day.
Share sent to a person	Analyst hours a day	    Cost a day	Saving a day
20%	                        6.7 h	            233 euros	1,167 euros
35%	                        9.2 h	            321 euros	1,079 euros
50%	                        11.7 h	            408 euros	992 euros

The row that applies is the share our review rule sends to a person on the evaluation set: [FILL]. The saving depends on the assumptions above (volume, wage, minutes per case), so it is an order of magnitude, not a forecast.

## Architecture

*The two components and how one feeds the other: a diagram plus a short walk-through of one request from input to output. Which models or APIs, and why those. What you reused from assignments 1–3, and whose it was.*

## Data

*Where each component's data comes from, with links, licences and sizes. Confirm that there is no personal data in your inputs or logs.*

## Evaluation

### Components

*Each component on its own: the metric, the result, the test data and its size.*

#### Component A: credit-risk model

*Written by P2 (Juan). Every number comes from `uv run python train_model.py --csv <lending club .gz> --max-rows 200000` and is saved in `models/metrics.json`; the plot is `models/calibration.png`.*

**Data and split.** 200,000 loans sampled from the 1,345,350 finished Lending Club loans (Fully Paid vs. Charged Off / Default), default rate 19.5%. Only the 9 application fields and the outcome are read. Stratified split: test 50,000 (the same 25% as before, so the `eval/` files built from `data/test_split.csv` are still held out: all 265 of their loans are in it), validation 30,000, train 120,000. Models, calibration and the threshold were chosen on validation; the test set is only reported.

**Models compared (test set).** Metrics that fit a probability used for a cost decision: AUC (ranking), PR-AUC (ranking on the minority class, defaults), Brier score (accuracy of the probability itself).

| Model | AUC | PR-AUC | Brier | Size |
|---|---|---|---|---|
| Logistic regression (C=1; C=0.1 the same) | 0.699 | 0.350 | 0.144 | 2 KB |
| Logistic regression, balanced class weights | 0.699 | 0.350 | 0.220 | 2 KB |
| Random forest (leaf 200 / depth 10 leaf 100) | 0.700 / 0.701 | 0.355 / 0.356 | 0.144 | 1.1 / 1.7 MB |
| Random forest, balanced class weights | 0.700 | 0.355 | 0.218 | 1.2 MB |
| Gradient boosting, depth 4, lr 0.08 (previous model) | 0.703 | 0.358 | 0.144 | 56 KB |
| Gradient boosting, depth 3, lr 0.05, 400 trees | 0.703 | 0.358 | 0.144 | 74 KB |
| Gradient boosting, depth 4, lr 0.03, 600 trees, L2 | 0.702 | 0.357 | 0.144 | 105 KB |
| Gradient boosting, depth 4, balanced class weights | 0.702 | 0.357 | 0.218 | 55 KB |
| **Gradient boosting, depth 6, lr 0.05, 300 trees + sigmoid calibration (chosen)** | **0.703** | **0.357** | **0.144** | **137 KB** |

Chosen: the highest validation PR-AUC under 5 MB (validation AUC 0.714, PR-AUC 0.368). Read honestly, all gradient-boosting settings are within 0.001 AUC of each other on test, below the noise of a 50,000-loan sample (about ±0.003), and only 0.004 above logistic regression. With 9 application fields, the interest rate and grade (already the lender's own risk price) carry most of the signal. Tuning depth or learning rate does not help; the next gain would need more information about the borrower (credit history), not a bigger model. Balanced class weights do not improve ranking and spoil the probabilities (Brier 0.22: where 19% of loans default they predict about 48%). That matters, because the thresholds below read the probability as a real default rate.

**Calibration.** Sigmoid (Platt) calibration on 3 folds of the training data, chosen over isotonic on validation Brier (0.14221 vs 0.14225). On the test set, in each of the 10 deciles of predicted risk the observed default rate is within 1.5 points of the prediction (lowest decile 4.3% predicted, 4.2% observed; highest 44.7% vs 43.7%). See `models/calibration.png`.

**Cost of errors.** For an average application ($14,612 at 13.2%, a mix of 36 and 60 months):

- *Approving a loan that defaults* loses LGD × amount. Central LGD 55%: Lending Club loans usually charge off after a year or more of payments, so part of the principal and interest has come back, and recoveries after charge-off are small. Cost: about **$8,000**.
- *Declining a good customer* loses the margin we would have earned: amount × (interest rate − 4% funding and operating cost) × years of balance outstanding (half the term for an amortising loan, × 0.8 because many good borrowers prepay). Cost: about **$2,200**.

Ratio 3.7, so approving pays off below a probability of default of 2,200 / (2,200 + 8,000) = **21.5%**. Because the figures are assumptions, we also ran a low-ratio case (LGD 45%, funding 3%, little prepayment: ratio 2.4, break-even 29%) and a high-ratio case (LGD 65%, funding 5%, more prepayment: ratio 5.5, break-even 15%). On validation the cost-minimising threshold was 30%, 21% and 16%, within one point of the theory in every case, which confirms that the probabilities are calibrated. On test, the central threshold cuts the cost of errors per application from $1,565 (approve everyone) to $1,173 (−25%).

**Thresholds this implies (for P1 and P5).** Approve below **15%** (the decision is "approve" even under the pessimistic costs), decline at **29%** or above (it is "decline" even under the optimistic costs), and send to a person in between, where the right decision depends on which cost figure is true. On the 50,000 test loans:

| Policy | Approved | Review | Declined | Default rate: approved / review / declined |
|---|---|---|---|---|
| 15% / 29% (recommended) | 43% | 37% | 20% | 9.4% / 21.3% / 37.4% |
| 12% / 40% (current placeholder) | 33% | 60% | 7% | 7.9% / 22.6% / 45.5% |

The current placeholder sends 60% of applications to a person (36 of the 50 normal cases in `eval/cases.csv`), and declines only above 40% although the break-even is about 21%. With 15% / 29%, 37% go to review, close to the 35% row of the business case. Limit: the break-even varies a lot by grade (central costs: 7% for grade A, whose thin margin barely covers a default, up to 45% for G), so a later version could use one threshold per grade.

**Drivers.** For each feature, the applicant's value is swapped for those of 100 real training applicants (saved in the model file) and the probabilities are averaged; the driver's effect is the applicant's probability minus that average. This replaced a single median value, which for categories such as grade or purpose compared every applicant with one arbitrary "typical" value. It takes about 20–30 ms per request.

### End to end

*How you wrote `eval/cases.csv` (counts by kind, two examples). Results against the baseline. Where the output is free text: the judge's rubric and your agreement with it on your own labels.*

### Cost and latency

*Per request: tokens, euros and seconds, for each component and in total. What it would cost per month at the volume in your business case.*

## Operations

*What you log. The quality signal and the drift check you monitor, the reference set, and the alert threshold with its justification. The drift demo: what you shifted, and the result with the alert firing (screenshot or numbers). The rate of false alarms on normal inputs.*

*Written by P6 (Francisco). Every number comes from `uv run python tune_alert.py` or `uv run python main.py drift-demo`, run on 25 Nov with the model in `main` at that date. Neither command calls Gemini.*

**What we log.** Every request is one record (`logbook.py`): the required fields (input summary without personal data, features, outputs of both components, versions, review flag, latency, tokens, cost) plus four fields for monitoring: `p_default`, `band`, `recommendation` and `review_reasons`. Thumbs up/down are a second table. On Render the logs go to Postgres, so they survive restarts.

**What the Monitoring tab shows.** Over time: share of cases sent to review, share of 👎 among rated cases, mean probability of default and mean model confidence. In the summary: requests, share sent to review, each review reason (grey zone, memo contradicts score, Gemini unavailable, invalid input) as a share of requests, median and 95th-percentile latency, cost per request and total cost, and the drift table.

**Drift check.** For 8 numbers of each request (loan amount, interest rate, DTI, income, term, grade, and the model's probability of default and confidence) we compare the last 100 requests with a reference set of 200 normal held-out Lending Club applications (`eval/reference.csv` -> `monitoring/reference_features.csv`) using the population stability index (PSI, 5 bins). Because PSI is high by chance with small samples, a feature alerts only if its PSI is above 0.25 **and** above 5 times the PSI expected by chance for those sample sizes, and only once there are at least 40 requests.

**Drift demo.** `eval/shifted.csv` holds 50 applications from another customer segment: larger loans, worse grades, more small-business loans. Result of `main.py drift-demo`:

| Feature | PSI, 50 normal cases | PSI, 50 shifted cases | Threshold |
|---|---|---|---|
| loan_amnt | 0.039 | 3.082 | 0.5 |
| int_rate | 0.216 | 2.684 | 0.5 |
| dti | 0.104 | 2.298 | 0.5 |
| annual_inc | 0.134 | 0.063 | 0.5 |
| term | 0.001 | 0.003 | 0.5 |
| grade_num | 0.150 | 4.600 | 0.5 |
| p_default | 0.231 | 4.287 | 0.5 |
| confidence | 0.236 | 4.302 | 0.5 |

Normal inputs: quiet. Shifted inputs: alert on 6 of 8 features.

**False alarms and detection (`tune_alert.py`, 200 random windows each).** The first placeholder rule (window 20, 2 x noise) fired on about half of the windows of normal traffic.

| NOISE_MULTIPLE | Window | False alarms, normal traffic | Detection, shifted traffic |
|---|---|---|---|
| 2 | 20 | 54% | 100% |
| 3 | 40 | 16% | 100% |
| 4 | 40 | 5% | 100% |
| **5** | **40** | **2%** | **100%** |
| 5 | 50 | 2% | 100% |
| 6 | 40 | 0.5% | 100% |

We keep **NOISE_MULTIPLE = 5 and MIN_WINDOW = 40**: it catches the shifted segment every time and false-alarms on about 1 window in 50. Going to 6 cuts false alarms to 0.5% but makes partial shifts harder to see. How big a shift must be (same rule, window 40): if 10% of the traffic comes from the new segment it is caught 2% of the time, 30% -> 17%, 50% -> 76%, 100% -> 100%. So the alert is meant for a real change of customer mix (half the traffic or more), not for a few unusual applications; those still reach a person through the review rule.

**Thumbs-down alert.** It fires when more than 30% of at least 40 rated requests are 👎 (`THUMBS_DOWN_ALERT = 0.3`, was 0.4). With a binomial model: if analysts normally give 👎 to 15% of memos (assumption: we have no real feedback yet), it fires by chance with probability 0.4%; at 10% it practically never does. If quality really drops to 50% 👎 it fires 99% of the time, and at 35% about 69%. The old 40% threshold caught a drop to 35% only 20% of the time.

**Limits.** (1) `data/test_split.csv` was not available to P6, so normal traffic was simulated with the 250 real held-out loans we have (reference + normal cases), split at random into a reference of 200 and windows of at most 50. Windows of 60 and 100 (the size the app actually checks) must be re-measured with `test_split.csv`. (2) The 15% usual 👎 rate is an assumption until there is real feedback. (3) If the model is retrained, `p_default` and `confidence` change: rerun `main.py reference` and `tune_alert.py`.

## Failures

*Failed end-to-end cases grouped by cause (which component, or the hand-off between them), with counts, and what you would fix first.*

## Decision

Which cases go to a person. A case goes to a person if any of these holds:

the default probability is in the grey zone, between 12% and 40% (thresholds confirmed on the evaluation set: [FILL final values]);
the memo's recommendation contradicts the model's band;
Gemini is unavailable (quota or overload), so there is no memo;
the input is not a valid application (missing fields or values out of range); it is refused without calling the model.

Cases a day for a person = share flagged x volume, for example [FILL]% x 200 = [FILL] cases a day.

How the thresholds are justified. Approving a loan that later defaults costs more than declining a good customer. If a bad approval costs 5 times a lost good customer (assumption), approving pays off only below a default probability of 1 / (1 + 5), about 17%. We start from 12% for approval and 40% for decline, and adjust them on the evaluation set: [FILL why].

Go / no-go. Draft: conditional go, as an assistant for analysts and not as an automatic decision-maker. Conditions and the evidence for each:

The product matches the correct decision on at least as many evaluation cases as the cutoff rule: [FILL result].
The memo is faithful to the model's numbers according to the judge, which agrees with our own labels: [FILL agreement].
Cost per request below 0.01 euros and less time per case than the 12-minute baseline: [FILL].
The drift alert stays quiet on normal traffic and fires on shifted traffic: shown in the Operations section.

Limits: the data is Lending Club (US loans, 2007 to 2018, anonymised), so this is a prototype and would need retraining on the lender's own data. Credit scoring is a high-risk use under the EU AI Act, so a person must stay in the loop and be able to override the system.

Final decision: [FILL go, no-go, or go with these conditions, after P5's results].

## Bonus: reinforcement learning

*Delete this section if you did not do it. The loop (feedback → reward → update → behaviour), the reward and what it could reward by accident, the exploration strategy and its cost, the evidence from real logs (and which part was simulated), the risks.*

## How to run it

*Anything beyond SETUP.md that the professor needs to run it: which keys, any download step, how long the first run takes.*

## AI use

*What you used and how, or "none".*
