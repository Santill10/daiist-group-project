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
