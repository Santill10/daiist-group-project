# Contributions

*Fill this in as you go, not on the last day. It is checked against the commits and against the peer assessment. A first name and a GitHub username are enough: this repo is public.*

## Members

# Contributions

*Fill this in as you go, not on the last day. It is checked against the commits and against the peer assessment. A first name and a GitHub username are enough: this repo is public.*

## Members

| First name | GitHub username | Main areas |
|---|---|---|
| Santiago | Santill10 | Project lead (P1): business case, REPORT.md, Render deployment, merges, final submission|
| Juan | jmunozqieu2022-ai | Component A: credit-risk model, application parser |
| Andrés | andresramirezzz | Component B: Gemini memo prompt, LLM judge, human labels |
| Ángel | Angelbenito1 | Integration and deploy: app, Render, check.py |
| Yago | ymoreno2022 | Evaluation: eval cases, evaluate.py, cost |
| Francisco | frocaieu | Logging and monitoring: drift alert, reference set |

## Log

*One line per piece of work: the date, who did it (more than one name is fine), and what. Code done in pair programming counts for both, even if only one of you committed it.*

- 2026-11-24, Santiago: chose the product (Loan Screening Copilot) and the business case, and split the work across the six roles in a shared team prompt.
- 2026-11-24, Santiago: set up the fork, the local environment and the first working chain of the two components in `pipeline.py` (credit model and Gemini memo), with a refusal path for invalid inputs.
- 2026-11-24, Santiago: built the evaluation files (`eval/cases.csv`, `reference.csv`, `shifted.csv`) and tuned the drift alert on real data (`tune_alert.py`, `monitor.py`).
- 2026-11-24, Santiago: deployed the app on Render with its Postgres database and ran `check.py` against the live URL.
- 2026-11-25, Francisco: monitoring (P6). Added the monitoring fields to `logbook.py` (p_default, band, recommendation, review reasons), the PD, p95-latency and review-reason signals to `monitor.py`, fixed `tune_alert.py` (MIN_WINDOW now follows the window, fallback without `test_split.csv`, partial-shift and thumbs-down tables), set `THUMBS_DOWN_ALERT` to 0.3 and wrote the Operations section of `REPORT.md`.

## Log

*One line per piece of work: the date, who did it (more than one name is fine), and what. Code done in pair programming counts for both, even if only one of you committed it.*

| Date | Who | What |
|---|---|---|
| | | |
