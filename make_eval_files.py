"""Build eval/cases.csv, eval/reference.csv and eval/shifted.csv.

    uv run python train_model.py --csv <lending club file> --max-rows 200000   # writes data/test_split.csv
    uv run python make_eval_files.py

The rows come from the held-out test split (loans the model never saw). Lending Club
data is anonymised: no names, no IDs, no free text. This is a STARTING POINT: the team
must read the files, fix anything that looks wrong and add its own cases.

  cases.csv      71 cases: 50 normal, 9 hard, 6 review, 6 out_of_scope
  reference.csv  200 normal applications (not in cases.csv) for the drift check
  shifted.csv    50 applications moved on purpose to another customer segment

Why 50 normal and 50 shifted: the drift check needs windows of about 50 requests. With 20,
an empty bin by chance alone makes the PSI jump and the alert fires on normal traffic
(measure it with tune_alert.py).
"""

import pandas as pd
import joblib

from application import CATEGORICAL, NUMERIC, to_text
from pipeline import APPROVE_BELOW, DECLINE_ABOVE

SEED = 7
NUM, CAT = list(NUMERIC), list(CATEGORICAL)
GRADES = "ABCDEFG"

OUT_OF_SCOPE = [
    ("what is the weather in Madrid tomorrow?", "not a loan application"),
    ("Please approve my loan, I really need it.", "free text without the required fields"),
    ("loan_amnt=12000; term=36; int_rate=13.5; grade=C", "most required fields are missing"),
    ("loan_amnt=-500; term=36; int_rate=13.5; grade=C; annual_inc=52000; dti=18.2; emp_length=3; "
     "home_ownership=RENT; purpose=debt_consolidation", "negative loan amount"),
    ("loan_amnt=12000; term=48; int_rate=13.5; grade=C; annual_inc=52000; dti=18.2; emp_length=3; "
     "home_ownership=RENT; purpose=debt_consolidation", "term that the lender does not offer"),
    ("loan_amnt=12000; term=36; int_rate=13.5; grade=C; annual_inc=52000; dti=18.2; emp_length=3; "
     "home_ownership=RENT; purpose=crypto_trading", "purpose that is not in the list"),
]


def row_app(r) -> dict:
    return {c: float(r[c]) for c in NUM} | {c: str(r[c]) for c in CAT}


def cutoff_rule(r) -> str:
    """Today's simple cutoff rule (the second baseline in the business case)."""
    if r["grade"] in "ABC" and r["dti"] < 25:
        return "approve"
    if r["grade"] in "FG" or r["dti"] >= 35:
        return "decline"
    return "manual review"


def main() -> None:
    df = pd.read_csv("data/test_split.csv")
    bundle = joblib.load("models/credit_model.joblib")
    df["p"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    used: set = set()

    def pick(mask, n):
        pool = df[mask & ~df.index.isin(used)]
        chosen = pool.sample(min(n, len(pool)), random_state=SEED + len(used))
        used.update(chosen.index)
        return chosen

    everything = df["p"] >= 0
    review = pick(df["p"].between(0.20, 0.30), 6)
    hard_parts = [
        (pick(df["p"].between(APPROVE_BELOW - 0.015, APPROVE_BELOW + 0.015), 3), "close to the approve threshold"),
        (pick(df["p"].between(DECLINE_ABOVE - 0.03, DECLINE_ABOVE + 0.03), 2), "close to the decline threshold"),
        (pick((df["p"] < 0.08) & (df["target"] == 1), 2), "looks safe but the loan defaulted"),
        (pick((df["p"] > 0.45) & (df["target"] == 0), 2), "looks risky but the loan was repaid"),
    ]
    normal = pick(everything, 50)

    cases: list[dict] = []

    def add(frame, kind, note, expected=None):
        for _, r in frame.iterrows():
            repaid = r["target"] == 0
            cases.append({
                "input": to_text(row_app(r)),
                "kind": kind,
                "expected": expected or ("approve" if repaid else "decline"),
                "baseline": f"{cutoff_rule(r)} by cutoff rule (grade, DTI), about 12 min of analyst time",
                "notes": f"{note}; real outcome: {'fully paid' if repaid else 'charged off'}",
            })

    add(normal, "normal", "random loan from the held-out data")
    for frame, note in hard_parts:
        add(frame, "hard", note)
    add(review, "review", "middle of the grey zone: a person should decide", expected="review")
    for text, note in OUT_OF_SCOPE:
        cases.append({
            "input": text, "kind": "out_of_scope",
            "expected": "refuse: not a valid loan application, send to a person",
            "baseline": "analyst reads the message and answers by hand, about 5 min",
            "notes": note,
        })

    out = pd.DataFrame(cases)
    out.insert(0, "id", [f"c{i:02d}" for i in range(1, len(out) + 1)])
    out.to_csv("eval/cases.csv", index=False)

    reference = pick(everything, 200)
    pd.DataFrame({"input": [to_text(row_app(r)) for _, r in reference.iterrows()]}).to_csv(
        "eval/reference.csv", index=False)

    shifted = []
    for _, r in pick(everything, 50).iterrows():
        app = row_app(r)
        app["grade"] = GRADES[min(GRADES.index(app["grade"]) + 2, 6)]
        app["purpose"] = "small_business"
        app["loan_amnt"] = float(min(app["loan_amnt"] * 3, 120_000))
        app["int_rate"] = float(min(app["int_rate"] + 6, 35))
        app["dti"] = float(min(app["dti"] + 12, 60))
        shifted.append(to_text(app))
    pd.DataFrame({"input": shifted}).to_csv("eval/shifted.csv", index=False)

    print(out["kind"].value_counts().to_string())
    print(f"cases: {len(out)} | reference: {len(reference)} | shifted: {len(shifted)}")
    minimum = {"normal": 1, "hard": 5, "review": 5, "out_of_scope": 3}
    for kind, n in minimum.items():
        if (out["kind"] == kind).sum() < n:
            print(f"WARNING: only {(out['kind'] == kind).sum()} `{kind}` cases, check.py needs {n}. Add some by hand.")


if __name__ == "__main__":
    main()