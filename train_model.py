"""Train the credit-risk model (component A) from a Lending Club CSV.

uv run python train_model.py --csv data/lending_club.csv --max-rows 200000

Writes models/credit_model.joblib (small: it is committed and loaded online)
and models/metrics.json (for REPORT.md). Target: 1 = Charged Off / Default.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from application import CATEGORICAL, NUMERIC

NUM, CAT = list(NUMERIC), list(CATEGORICAL)
BAD, GOOD = {"Charged Off", "Default"}, {"Fully Paid"}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


def load(csv: str, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(csv, usecols=lambda c: c in NUM + CAT + ["loan_status"], low_memory=False)
    missing = set(NUM + CAT + ["loan_status"]) - set(df.columns)
    if missing:
        raise SystemExit(f"Columns missing in the CSV: {sorted(missing)}")
    df = df[df["loan_status"].isin(BAD | GOOD)].copy()
    df["target"] = df["loan_status"].isin(BAD).astype(int)
    df["term"], df["int_rate"] = _num(df["term"]), _num(df["int_rate"])
    emp = df["emp_length"].astype(str)
    df["emp_length"] = _num(emp).mask(emp.str.contains("<"), 0)
    for c in ("loan_amnt", "annual_inc", "dti"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["grade"] = df["grade"].astype(str).str.upper()
    df["home_ownership"] = df["home_ownership"].astype(str).str.upper()
    df["purpose"] = df["purpose"].astype(str).str.lower()
    df = df.dropna(subset=NUM)
    for c in CAT:
        df = df[df[c].isin(CATEGORICAL[c])]
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    return df


def make(estimator, scale: bool) -> Pipeline:
    pre = ColumnTransformer(
        [("num", StandardScaler() if scale else "passthrough", NUM),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        sparse_threshold=0,  # dense output: HistGradientBoosting needs it
    )
    return Pipeline([("pre", pre), ("model", estimator)])


CANDIDATES = {
    "logistic_regression": lambda: make(LogisticRegression(max_iter=1000), True),
    "random_forest": lambda: make(
        RandomForestClassifier(n_estimators=100, min_samples_leaf=200, n_jobs=-1, random_state=42), False),
    "hist_gradient_boosting": lambda: make(
        HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=42), False),
}


def scores(model, X, y) -> dict:
    p = model.predict_proba(X)[:, 1]
    return {"auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-rows", type=int, default=200_000)
    args = ap.parse_args()

    df = load(args.csv, args.max_rows)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[NUM + CAT], df["target"], test_size=0.25, stratify=df["target"], random_state=42)
    print(f"{len(df):,} loans, default rate {df['target'].mean():.1%}")

    # Rows the model never saw: make_eval_files.py builds the evaluation files from them.
    Path("data").mkdir(exist_ok=True)
    X_te.assign(target=y_te).to_csv("data/test_split.csv", index=False)

    results = {}
    for name, build in CANDIDATES.items():
        results[name] = scores(build().fit(X_tr, y_tr), X_te, y_te)
        print(f"{name:24s} {results[name]}")

    best = max(results, key=lambda n: results[n]["pr_auc"])
    final = CalibratedClassifierCV(CANDIDATES[best](), method="sigmoid", cv=3, ensemble=False)
    final.fit(X_tr, y_tr)
    results[best + "_calibrated"] = scores(final, X_te, y_te)
    print(f"best: {best}; calibrated: {results[best + '_calibrated']}")

    defaults = {c: float(X_tr[c].median()) for c in NUM} | {c: X_tr[c].mode()[0] for c in CAT}
    bundle = {"model": final, "version": f"credit-{best}-v1", "defaults": defaults,
              "features": NUM + CAT}
    Path("models").mkdir(exist_ok=True)
    joblib.dump(bundle, "models/credit_model.joblib", compress=3)
    Path("models/metrics.json").write_text(json.dumps(
        {"chosen": best, "n_loans": len(df), "default_rate": round(float(df["target"].mean()), 4),
         "test_metrics": results}, indent=2))
    print("size (MB):", round(Path("models/credit_model.joblib").stat().st_size / 1e6, 2))


if __name__ == "__main__":
    main()