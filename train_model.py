"""Train the credit-risk model (component A) from a Lending Club CSV.

uv run python train_model.py --csv "<path to the .gz>" --max-rows 200000

Writes:
  models/credit_model.joblib   the calibrated model (committed, loaded online, kept under 5 MB)
  models/metrics.json          every candidate's metrics, calibration curves, cost analysis
  models/calibration.png       calibration curves (only if matplotlib is installed: `uv sync` dev group)
  data/test_split.csv          held-out loans (git-ignored), used by make_eval_files.py

Target: 1 = Charged Off / Default, 0 = Fully Paid. Only the 9 inputs and loan_status are
read: no free text or identifying columns (desc, emp_title, zip_code, member_id...).

Split (stratified): test 25% exactly as before, so data/test_split.csv and the eval files built
from it stay the same. The other 75% is split again into train (80%) and validation (20%).
Models are chosen, calibrated and the threshold is picked on validation; test is only reported.
"""

import argparse
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from application import CATEGORICAL, NUMERIC

NUM, CAT = list(NUMERIC), list(CATEGORICAL)
BAD, GOOD = {"Charged Off", "Default"}, {"Fully Paid"}
MAX_MB = 5.0           # Render free plan: the model must load fast
BACKGROUND_ROWS = 100  # reference applicants for the driver explanation (pipeline.component_a)


# ---------- data ----------

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


# ---------- candidates ----------

def make(estimator, scale: bool) -> Pipeline:
    pre = ColumnTransformer(
        [("num", StandardScaler() if scale else "passthrough", NUM),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        sparse_threshold=0,  # dense output: HistGradientBoosting needs it
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def _lr(**kw):
    return lambda: make(LogisticRegression(max_iter=2000, **kw), True)


def _rf(**kw):
    return lambda: make(RandomForestClassifier(n_jobs=-1, random_state=42, **kw), False)


def _hgb(**kw):
    return lambda: make(HistGradientBoostingClassifier(random_state=42, **kw), False)


# Grid kept small on purpose: each idea (regularisation, depth, learning rate, class weights)
# gets one or two settings. "balanced" class weights give probabilities that are too high:
# calibration fixes that, so they compete on ranking (AUC, PR-AUC) only.
CANDIDATES = {
    "logistic_regression":            _lr(),
    "logistic_regression_C0.1":       _lr(C=0.1),
    "logistic_regression_balanced":   _lr(class_weight="balanced"),
    "random_forest":                  _rf(n_estimators=100, min_samples_leaf=200),
    "random_forest_depth10":          _rf(n_estimators=150, max_depth=10, min_samples_leaf=100),
    "random_forest_balanced":         _rf(n_estimators=100, min_samples_leaf=200, class_weight="balanced_subsample"),
    "hist_gradient_boosting":         _hgb(max_depth=4, learning_rate=0.08, max_iter=200),
    "hgb_depth3_lr0.05_400":          _hgb(max_depth=3, learning_rate=0.05, max_iter=400),
    "hgb_depth6_lr0.05_300":          _hgb(max_depth=6, learning_rate=0.05, max_iter=300, min_samples_leaf=100),
    "hgb_depth4_lr0.03_600_l2":       _hgb(max_depth=4, learning_rate=0.03, max_iter=600, l2_regularization=1.0),
    "hgb_depth4_lr0.08_balanced":     _hgb(max_depth=4, learning_rate=0.08, max_iter=200, class_weight="balanced"),
}


def size_mb(obj) -> float:
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return round(buf.tell() / 1e6, 3)


def scores(p: np.ndarray, y) -> dict:
    return {"auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "log_loss": round(log_loss(y, p), 4)}


def calibration_curve(p: np.ndarray, y, bins: int = 10) -> list[dict]:
    """Quantile bins: mean predicted probability vs. observed default rate."""
    d = pd.DataFrame({"p": p, "y": np.asarray(y)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    g = d.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return [{"predicted": round(float(r.predicted), 4), "observed": round(float(r.observed), 4), "n": int(r.n)}
            for r in g.itertuples()]


# ---------- cost of errors ----------

# Costs in dollars (the data is US loans), averaged over the loans being scored.
#   Approving a loan that defaults loses  LGD x loan amount  (LGD: share of the amount lost
#     after the payments made before default and recoveries).
#   Declining a good customer loses the margin we would have earned:
#     loan x (int_rate - funding and operating cost) x years of balance outstanding,
#     where an amortising loan has on average half its principal outstanding over its term
#     and PREPAY shortens that because many good borrowers repay early.
# Three scenarios: a decision is automated only where it is the same under all three.
COST_SCENARIOS = {
    #              LGD   funding+ops  prepay
    "low_ratio":  (0.45, 0.03, 0.90),   # good recoveries, cheap funding: a lost customer weighs more
    "central":    (0.55, 0.04, 0.80),
    "high_ratio": (0.65, 0.05, 0.70),   # poor recoveries, dear funding: a default weighs more
}


def loan_costs(X: pd.DataFrame, lgd: float, funding: float, prepay: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-loan cost of a bad approval and of a lost good customer."""
    amount = X["loan_amnt"].to_numpy()
    years_outstanding = 0.5 * (X["term"].to_numpy() / 12) * prepay
    margin = np.clip(X["int_rate"].to_numpy() / 100 - funding, 0, None)
    return lgd * amount, amount * margin * years_outstanding


def cost_curve(p, y, cost_bad: float, cost_good: float, grid) -> pd.DataFrame:
    """Cost per application of a single threshold t (approve if p < t), with average costs."""
    y = np.asarray(y)
    rows = []
    for t in grid:
        approve = p < t
        cost = cost_bad * (approve & (y == 1)).sum() + cost_good * (~approve & (y == 0)).sum()
        rows.append({"threshold": round(float(t), 3), "cost_per_app": cost / len(y),
                     "approve_rate": approve.mean()})
    return pd.DataFrame(rows)


def band_stats(p, y, approve_below, decline_above) -> dict:
    y = np.asarray(y)
    a, d = p < approve_below, p >= decline_above
    r = ~a & ~d
    rate = lambda m: round(float(y[m].mean()), 3) if m.any() else None  # noqa: E731
    return {"approve_below": approve_below, "decline_above": decline_above,
            "share_approve": round(float(a.mean()), 3), "share_review": round(float(r.mean()), 3),
            "share_decline": round(float(d.mean()), 3), "default_rate_approved": rate(a),
            "default_rate_review": rate(r), "default_rate_declined": rate(d)}


def cost_analysis(p_val, X_val, y_val, p_te, X_te, y_te) -> dict:
    grid = np.round(np.arange(0.02, 0.61, 0.01), 2)
    out = {"assumptions": {k: {"lgd": v[0], "funding_and_ops": v[1], "prepay_factor": v[2]}
                           for k, v in COST_SCENARIOS.items()},
           "mean_loan_usd": round(float(X_val["loan_amnt"].mean())),
           "mean_int_rate": round(float(X_val["int_rate"].mean()), 2), "scenarios": {}}
    for name, params in COST_SCENARIOS.items():
        cb, cg = (float(v.mean()) for v in loan_costs(X_val, *params))
        curve = cost_curve(p_val, y_val, cb, cg, grid)
        best = float(curve.loc[curve["cost_per_app"].idxmin(), "threshold"])
        te = cost_curve(p_te, y_te, cb, cg, [best, 1.01]).set_index("threshold")  # 1.01: approve everyone
        out["scenarios"][name] = {
            "cost_bad_approval_usd": round(cb), "cost_lost_good_customer_usd": round(cg),
            "cost_ratio": round(cb / cg, 2),
            "breakeven_pd_theory": round(cg / (cb + cg), 3),
            "best_threshold_validation": best,
            "test_cost_per_app_usd": round(float(te.loc[best, "cost_per_app"]), 1),
            "test_cost_per_app_approve_all_usd": round(float(te.loc[1.01, "cost_per_app"]), 1),
            "test_approve_rate": round(float(te.loc[best, "approve_rate"]), 3),
        }
    # Thin margins on cheap loans: the break-even differs a lot by grade (central scenario).
    cb, cg = loan_costs(X_val, *COST_SCENARIOS["central"])
    g = pd.DataFrame({"grade": X_val["grade"].to_numpy(), "cb": cb, "cg": cg}).groupby("grade").mean()
    out["breakeven_pd_by_grade_central"] = {k: round(float(v), 3) for k, v in (g["cg"] / (g["cb"] + g["cg"])).items()}

    s = out["scenarios"]
    approve_below = round(s["high_ratio"]["breakeven_pd_theory"], 2)
    decline_above = round(s["low_ratio"]["breakeven_pd_theory"], 2)
    out["recommended_policy"] = {
        "single_threshold": round(s["central"]["breakeven_pd_theory"], 2),
        "rule": "approve below the high-ratio break-even, decline at or above the low-ratio break-even, "
                "review in between: the grey zone is where the right decision depends on which cost figure is true",
        "test": band_stats(p_te, y_te, approve_below, decline_above),
    }
    out["current_pipeline_policy_test"] = band_stats(p_te, y_te, 0.12, 0.40)
    return out


# ---------- plot ----------

def plot_calibration(curves: dict, chosen: str, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (uv sync installs it with the dev group): no plot.")
        return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 0.8], [0, 0.8], ls="--", c="grey", lw=1, label="perfect calibration")
    for name, pts in curves.items():
        x, y = [b["predicted"] for b in pts], [b["observed"] for b in pts]
        bold = name == chosen + "_calibrated"
        ax.plot(x, y, marker="o", ms=5 if bold else 3, lw=2.5 if bold else 0.9,
                alpha=1 if bold else 0.55, label=name, zorder=3 if bold else 2)
    ax.set_xlabel("mean predicted probability of default (decile)")
    ax.set_ylabel("observed default rate")
    ax.set_title("Calibration on the held-out test set")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-rows", type=int, default=200_000)
    args = ap.parse_args()

    df = load(args.csv, args.max_rows)
    X_rest, X_te, y_rest, y_te = train_test_split(
        df[NUM + CAT], df["target"], test_size=0.25, stratify=df["target"], random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rest, y_rest, test_size=0.2, stratify=y_rest, random_state=42)
    print(f"{len(df):,} loans, default rate {df['target'].mean():.1%} | "
          f"train {len(X_tr):,}, validation {len(X_val):,}, test {len(X_te):,}")

    # Rows the model never saw: make_eval_files.py builds the evaluation files from them.
    Path("data").mkdir(exist_ok=True)
    X_te.assign(target=y_te).to_csv("data/test_split.csv", index=False)

    results, curves = {}, {}
    for name, build in CANDIDATES.items():
        m = build().fit(X_tr, y_tr)
        p_val, p_te = m.predict_proba(X_val)[:, 1], m.predict_proba(X_te)[:, 1]
        results[name] = {"validation": scores(p_val, y_val), "test": scores(p_te, y_te), "size_mb": size_mb(m)}
        curves[name] = calibration_curve(p_te, y_te)
        print(f"{name:32s} val {results[name]['validation']} | {results[name]['size_mb']} MB")

    eligible = [n for n in results if results[n]["size_mb"] <= MAX_MB]
    best = max(eligible, key=lambda n: (results[n]["validation"]["pr_auc"], results[n]["validation"]["auc"]))
    print(f"best on validation PR-AUC (size <= {MAX_MB} MB): {best}")

    # Calibrate on the training data (3-fold), choose the method on validation by Brier score.
    calibrated = {}
    for method in ("sigmoid", "isotonic"):
        c = CalibratedClassifierCV(CANDIDATES[best](), method=method, cv=3, ensemble=False).fit(X_tr, y_tr)
        calibrated[method] = (c, brier_score_loss(y_val, c.predict_proba(X_val)[:, 1]))
        print(f"calibration {method:8s} validation Brier {calibrated[method][1]:.5f}")
    method = min(calibrated, key=lambda k: calibrated[k][1])
    final = calibrated[method][0]

    p_val, p_te = final.predict_proba(X_val)[:, 1], final.predict_proba(X_te)[:, 1]
    key = best + "_calibrated"
    results[key] = {"validation": scores(p_val, y_val), "test": scores(p_te, y_te),
                    "size_mb": size_mb(final), "calibration_method": method}
    curves[key] = calibration_curve(p_te, y_te)
    print(f"final ({method}): test {results[key]['test']}")

    costs = cost_analysis(p_val, X_val, y_val, p_te, X_te, y_te)
    for name, s in costs["scenarios"].items():
        print(f"cost {name:10s} bad ${s['cost_bad_approval_usd']:,} vs good ${s['cost_lost_good_customer_usd']:,} "
              f"ratio {s['cost_ratio']:.2f}  break-even {s['breakeven_pd_theory']:.1%}  "
              f"best threshold (val) {s['best_threshold_validation']:.0%}")
    print("break-even by grade (central):", costs["breakeven_pd_by_grade_central"])
    print("recommended policy on test:", costs["recommended_policy"]["test"])
    print("current 12%/40% policy on test:", costs["current_pipeline_policy_test"])

    defaults = {c: float(X_tr[c].median()) for c in NUM} | {c: X_tr[c].mode()[0] for c in CAT}
    background = X_tr.sample(BACKGROUND_ROWS, random_state=0).reset_index(drop=True)
    bundle = {"model": final, "version": f"credit-{best}-{method}-v2", "defaults": defaults,
              "background": background, "features": NUM + CAT}
    Path("models").mkdir(exist_ok=True)
    joblib.dump(bundle, "models/credit_model.joblib", compress=3)
    Path("models/metrics.json").write_text(json.dumps(
        {"chosen": best, "calibration_method": method, "version": bundle["version"],
         "selection": "highest validation PR-AUC among models under 5 MB; test set only reported",
         "n_loans": len(df), "n_train": len(X_tr), "n_validation": len(X_val), "n_test": len(X_te),
         "default_rate": round(float(df["target"].mean()), 4),
         "models": results, "calibration_curves_test": curves, "cost_analysis": costs}, indent=2))
    plot_calibration(curves, best, Path("models/calibration.png"))
    print("size (MB):", round(Path("models/credit_model.joblib").stat().st_size / 1e6, 2))


if __name__ == "__main__":
    main()
