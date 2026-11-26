"""
Monitoring: quality signals over time, drift against the reference set, alerts.

Thresholds chosen by P6 with `uv run python tune_alert.py` (numbers in REPORT.md,
Operations). Drift: with MIN_WINDOW 40 and NOISE_MULTIPLE 5 the alert fired on 2% of
random windows of normal traffic and on 100% of windows of the shifted segment.
Thumbs-down: > 30% 👎 among at least 40 ratings fires by chance with probability under
0.5% if the usual 👎 rate is 15% or less, and catches a real rise to 50% 99% of the time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE_FILE = ROOT / "monitoring" / "reference_features.csv"

PSI_BINS = 5           # quantile bins of the reference; few, because samples are small
PSI_ALERT = 0.25       # usual rule of thumb for a large shift; alert when a feature's PSI exceeds this...
NOISE_MULTIPLE = 5     # ...and this many times the PSI expected by chance (psi_noise)
WINDOW = 100           # the drift check looks at the last WINDOW requests...
MIN_WINDOW = 40        # ...and stays silent with fewer than this
THUMBS_DOWN_ALERT = 0.3  # alert when more than this share of rated requests is 👎 (needs MIN_WINDOW ratings)


def _bin(values: np.ndarray, reference: np.ndarray, bins: int) -> np.ndarray:
    """Bin index of each value, with bins defined by the reference.

    A feature with few distinct values in the reference (counts, labels coded as
    numbers) gets one bin per value plus one for anything else; a continuous one
    gets quantile bins, so each holds about the same share of the reference.
    """
    distinct = np.unique(reference)
    if len(distinct) <= bins:
        index = np.searchsorted(distinct, values)
        found = (index < len(distinct)) & (distinct[np.minimum(index, len(distinct) - 1)] == values)
        return np.where(found, index, len(distinct))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right")


def psi(reference, current, bins: int = PSI_BINS) -> float:
    """Population stability index of `current` against `reference`.

    0 means the same distribution; the usual rule of thumb is < 0.1 stable,
    0.1-0.25 moderate, > 0.25 a large shift. With small samples, though, PSI is
    large by chance alone: see `psi_noise`.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    ref_bins, cur_bins = _bin(reference, reference, bins), _bin(current, reference, bins)
    n_bins = max(ref_bins.max(), cur_bins.max()) + 1
    expected = np.bincount(ref_bins, minlength=n_bins) / len(reference)
    actual = np.bincount(cur_bins, minlength=n_bins) / len(current)
    expected, actual = np.clip(expected, 1e-4, None), np.clip(actual, 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_noise(n_reference: int, n_current: int, bins: int = PSI_BINS) -> float:
    """PSI you would expect by chance, with no drift at all, for these sample sizes.

    About (bins - 1) * (1/n_reference + 1/n_current): with 40 reference inputs and
    20 live ones that is already 0.3. An alert below this is a coin toss.
    """
    return (bins - 1) * (1 / n_reference + 1 / n_current)


def load_reference() -> pd.DataFrame | None:
    """Features of the reference inputs (`main.py reference` writes them)."""
    return pd.read_csv(REFERENCE_FILE) if REFERENCE_FILE.is_file() else None


def drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI per feature, and whether it alerts. Both frames have one column per feature (no `f_` prefix).

    The rule: alert when PSI exceeds PSI_ALERT *and* NOISE_MULTIPLE times what
    chance alone would give with these sample sizes.
    """
    rows = []
    for feature in reference.columns:
        if feature in current.columns:
            value = psi(reference[feature], current[feature])
            noise = psi_noise(reference[feature].notna().sum(), max(current[feature].notna().sum(), 1))
            threshold = max(PSI_ALERT, NOISE_MULTIPLE * noise)
            rows.append({"feature": feature, "psi": round(value, 3), "threshold": round(threshold, 3), "alert": bool(value > threshold)})
    return pd.DataFrame(rows, columns=["feature", "psi", "threshold", "alert"])


def live_features(logs: pd.DataFrame) -> pd.DataFrame:
    """The features of the last WINDOW logged requests, without the `f_` prefix."""
    columns = [c for c in logs.columns if c.startswith("f_")]
    return logs[columns].tail(WINDOW).rename(columns=lambda c: c[2:])


def quality(logs: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Quality signals per period, in long format (time, signal, value) for a line plot."""
    if logs.empty:
        return pd.DataFrame(columns=["time", "signal", "value"])
    df = logs.set_index("timestamp")
    rated = df["feedback"].dropna().astype(float)
    signals = pd.DataFrame({
        "share sent to review": df["review"].astype(float).resample(freq).mean(),
        "share 👎 (of rated)": (rated < 0).astype(float).resample(freq).mean(),
    })
    if "f_confidence" in df:
        signals["mean confidence"] = df["f_confidence"].resample(freq).mean()
    if "f_p_default" in df:
        signals["mean probability of default"] = df["f_p_default"].resample(freq).mean()
    long = signals.reset_index(names="time").melt("time", var_name="signal", value_name="value")
    return long.dropna()


def summary(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame({"metric": ["requests"], "value": [0]})
    rated = logs["feedback"].dropna().astype(float)
    rows = {
        "requests": len(logs),
        "sent to review": f"{logs['review'].mean():.0%}",
        "rated 👍/👎": len(rated),
        "share 👎 (of rated)": f"{(rated < 0).mean():.0%}" if len(rated) else "-",
        "median latency (s)": round(float(logs["latency_s"].median()), 2),
        "mean cost per request (€)": round(float(logs["cost_eur"].mean()), 5),
        "total cost (€)": round(float(logs["cost_eur"].sum()), 4),
        "p95 latency (s)": round(float(logs["latency_s"].quantile(0.95)), 2),
        "mean probability of default": f"{logs['f_p_default'].mean():.1%}" if "f_p_default" in logs else "-",
    }
    for reason, share in review_reasons(logs).items():
        rows[f"review reason: {reason}"] = f"{share:.0%} of requests"
    return pd.DataFrame({"metric": list(rows), "value": [str(v) for v in rows.values()]})


def review_reasons(logs: pd.DataFrame) -> pd.Series:
    """Share of all requests that carry each review reason (a request can have several)."""
    if logs.empty:
        return pd.Series(dtype=float)
    from_outputs = (logs["outputs"] if "outputs" in logs else pd.Series([{}] * len(logs), index=logs.index)).map(
        lambda o: (o or {}).get("component_b", {}).get("review_reasons", []) if isinstance(o, dict) else [])
    if "review_reasons" in logs:  # field added by logbook.monitoring_fields; older logs only have outputs
        reasons = logs["review_reasons"].where(logs["review_reasons"].map(lambda r: isinstance(r, list)), from_outputs)
    else:
        reasons = from_outputs
    return reasons.explode().dropna().value_counts() / len(logs)


def alerts(logs: pd.DataFrame, reference: pd.DataFrame | None) -> list[str]:
    """The alert rules. Each fired alert is one message."""
    fired = []
    window = logs.tail(WINDOW)
    if reference is not None and len(window) >= MIN_WINDOW:
        report = drift(reference, live_features(window))
        for row in report[report["alert"]].itertuples():
            fired.append(f"Drift in `{row.feature}`: PSI {row.psi} > {row.threshold} over the last {len(window)} requests.")
    rated = window["feedback"].dropna().astype(float) if not window.empty else pd.Series(dtype=float)
    if len(rated) >= MIN_WINDOW and (rated < 0).mean() > THUMBS_DOWN_ALERT:
        fired.append(f"{(rated < 0).mean():.0%} of the last {len(rated)} rated requests are 👎 (> {THUMBS_DOWN_ALERT:.0%}).")
    return fired
