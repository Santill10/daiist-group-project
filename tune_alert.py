"""False alarms vs. detection of the alerts in monitor.py, measured on your data.

    uv run python tune_alert.py

Nothing here calls Gemini. It prints three tables to choose and justify the thresholds
in REPORT.md (Operations):

1. Drift rule: for each NOISE_MULTIPLE and window size, how often the alert fires on
   random windows of NORMAL traffic (false alarms: you want ~0) and of SHIFTED traffic
   (detection: you want ~100%). MIN_WINDOW is set to the window size, so every row is a
   real test of the rule (before, windows under MIN_WINDOW could never fire).
2. Partial drift: windows where only part of the traffic comes from the shifted segment,
   with the thresholds currently in monitor.py. It says how big a shift must be to be caught.
3. Thumbs-down rule: with a binomial model, how often the 👎 alert fires by chance when the
   product works as usual, and how often it fires when quality really drops.

Normal traffic comes from data/test_split.csv (held-out loans, written by train_model.py,
git-ignored). If that file is missing, a fallback uses the 250 real held-out loans we do
have (eval/reference.csv + the `normal` cases of eval/cases.csv): for each window they are
shuffled, 200 play the reference and the window is drawn from the other 50, so normal
windows are never compared with themselves. Windows are then at most 50 and the numbers
are provisional until test_split.csv is available. (Resampling the same 50 cases with
replacement does not work: duplicated rows make the PSI look like drift.)
"""

import math

import numpy as np
import pandas as pd

import monitor
import pipeline
from application import CATEGORICAL, NUMERIC, to_text

NUM, CAT = list(NUMERIC), list(CATEGORICAL)
N_WINDOWS = 200
MULTIPLES = (2, 3, 4, 5, 6)
WINDOWS = (20, 30, 40, 50, 60, 100)
SHIFT_SHARES = (0.1, 0.2, 0.3, 0.5, 1.0)


def text(row) -> str:
    return to_text({c: float(row[c]) for c in NUM} | {c: str(row[c]) for c in CAT})


def features(inputs) -> pd.DataFrame:
    return pd.DataFrame([pipeline.input_features(t) for t in inputs])


def as_logs(f: pd.DataFrame) -> pd.DataFrame:
    logs = f.add_prefix("f_").reset_index(drop=True)
    logs["feedback"] = np.nan  # no thumbs: only the drift rule is tested
    logs["review"] = False
    return logs


class NormalTraffic:
    """Random windows of normal traffic, each with the reference it is compared with."""

    def __init__(self, reference: pd.DataFrame):
        self.reference = reference
        try:
            rows = pd.read_csv("data/test_split.csv").sample(2000, random_state=0)
            self.pool = features([text(r) for _, r in rows.iterrows()])
            self.split = False
            self.source = "data/test_split.csv (2000 held-out loans)"
        except FileNotFoundError:
            cases = pd.read_csv("eval/cases.csv")
            extra = features(cases.loc[cases["kind"] == "normal", "input"])
            self.pool = pd.concat([reference, extra], ignore_index=True)
            self.split = True
            self.source = (f"PROVISIONAL fallback, data/test_split.csv not found: {len(self.pool)} real loans "
                           f"(reference + normal cases), split at random into reference {len(reference)} / window")
        self.max_window = len(self.pool) - len(reference) if self.split else len(self.pool)

    def window(self, size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(window, reference to compare it with)."""
        if not self.split:
            return self.pool.sample(size, random_state=seed), self.reference
        order = np.random.default_rng(seed).permutation(len(self.pool))
        ref = self.pool.iloc[order[: len(self.reference)]]
        return self.pool.iloc[order[len(self.reference):][:size]], ref


def fires(f: pd.DataFrame, reference: pd.DataFrame) -> bool:
    return bool(monitor.alerts(as_logs(f), reference))


def drift_table(normal: NormalTraffic, shifted, reference, seeds) -> None:
    print(f"1. Drift rule (PSI_ALERT = {monitor.PSI_ALERT}, PSI_BINS = {monitor.PSI_BINS}; MIN_WINDOW = window)")
    print("NOISE_MULTIPLE | window | false alarms on normal | detection on shifted")
    for multiple in MULTIPLES:
        monitor.NOISE_MULTIPLE = multiple
        for window in WINDOWS:
            if window > normal.max_window:
                continue
            monitor.MIN_WINDOW = window
            false = sum(fires(*normal.window(window, int(s))) for s in seeds)
            # shifted.csv has 50 inputs: larger windows are filled by resampling them
            caught = sum(
                fires(shifted.sample(window, replace=window > len(shifted), random_state=int(s)), reference)
                for s in seeds
            )
            print(f"{multiple:>14} | {window:>6} | {false:>8}/{N_WINDOWS} = {false / N_WINDOWS:>4.0%}      "
                  f"| {caught:>4}/{N_WINDOWS} = {caught / N_WINDOWS:>4.0%}")


def partial_table(normal: NormalTraffic, shifted, reference, seeds, settings) -> None:
    multiple, window = settings
    monitor.NOISE_MULTIPLE, monitor.MIN_WINDOW = multiple, window
    print(f"\n2. Partial drift with NOISE_MULTIPLE = {multiple}, window = {window}: share of the window "
          "taken from the shifted segment")
    print("shifted share | detection")
    for share in SHIFT_SHARES:
        n_shift = round(share * window)
        caught = 0
        for s in seeds:
            calm, ref = normal.window(window - n_shift, int(s)) if n_shift < window else (normal.pool.iloc[:0], reference)
            part = pd.concat([calm, shifted.sample(n_shift, replace=n_shift > len(shifted), random_state=int(s) + 1)])
            caught += fires(part, ref)
        print(f"{share:>13.0%} | {caught:>4}/{N_WINDOWS} = {caught / N_WINDOWS:>4.0%}")


def binom_tail(n: int, p: float, k_min: int) -> float:
    """P(X >= k_min) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(k_min, n + 1))


def thumbs_table() -> None:
    print("\n3. Thumbs-down rule: alert when the share of 👎 among the last n rated requests is > threshold")
    print("   false alarm = chance it fires when the true 👎 rate is the usual one; "
          "detection = chance it fires when the true rate has really gone up")
    print("ratings n | threshold | false alarm (true 👎 10%) | (true 👎 15%) | detection (true 👎 35%) | (true 👎 50%)")
    for n in (20, 30, 40, 60):
        for threshold in (0.25, 0.3, 0.4):
            k_min = math.floor(threshold * n) + 1  # strictly more than threshold, as in monitor.alerts
            cells = [binom_tail(n, p, k_min) for p in (0.10, 0.15, 0.35, 0.50)]
            print(f"{n:>9} | {threshold:>9.0%} | {cells[0]:>24.1%} | {cells[1]:>12.1%} | "
                  f"{cells[2]:>22.1%} | {cells[3]:>11.1%}")


def main() -> None:
    rng = np.random.default_rng(0)
    current = (monitor.NOISE_MULTIPLE, monitor.MIN_WINDOW)
    reference = features(pd.read_csv("eval/reference.csv")["input"])
    shifted = features(pd.read_csv("eval/shifted.csv")["input"])
    normal = NormalTraffic(reference)
    print(f"reference {len(reference)} inputs | shifted {len(shifted)} | normal traffic: {normal.source}")
    print(f"features watched: {', '.join(reference.columns)}\n")

    seeds = rng.integers(0, 10**6, N_WINDOWS)
    drift_table(normal, shifted, reference, seeds)
    partial_table(normal, shifted, reference, seeds, current)
    thumbs_table()


if __name__ == "__main__":
    main()
