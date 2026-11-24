"""False alarms vs. detection of the drift alert (monitor.py), measured on your data.

    uv run python tune_alert.py

Needs data/test_split.csv (written by train_model.py), eval/reference.csv and
eval/shifted.csv (written by make_eval_files.py). Prints, for each NOISE_MULTIPLE and
window size, how often the alert fires on random windows of NORMAL traffic (false
alarms: you want ~0) and on SHIFTED traffic (detection: you want ~100%). Use the table
to choose and justify the threshold in REPORT.md. Nothing here calls Gemini.
"""

import numpy as np
import pandas as pd

import monitor
import pipeline
from application import CATEGORICAL, NUMERIC, to_text

NUM, CAT = list(NUMERIC), list(CATEGORICAL)
N_WINDOWS = 200


def text(row) -> str:
    return to_text({c: float(row[c]) for c in NUM} | {c: str(row[c]) for c in CAT})


def features(inputs) -> pd.DataFrame:
    return pd.DataFrame([pipeline.input_features(t) for t in inputs])


def as_logs(f: pd.DataFrame) -> pd.DataFrame:
    logs = f.add_prefix("f_").copy()
    logs["feedback"] = np.nan  # no thumbs: only the drift rule is tested
    logs["review"] = False
    return logs


def main() -> None:
    rng = np.random.default_rng(0)
    reference = features(pd.read_csv("eval/reference.csv")["input"])
    shifted = features(pd.read_csv("eval/shifted.csv")["input"])
    pool_rows = pd.read_csv("data/test_split.csv").sample(2000, random_state=0)
    normal = features([text(r) for _, r in pool_rows.iterrows()])
    print(f"reference {len(reference)} inputs | normal pool {len(normal)} | shifted {len(shifted)}")
    print(f"features watched: {', '.join(reference.columns)}\n")

    seeds = rng.integers(0, 10**6, N_WINDOWS)
    print("NOISE_MULTIPLE | window | false alarms on normal | detection on shifted")
    for multiple in (2, 3, 4, 5):
        monitor.NOISE_MULTIPLE = multiple
        for window in (20, 30, 40, 50):
            false = sum(bool(monitor.alerts(as_logs(normal.sample(window, random_state=int(s))), reference)) for s in seeds)
            caught = sum(
                bool(monitor.alerts(as_logs(shifted.sample(min(window, len(shifted)), random_state=int(s))), reference))
                for s in seeds
            )
            print(f"{multiple:>14} | {window:>6} | {false:>8}/{N_WINDOWS} = {false / N_WINDOWS:>4.0%}      | {caught:>4}/{N_WINDOWS} = {caught / N_WINDOWS:>4.0%}")


if __name__ == "__main__":
    main()