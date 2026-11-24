"""A loan application written as text <-> dict, with validation.

Input format (one line, any order, ';' separated):
loan_amnt=12000; term=36; int_rate=13.5; grade=C; annual_inc=52000; dti=18.2;
emp_length=3; home_ownership=RENT; purpose=debt_consolidation

Only these known fields are read; anything else in the text is ignored, so
free text (names, emails...) can never reach the logs through input_summary.
"""

import re

NUMERIC = {  # name: (min, max) accepted
    "loan_amnt": (500, 500_000),
    "term": (36, 60),
    "int_rate": (0.5, 40.0),
    "annual_inc": (1_000, 10_000_000),
    "dti": (0.0, 100.0),
    "emp_length": (0, 10),
}
CATEGORICAL = {
    "grade": set("ABCDEFG"),
    "home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "purpose": {
        "debt_consolidation", "credit_card", "home_improvement", "other",
        "major_purchase", "small_business", "car", "medical", "moving",
        "vacation", "house", "wedding", "renewable_energy", "educational",
    },
}
FIELDS = list(NUMERIC) + list(CATEGORICAL)
_PAIR = re.compile(r"([A-Za-z_]+)\s*=\s*([^;\n]+)")


class ApplicationError(ValueError):
    """The text is not a valid loan application."""


def parse_application(text: str) -> dict:
    pairs = {k.lower(): v.strip() for k, v in _PAIR.findall(text or "")}
    missing = [f for f in FIELDS if f not in pairs]
    if missing:
        raise ApplicationError("missing fields: " + ", ".join(missing))

    app: dict = {}
    for name, (low, high) in NUMERIC.items():
        try:
            value = float(pairs[name].replace(",", ""))
        except ValueError:
            raise ApplicationError(f"{name} must be a number") from None
        if not low <= value <= high:
            raise ApplicationError(f"{name}={value:g} is outside the accepted range {low:g}-{high:g}")
        app[name] = value
    if app["term"] not in (36, 60):
        raise ApplicationError("term must be 36 or 60")

    for name, allowed in CATEGORICAL.items():
        value = pairs[name].strip()
        value = value.lower() if name == "purpose" else value.upper()
        if value not in allowed:  # never echo the value back: it could be free text
            raise ApplicationError(f"{name} is not valid; allowed: {', '.join(sorted(allowed))}")
        app[name] = value
    return app


def to_text(app: dict) -> str:
    """Normalised text of a parsed application: safe to log."""
    return "; ".join(
        f"{n}={app[n]:g}" if isinstance(app[n], float) else f"{n}={app[n]}" for n in FIELDS
    )