"""Loan screening copilot.

Component A (tabular ML): a calibrated credit-risk model scores a loan application.
Component B (agentic LLM): Gemini writes a short credit memo from A's score and drivers.
A's output is B's input, and the review rule uses both.

    run(raw_input) -> {
        "input_summary": str,          # short, no personal data: it is logged
        "features": {name: number},    # what the drift check watches
        "outputs": {component: ...},   # each component's output, JSON-serialisable
        "versions": {component: str},  # model and prompt versions
        "review": bool,                # does this case go to a person?
        "tokens": int,                 # tokens spent on this request (0 if none)
        "cost_eur": float,             # cost of this request in euros
    }

`raw_input` is what the App tab's input gives you: text, or a file path.
The `input` column of eval/cases.csv uses the same convention.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd

from application import ApplicationError, parse_application, to_text

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "credit_model.joblib"
CACHE_DIR = ROOT / "cache"
PROMPT_VERSION = "memo-v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
MAX_ATTEMPTS = 3  # Gemini sometimes answers 503/429 for a moment: retry before giving up

VERSIONS = {"component_a": "credit-model-not-loaded", "component_b": f"gemini-{PROMPT_VERSION}"}

# PLACEHOLDER thresholds on the probability of default. They must be tuned on the
# evaluation set (cost of a bad approval vs. cost of a lost good customer) and
# justified in REPORT.md.
APPROVE_BELOW = 0.12
DECLINE_ABOVE = 0.40

GRADE_NUM = {g: float(i + 1) for i, g in enumerate("ABCDEFG")}
_BUNDLE = None

PROMPT = """You are an assistant for a credit analyst at a consumer lender. Write a short memo for the analyst.

Application (anonymised): {application}
Model estimate: probability of default {p:.1%} (the portfolio average is about 19%).
Largest drivers, as the change in that probability in percentage points compared with an average applicant: {drivers}
Policy: approve if the probability is below {approve:.0%}, decline if it is {decline:.0%} or above, otherwise send to an analyst (review).

Rules:
- Use only the numbers above. Do not invent facts about the applicant.
- 60 to 100 words, plain language, mention the two or three drivers that matter most.
- Recommend exactly one of: approve, review, decline.
Reply as JSON: {{"recommendation": "...", "memo": "..."}}"""


# ---------- input ----------

def read_input(raw_input: str) -> str:
    """Text as it is; for a file path, the file's text."""
    path = Path(str(raw_input))
    if len(str(raw_input)) < 300 and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return str(raw_input)


def input_features(text: str) -> dict[str, float]:
    """Numbers that describe the input, for the drift check."""
    try:
        app = parse_application(text)
    except ApplicationError:
        return {"n_words": float(len(text.split()))}
    a = component_a(app)  # the model's own output is a feature too: its drift matters most
    return {
        "loan_amnt": app["loan_amnt"],
        "int_rate": app["int_rate"],
        "dti": app["dti"],
        "annual_inc": app["annual_inc"],
        "term": app["term"],
        "grade_num": GRADE_NUM[app["grade"]],
        "p_default": a["probability_default"],
        "confidence": a["confidence"],
    }


# ---------- component A: credit-risk model ----------

def _bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(MODEL_PATH)
        VERSIONS["component_a"] = _BUNDLE["version"]
    return _BUNDLE


def component_a(app: dict) -> dict:
    """Probability of default, and the 3 features that move it most.

    Drivers: for each feature, swap the applicant's value for the values of 100 real
    applicants from the training data (the bundle's "background"), keep everything else,
    and average the probabilities. effect_pp = this applicant's probability minus that
    average: how much this feature's value moves the risk compared with typical applicants.
    Averaging over many applicants is steadier than one median value (a single "typical"
    grade or purpose can sit on a step of the trees). Works with any model; ~900 rows, a few ms.
    """
    b = _bundle()
    cols = b["features"]
    row = pd.DataFrame([{c: app[c] for c in cols}])
    p = float(b["model"].predict_proba(row)[0, 1])

    bg = b.get("background")
    if bg is None:  # older bundles: one typical value per feature
        alt = pd.concat([row] * len(cols), ignore_index=True)
        for i, c in enumerate(cols):
            alt.loc[i, c] = b["defaults"][c]
        base_probs = b["model"].predict_proba(alt)[:, 1]
    else:
        frames = []
        for c in cols:
            f = pd.concat([row] * len(bg), ignore_index=True)
            f[c] = bg[c].to_numpy()
            frames.append(f)
        probs = b["model"].predict_proba(pd.concat(frames, ignore_index=True))[:, 1]
        base_probs = probs.reshape(len(cols), len(bg)).mean(axis=1)
    effects = sorted(zip(cols, base_probs), key=lambda cp: -abs(p - cp[1]))[:3]
    drivers = [{"feature": c, "value": app[c], "effect_pp": round(100 * (p - float(q)), 1)} for c, q in effects]

    band = "approve" if p < APPROVE_BELOW else "decline" if p >= DECLINE_ABOVE else "review"
    return {
        "probability_default": round(p, 4),
        "confidence": round(max(p, 1 - p), 3),
        "band": band,
        "drivers": drivers,
    }


# ---------- component B: Gemini credit memo ----------

def _call_gemini(key: str, model: str, prompt: str) -> dict:
    """One call to Gemini. Raises on any failure; component_b decides what to do."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)  # keep a reference: a temporary client can be closed before the call
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
    )
    data = json.loads(resp.text)
    u = resp.usage_metadata
    tokens_in = getattr(u, "prompt_token_count", 0) or 0
    tokens_out = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
    eur_in = float(os.environ.get("LLM_EUR_PER_M_INPUT", "0.28"))
    eur_out = float(os.environ.get("LLM_EUR_PER_M_OUTPUT", "2.30"))
    return {
        "recommendation": str(data["recommendation"]).strip().lower(),
        "memo": str(data["memo"]).strip(),
        "model": model,
        "tokens": tokens_in + tokens_out,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_eur": (tokens_in * eur_in + tokens_out * eur_out) / 1e6,
    }


def component_b(app: dict, a: dict) -> dict:
    """Memo + recommendation. Never raises: on any failure it returns an 'error'."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"error": "GEMINI_API_KEY is not set (put it in .env, or in Render's Environment page)"}

    model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    VERSIONS["component_b"] = f"{model}/{PROMPT_VERSION}"
    drivers = "; ".join(
        f"{d['feature']}={d['value']:g} ({d['effect_pp']:+.1f} pp)" if isinstance(d["value"], float)
        else f"{d['feature']}={d['value']} ({d['effect_pp']:+.1f} pp)"
        for d in a["drivers"]
    )
    prompt = PROMPT.format(application=to_text(app), p=a["probability_default"], drivers=drivers,
                           approve=APPROVE_BELOW, decline=DECLINE_ABOVE)

    cache_file = CACHE_DIR / (hashlib.sha256((model + prompt).encode()).hexdigest()[:24] + ".json")
    use_cache = os.environ.get("LLM_CACHE") == "1"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    out = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            out = _call_gemini(key, model, prompt)
            break
        except Exception as exc:  # network, quota, overload, bad JSON...
            text = str(exc)
            transient = any(code in text for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if not transient or attempt == MAX_ATTEMPTS - 1:
                return {"error": f"{type(exc).__name__}: {text[:200]}"}
            time.sleep(1.5 * (attempt + 1))

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file.write_text(json.dumps(out))
    return out


# ---------- the chain ----------

def run(raw_input: str) -> dict:
    text = read_input(raw_input)
    try:
        app = parse_application(text)
    except ApplicationError as exc:
        # Out of scope: refuse, send to a person, no LLM call. The text itself is
        # NOT logged because we cannot be sure it has no personal data.
        reason = str(exc)
        message = f"Not a valid loan application: {reason}. Sent to a person."
        return {
            "input_summary": f"invalid input ({len(text.split())} words)",
            "features": input_features(text),
            "outputs": {
                "component_a": {"error": reason},
                "component_b": {"recommendation": "review", "review_reasons": ["invalid_input"],
                                "memo": message, "reply": message},
            },
            "versions": dict(VERSIONS),
            "review": True,
            "tokens": 0,
            "cost_eur": 0.0,
        }

    a = component_a(app)
    b = component_b(app, a)

    reasons = []
    if a["band"] == "review":
        reasons.append("grey_zone")
    if "error" in b:
        reasons.append("llm_unavailable")
        memo = (f"Automatic memo unavailable ({b['error']}). Model estimate: "
                f"{a['probability_default']:.1%} probability of default. Sent to a person.")
        b = {**b, "recommendation": "review", "memo": memo, "tokens": 0, "cost_eur": 0.0}
    elif b["recommendation"] not in ("approve", "review", "decline") or b["recommendation"] != a["band"]:
        reasons.append("memo_contradicts_score")

    b = {**b, "review_reasons": reasons, "reply": b["memo"]}
    return {
        "input_summary": to_text(app),
        "features": input_features(text),
        "outputs": {"component_a": a, "component_b": b},
        "versions": dict(VERSIONS),
        "review": bool(reasons),
        "tokens": int(b.get("tokens", 0)),
        "cost_eur": float(b.get("cost_eur", 0.0)),
    }