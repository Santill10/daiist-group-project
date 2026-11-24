"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

Keep the logic in pipeline.py (or your own modules); this file only wires it to
the interface and the logs. Keep `serve()` and the `/run` endpoint: check.py
calls `serve()` locally and `/run` on your deployed app, with one input from
eval/cases.csv (text, or a file path).
"""

import os
import time

import gradio as gr

import env
import logbook
import monitor
import pipeline

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "Loan Screening Copilot"
DESCRIPTION = (
    "For credit analysts at a consumer lender. A risk model scores the loan application and "
    "Gemini writes a short credit memo; unclear or invalid cases go to a person."
)

REVIEW_LABELS = {
    "grey_zone": "risk score in the grey zone",
    "llm_unavailable": "memo unavailable",
    "memo_contradicts_score": "memo disagrees with the score",
    "invalid_input": "not a valid application",
}


def serve(raw_input: str) -> dict:
    """Run one request through the pipeline, log it, and return the logged record."""
    started = time.perf_counter()
    result = pipeline.run(raw_input)
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


# --- App tab -----------------------------------------------------------------

def on_run(raw_input):
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Give me an input first.")
    record = serve(raw_input)
    outputs = record["outputs"]
    reasons = outputs["component_b"].get("review_reasons", [])
    why = ", ".join(REVIEW_LABELS.get(r, r) for r in reasons)
    status = f"⚠️ **Sent to a person for review** ({why})." if record["review"] else "✅ Handled automatically."
    return (
        outputs["component_a"],
        outputs["component_b"]["reply"],
        f"{status}  Latency {record['latency_s']} s, cost €{record['cost_eur']:.4f}.",
        record["id"],
    )


def on_feedback(request_id, value):
    if not request_id:
        gr.Warning("Run a request first.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Thanks for the feedback.")


# --- Monitoring tab ----------------------------------------------------------

def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert_text = "No reference set yet: run `uv run python main.py reference` and commit `monitoring/`."
    elif fired:
        alert_text = "### 🚨 Alerts\n" + "\n".join(f"- {message}" for message in fired)
    else:
        alert_text = "### ✅ No alerts"
    if reference is not None and len(logs) >= monitor.MIN_WINDOW:
        drift_table = monitor.drift(reference, monitor.live_features(logs))
    else:
        drift_table = monitor.drift(reference, reference.head(0)) if reference is not None else None
    return alert_text, monitor.summary(logs), monitor.quality(logs), drift_table


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tab("App"):
        request_id = gr.State("")
        raw_input = gr.Textbox(
            label="Loan application (one line)",
            placeholder="loan_amnt=12000; term=36; int_rate=13.5; grade=C; annual_inc=52000; dti=18.2; emp_length=3; home_ownership=RENT; purpose=debt_consolidation",
            lines=3,
        )
        gr.Examples(
            examples=[
                ["loan_amnt=8000; term=36; int_rate=7.5; grade=A; annual_inc=95000; dti=8.0; emp_length=8; home_ownership=MORTGAGE; purpose=credit_card"],
                ["loan_amnt=24000; term=60; int_rate=12.3; grade=C; annual_inc=150000; dti=15.1; emp_length=0; home_ownership=RENT; purpose=debt_consolidation"],
                ["loan_amnt=20000; term=60; int_rate=24.9; grade=F; annual_inc=32000; dti=32.0; emp_length=1; home_ownership=RENT; purpose=small_business"],
                ["what is the weather in Madrid tomorrow?"],
            ],
            inputs=raw_input,
            label="Examples (fictional applications; the last one is out of scope)",
        )
        run_button = gr.Button("Run", variant="primary")
        with gr.Row():
            out_a = gr.JSON(label="Component A: risk model")
            out_b = gr.Textbox(label="Component B: credit memo", lines=6)
        status = gr.Markdown()
        with gr.Row():
            up = gr.Button("👍")
            down = gr.Button("👎")
        run_button.click(on_run, raw_input, [out_a, out_b, status, request_id], api_name="run")
        up.click(lambda rid: on_feedback(rid, +1), request_id, None, api_name=False)
        down.click(lambda rid: on_feedback(rid, -1), request_id, None, api_name=False)

    with gr.Tab("Monitoring"):
        refresh = gr.Button("Refresh")
        alert_box = gr.Markdown()
        summary_table = gr.Dataframe(label="Summary", interactive=False)
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", label="Quality over time", height=300)
        drift_table = gr.Dataframe(label=f"Drift: PSI of the last {monitor.WINDOW} requests against the reference set", interactive=False)
        gr.Markdown(f"<small>{LOG_TARGET}</small>")
        outputs = [alert_box, summary_table, quality_plot, drift_table]
        refresh.click(on_refresh, None, outputs, api_name=False)
        demo.load(on_refresh, None, outputs, api_name=False)


if __name__ == "__main__":
    # Render gives the port in $PORT; locally it is 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))