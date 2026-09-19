"""Gradio interface for the affordance-detection demo.

Select an object, ask an affordance in natural language, inspect the highlighted
region and a confidence summary, rotate/zoom, and compare against your previous
query. The interface never claims certainty: low-confidence results are flagged.

Run:
    python -m src.app.demo --checkpoint results/checkpoints/openad_pn2.t7
    python -m src.app.demo --checkpoint pre.t7 --finetuned results/checkpoints/finetuned_best.pth
"""

from __future__ import annotations

import argparse

import numpy as np

from ..paths import DEFAULT_CHECKPOINT, REF_OPENAD
from .backend import STATE_ERROR, STATE_INVALID, STATE_LOW_CONFIDENCE, DemoBackend

DEFAULT_CONFIG = REF_OPENAD / "config" / "openad_pn2" / "full_shape_open_vocab_cfg.py"

EXAMPLE_QUERIES = [
    "grasp",
    "Where can I sit on this?",
    "the part used for pouring",
    "cut",
    "float this in the air",  # unsupported -> demonstrates the low-confidence state
]

_STATUS_PREFIX = {
    STATE_INVALID: "### ⌨️ Waiting for a query\n\n",
    STATE_LOW_CONFIDENCE: "### ⚠️ Low-confidence result\n\n",
    STATE_ERROR: "### ❌ Error\n\n",
}


def _figure(points: np.ndarray | None, scores: np.ndarray | None, title: str):
    import plotly.graph_objects as go

    if points is None or len(points) == 0:
        fig = go.Figure()
    else:
        fig = go.Figure(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2, color=scores, cmin=0.0, cmax=1.0,
                    colorscale="Viridis", colorbar=dict(title="score"),
                ),
            )
        )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def build_demo(backend: DemoBackend):
    """Build the Gradio Blocks app around a loaded :class:`DemoBackend`."""
    import gradio as gr

    def run(sample_label, query, model_name, previous):
        index = int(sample_label.split(":")[0])
        pred = backend.predict(index, query, model_name)
        summary = _STATUS_PREFIX.get(pred.state, "") + pred.summary
        current_fig = _figure(pred.points, pred.scores, f"Current: {query}")

        if previous:
            prev_fig = _figure(previous["points"], previous["scores"], f"Previous: {previous['query']}")
            prev_md = previous["summary"]
        else:
            prev_fig = _figure(None, None, "Previous")
            prev_md = "_No previous query yet._"

        new_previous = {
            "points": pred.points, "scores": pred.scores, "summary": summary, "query": query,
        }
        return current_fig, summary, prev_fig, prev_md, new_previous

    with gr.Blocks(title="Open-Vocabulary 3D Affordance Detection") as demo:
        gr.Markdown(
            "# Open-Vocabulary 3D Affordance Detection\n"
            "Ask where to perform an action on a 3D object; the model highlights the "
            "supporting points. **This is a research prototype — predictions can be "
            "wrong. Use the confidence summary and compare phrasings.**"
        )
        previous = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                sample = gr.Dropdown(
                    choices=backend.sample_labels, value=backend.sample_labels[0],
                    label="Object",
                )
                model = gr.Radio(
                    choices=backend.model_names, value=backend.model_names[0], label="Model",
                )
                query = gr.Textbox(label="Affordance query", placeholder="e.g. grasp")
                gr.Examples(EXAMPLE_QUERIES, inputs=query, label="Example queries")
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    reset_btn = gr.Button("Reset")
            with gr.Column(scale=2):
                current_plot = gr.Plot(label="Prediction")
                current_summary = gr.Markdown()
                with gr.Accordion("Compare with previous query", open=False):
                    previous_plot = gr.Plot(label="Previous")
                    previous_summary = gr.Markdown("_No previous query yet._")

        run_btn.click(
            run,
            inputs=[sample, query, model, previous],
            outputs=[current_plot, current_summary, previous_plot, previous_summary, previous],
        )
        reset_btn.click(
            lambda: (_figure(None, None, "Current"), "", _figure(None, None, "Previous"),
                     "_No previous query yet._", None),
            outputs=[current_plot, current_summary, previous_plot, previous_summary, previous],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="pretrained checkpoint")
    parser.add_argument("--finetuned", default=None, help="optional fine-tuned checkpoint")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("no checkpoint given (pass --checkpoint or set OPENAD_CHECKPOINT)")
    return args


def main() -> None:
    args = parse_args()
    checkpoints = {"pretrained": args.checkpoint}
    if args.finetuned:
        checkpoints["finetuned"] = args.finetuned

    backend = DemoBackend(
        args.config, checkpoints, split=args.split, num_samples=args.num_samples,
        data_root=args.data_root, device=args.device,
    ).load()

    build_demo(backend).launch(share=args.share)


if __name__ == "__main__":
    main()
