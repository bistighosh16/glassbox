import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from glassbox.generate import generate_stream
from glassbox.model import Transformer
from glassbox.ops import softmax
from glassbox.tokenizer import Tokenizer
from glassbox.trace import logit_lens


st.set_page_config(
    page_title="GlassBox AI — See Inside the Model",
    page_icon="🔮",
    layout="wide",
)
st.markdown(
    """
    <style>
        .main { background-color: #090a0f; color: #f8fafc; }
        .stApp { background-color: #090a0f; }
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace;
            color: #38bdf8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = ROOT / "models" / "SmolLM2-135M"


@st.cache_resource
def get_model_and_tokenizer(model_dir: str):
    return Transformer(model_dir), Tokenizer(model_dir)


if not (MODEL_DIR / "config.json").is_file() or not (
    (MODEL_DIR / "model.safetensors").is_file()
    or (MODEL_DIR / "model.safetensors.index.json").is_file()
):
    with st.spinner("Downloading SmolLM2-135M model weights (~270 MB)..."):
        from download_model import main as download_model_weights

        download_model_weights()

model, tokenizer = get_model_and_tokenizer(str(MODEL_DIR))
config = model.config

st.title("GlassBox 🔮 — See Inside the Model")
st.caption("Pure NumPy Transformer Engine · Mechanistic Interpretability Dashboard")

st.sidebar.header("Controls & Prompt")
prompt = st.sidebar.text_area(
    "Prompt", value="The capital of France is", height=100
)
max_tokens = st.sidebar.number_input(
    "Maximum new tokens", min_value=1, max_value=50, value=20
)
st.sidebar.subheader("Sampling Strategy")
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.05
)
top_k = st.sidebar.number_input(
    "Top-k", min_value=1, max_value=200, value=40
)
top_p = st.sidebar.slider(
    "Top-p", min_value=0.05, max_value=1.0, value=0.9, step=0.05
)

if "steps" not in st.session_state:
    st.session_state.steps = []
    st.session_state.generated_text = ""

if st.sidebar.button("🚀 Generate Tokens", type="primary", use_container_width=True):
    started = time.perf_counter()
    with st.spinner("Generating tokens and collecting model traces..."):
        generated_steps = list(
            generate_stream(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                use_cache=True,
                trace=True,
            )
        )
    prompt_ids = tokenizer.encode(prompt)
    generated_ids = [step.token_id for step in generated_steps]
    st.session_state.steps = generated_steps
    st.session_state.generated_text = tokenizer.decode(prompt_ids + generated_ids)
    st.session_state.generation_seconds = time.perf_counter() - started

steps = st.session_state.steps
if steps:
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Generated tokens", len(steps))
    elapsed = max(st.session_state.generation_seconds, 0.001)
    metric_right.metric("Average speed", f"{len(steps) / elapsed:.1f} tokens/s")
    st.subheader("Generated text")
    st.write(st.session_state.generated_text)

    step_idx = st.slider(
        "Inspect generation step",
        min_value=0,
        max_value=len(steps) - 1,
        value=0,
    )
    current_step = steps[step_idx]
    context_ids = tokenizer.encode(prompt) + [
        step.token_id for step in steps[:step_idx]
    ]
    context_tokens = [
        tokenizer.decode(token_id).replace(" ", "·") for token_id in context_ids
    ]

    st.markdown("---")
    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.subheader("Attention")
        layer_idx = st.selectbox(
            "Layer",
            options=list(range(config.num_hidden_layers)),
            index=min(15, config.num_hidden_layers - 1),
        )
        head_idx = st.selectbox(
            "Attention head",
            options=list(range(config.num_attention_heads)),
            index=0,
        )
        layer_trace = current_step.traces[layer_idx]
        attention_row = layer_trace.attn_weights[0, head_idx, -1, :]
        attention_tokens = context_tokens[-len(attention_row) :]
        attention_fig = go.Figure(
            go.Heatmap(
                z=[attention_row.tolist()],
                x=attention_tokens,
                y=["Next token"],
                colorscale="Viridis",
                colorbar={"title": "Attention"},
            )
        )
        attention_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#090a0f",
            plot_bgcolor="#090a0f",
            font=dict(color="#cbd5e1"),
            xaxis_title="Tokens attended to",
        )
        st.plotly_chart(attention_fig, use_container_width=True)

    with col_right:
        st.subheader("Next-token probabilities")
        logits = current_step.output_logits
        probabilities = softmax(logits / max(float(temperature), 0.01))
        top_indices = np.argsort(probabilities)[-10:][::-1]
        top_tokens = [
            tokenizer.decode(int(index)).replace(" ", "·")
            for index in top_indices
        ]
        probability_fig = go.Figure(
            go.Bar(
                x=[float(probabilities[index] * 100) for index in top_indices],
                y=top_tokens,
                orientation="h",
                marker=dict(color="#38bdf8"),
            )
        )
        probability_fig.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Probability (%)",
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#090a0f",
            plot_bgcolor="#090a0f",
            font=dict(color="#cbd5e1"),
        )
        st.plotly_chart(probability_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Logit Lens — Predictions Layer by Layer")
    lens_results = logit_lens(model, tokenizer, current_step.traces)
    lens_columns = st.columns(5)
    for index, result in enumerate(lens_results):
        with lens_columns[index % len(lens_columns)]:
            st.metric(
                f"Layer {result.layer_idx}",
                result.top_token_str,
                f"{result.top_prob:.1%}",
            )
else:
    st.info("Enter a prompt and select **Generate Tokens** to inspect a run.")
