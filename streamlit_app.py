import os
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# Set dark theme page configuration
st.set_page_config(
    page_title="GlassBox AI 🔮 — See Inside the Model",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark instrument panel aesthetic
st.markdown("""
<style>
    .main { background-color: #090a0f; color: #f8fafc; }
    .stApp { background-color: #090a0f; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #38bdf8; }
    .lens-card { background: #181d2d; border: 1px solid #283046; border-radius: 6px; padding: 8px; text-align: center; }
    .first-match { border: 2px solid #facc15 !important; background: #232215 !important; }
</style>
""", unsafe_allow_html=True)

# 1. Download Model Weights if not present
model_dir = Path("models/SmolLM2-135M")
if not (model_dir / "model.safetensors").exists():
    with st.spinner("Downloading SmolLM2-135M model weights (~270MB)..."):
        import download_model

from glassbox.loader import load_model
from glassbox.tokenizer import SmolLMTokenizer
from glassbox.trace import TraceCollector
from glassbox.ops import softmax

# 2. Load Model & Tokenizer with caching
@st.cache_resource
def get_model_and_tokenizer():
    model, config = load_model("models/SmolLM2-135M")
    tokenizer = SmolLMTokenizer("models/SmolLM2-135M")
    return model, config, tokenizer

model, config, tokenizer = get_model_and_tokenizer()

st.title("GlassBox 🔮 — See Inside the Model")
st.caption("Pure NumPy Transformer Engine · Mechanistic Interpretability Dashboard")

# Sidebar Controls
st.sidebar.header("Controls & Prompt")
prompt = st.sidebar.text_area("Prompt", value="The capital of France is", height=100)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=50, value=20)
use_cache = st.sidebar.checkbox("KV Cache Enabled", value=True)

st.sidebar.subheader("Sampling Strategy")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.05)
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=100, value=40)
top_p = st.sidebar.slider("Top-P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

# Session state for trace data
if "trace" not in st.session_state:
    st.session_state.trace = None
if "generated_tokens" not in st.session_state:
    st.session_state.generated_tokens = []
if "speed" not in st.session_state:
    st.session_state.speed = 0.0

if st.sidebar.button("🚀 Generate Tokens", type="primary", use_container_width=True):
    collector = TraceCollector()
    input_ids = tokenizer.encode(prompt)
    
    t0 = time.time()
    # Simple generation loop
    curr_ids = list(input_ids)
    
    with st.spinner("Running GlassBox NumPy Engine..."):
        for step in range(max_tokens):
            step_trace = collector.start_step(step, curr_ids)
            logits = model.forward(curr_ids, trace_collector=collector, cache=None)
            
            # Extract last token logits
            next_logits = logits[0, -1, :]
            step_trace.logits = next_logits
            
            # Simple greedy or sampled next token
            if temperature == 0:
                next_id = int(np.argmax(next_logits))
            else:
                scaled = next_logits / temperature
                probs = softmax(scaled)
                next_id = int(np.random.choice(len(probs), p=probs))
                
            curr_ids.append(next_id)
            step_trace.output_token = next_id
            if next_id == tokenizer.eos_token_id:
                break
                
    elapsed = time.time() - t0
    st.session_state.generated_tokens = [tokenizer.decode([tid]) for tid in curr_ids]
    st.session_state.trace = collector
    st.session_state.speed = round(len(curr_ids) / max(0.001, elapsed), 1)

# Display Dashboard if Trace exists
if st.session_state.trace and len(st.session_state.trace.steps) > 0:
    steps = st.session_state.trace.steps
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Tok/s", f"{st.session_state.speed}")
    c2.metric("Total Tokens", f"{len(st.session_state.generated_tokens)}")
    c3.metric("KV Cache", "ON" if use_cache else "OFF")
    
    # Step Selector Slider
    step_idx = st.slider("Step Inspector Slider", min_value=0, max_value=len(steps)-1, value=0)
    current_step = steps[step_idx]
    
    st.markdown("---")
    
    # Heatmap & Probabilities Split
    col_left, col_right = st.columns([6, 4])
    
    with col_left:
        st.subheader("Attention Matrix")
        layer_idx = st.selectbox("Layer", options=list(range(config.num_hidden_layers)), index=15)
        head_idx = st.selectbox("Head", options=list(range(config.num_attention_heads)), index=0)
        
        attn_matrix = current_step.attention[layer_idx][head_idx]
        tokens_so_far = [tokenizer.decode([tid]).replace(" ", "·") for tid in current_step.tokens]
        
        # Plotly Viridis Heatmap
        fig_attn = px.imshow(
            np.sqrt(np.clip(attn_matrix, 0, 1)),
            x=tokens_so_far,
            y=tokens_so_far[1:],
            color_continuous_scale="Viridis",
            labels=dict(x="Key Tokens (Past)", y="Query Tokens (Generated)", color="Attention")
        )
        fig_attn.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#090a0f",
            plot_bgcolor="#090a0f",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_attn, use_container_width=True)

    with col_right:
        st.subheader("Next-Token Probabilities")
        step_logits = current_step.logits
        probs = softmax(step_logits / max(0.01, temperature))
        top_indices = np.argsort(probs)[-10:][::-1]
        
        top_toks = [tokenizer.decode([i]).replace(" ", "·") for i in top_indices]
        top_probs = [probs[i] * 100 for i in top_indices]
        
        fig_probs = go.Figure(go.Bar(
            x=top_probs,
            y=top_toks,
            orientation='h',
            marker=dict(color='#38bdf8')
        ))
        fig_probs.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#090a0f",
            plot_bgcolor="#090a0f",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_probs, use_container_width=True)

    # Bottom: Logit Lens
    st.markdown("---")
    st.subheader("Logit Lens — Predictions Layer by Layer")
    st.caption("🎯 Gold Highlight = First layer where prediction matches final layer output")
    
    lens_data = current_step.logit_lens
    final_token = lens_data[-1]["top_token"]
    
    # Find first match layer
    first_match = next((i for i, l in enumerate(lens_data) if l["top_token"] == final_token), -1)
    
    # Display 30 layers in 2 rows of 15 columns
    cols_row1 = st.columns(15)
    for i in range(15):
        layer_info = lens_data[i]
        tok_str = (layer_info.get("top_token_str") or tokenizer.decode([layer_info["top_token"]])).replace(" ", "·")
        prob_pct = round(layer_info["prob"] * 100, 1)
        
        is_gold = (i == first_match)
        border_style = "2px solid #facc15" if is_gold else "1px solid #283046"
        bg_style = "#232215" if is_gold else "#181d2d"
        badge = "🎯" if is_gold else ""
        
        cols_row1[i].markdown(f"""
            <div style="background:{bg_style}; border:{border_style}; border-radius:4px; padding:4px; text-align:center;">
                <div style="font-size:10px; color:#94a3b8;">L{i} {badge}</div>
                <div style="font-size:12px; font-weight:bold; color:#f8fafc;">{tok_str}</div>
                <div style="font-size:10px; color:#38bdf8;">{prob_pct}%</div>
            </div>
        """, unsafe_allow_html=True)

    cols_row2 = st.columns(15)
    for i in range(15, 30):
        layer_info = lens_data[i]
        tok_str = (layer_info.get("top_token_str") or tokenizer.decode([layer_info["top_token"]])).replace(" ", "·")
        prob_pct = round(layer_info["prob"] * 100, 1)
        
        is_gold = (i == first_match)
        border_style = "2px solid #facc15" if is_gold else "1px solid #283046"
        bg_style = "#232215" if is_gold else "#181d2d"
        badge = "🎯" if is_gold else ""
        
        cols_row2[i-15].markdown(f"""
            <div style="background:{bg_style}; border:{border_style}; border-radius:4px; padding:4px; text-align:center;">
                <div style="font-size:10px; color:#94a3b8;">L{i} {badge}</div>
                <div style="font-size:12px; font-weight:bold; color:#f8fafc;">{tok_str}</div>
                <div style="font-size:10px; color:#38bdf8;">{prob_pct}%</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("👈 Set your prompt and click '🚀 Generate Tokens' in the sidebar to run the engine!")