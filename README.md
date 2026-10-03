# 🔮 GlassBox: See Inside the Model

**Made By 🔮 Vivi** A small language model rebuilt in pure NumPy, with a live visualizer for its internals.

GlassBox runs the open-weight **SmolLM2-135M** locally and shows what happens as it generates text: where it attends, which next tokens it considers, and how its predictions change across layers. There is no PyTorch runtime dependency; the inference code is designed to be small enough to read and explore.

## ✨ Why GlassBox?

Many language-model demos hide the model behind an API. GlassBox takes the opposite approach: it pairs a real, open-weight model with an inspectable NumPy inference engine and a visual guide to its generation process. It is a learning tool, as well as a starting point for experiments with model internals.

## 🧭 Features

- **Attention heatmap:** inspect any layer and head, average across heads, normalize rows, or toggle the first-token attention sink. Hover a cell to inspect its query, key, and weight.
- **Next-token probabilities:** explore a generation step's top 200 candidates and see which token the model selected.
- **What-if branching:** click a probability bar to force that token and continue generation; return to the original prompt at any time.
- **Token stream:** click a generated token to inspect its step. Surprise-colored chips show the model's confidence, while attention-weighted highlighting shows which earlier tokens received attention. Attention is not proof of why the model answered.
- **Head gallery:** compare all attention heads in the selected layer.
- **Logit lens:** follow the top prediction across all 30 transformer blocks.
- **Sampling and speed controls:** adjust temperature, top-p, and top-k for the next run; compare KV-cache behavior and generation speed.
- **Local visualizer:** a FastAPI server streams generation traces to a browser UI with no frontend build step.

## 🚀 Quick Start

Requires Python 3.11 or later. Model weights are downloaded separately and need about 270 MB of disk space.

```powershell
git clone https://github.com/bistighosh16/glassbox.git
cd glassbox
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_model.py
python run_server.py
```

On macOS or Linux, activate the environment with `source venv/bin/activate`. Then open [http://localhost:8000](http://localhost:8000), launch GlassBox, enter a prompt, and select **generate**.

The model weights are not stored in this repository. `download_model.py` fetches them from Hugging Face.

## 🏗️ Model and Architecture

SmolLM2-135M has **30 layers**, a **576-wide hidden state**, **9 attention heads** with **3 key/value heads** (grouped-query attention), a **49,152-token vocabulary**, and about **134.5 million parameters**.

```text
Prompt -> Tokenizer -> Embeddings
			 |
		 30 transformer blocks
		 RMSNorm, RoPE, GQA, SwiGLU
			 |
	     Final norm -> LM head -> logits
				      |
			  temperature / top-k / top-p
				      |
				  next token

	      KV cache: keys and values between steps

     attention and hidden-state traces -> FastAPI WebSocket
				      -> browser visualizer
```

The loader reads the model weights, including conversion of stored bfloat16 values for NumPy. The tokenizer and ops modules prepare and process tokens; the model and generation modules run the forward pass and sampling. Trace data feeds the attention heatmap, probability bars, and logit lens.

## 🧪 Verification

The tests compare NumPy results with the Hugging Face/PyTorch reference implementation. On the checked cases, the forward-pass maximum logit difference is about `8.4e-5`, all 30 block hidden states have relative error below `1e-4`, and 25-token greedy generation matches. KV-cache checks compare cached and uncached decoding as well as reference logits.

These checks cover specific prompts and cases, not a broad benchmark. To run them:

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
python -X utf8 tests/check_greedy_e2e.py
python -X utf8 tests/check_lens_layers_vs_hf.py
```

## ⚠️ Limitations

- SmolLM2-135M is a small base model; generated text can drift or repeat.
- Browser probability bars use the top 200 candidates, not the full vocabulary.
- Attention shows where weight was assigned; it does not explain the model's reasoning.
- Inference uses CPU NumPy. GPU execution, batching, and quantization are not implemented.

## 🗂️ Project Layout

```text
glassbox/    NumPy model, tokenizer, generation, tracing, and server
glassbox/web/ Browser visualizer (HTML, CSS, and JavaScript)
tests/       Numerical, cache, generation, and UI-payload checks
docs/        Model and architecture notes
```

## 🎉 Built For

Hacktoberfest X Hacktropica at Asansol Engineering College, an MLH community fest focused on open-source AI.

## 🙏 Credits and License

The model is [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) by Hugging Face (HuggingFaceTB); see its model card for model licensing. GlassBox code is available under the [MIT License](LICENSE).