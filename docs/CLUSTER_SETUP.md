# Fresh environment on a GPU cluster (recommended)

Use a **project virtualenv** so you don’t mix packages into `~/.local` or the course image in confusing ways.

## 1. Clone / update

```bash
cd ~/attention_guided_steering   # or your path
git pull origin feature/custom-steering   # or your branch
```

## 2. Create and activate venv

```bash
cd ~/attention_guided_steering
python3 -m venv .venv
source .venv/bin/activate
```

Every new shell session:

```bash
cd ~/attention_guided_steering && source .venv/bin/activate
```

Confirm:

```bash
which python    # should end in .../attention_guided_steering/.venv/bin/python
python -c "import sys; print(sys.executable)"
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This installs **torch**, **transformers**, **torchmetrics**, **xRFM** (built from the git URL in `requirements.txt`), etc.

**Note:** `pip install -r requirements.txt` can take a while and may download large CUDA wheels for PyTorch.

## 4. Auth and env vars

```bash
# Gated Llama weights (required for most models in this repo)
huggingface-cli login
# or: export HF_TOKEN="hf_..."

# Optional: HF download cache on a large filesystem (must be a REAL writable path).
# Do not copy "/path/to/hf_cache" from docs — that will error. Examples:
#   export CACHE_DIR="$HOME/hf_cache"
#   export CACHE_DIR="$HOME/scratch/hf_cache"
# Or omit CACHE_DIR entirely → defaults to ~/.cache/huggingface

# Only for GPT-4o evaluation (script 3), not for transfer/collect
export OPENAI_API_KEY="sk-..."
```

## 5. Sanity checks

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import torchmetrics; import rfm; print('torchmetrics + rfm ok')"
```

## 6. Run transfer smoke test (70B)

From repo root, with venv **activated**:

```bash
python transfer/collect_paired_activations.py -m llama_3.1_70b -c fears --concept fire --max_prompts 5 -t -1 --datasize single
```

Use `--max_prompts 5` first to verify load + forward; then raise to 200.

**If you see CUDA OOM while loading 70B on a ~22GB GPU:** weights use `device_map="auto"` by default (CPU offload). Optionally cap GPU bytes so forwards have headroom:

```bash
export STEERING_GPU_MEMORY_CAP_GB=18
```

On a **40GB+** GPU you can force all weights on GPU 0: `export STEERING_DEVICE_MAP=cuda`.

**8B (4-bit Llama):** `utils.select_llm` defaults to **`device_map="cuda"`** for any `*_8b` model id so **BitsAndBytes** does not place quantized layers on CPU (which raises `ValueError` from `quantizer_bnb_4bit`). Use `STEERING_DEVICE_MAP=auto` only if you intentionally follow Hugging Face’s 4-bit CPU/GPU offload setup.

## 7. Transfer smoke test (8B, lighter)

Uses **`llama_3.1_8b`** / **`llama_3.3_8b`** (see `transfer/README.md`). **`llama_3.3_8b`** uses **`LLAMA_33_8B_HF_REPO`** (default public [allura-forge/Llama-3.3-8B-Instruct](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct)); not `meta-llama/Meta-Llama-3-8B` (that is Llama 3.0 **base**).

```bash
python transfer/collect_paired_activations.py -m llama_3.1_8b -c fears --concept fire --max_prompts 5 -t -1 --datasize single
python transfer/collect_paired_activations.py -m llama_3.3_8b -c fears --concept fire --max_prompts 5 -t -1 --datasize single
```

## Avoid

- **`pip install -r requirements.txt`** without a venv → can install into **`~/.local`** and confuse `which python`.
- Relying on **`~/.local/bin`** for CLIs — not needed if you run **`python script.py`**; optional: `export PATH="$HOME/.local/bin:$PATH"`.

## If something still fails

Paste:

- `which python`
- `python -c "import transformers; print(transformers.__version__)"`
- The full traceback
