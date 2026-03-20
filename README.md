## Attention-guided steering

Extension of **[pdavar/attention_guided_steering](https://github.com/pdavar/attention_guided_steering)** — same `0_`–`4_` pipeline, plus **`transfer/`** (cross-version steering) and small changes in **`utils.py`** / **`args.py`**. Long-form notes: **`docs/internal/REFERENCE.md`**.

**Setup:** `make prep` → `source .venv/bin/activate` — **`docs/SETUP.md`**. **Transfer:** **`transfer/README.md`** and `make help` / `make transfer-one`.

### 1) Environment

Python ≥3.10, NVIDIA GPU, CUDA. `pip install -r requirements.txt` in a venv; `huggingface-cli login` for gated models. Evaluations: `export OPENAI_API_KEY=...`. Details: **`docs/SETUP.md`**.

### 2) Shared CLI flags (`args.py`)

- `--rep_token/-t` (`max_attn_per_layer`)
- `--model_name/-m` — see `utils.select_llm` (e.g. `llama_3.0_8b`, `llama_3.1_8b_hf`, `llama_3.1_8b`, `llama_3.3_8b`, 70B ids, Qwen)
- `--concept_type/-c`, `--control_method/-cm`, `--version/-v`, `--label/-l`, `--only-concept` (single concept)

### 3) Pipeline steps

| Step | Script | Example |
|------|--------|---------|
| 0 | `0_visualize_attn.py` | `python 0_visualize_attn.py -t max_attn_per_layer -m llama_3.1_8b -c fears -l soft --only-concept fire` |
| 1 | `1_get_directions.py` | `python 1_get_directions.py -t max_attn_per_layer -m llama_3.1_8b -c fears -cm rfm -v 1 -l soft` |
| 2 | `2_steer.py` | `python 2_steer.py -t max_attn_per_layer -m llama_3.1_8b -c fears -cm rfm -v 1 -l soft` |
| 3 | `3_evaluate_steered_outputs.py` | needs `OPENAI_API_KEY` |
| 4 | `4_visualize_scores.py` | `python 4_visualize_scores.py` |

Outputs under `data/attention_to_prompt`, `data/directions`, `data/cached_outputs`, `data/csvs`.

### 4) Typical end-to-end (single model)

```bash
python 0_visualize_attn.py -m llama_3.1_8b -c fears
python 1_get_directions.py -m llama_3.1_8b -c fears
python 2_steer.py -m llama_3.1_8b -c fears
OPENAI_API_KEY=... python 3_evaluate_steered_outputs.py -m llama_3.1_8b -c fears
python 4_visualize_scores.py
```

### 5) Transfer (another model / version)

**`transfer/README.md`** — one-concept `make transfer-one`, multi-concept **`./transfer/run_full_pipeline_many.sh`**, **runtime estimates**, and batch / HTML eval scripts.

### 6) Notes

Custom concepts: `-c custom` + `data/concepts/custom.txt`. `run_first_five` in some scripts limits concepts for quick tests.
