## Attention-guided steering

Extension of **[pdavar/attention_guided_steering](https://github.com/pdavar/attention_guided_steering)** — same `0_`–`4_` pipeline, plus **`transfer/`** (cross-checkpoint steering) and small changes in **`utils.py`** / **`args.py`**.

**Setup:** `make prep` → `source .venv/bin/activate` — **`docs/SETUP.md`**. **`make help`** lists transfer shortcuts.

### Single-concept transfer (source model → target model)

Transfer learns a **per-layer linear map** \(W_\ell\) from **paired activations** on the same prompts, then steers the **target** using **source** RFM directions mapped through \(W_\ell\).

**Requirement:** **same hidden size** (e.g. 8B→8B, 70B→70B). IDs must match `utils.select_llm` / script `choices` (e.g. `llama_3.0_8b`, `llama_3.1_8b_hf`, `llama_3.1_8b`, `llama_3.3_8b`, 70B variants).

**One command (recommended)** — from repo root, after venv + `huggingface-cli login` if needed:

```bash
make transfer-one \
  SOURCE_MODEL=llama_3.0_8b \
  TARGET_MODEL=llama_3.1_8b_hf \
  CONCEPT_TYPE=fears \
  CONCEPT=fire \
  REP_TOK=-1
```

Override **`CONCEPT=`** (one name from `data/concepts/<CONCEPT_TYPE>.txt`), **`CONCEPT_TYPE=`**, and **`SOURCE_MODEL=` / `TARGET_MODEL=`** for other pairs. Optional: **`MAX_PROMPTS=`**, **`PROMPT=`** (test question for the final `steer_with_transferred` step).

What **`make transfer-one`** does:

1. **`1_get_directions.py`** on the **source** with **`--only-concept $(CONCEPT)`** → `data/directions/…`
2. **`transfer/collect_paired_activations.py`** on **source**, then **target** → `data/paired_activations/acts_<model>_<type>_<concept>.npz`
3. **`transfer/merge_and_fit_mapping.py`** → `data/transfer_mappings/W_<src>_to_<tgt>_<type>_<concept>_W.pkl`
4. **`transfer/steer_with_transferred.py`** → prints **target** generation with transferred steering

**Max-attention directions:** run **`0_visualize_attn.py`** on the source for that concept first, then use **`-t max_attn_per_layer`** for directions, steering, and (for consistency) prefer the same token choice when collecting — see **`transfer/README.md`** (collection still defaults to last token in places).

**After you have `*_W.pkl`:** run **`transfer/steer_with_transferred.py`** with any prompt (see **`transfer/README.md`** for flags and more model pairs).

**Batch eval / baseline vs transfer:** **`transfer/README.md`** (`batch_triple_compare.py`, `evaluate_jsonl.py`, `run_full_pipeline_many.sh`).

---

### Environment

Python ≥3.10, NVIDIA GPU, CUDA. `pip install -r requirements.txt` in a venv; **`export OPENAI_API_KEY=...`** for GPT-based eval scripts. Gated Llama: **`huggingface-cli login`**.

### Shared CLI flags (`args.py`)

- `--rep_token/-t` (e.g. `max_attn_per_layer`, `-1`)
- `--model_name/-m` — see `utils.select_llm`
- `--concept_type/-c`, `--control_method/-cm`, `--version/-v`, `--label/-l`, **`--only-concept`** (single concept)

### Original pipeline (single model, no transfer)

| Step | Script |
|------|--------|
| 0 | `0_visualize_attn.py` |
| 1 | `1_get_directions.py` |
| 2 | `2_steer.py` |
| 3 | `3_evaluate_steered_outputs.py` |
| 4 | `4_visualize_scores.py` |

```bash
python 0_visualize_attn.py -m llama_3.1_8b -c fears
python 1_get_directions.py -m llama_3.1_8b -c fears
python 2_steer.py -m llama_3.1_8b -c fears
OPENAI_API_KEY=... python 3_evaluate_steered_outputs.py -m llama_3.1_8b -c fears
python 4_visualize_scores.py
```

### Notes

Custom concepts: `-c custom` + `data/concepts/custom.txt`. `run_first_five` in some scripts limits concepts for quick tests. Multi-concept pipeline, runtimes, batch triple + eval: **`transfer/README.md`**.
