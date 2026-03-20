# Cross-version transfer (Llama 3.1 ↔ 3.3)

**Upstream:** This folder is an **add-on** to the main [attention-guided steering](https://github.com/pdavar/attention_guided_steering) pipeline (`1_get_directions.py`, `2_steer.py`, `NeuralController`, etc.). It does not replace those scripts; it reuses their training prompts and direction files. Supporting changes also live in **shared** modules (e.g. **`utils.select_llm`**, **`args.py`** model list)—see **`docs/UPSTREAM.md`** for what is new vs. edited upstream files.

**Idea:** Learn a per-layer linear map from paired activations on the same prompts, then map source steering directions into the target model’s space:  
`v_tgt = normalize(X.T @ v_src)` where `X` solves `A_tgt ≈ A_src @ X` (rows = prompts).

**Sizes:** Same workflow for **70B** (`llama_3.1_70b` ↔ `llama_3.3_70b`) or **8B** (`llama_3.1_8b` ↔ `llama_3.3_8b`). Hidden size matches within each pair, so \(W_\ell \in \mathbb{R}^{d \times d}\) per layer.

**VRAM:** Each script loads **one** model at a time. Run collection twice, then merge on CPU. **8B** is the lighter pilot (~22GB L4 friendly); **70B** may need `device_map="auto"` / CPU offload (see below).

**Llama 3.3 8B weights:** Meta does **not** publish `meta-llama/Llama-3.3-8B-Instruct` on Hugging Face (that URL 404s). `utils.select_llm("llama_3.3_8b")` loads **`LLAMA_33_8B_HF_REPO`** with **dynamic NF4** (`BitsAndBytesConfig`). Default repo is public **[allura-forge/Llama-3.3-8B-Instruct](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct)** (community; review the model card). Override: `export LLAMA_33_8B_HF_REPO='your/repo'`.

**Not the same as [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B):** that hub entry is **Llama 3.0** and the **base** (not Instruct) model. This codebase expects an **Instruct** chat checkpoint (e.g. gated **`meta-llama/Meta-Llama-3-8B-Instruct`** if you point `LLAMA_33_8B_HF_REPO` there for experiments — note that is **3.0**, not 3.3).

**TODO:** Support **`max_attn_per_layer`** in collection (same token choice as `1_get_directions`) instead of only last token (`-t -1`). See `docs/TRANSFER_PLAN.md`.

**~22GB GPUs (e.g. L4):** Prefer the **8B** pair first. For **70B**, 4-bit may OOM with everything on CUDA. The repo uses **`device_map="auto"`** by default so weights can spill to CPU. Optionally: `export STEERING_GPU_MEMORY_CAP_GB=18` to reserve VRAM for activations. Large GPU only: `export STEERING_DEVICE_MAP=cuda` restores all weights on GPU 0.

## Prerequisites

- Install deps from repo root: `pip install -r requirements.txt` (includes **`torchmetrics`**, required by `direction_utils` / `utils.select_llm`).
- Source **RFM directions** already extracted on the source model (e.g. `1_get_directions.py` for `llama_3.1_70b` or `llama_3.1_8b`).
- Repo root as cwd (or any cwd; scripts add repo to `sys.path`).
- Same `--concept`, `--concept_type`, `--max_prompts`, `--datasize`, `--seed` for both collection runs.

## Steps

### 1. Collect activations (two jobs / sequential runs)

**8B (lighter):**

```bash
python transfer/collect_paired_activations.py -m llama_3.1_8b -c fears --concept fire \
  --max_prompts 200 -t -1 --datasize single

python transfer/collect_paired_activations.py -m llama_3.3_8b -c fears --concept fire \
  --max_prompts 200 -t -1 --datasize single
```

**70B:**

```bash
python transfer/collect_paired_activations.py -m llama_3.1_70b -c fears --concept fire \
  --max_prompts 200 -t -1 --datasize single

python transfer/collect_paired_activations.py -m llama_3.3_70b -c fears --concept fire \
  --max_prompts 200 -t -1 --datasize single
```

Outputs under `data/paired_activations/` (`.npz` + `.meta.json`).  
`-t -1` = last token (simple alignment). Steering still loads source directions with your usual **`max_attn_per_layer`** `.pkl` files; the **mismatch** with last-token activations for \(W\) is a known limitation until TODO above is implemented.

**Custom concepts:** use `-c custom --concept "<full prefix line>"` and `--datasize triple` if you use `triple` in `1_get_directions.py`.

### 2. Fit maps

**8B:**

```bash
python transfer/merge_and_fit_mapping.py \
  --src_npz data/paired_activations/acts_llama_3.1_8b_fears_fire.npz \
  --tgt_npz data/paired_activations/acts_llama_3.3_8b_fears_fire.npz \
  --source_model llama_3.1_8b --target_model llama_3.3_8b \
  -c fears --concept fire --ridge 1e-2
```

**70B:**

```bash
python transfer/merge_and_fit_mapping.py \
  --src_npz data/paired_activations/acts_llama_3.1_70b_fears_fire.npz \
  --tgt_npz data/paired_activations/acts_llama_3.3_70b_fears_fire.npz \
  --source_model llama_3.1_70b --target_model llama_3.3_70b \
  -c fears --concept fire --ridge 1e-2
```

Writes `data/transfer_mappings/..._W.pkl` and a small meta `.npz`.

### 3. Steer target model with transferred directions

**8B:**

```bash
python transfer/steer_with_transferred.py \
  --w_pkl data/transfer_mappings/W_llama_3.1_8b_to_llama_3.3_8b_fears_fire_W.pkl \
  --source_model llama_3.1_8b --target_model llama_3.3_8b \
  -c fears --concept fire -t max_attn_per_layer -l soft \
  --prompt "What is the scariest thing in the world?"
```

**70B:**

```bash
python transfer/steer_with_transferred.py \
  --w_pkl data/transfer_mappings/W_llama_3.1_70b_to_llama_3.3_70b_fears_fire_W.pkl \
  --source_model llama_3.1_70b --target_model llama_3.3_70b \
  -c fears --concept fire -t max_attn_per_layer -l soft \
  --prompt "What is the scariest thing in the world?"
```

## Code map

| File | Role |
|------|------|
| `datasets.training_user_contents_and_labels` | Same user messages as training data, tokenizer-agnostic |
| `transfer/collect_paired_activations.py` | One model, all steered layers, `output_hidden_states` |
| `transfer/merge_and_fit_mapping.py` | Ridge regression per layer |
| `transfer/steer_with_transferred.py` | `NeuralController` + swapped directions |

## See also

- **`docs/RUN_TRANSFER_8B_CLUSTER.md`** — step-by-step checklist for running the 8B pipeline on a GPU cluster (Slurm-style splitting, Path A vs B, `fire` vs `run_first_five`).
- **`docs/TRANSFER_PLAN.md`** — background and design notes.
