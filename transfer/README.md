# Cross-version 70B transfer (Llama 3.1 ↔ 3.3)

**Idea:** Learn a per-layer linear map from paired activations on the same prompts, then map source steering directions into the target model’s space:  
`v_tgt = normalize(X.T @ v_src)` where `X` solves `A_tgt ≈ A_src @ X` (rows = prompts).

**VRAM:** Each script loads **one** 70B at a time. Run collection twice, then merge on CPU.

## Prerequisites

- Install deps from repo root: `pip install -r requirements.txt` (includes **`torchmetrics`**, required by `direction_utils` / `utils.select_llm`).
- Source **RFM directions** already extracted on the source model (e.g. `1_get_directions.py` for `llama_3.1_70b`).
- Repo root as cwd (or any cwd; scripts add repo to `sys.path`).
- Same `--concept`, `--concept_type`, `--max_prompts`, `--datasize`, `--seed` for both collection runs.

## Steps

### 1. Collect activations (two jobs / sequential runs)

```bash
python transfer/collect_paired_activations.py -m llama_3.1_70b -c fears --concept fire \
  --max_prompts 200 -t -1 --datasize single

python transfer/collect_paired_activations.py -m llama_3.3_70b -c fears --concept fire \
  --max_prompts 200 -t -1 --datasize single
```

Outputs under `data/paired_activations/` (`.npz` + `.meta.json`).  
`-t -1` = last token (simple alignment). For `max_attn_per_layer` directions, you can still collect with `-t -1` to fit `W`; the steering script loads source directions from disk using your usual `-t max_attn_per_layer`.

**Custom concepts:** use `-c custom --concept "<full prefix line>"` and `--datasize triple` if you use `triple` in `1_get_directions.py`.

### 2. Fit maps

```bash
python transfer/merge_and_fit_mapping.py \
  --src_npz data/paired_activations/acts_llama_3.1_70b_fears_fire.npz \
  --tgt_npz data/paired_activations/acts_llama_3.3_70b_fears_fire.npz \
  --source_model llama_3.1_70b --target_model llama_3.3_70b \
  -c fears --concept fire --ridge 1e-2
```

Writes `data/transfer_mappings/..._W.pkl` and a small meta `.npz`.

### 3. Steer target model with transferred directions

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

`docs/TRANSFER_PLAN.md` — background and design notes.
