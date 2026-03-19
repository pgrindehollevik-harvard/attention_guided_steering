# Steering Vector Transfer: 8B → 70B

Plan for transferring steering vectors from small (8B) to large (70B) models via learned linear mapping.

## Goal

Learn \( W \) so that \( v_{70B} \approx W \cdot v_{8B} \) using paired activations (same prompts, both models). This extends the universal steering paper (Beaglehole et al., arxiv 2502.03708), which re-extracts on each model size.

## Dimensions

| Model | Layers | Hidden size |
|-------|--------|-------------|
| Llama 3.1 8B | 32 | 4096 |
| Llama 3.1/3.3 70B | 80 | 8192 |

## Implementation Plan

### Phase 1: Paired activation collection

**Script: `transfer/0_collect_paired_activations.py`**

- Load same prompts from existing dataset (e.g. `datasets.py` → fears/moods/etc.)
- Run forward pass on **both** 8B and 70B with `output_hidden_states=True`
- Extract last-token activations per layer (same logic as `direction_utils.get_hidden_states_and_attns`)
- Handle layer alignment: 8B has 32 layers, 70B has 80 layers
  - **Option A**: Map 8B layer \( \ell \) → 70B layer \( \lfloor \ell \cdot 80/32 \rfloor \) (or nearest)
  - **Option B**: Collect all 80 layers from 70B, fit separate \( W \) per 8B-layer → 70B-layer pair
- Save: `data/paired_activations/{concept}_{model_small}_{model_large}.npz`
  - Keys: `acts_small` (n_prompts, n_layers_small, 4096), `acts_large` (n_prompts, n_layers_large, 8192)

**Reuse:**
- `datasets.py` → `get_dataset_fn`, dataset format
- `direction_utils.get_hidden_states_and_attns` logic (or refactor to share)
- `utils.select_llm` for model loading

### Phase 2: Learn linear mapping \( W \)

**Script: `transfer/1_learn_mapping.py`**

- Load paired activations
- For each (small_layer, large_layer) pair:
  - Fit \( W \in \mathbb{R}^{8192 \times 4096} \) s.t. \( \text{acts\_large}[:, L] \approx W \cdot \text{acts\_small}[:, \ell] \)
  - Minimize \( \| X_{large} - W X_{small}^T \|^2 \) → closed form: \( W = X_{large}^T (X_{small}^T)^+ \) or ridge regression
- Save: `data/transfer_mappings/W_{model_small}_{model_large}_layer{small}_{large}.npy`

**Layer alignment strategy:**
- Start simple: 1:1 by relative depth, e.g. 8B layer 16 → 70B layer 40
- Mapping: `large_layer = round(small_layer * (n_layers_large - 1) / (n_layers_small - 1))`

### Phase 3: Transfer and evaluate

**Script: `transfer/2_steer_with_transferred.py`**

- Load 8B concept vectors from `data/directions/`
- Load learned \( W \) per layer
- Compute \( v_{70B} = W \cdot v_{8B} \), normalize
- Steer 70B using transferred vectors (reuse `generation_utils.hook_model`, `NeuralController._controlled_generate`)
- Compare to: (a) original 70B, (b) re-extracted 70B vectors (if available)

**Script: `transfer/3_evaluate_transfer.py`**

- Run evaluation prompts (from `data/evaluation_prompts/`)
- Compare steered outputs: transferred vs re-extracted vs no steering
- Option: use GPT-4o judge (like `3_evaluate_steered_outputs.py`) for automated scoring

### Phase 4: Generalization (optional)

- Fit \( W \) on one concept (e.g. "fear of fire")
- Test on other concepts: does the same \( W \) transfer other 8B vectors?
- If not, may need concept-specific or per-concept mappings

## File structure

```
transfer/
├── 0_collect_paired_activations.py
├── 1_learn_mapping.py
├── 2_steer_with_transferred.py
├── 3_evaluate_transfer.py
└── transfer_utils.py          # shared helpers (layer alignment, load/save)
data/
├── paired_activations/        # new
└── transfer_mappings/         # new
```

## Dependencies

- Same as main repo: `torch`, `transformers`, `numpy`, etc.
- 70B model requires ~40GB+ VRAM (4-bit) or multi-GPU

## CLI flags (suggested)

- `--model_small` (default: `llama_3.1_8b`)
- `--model_large` (default: `llama_3.1_70b` or `llama_3.3_70b`)
- `--concept` / `--concept_type`
- `--n_prompts` (for paired collection, default: 200)

## Order of work

1. **0_collect_paired_activations.py** — collect data
2. **transfer_utils.py** — layer alignment, load/save helpers
3. **1_learn_mapping.py** — fit \( W \)
4. **2_steer_with_transferred.py** — apply transferred vectors
5. **3_evaluate_transfer.py** — compare outputs
