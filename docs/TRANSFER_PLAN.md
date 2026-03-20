# Steering Vector Transfer (primary focus)

**Scope:** Transfer steering vectors between **Llama 3.1** and **Llama 3.3** at the **same parameter scale** (e.g. **70B↔70B** or **8B↔8B**; same hidden size per pair). Following feedback from Parmida: remove the dimensionality gap first and ask whether a map \(v_{\text{3.3}} \approx W\, v_{\text{3.1}}\) exists. A **8B pilot** (`llama_3.1_8b` ↔ `llama_3.3_8b`) is supported for smaller GPUs (~22GB L4); see `transfer/README.md`.
---

## Scientific question

Do concept steering directions learned on **Llama 3.1 70B-Instruct** align with those on **Llama 3.3 70B-Instruct** via a linear map in activation space?

- If **yes** (good \(W\) from paired activations → transferred vectors steer 3.3 well): cross-*version* transfer within 70B is plausible; then 8B→70B is the next hard step.
- If **no**: version / training drift may dominate; mapping across sizes may need different tools.

---

## Model facts (verify in `config` in code)

| Model | Typical hidden \(d\) | Layers |
|-------|---------------------|--------|
| Llama 3.1 8B | 4096 | 32 |
| Llama 3.3 8B | 4096 | 32 |
| Llama 3.1 70B | 8192 | 80 |
| Llama 3.3 70B | 8192 | 80 |

Same \(d\) within a size tier ⇒ per layer, \(W_\ell \in \mathbb{R}^{d \times d}\) maps activations (and directions) in the **same** space. You still need **layer index alignment** if `num_hidden_layers` ever differs; start with **same index \(\ell\)** for both if configs match.

---

## VRAM: avoid OOM with two 70Bs

Two 4-bit 70B models loaded together often need **~70–90+ GB** VRAM, which many setups do not have. Default design: **never hold both full models on GPU at once.**

### Plan: two-pass activation collection

1. **Pass A — source model only** (e.g. 3.1 70B)  
   - Load model → for each prompt: forward, `output_hidden_states=True`, extract chosen token’s hidden state per layer → append to CPU/disk buffers.  
   - Save `acts_3.1.npz` (or shard by chunk).  
   - `del model`; `torch.cuda.empty_cache()`.

2. **Pass B — target model only** (e.g. 3.3 70B)  
   - **Identical prompt list** (same strings, same chat template per model if tokenizers differ — see below).  
   - Save `acts_3.3.npz`.

3. **Offline**  
   - Align prompts by index, fit \(W_\ell\), evaluate transfer. **No** GPU needed for fitting if matrices fit in RAM (or use low-rank / chunked solves).

### Single-GPU RAM vs 70B 4-bit

A **~22GB** consumer/datacenter card (e.g. L4) often **cannot** hold the full 70B 4-bit model with `device_map="cuda"`. The repo defaults to **`device_map="auto"`** so **Accelerate** can place some layers on **CPU** (slower but runs). Optionally cap GPU use to leave headroom for activations: `export STEERING_GPU_MEMORY_CAP_GB=18`. To force the old all-on-GPU behavior on a large card: `export STEERING_DEVICE_MAP=cuda`.

### Other VRAM knobs

| Technique | Role |
|-----------|------|
| **4-bit / NF4** (already in `utils.select_llm`) | Keep inference footprint low. |
| **Batch size 1** | One prompt per forward during collection. |
| **`torch.inference_mode()`** | No autograd graph. |
| **Store float16/float32 on CPU or disk** | Activations: `(n_prompts, n_layers, d)`; prefer `float16` on disk to halve size. |
| **Optional: `device_map` CPU offload** | If one model barely fits, offload some layers to CPU (slower, saves VRAM). |
| **Subset of layers** | For debugging, collect/fit only layers \(\{1,\ldots,L\}\) you actually steer. |

### Tokenizer / template caveat

3.1 and 3.3 may use the same chat template, but **token IDs can differ** slightly. For strict pairing:

- Store **raw prompt strings** (after `apply_chat_template` for **that** model), or  
- Store **text** + apply each model’s template when collecting so each model sees the right special tokens.

Paired rows must mean “same user intent,” not necessarily identical token sequences.

### Fitting \(W\): memory on CPU

Full \(W_\ell \in \mathbb{R}^{8192 \times 8192}\) is ~268M floats/layer (~1 GB/layer in fp32). **80 layers** ⇒ large aggregate storage. Practical options for v1:

- **Low-rank \(W_\ell \approx U_\ell V_\ell^\top\)** with small rank \(r\) (e.g. 256–1024), or  
- **Ridge regression** with randomized / iterative solvers if you only need \(W v\) for known \(v\).

Document chosen approach in code comments; start with **one layer + one concept** to validate pipeline before scaling.

---

## TODO (not implemented yet)

- **`collect_paired_activations.py` + `max_attn_per_layer`:** The main pipeline selects the representation token **per layer** from attention to the concept prefix (`0_visualize_attn` → `max_attn_per_layer`). Collection currently defaults to **last token** (`-t -1`) for simplicity. **To-do:** add `-t max_attn_per_layer` (and per-model attention `.npy` inputs) so activations used to fit \(W\) match the paper/repo steering setup.

---

## Implementation status

Implemented in-repo (see **`transfer/README.md`** for commands):

| Piece | Location |
|-------|----------|
| Tokenizer-agnostic training prompts | `datasets.training_user_contents_and_labels` |
| Collect activations (one model / run) | `transfer/collect_paired_activations.py` |
| Ridge fit per layer | `transfer/merge_and_fit_mapping.py` |
| Steer target with mapped directions | `transfer/steer_with_transferred.py` |
| Paths / layer list helper | `transfer/transfer_utils.py` |

`evaluate_transfer.py` is still optional (reuse `3_evaluate_steered_outputs.py` patterns if needed).

---

## Implementation phases (repo)

### Phase 1 — `transfer/collect_paired_activations.py`

- CLI: `--model`, `--out_path`, `--concept_type`, `--max_prompts`, `--rep_token` (e.g. last token vs max-attn — mirror main pipeline).
- Loop: load **one** model (8B or 70B) → run prompts → save activations → unload.
- Run twice (3.1 then 3.3) with shared manifest (e.g. JSON list of prompt indices + text).

**Output:** e.g. `data/paired_activations/acts_llama_3.1_70b_*.npz` and `acts_llama_3.3_70b_*.npz` (or `..._8b_...` for the 8B pilot).

### Phase 2 — `transfer/merge_and_fit_mapping.py`

- Load both NPZs, assert aligned row counts.
- Per layer \(\ell\): fit \(W_\ell\) (full or low-rank) minimizing \(\|A^{(3.3)}_\ell - A^{(3.1)}_\ell W_\ell^\top\|_F\) or equivalent (define convention to match your steering hook).
- Save mappings under `data/transfer_mappings/`.

### Phase 3 — `transfer/steer_with_transferred.py`

- Load **target** model (e.g. 3.3) + native directions vs **transferred** directions: \( \tilde{v}^{(3.3)}_\ell = W_\ell v^{(3.1)}_\ell \) (normalize after apply — match existing steering code).
- Reuse `generation_utils.hook_model` / `NeuralController` patterns from `2_steer.py`.

### Phase 4 — `transfer/evaluate_transfer.py` (optional v1)

- Same eval prompts as main repo; compare GPT-4o or human checklist: native 3.3 vs transferred-from-3.1 vs baseline.

### Shared — `transfer/transfer_utils.py`

- Manifest I/O, layer indexing, dtype/shape checks, paths under `data/`.

---

## Suggested file layout

```
transfer/
├── collect_paired_activations.py
├── merge_and_fit_mapping.py
├── steer_with_transferred.py
├── evaluate_transfer.py          # optional first PR
└── transfer_utils.py
data/
├── paired_activations/
└── transfer_mappings/
```

---

## CLI sketch

- `collect_paired_activations.py`: `--model_name` includes `llama_3.1_8b | llama_3.3_8b | llama_3.1_70b | llama_3.3_70b`, optional `--manifest prompts.jsonl`, `--out_dir ...`
- `merge_and_fit_mapping.py`: `--src_npz`, `--tgt_npz`, `--out_dir`, `--rank` (optional low-rank)

---

## Order of work

1. `transfer_utils.py` + manifest format  
2. `collect_paired_activations.py` (two sequential runs)  
3. `merge_and_fit_mapping.py`  
4. `steer_with_transferred.py` (one layer / one concept smoke test)  
5. `evaluate_transfer.py` when steering looks sane  

---

## Done in-repo: 8B ↔ 8B version transfer (pilot)

- `llama_3.1_8b` (Unsloth 4-bit) ↔ `llama_3.3_8b` (gated Meta + dynamic NF4 in `utils.select_llm`). Same \(d\)=4096 and depth as 3.1 8B ⇒ same ridge pipeline as 70B.

## Deferred: 8B → 70B (later repo phase or separate doc)

- Requires \(W \in \mathbb{R}^{8192 \times 4096}\) and **depth alignment** (32 vs 80 layers).  
- Revisit after same-scale version transfer (8B or 70B) looks promising.
