# Transfer (cross-version steering)

Maps source steering directions into a target model via per-layer linear maps fit on paired activations. Upstream: [pdavar/attention_guided_steering](https://github.com/pdavar/attention_guided_steering).

## Quick run

From repo root:

```bash
make help
make transfer-official-8b-pipeline    # default: llama_3.0_8b -> llama_3.1_8b_hf, concept fire
```

Override models/concept: `make transfer-collect-both SOURCE_MODEL=llama_3.1_8b TARGET_MODEL=llama_3.3_8b CONCEPT=fire`.

## Scripts

| Script | Role |
|--------|------|
| `collect_paired_activations.py` | One model, `output_hidden_states`, save `data/paired_activations/*.npz` |
| `merge_and_fit_mapping.py` | Ridge \(W_\ell\); writes `data/transfer_mappings/*_W.pkl` |
| `steer_with_transferred.py` | Load target + mapped directions; generate |

**Prereq:** RFM `.pkl` for **source** model (`1_get_directions.py` with same `-m` as `SOURCE_MODEL`). Use the same **`-t` / `REP_TOK`** for collect, merge paths, and steer.

**TODO:** `collect_paired_activations.py` does not yet support `max_attn_per_layer` (collection uses `-t -1` unless extended).

## Model pairs (same hidden size)

- **70B:** `llama_3.1_70b` ↔ `llama_3.3_70b`
- **8B Unsloth + hub:** `llama_3.1_8b` ↔ `llama_3.3_8b` (`LLAMA_33_8B_HF_REPO` for 3.3 hub id)
- **8B official Meta (gated):** `llama_3.0_8b` ↔ `llama_3.1_8b_hf`

Long-form (VRAM, HF 403, 3.2 vs 8B, cluster): **`docs/internal/REFERENCE.md`**.
