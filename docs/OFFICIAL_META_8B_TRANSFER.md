# Official Meta 8B models on Hugging Face (transfer / steering)

## Llama 3.2 does **not** have an official 8B **text** Instruct on HF

Under [`meta-llama`](https://huggingface.co/meta-llama), **Llama 3.2** is published as **1B** and **3B** text models (plus **11B** and **90B** **Vision**). There is **no** `meta-llama/Llama-3.2-8B-Instruct` (or similar) in that family.

So you **cannot** do “3.1 8B Instruct → 3.2 8B Instruct” using **only** official `meta-llama` repos at 8B: the target checkpoint does not exist on the hub.

## Official **8B text Instruct** checkpoints that *do* exist (gated)

| `model_name` in this repo | Hugging Face repo | Notes |
|---------------------------|-------------------|--------|
| `llama_3.0_8b` | [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Llama **3.0** Instruct (not 3.2). |
| `llama_3.1_8b_hf` | [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | Same weights as 3.1, loaded from Meta (dynamic NF4). |
| `llama_3.1_8b` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Same **model**, Unsloth-quantized hub (not the `meta-llama/` org). |

Both gated Meta IDs require **`huggingface-cli login`** and accepting the license on each model card.

### 403 `GatedRepoError` / “not in the authorized list”

**Access is per model repo.** Being approved for [`Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) does **not** automatically grant [`Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). Open **each** model page while logged in, accept the license, and request access if the UI asks. Approval can take a short time.

After Meta/HF grants access:

1. Confirm: `huggingface-cli whoami` and that you can open the model **Files** tab in the browser.
2. Re-run **only** the failed collect job (e.g. `-m llama_3.0_8b`). You can keep the successful `acts_llama_3.1_8b_hf_*.npz` — no need to re-collect 3.1 unless you change prompts/seed.

## Recommended **official-only** same-scale version transfer (8B)

Use **Llama 3.0 ↔ 3.1** at 8B Instruct (same rough architecture tier for a linear map pilot):

1. **Collect** (two jobs): `-m llama_3.0_8b` and `-m llama_3.1_8b_hf` (same `--concept`, `--max_prompts`, `-t`, `--seed`, `--datasize`).
2. **Merge** with `--source_model` / `--target_model` matching those ids.
3. **Directions** on the source: run `1_get_directions.py` with the **same** `-m` as the source model id (e.g. `llama_3.0_8b`).

Always confirm `num_hidden_layers` and `hidden_size` match in both configs before trusting a full run; if Meta ever changes a tier, re-check.

## If you need **Llama 3.2** under `meta-llama` only

- **3.2 3B Instruct** exists (`meta-llama/Llama-3.2-3B-Instruct`) but is **not** 8B — the transfer code assumes **same hidden size** per layer as 3.1 8B, so it is **not** a drop-in replacement for the 8B pipeline.
- For **vision** models, hooks and shapes differ; treat as a separate project.

## See also

- `transfer/README.md` — commands for collect / merge / steer.
- `docs/RUN_TRANSFER_8B_CLUSTER.md` — cluster checklist.
