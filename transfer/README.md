## Transfer (cross-version steering)

Maps **source** RFM directions into a **target** model with per-layer linear maps \(W_\ell\) fit on paired activations. Same idea as upstream [pdavar/attention_guided_steering](https://github.com/pdavar/attention_guided_steering); scripts live in **`transfer/`**.

**Cluster / env:** `make prep`, `source .venv/bin/activate`, `huggingface-cli login` — **`docs/SETUP.md`**. For single-concept **source → target**, start from the root **`README.md`**.

---

### How long does it take? (8B-class GPU, rough)

| Phase | Per concept (order of magnitude) |
|-------|-------------------------------------|
| `0_visualize_attn` + `1_get_directions` (max-attn path) | ~3–8 min |
| `collect_paired_activations` ×2 (src + tgt, 200 prompts) | ~2–5 min |
| `merge_and_fit_mapping` | seconds |

**Build only:** multiply by number of concepts (e.g. **4 concepts → ~20–50 min**).

**After artifacts exist:** `batch_triple_compare.py` scales as **concepts × prompts × coefs** (each cell = several forward passes across two models, often **~20–60+ min** for 4 concepts × 5 YAML prompts × 4 coefs). **`evaluate_jsonl`** with GPT is ~1–3 s per row (API latency).

**One-shot script** `run_full_pipeline_many.sh` = full build for all concepts **then** triple **then** eval → expect **roughly 1–3 hours** for a multi-concept demo on one 8B GPU; use fewer concepts or `SKIP_TRIPLE=1` / `SKIP_EVAL=1` to shorten.

---

### Quick commands

| Goal | Command |
|------|---------|
| One concept, default 3.0→3.1 HF | `make transfer-one` (override `CONCEPT=…`) |
| Full build + triple + CSV/HTML for many concepts | `CONCEPTS="fire,bathing" ./transfer/run_full_pipeline_many.sh` |
| Same via Make | `make transfer-full-many PIPELINE_CONCEPTS=fire,bathing` |

Defaults in the Makefile: `SOURCE_MODEL=llama_3.0_8b`, `TARGET_MODEL=llama_3.1_8b_hf`, `REP_TOK=-1`. For **max-attn** directions, set `REP_TOK=max_attn_per_layer` in the shell script env or run the `0`/`1` steps manually (below).

---

### Pipeline (manual, matches `make transfer-one`)

| Step | What | Output |
|------|------|--------|
| 1 | `1_get_directions.py` on **source** with `--only-concept X` | `data/directions/*.pkl` |
| 2 | `collect_paired_activations.py` — **source** then **target** | `data/paired_activations/*.npz` |
| 3 | `merge_and_fit_mapping.py` | `data/transfer_mappings/*_W.pkl` |
| 4 | `steer_with_transferred.py` | generations |

**Max-attn readout:** before step 1, run `0_visualize_attn.py -m <source> -c <type> -l soft --only-concept X`. Use **`-t max_attn_per_layer`** consistently for directions and steering. Collection still uses **`-1`** today (see TODO below).

---

### Scripts

| Script | Role |
|--------|------|
| `collect_paired_activations.py` | Forward pass, save activations → `.npz` |
| `merge_and_fit_mapping.py` | Ridge \(W_\ell\) → `*_W.pkl` |
| `steer_with_transferred.py` | Single prompt, target + mapped dirs |
| `batch_steer_transferred.py` | One concept, YAML prompts → JSONL |
| `multi_concept_batch_steer.py` | Many concepts, one target load → JSONL |
| `batch_triple_compare.py` | **Baseline (target)** + **native source** + **native target** (own RFM on target) + **transfer (target)**; multi-`coef`; optional `--prompts_file` |
| `evaluate_jsonl.py` | JSONL → CSV; optional **HTML** + **GPT** judge (`OPENAI_API_KEY`). GPT uses same `data/evaluation_prompts/*` families as `3_evaluate_steered_outputs.py` — set **`--concept_type`** to match the run (fears / moods / personas / …). |
| `preview_jsonl.py` | Pretty-print JSONL → `less -R` |
| `run_full_pipeline_many.sh` | Loop concepts (0/1/collect/merge) + triple + eval |

**Activation token for \(W\):** `collect_paired_activations.py` supports **`-t -1`** (default in Make) or **`-t max_attn_per_layer`** (uses `data/attention_to_prompt/attentions_meanhead_<model>_<concept>_paired_statements.npy` from `0_visualize_attn`). `run_full_pipeline_many.sh` sets **`COLLECT_REP_TOK`** to match **`REP_TOK`** when the latter is `max_attn_per_layer`, else `-1`.

---

### Model pairs (same hidden size)

- **8B Meta gated:** `llama_3.0_8b` ↔ `llama_3.1_8b_hf`
- **8B Unsloth + hub:** `llama_3.1_8b` ↔ `llama_3.3_8b`
- **70B:** `llama_3.1_70b` ↔ `llama_3.3_70b`

---

### Examples

**Transferred steer (single prompt):**

```bash
python transfer/steer_with_transferred.py \
  --w_pkl data/transfer_mappings/W_llama_3.0_8b_to_llama_3.1_8b_hf_fears_fire_W.pkl \
  --source_model llama_3.0_8b --target_model llama_3.1_8b_hf \
  -c fears --concept fire -t max_attn_per_layer -l soft \
  --prompt "Why are some people afraid of fire?" --coef 0.7 --max_tokens 200
```

**JSONL batch + HTML report:** set `OPENAI_API_KEY`, then run `evaluate_jsonl.py --out_report …` (see script `--help`).

**Extra prompts file:** copy `data/transfer_eval_prompts_extra.example.txt` → edit → pass `PROMPTS_FILE=...` into `run_full_pipeline_many.sh` or `batch_triple_compare.py`.

Optional local notes (not tracked on GitHub): sample generations / demo bullets — see root **`.gitignore`**.
