# Evidence package: steering works (for colleagues)

Use this when you want **reproducible, multi-example** evidence—not just one manual prompt.

## What you are showing

1. **Same model, same prompts:** for each case you show **baseline** (no hook) vs **steered** (hook on), so any difference is from steering, not from a different model load (multi-concept script loads the target **once**).
2. **Many prompts:** `test_prompts.yaml` versions **1–5** for `fears` (or another `concept_type` key).
3. **Many concepts (optional):** repeat for several concepts (e.g. `fire`, `bathing`, `spiders`) so it’s not cherry-picked.

Steering is **not** required to print the concept word (e.g. “fire”) on generic fear questions; see main README / `docs/sample_outputs.md`. For lexical checks, use prompts that **mention** the concept.

---

## Prerequisite: artifacts per concept (transfer path)

For each concept you want in the bundle:

| Artifact | How |
|----------|-----|
| Source RFM directions | `1_get_directions.py` … `--only-concept <name>` (same `-m`, `-t`, `-l` as steering) |
| Attention `.npy` (if `-t max_attn_per_layer`) | `0_visualize_attn.py` … `--only-concept <name>` |
| Paired activations | `transfer/collect_paired_activations.py` (source + target) |
| `W_*_W.pkl` | `transfer/merge_and_fit_mapping.py` |

Makefile shortcut (repeat with `CONCEPT=...`): `make transfer-one` or individual targets.

---

## One command: multi-concept JSONL

From repo root (GPU, venv active):

```bash
python transfer/multi_concept_batch_steer.py \
  --source_model llama_3.0_8b --target_model llama_3.1_8b_hf \
  -c fears -t max_attn_per_layer -l soft \
  --concepts fire,bathing,heights,spiders \
  --coef 0.7 \
  --overwrite \
  --out_jsonl data/transfer_runs/colleague_demo_fears.jsonl
```

- Skips concepts **missing** directions or `W` (prints `[skip]`).
- **5 prompts × N concepts** rows (default versions 1–5).

**Browse:**

```bash
python transfer/preview_jsonl.py data/transfer_runs/colleague_demo_fears.jsonl | less -R
```

**Share:** the JSONL + this doc + pinned commit hash / branch name.

---

## Evaluate (after JSONL)

Turn generations into a **CSV** for slides / spreadsheets.

**Free metrics** (length + crude repetition; no API):

```bash
python transfer/evaluate_jsonl.py \
  --in_jsonl data/transfer_runs/colleague_demo_fears.jsonl \
  --out_csv data/transfer_runs/colleague_demo_fears_eval.csv \
  --mode metrics
```

**GPT judge** on **steered** assistant text (same rubric as `3_evaluate_steered_outputs.py` / `data/evaluation_prompts/phobia_eval_v{version}.txt`). Requires `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY=...   # or use your cluster secret mechanism
python transfer/evaluate_jsonl.py \
  --in_jsonl data/transfer_runs/colleague_demo_fears.jsonl \
  --out_csv data/transfer_runs/colleague_demo_fears_eval.csv \
  --mode both
```

`--mode both` adds columns `gpt_steered_score` (0/1) and a short `gpt_steered_raw` explanation.

**Why only `fire` in multi-concept?** Other concepts were skipped because **`rfm_<concept>_tokenidx_max_attn_per_layer_...pkl`** (and/or `W_...`) was missing. Run `0_visualize_attn` + `1_get_directions` + transfer collect/merge for each concept, or use **`--concepts fire`** until those exist.

---

## Optional: native steering (single model, no transfer)

For “does steering work **at all**?” without transfer:

1. Set **`run_first_five = False`** in `2_steer.py` (or run only the concepts you need by temporarily trimming the concepts file—careful).
2. Run `2_steer.py` with your `-m`, `-t`, `-c`, `-v`, `-l`.
3. Output pickle path is given by `utils.get_steered_output_filename(...)`.
4. `3_evaluate_steered_outputs.py` can score outputs (**requires `OPENAI_API_KEY`**).

---

## Suggested talking points for colleagues

- **Coefficient:** ~**0.65–0.8** is usually the readable band; **≥0.9** often shows **degeneration** (loops, fixation on random context)—useful as a stress test, not as “best quality.”
- **Transfer:** directions are from **source**; **target** sees **mapped** vectors via **W**; effect can be **softer** than native source steering.
- **Max-attn vs `-1`:** max-attn readout often matches the paper-style setup; `-1` is simpler and self-consistent with current activation collection.

---

## Files

| File | Role |
|------|------|
| `transfer/multi_concept_batch_steer.py` | Many concepts → one JSONL |
| `transfer/batch_steer_transferred.py` | One concept → JSONL |
| `transfer/preview_jsonl.py` | Human-readable view (`\| less -R`) |
| `transfer/evaluate_jsonl.py` | JSONL → CSV (metrics ± GPT judge) |
