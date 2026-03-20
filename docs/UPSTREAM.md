# Upstream vs. extensions

## Original project

- **Repository:** [github.com/pdavar/attention_guided_steering](https://github.com/pdavar/attention_guided_steering)
- **Role:** Reference implementation for attention-guided steering — visualization of attention to concept prefixes, RFM (or related) direction extraction, steering at generation time, GPT-4o evaluation, and score summaries.

The **numbered scripts** (`0_`–`4_`), `neural_controllers.py`, `datasets.py`, and `data/` are still the core upstream layout. This extension **does** touch a few shared files (not only `transfer/`) so that new models and the transfer pipeline integrate with the same `select_llm`, CLI, and docs.

## What we added (new paths)

| Path | Purpose |
|------|--------|
| **`transfer/`** | Cross-*version* steering: collect paired activations, fit per-layer maps, steer target model with mapped directions. Uses `utils`, `datasets`, `NeuralController`. |
| **`docs/TRANSFER_PLAN.md`**, **`docs/CLUSTER_SETUP.md`**, **`docs/UPSTREAM.md`** | Design notes, cluster/venv guidance, this file. |
| **`transfer/README.md`** | Operator guide for the transfer workflow. |

## What we changed (upstream files, not only `transfer/`)

These files **also exist in the original repo**; they are edited here so extensions work end-to-end. Intent: stay compatible with upstream usage (same flags and scripts) while adding options.

| File | Kind of change |
|------|----------------|
| **`utils.py`** | Model loading (`select_llm`): HF cache handling, Llama rope config compatibility, `device_map` / optional GPU memory cap, **extra `model_name` values** (e.g. `llama_3.3_8b` / 70B ids), `get_coefs` / response parsing where needed for those ids. |
| **`args.py`** | Widen `--model_name` **choices** so the main pipeline scripts can pass new ids through to `utils.select_llm`. |
| **`README.md`** | Upstream credit + environment notes (cluster, VRAM) that apply to the whole repo, not only `transfer/`. |

So: **`transfer/` is the main feature area**, but **shared infrastructure** (`utils.py`, `args.py`, top-level `README.md`) is intentionally updated too — otherwise transfer and new models would not plug into the same codebase honestly.

When contributing or publishing results, **cite and link the upstream repo**; call out **both** new directories and edits to the files above as “extensions on top of pdavar/attention_guided_steering.”
