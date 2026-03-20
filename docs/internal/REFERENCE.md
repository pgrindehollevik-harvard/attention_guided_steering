# Internal reference (long-form)

> For day-to-day use: **`docs/SETUP.md`** and **`make help`**.

## Upstream vs extensions


## Original project

- **Repository:** [github.com/pdavar/attention_guided_steering](https://github.com/pdavar/attention_guided_steering)
- **Role:** Reference implementation for attention-guided steering — visualization of attention to concept prefixes, RFM (or related) direction extraction, steering at generation time, GPT-4o evaluation, and score summaries.

The **numbered scripts** (`0_`–`4_`), `neural_controllers.py`, `datasets.py`, and `data/` are still the core upstream layout. This extension **does** touch a few shared files (not only `transfer/`) so that new models and the transfer pipeline integrate with the same `select_llm`, CLI, and docs.

## What we added (new paths)

| Path | Purpose |
|------|--------|
| **`transfer/`** | Cross-*version* steering: collect paired activations, fit per-layer maps, steer target model with mapped directions. Uses `utils`, `datasets`, `NeuralController`. |
| **`docs/internal/REFERENCE.md`** (this file) | Long-form: cluster, HF gates, transfer design. |
| **`docs/SETUP.md`** | Short setup + Makefile pointer. |
| **`transfer/README.md`** | Transfer CLI summary. |

## What we changed (upstream files, not only `transfer/`)

These files **also exist in the original repo**; they are edited here so extensions work end-to-end. Intent: stay compatible with upstream usage (same flags and scripts) while adding options.

| File | Kind of change |
|------|----------------|
| **`utils.py`** | Model loading (`select_llm`): HF cache handling, Llama rope config compatibility, `device_map` / optional GPU memory cap, **extra `model_name` values** (e.g. `llama_3.3_8b` / 70B ids), `get_coefs` / response parsing where needed for those ids. |
| **`args.py`** | Widen `--model_name` **choices** so the main pipeline scripts can pass new ids through to `utils.select_llm`. |
| **`README.md`** | Upstream credit + environment notes (cluster, VRAM) that apply to the whole repo, not only `transfer/`. |

So: **`transfer/` is the main feature area**, but **shared infrastructure** (`utils.py`, `args.py`, top-level `README.md`) is intentionally updated too — otherwise transfer and new models would not plug into the same codebase honestly.

When contributing or publishing results, **cite and link the upstream repo**; call out **both** new directories and edits to the files above as “extensions on top of pdavar/attention_guided_steering.”

---

## Cluster / venv setup


Use a **project virtualenv** so you don’t mix packages into `~/.local` or the course image in confusing ways.

## 1. Clone / update

```bash
cd ~/attention_guided_steering   # or your path
git pull origin feature/custom-steering   # or your branch
```

## 2. Create and activate venv

```bash
cd ~/attention_guided_steering
python3 -m venv .venv
source .venv/bin/activate
```

Every new shell session:

```bash
cd ~/attention_guided_steering && source .venv/bin/activate
```

Confirm:

```bash
which python    # should end in .../attention_guided_steering/.venv/bin/python
python -c "import sys; print(sys.executable)"
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This installs **torch**, **transformers**, **torchmetrics**, **xRFM** (built from the git URL in `requirements.txt`), etc.

**Note:** `pip install -r requirements.txt` can take a while and may download large CUDA wheels for PyTorch.

## 4. Auth and env vars

```bash
# Gated Llama weights (required for most models in this repo)
huggingface-cli login
# or: export HF_TOKEN="hf_..."

# Optional: HF download cache on a large filesystem (must be a REAL writable path).
# Do not copy "/path/to/hf_cache" from docs — that will error. Examples:
#   export CACHE_DIR="$HOME/hf_cache"
#   export CACHE_DIR="$HOME/scratch/hf_cache"
# Or omit CACHE_DIR entirely → defaults to ~/.cache/huggingface

# Only for GPT-4o evaluation (script 3), not for transfer/collect
export OPENAI_API_KEY="sk-..."
```

## 5. Sanity checks

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import torchmetrics; import rfm; print('torchmetrics + rfm ok')"
```

## 6. Run transfer smoke test (70B)

From repo root, with venv **activated**:

```bash
python transfer/collect_paired_activations.py -m llama_3.1_70b -c fears --concept fire --max_prompts 5 -t -1 --datasize single
```

Use `--max_prompts 5` first to verify load + forward; then raise to 200.

**If you see CUDA OOM while loading 70B on a ~22GB GPU:** weights use `device_map="auto"` by default (CPU offload). Optionally cap GPU bytes so forwards have headroom:

```bash
export STEERING_GPU_MEMORY_CAP_GB=18
```

On a **40GB+** GPU you can force all weights on GPU 0: `export STEERING_DEVICE_MAP=cuda`.

**8B (4-bit Llama):** `utils.select_llm` defaults to **`device_map="cuda"`** for any `*_8b` model id so **BitsAndBytes** does not place quantized layers on CPU (which raises `ValueError` from `quantizer_bnb_4bit`). Use `STEERING_DEVICE_MAP=auto` only if you intentionally follow Hugging Face’s 4-bit CPU/GPU offload setup.

## 7. Transfer smoke test (8B, lighter)

Uses **`llama_3.1_8b`** / **`llama_3.3_8b`** (see `transfer/README.md`). **`llama_3.3_8b`** uses **`LLAMA_33_8B_HF_REPO`** (default public [allura-forge/Llama-3.3-8B-Instruct](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct)); not `meta-llama/Meta-Llama-3-8B` (that is Llama 3.0 **base**).

```bash
python transfer/collect_paired_activations.py -m llama_3.1_8b -c fears --concept fire --max_prompts 5 -t -1 --datasize single
python transfer/collect_paired_activations.py -m llama_3.3_8b -c fears --concept fire --max_prompts 5 -t -1 --datasize single
```

## Avoid

- **`pip install -r requirements.txt`** without a venv → can install into **`~/.local`** and confuse `which python`.
- Relying on **`~/.local/bin`** for CLIs — not needed if you run **`python script.py`**; optional: `export PATH="$HOME/.local/bin:$PATH"`.

## If something still fails

Paste:

- `which python`
- `python -c "import transformers; print(transformers.__version__)"`
- The full traceback

---

## Official Meta 8B & Hugging Face gates


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

### 403 “Please enable access to public gated repositories” (fine-grained token)

**Fine-grained** API tokens must explicitly allow **gated** public repos. In [Hugging Face token settings](https://huggingface.co/settings/tokens), edit the token and enable **access to public gated repositories** (exact label may vary by HF UI). Then `huggingface-cli login` again or set `HF_TOKEN`. A **classic** read token also works for gated Meta models after you accept each model card in the browser.

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
- Cluster / 8B runbook — sections below in this file.

---

## 8B transfer runbook (cluster)


**8B cluster checklist** (e.g. `llama_3.1_8b` ↔ `llama_3.3_8b`, or official `llama_3.0_8b` ↔ `llama_3.1_8b_hf`). See **Upstream** section above.

**Official Meta 8B:** there is no `meta-llama` Llama **3.2** **8B** Instruct; use **3.0 ↔ 3.1** Instruct at 8B for a gated-only pair (see **Official Meta 8B** section above).

## 0. One-time setup (login node or first GPU job)

```bash
cd /path/to/attention_guided_steering
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

- **Hugging Face:** `huggingface-cli login` (or `export HF_TOKEN=...`) if you use a **gated** `LLAMA_33_8B_HF_REPO`.
- **Llama 3.3 8B hub id:** There is **no** official `meta-llama/Llama-3.3-8B-Instruct` on HF. By default, `llama_3.3_8b` loads **[allura-forge/Llama-3.3-8B-Instruct](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct)** (public). Override: `export LLAMA_33_8B_HF_REPO='org/model'`.  
  **Do not** point the pipeline at [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) — that is **Llama 3.0 base**, not Instruct. For gated **Meta Llama 3.0 Instruct** 8B only, you could set `LLAMA_33_8B_HF_REPO=meta-llama/Meta-Llama-3-8B-Instruct` (that is still **not** a 3.3 checkpoint).
- **Llama 3.1 8B** uses the Unsloth 4-bit checkpoint; **3.3 8B** uses **dynamic NF4** in `utils.select_llm` (needs `bitsandbytes` + GPU).

Optional on tight GPUs (usually unnecessary for 8B, but safe):

```bash
export STEERING_GPU_MEMORY_CAP_GB=18   # leave headroom for activations
```

Use **repo root** as working directory (scripts fix `cwd` where needed).

---

## 1. Choose how you align the “representation token”

| Path | Collection (`collect_paired_activations`) | Source directions (`1_get_directions`) | Steer (`steer_with_transferred`) |
|------|-------------------------------------------|------------------------------------------|----------------------------------|
| **A — Simplest** | `-t -1` | `-t -1` | `-t -1` |
| **B — Match main repo default** | still `-t -1` today* | `-t max_attn_per_layer` | `-t max_attn_per_layer` |

\*Collection does not yet implement `max_attn_per_layer` (see `transfer/README.md` TODO). Path B is fine for steering but activations for \(W\) use the **last token**, so the map is not perfectly aligned with per-layer attn tokens—OK for a first experiment, not ideal for paper-perfect parity.

**Recommendation:** Start with **Path A** (`-1` everywhere) for a clean end-to-end test; switch to **B** when you care about matching `2_steer.py` defaults.

---

## 2. Concept name and `run_first_five`

`data/concepts/fears.txt` has **`Fire`** → internally **`fire`** (lowercased).

Scripts **`0_visualize_attn.py`** and **`1_get_directions.py`** use `run_first_five = True` and only process the **first five** concepts in the file—**`fire` is not in that set**.

**Options for `fire`:**

1. Temporarily set `run_first_five = False` in **`0_visualize_attn.py`** and **`1_get_directions.py`** for those jobs (heavy: many concepts), **or**
2. For a **smoke test**, use the first concept **`bathing`** everywhere below (`--concept bathing`), **or**
3. Temporarily move **`Fire`** to the **top** of `data/concepts/fears.txt` for a one-off run (remember to revert).

---

## 3. Path A — Full 8B transfer (example: `fire`, `-t -1`)

Replace `fire` with `bathing` if you keep `run_first_five = True` and don’t want to edit files.

### 3a. Source directions (3.1 8B only)

No `0_visualize_attn` needed for `-t -1`.

```bash
source .venv/bin/activate
cd /path/to/attention_guided_steering

python 1_get_directions.py -m llama_3.1_8b -c fears -t -1 -cm rfm -v 1 -l soft
```

Confirm a file appears under `data/directions/`, e.g.  
`rfm_<concept>_tokenidx_-1_block_softlabels_llama_3.1_8b.pkl`.

### 3b. Collect paired activations (two sequential jobs)

Same flags except `-m` (good for two Slurm jobs back-to-back):

```bash
# Job 1
python transfer/collect_paired_activations.py -m llama_3.1_8b -c fears --concept fire \
  --max_prompts 200 -t -1 --datasize single --seed 0

# Job 2 (after job 1 finishes; only one 8B in VRAM at a time)
python transfer/collect_paired_activations.py -m llama_3.3_8b -c fears --concept fire \
  --max_prompts 200 -t -1 --datasize single --seed 0
```

Outputs:

- `data/paired_activations/acts_llama_3.1_8b_fears_fire.npz` (+ `.meta.json`)
- `data/paired_activations/acts_llama_3.3_8b_fears_fire.npz` (+ `.meta.json`)

### 3c. Fit maps (CPU is fine; can run on login node)

```bash
python transfer/merge_and_fit_mapping.py \
  --src_npz data/paired_activations/acts_llama_3.1_8b_fears_fire.npz \
  --tgt_npz data/paired_activations/acts_llama_3.3_8b_fears_fire.npz \
  --source_model llama_3.1_8b --target_model llama_3.3_8b \
  -c fears --concept fire --ridge 1e-2
```

Writes e.g. `data/transfer_mappings/W_llama_3.1_8b_to_llama_3.3_8b_fears_fire_W.pkl`.

### 3d. Steer the **target** (3.3 8B) with transferred directions

```bash
python transfer/steer_with_transferred.py \
  --w_pkl data/transfer_mappings/W_llama_3.1_8b_to_llama_3.3_8b_fears_fire_W.pkl \
  --source_model llama_3.1_8b --target_model llama_3.3_8b \
  -c fears --concept fire -t -1 -l soft \
  --prompt "What is the scariest thing in the world?"
```

`-t` here must match the **source** `.pkl` from step 3a (`-1` in this path).

---

## 4. Path B — `max_attn_per_layer` for directions + steer

1. Run **`0_visualize_attn.py`** with `-m llama_3.1_8b -c fears` so `data/attention_to_prompt/attentions_meanhead_llama_3.1_8b_<concept>_paired_statements.npy` exists for your concept (see §2).
2. Run **`1_get_directions.py`** with `-t max_attn_per_layer` (same `-m`/`-c`).
3. Keep **`collect_paired_activations`** as in §3b (still `-t -1` unless/until collection supports max-attn).
4. **Merge** as in §3c.
5. **`steer_with_transferred.py`** with `-t max_attn_per_layer`.

---

## 5. Slurm-style splitting (typical)

- **Job A:** `1_get_directions.py` (3.1 8B) — GPU.
- **Job B:** `collect_paired_activations.py` `-m llama_3.1_8b` — GPU.
- **Job C:** `collect_paired_activations.py` `-m llama_3.3_8b` — GPU (after B).
- **Job D:** `merge_and_fit_mapping.py` — CPU, small memory.
- **Job E:** `steer_with_transferred.py` — GPU (3.3 8B).

Share a filesystem (or stage `data/` artifacts) so NPZs and `*_W.pkl` are visible to each job.

---

## 6. Quick smoke test (few prompts)

Use `--max_prompts 5` in both collect commands; keep the same value for both models. Directions step still needs the matching concept file for that concept.

---

## 7. If something fails

| Symptom | What to check |
|--------|----------------|
| `invalid choice: 'llama_3.1_8b'` from `collect_paired_activations.py` | Cluster checkout is **behind** the branch that adds 8B flags. `git pull` (e.g. `feature/custom-steering`) and confirm `transfer/collect_paired_activations.py` lists `llama_3.1_8b` in `--model_name` choices. |
| `ValueError: Some modules are dispatched on the CPU or the disk` (BitsAndBytes / `quantizer_bnb_4bit`) | Happens when `device_map="auto"` puts **4-bit** layers on CPU. **Fix:** `git pull` — current `utils.py` defaults **8B** loads to `device_map="cuda"`. Or set `export STEERING_DEVICE_MAP=cuda` before running. Avoid `STEERING_DEVICE_MAP=auto` + `STEERING_GPU_MEMORY_CAP_GB` for 8B unless you know you need CPU offload (BNB needs a special offload path). |
| 404 / `Repository Not Found` for 3.3 8B | `git pull` — use default `LLAMA_33_8B_HF_REPO` or set it explicitly. Official `meta-llama/Llama-3.3-8B-Instruct` does not exist on HF. |
| 403 `GatedRepoError` / “not in the authorized list” for `Meta-Llama-3-8B-Instruct` | **Separate** HF gate from 3.1 Instruct. Log in on the website, open the [3.0 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) card, accept + request access; wait for approval, then `huggingface-cli login` on the cluster and re-run **only** the `llama_3.0_8b` collect. |
| 401 / gated `LLAMA_33_8B_HF_REPO` | HF token + accept license for that repo. |
| `ModuleNotFoundError: torchmetrics` | `pip install -r requirements.txt` in the venv you use for `python`. |
| `FileNotFoundError` on `.pkl` in steer | Run `1_get_directions` on **source** model; match `-t` and concept string. |
| CUDA OOM on 8B | Rare at 4-bit on one GPU; if it happens, free other jobs on the device or request a larger-GPU node. Do **not** rely on CPU offload for 4-bit without the HF doc path above. |

More detail: **`transfer/README.md`** and **`make help`**.

---

## Transfer plan & design


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
| Llama 3.0 8B Instruct (`Meta-Llama-3-8B-Instruct`) | 4096 | 32 |
| Llama 3.1 8B | 4096 | 32 |
| Llama 3.3 8B (no official `meta-llama` HF Instruct; use community or other id) | 4096 | 32 |
| Llama 3.2 text on HF | 1B / 3B only (no official 8B Instruct under `meta-llama`) | — |
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

- `llama_3.1_8b` (Unsloth 4-bit) ↔ `llama_3.3_8b` (dynamic NF4; hub id from `LLAMA_33_8B_HF_REPO`, default community **Llama 3.3 8B Instruct** mirror — Meta has no official `meta-llama/Llama-3.3-8B-Instruct` on HF). Same \(d\)=4096 and depth as 3.1 8B ⇒ same ridge pipeline as 70B.

## Deferred: 8B → 70B (later repo phase or separate doc)

- Requires \(W \in \mathbb{R}^{8192 \times 4096}\) and **depth alignment** (32 vs 80 layers).  
- Revisit after same-scale version transfer (8B or 70B) looks promising.
