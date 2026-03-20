# Run the 8B transfer pipeline on a GPU cluster

This is the **operator checklist** for 8B-sized runs (e.g. `llama_3.1_8b` ↔ `llama_3.3_8b`, or **official-only** `llama_3.0_8b` ↔ `llama_3.1_8b_hf`). It extends the upstream pipeline; see **`docs/UPSTREAM.md`**.

**Official Meta 8B only:** read **`docs/OFFICIAL_META_8B_TRANSFER.md`** first — there is no `meta-llama` Llama **3.2** **8B** Instruct; use **3.0 ↔ 3.1** Instruct at 8B for a gated-only pair.

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

More cluster notes: **`docs/CLUSTER_SETUP.md`**. Transfer overview: **`transfer/README.md`**.
