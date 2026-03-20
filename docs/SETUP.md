# Setup (short)

## GPU machine: one-line prep, then experiments

**First time** (or new clone) at repo root:

```bash
make prep
source .venv/bin/activate
huggingface-cli login    # once, for gated Meta models
```

**403 with a fine-grained token:** If login “succeeds” but `hf_hub_download` still returns **403** and says *“enable access to public gated repositories”*, open **[HF token settings](https://huggingface.co/settings/tokens)** → edit your token → turn on **access to public gated repositories** (wording may vary). Meta Llama repos are *gated*; fine-grained tokens do not include that unless you enable it. Alternatively use a **classic** read token for CLI login.

After that, use **`make help`** for transfer shortcuts, or run **any** script by hand — Make does not replace the CLI.

---

## How `make` works here

- **Makefile** = list of **targets** (names) and **recipes** (shell commands). `make prep` runs the `prep:` recipe.
- **Variables** at the top set defaults (`SOURCE_MODEL`, `CONCEPT`, …). Override per run:  
  `make transfer-merge CONCEPT=bathing MAX_PROMPTS=50`
- **`$(PYTHON)`** is chosen when Make starts: if `.venv/bin/python` exists (after `make prep`), that binary is used so transfer targets don’t accidentally use the wrong interpreter.
- **`.PHONY`** targets are not files — Make always runs them (no “already built” skip).
- **Custom workflows:** keep using full commands, e.g.  
  `python transfer/collect_paired_activations.py -m llama_3.1_70b ...`  
  Make is optional for repeated, default-heavy flows.

---

## Transfer shortcuts

```bash
make help
make transfer-one    # one CONCEPT end-to-end: dirs (--only-concept) + collect + merge + steer
```

Default: `CONCEPT=fire`, `SOURCE_MODEL=llama_3.0_8b`, `TARGET_MODEL=llama_3.1_8b_hf`, `REP_TOK=-1`. Override: `make transfer-one CONCEPT=bathing`.

`1_get_directions` is limited to that concept via **`--only-concept`** (no more `run_first_five` trap for a single run).

---

## Manual env (if you skip `make prep`)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel && pip install -r requirements.txt
```

- **CUDA:** `python -c "import torch; print(torch.cuda.is_available())"`
- **Cache:** optional `export CACHE_DIR="$HOME/hf_cache"` (real path only)

---

## More detail

- **`docs/internal/REFERENCE.md`** — VRAM, HF 403, long runbooks  
- **`transfer/README.md`** — transfer scripts overview  
- **`docs/sample_outputs.md`** — saved baseline vs steered examples (reference)  
- **`docs/colleague_steering_evidence.md`** — multi-concept JSONL demo checklist
