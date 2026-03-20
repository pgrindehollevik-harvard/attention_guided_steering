# Setup (short)

## GPU machine: one-line prep, then experiments

**First time** (or new clone) at repo root:

```bash
make prep
source .venv/bin/activate
huggingface-cli login    # once, for gated Meta models
```

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
make transfer-official-8b-pipeline   # default official 8B pair + fire; see Makefile for vars
```

**Note:** `1_get_directions.py` uses `run_first_five = True` — for concept **`fire`**, use **`bathing`** or change that flag; see `docs/internal/REFERENCE.md`.

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
