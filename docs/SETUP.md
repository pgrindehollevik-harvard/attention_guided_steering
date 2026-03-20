# Setup (short)

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel && pip install -r requirements.txt
huggingface-cli login   # gated Meta models
```

- **CUDA check:** `python -c "import torch; print(torch.cuda.is_available())"`
- **Cache:** optional real path `export CACHE_DIR="$HOME/hf_cache"` (never use a fake `/path/to/...`)

## Transfer experiments (few commands)

From repo root, **`make help`** lists targets.

Typical **official 8B** pair (`llama_3.0_8b` → `llama_3.1_8b_hf`), concept `fire`:

```bash
make transfer-official-8b-pipeline
```

Override variables as needed, e.g. `make transfer-collect-both CONCEPT=bathing MAX_PROMPTS=50`.

**Note:** `1_get_directions.py` still has `run_first_five = True` — for concept **`fire`**, either use **`bathing`** for a quick test or set `run_first_five = False` before `make transfer-dirs`. See `docs/internal/REFERENCE.md`.

## More detail

- **Long-form notes (upstream, VRAM, HF 403, runbooks):** `docs/internal/REFERENCE.md`
- **Transfer CLI details:** `transfer/README.md`
