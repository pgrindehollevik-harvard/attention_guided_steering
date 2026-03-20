# =============================================================================
# How this Makefile works (short)
# -----------------------------------------------------------------------------
# - Each block under "name:" is a *target*. Running `make name` executes its
#   recipe (the indented shell lines).
# - `VAR ?= value` sets a default you can override: `make transfer-merge CONCEPT=bathing`
# - `$(VAR)` substitutes into commands. ACT_SRC / W_PKL are path templates.
# - `.PHONY:` means "not a file" — always run the recipe.
# - You can still run any script directly: `python transfer/collect_paired_activations.py ...`
#   Make is optional sugar for common experiment sequences.
# =============================================================================

# Prefer project venv after `make prep` (falls back to `python` on PATH)
PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python3 2>/dev/null || echo python; fi)

VENV ?= .venv

SOURCE_MODEL ?= llama_3.0_8b
TARGET_MODEL ?= llama_3.1_8b_hf
CONCEPT_TYPE ?= fears
CONCEPT      ?= fire
MAX_PROMPTS  ?= 200
REP_TOK      ?= -1
SEED         ?= 0
DATASIZE     ?= single
RIDGE        ?= 1e-2
LABEL        ?= soft
V            ?= 1
PROMPT       ?= What is the scariest thing in the world?

ACT_SRC = data/paired_activations/acts_$(SOURCE_MODEL)_$(CONCEPT_TYPE)_$(CONCEPT).npz
ACT_TGT = data/paired_activations/acts_$(TARGET_MODEL)_$(CONCEPT_TYPE)_$(CONCEPT).npz
W_PKL   = data/transfer_mappings/W_$(SOURCE_MODEL)_to_$(TARGET_MODEL)_$(CONCEPT_TYPE)_$(CONCEPT)_W.pkl

.PHONY: help prep transfer-dirs transfer-collect-src transfer-collect-tgt transfer-collect-both transfer-merge transfer-steer transfer-official-8b-pipeline

help:
	@echo "GPU box — first time / fresh clone:"
	@echo "  make prep                       # venv + pip install + quick import check"
	@echo "  source .venv/bin/activate       # each new shell (or use module system + point PYTHON=...)"
	@echo "  huggingface-cli login           # gated Meta weights (once per account)"
	@echo ""
	@echo "Transfer (defaults SOURCE=$(SOURCE_MODEL) TARGET=$(TARGET_MODEL) CONCEPT=$(CONCEPT)):"
	@echo "  make transfer-dirs              GPU — RFM directions for SOURCE"
	@echo "  make transfer-collect-both      GPU — activations src then tgt"
	@echo "  make transfer-merge             CPU OK — ridge -> $(W_PKL)"
	@echo "  make transfer-steer             GPU — steer target"
	@echo "  make transfer-official-8b-pipeline   all of the above in order"
	@echo ""
	@echo "Override any variable on the command line, e.g.:"
	@echo "  make transfer-collect-both CONCEPT=bathing MAX_PROMPTS=50"
	@echo "Using python: $(PYTHON)"

# One-shot environment prep on a GPU machine (idempotent)
prep:
	@if [ ! -x "$(VENV)/bin/python" ]; then \
		echo "Creating $(VENV) ..."; \
		python3 -m venv "$(VENV)"; \
	fi
	$(VENV)/bin/pip install -U pip setuptools wheel
	$(VENV)/bin/pip install -r requirements.txt
	@echo ""
	@$(VENV)/bin/python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
		|| { echo "(import check failed — fix errors above)"; exit 1; }
	@$(VENV)/bin/python -c "import torchmetrics, rfm" 2>/dev/null && echo "torchmetrics + rfm OK" || echo "WARN: torchmetrics/rfm import failed"
	@echo ""
	@echo "Prep done. Next in this shell:"
	@echo "  source $(VENV)/bin/activate"
	@echo "  huggingface-cli login    # if you use gated meta-llama models"

transfer-dirs:
	$(PYTHON) 1_get_directions.py -m $(SOURCE_MODEL) -c $(CONCEPT_TYPE) -t $(REP_TOK) -cm rfm -v $(V) -l $(LABEL)

transfer-collect-src:
	$(PYTHON) transfer/collect_paired_activations.py -m $(SOURCE_MODEL) -c $(CONCEPT_TYPE) \
		--concept $(CONCEPT) --max_prompts $(MAX_PROMPTS) -t $(REP_TOK) --datasize $(DATASIZE) --seed $(SEED)

transfer-collect-tgt:
	$(PYTHON) transfer/collect_paired_activations.py -m $(TARGET_MODEL) -c $(CONCEPT_TYPE) \
		--concept $(CONCEPT) --max_prompts $(MAX_PROMPTS) -t $(REP_TOK) --datasize $(DATASIZE) --seed $(SEED)

transfer-collect-both: transfer-collect-src transfer-collect-tgt

transfer-merge:
	$(PYTHON) transfer/merge_and_fit_mapping.py \
		--src_npz $(ACT_SRC) --tgt_npz $(ACT_TGT) \
		--source_model $(SOURCE_MODEL) --target_model $(TARGET_MODEL) \
		-c $(CONCEPT_TYPE) --concept $(CONCEPT) --ridge $(RIDGE)

transfer-steer:
	$(PYTHON) transfer/steer_with_transferred.py \
		--w_pkl $(W_PKL) \
		--source_model $(SOURCE_MODEL) --target_model $(TARGET_MODEL) \
		-c $(CONCEPT_TYPE) --concept $(CONCEPT) -t $(REP_TOK) -l $(LABEL) \
		--prompt "$(PROMPT)"

transfer-official-8b-pipeline: transfer-dirs transfer-collect-both transfer-merge transfer-steer
