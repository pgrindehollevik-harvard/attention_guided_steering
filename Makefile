# Transfer pipeline — override any variable: make transfer-merge CONCEPT=bathing
PYTHON       ?= python
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

.PHONY: help transfer-dirs transfer-collect-src transfer-collect-tgt transfer-collect-both transfer-merge transfer-steer transfer-official-8b-pipeline

help:
	@echo "Transfer workflow (default: SOURCE=$(SOURCE_MODEL) TARGET=$(TARGET_MODEL) CONCEPT=$(CONCEPT))"
	@echo "  make transfer-dirs              GPU — RFM directions for SOURCE (see docs/SETUP re: run_first_five)"
	@echo "  make transfer-collect-src       GPU — activations, source model"
	@echo "  make transfer-collect-tgt       GPU — activations, target model"
	@echo "  make transfer-collect-both      GPU — src then tgt (sequential)"
	@echo "  make transfer-merge             CPU OK — ridge maps -> $(W_PKL)"
	@echo "  make transfer-steer             GPU — steer target with transferred dirs"
	@echo "  make transfer-official-8b-pipeline   all of the above in order"
	@echo "Override: SOURCE_MODEL TARGET_MODEL CONCEPT_TYPE CONCEPT MAX_PROMPTS REP_TOK SEED PROMPT ..."

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
