#!/usr/bin/env bash
# Run per-concept transfer prereqs (visualize → directions → collect ×2 → merge),
# then batch_triple_compare + evaluate_jsonl (metrics + optional HTML/GPT).
# Runtime (8B, one GPU): order of ~1–3 h for several concepts — see transfer/README.md.
#
# Usage (repo root):
#   chmod +x transfer/run_full_pipeline_many.sh   # once
#   CONCEPTS="fire,bathing,heights" ./transfer/run_full_pipeline_many.sh
#   (comma-separated — supports multi-word names like "bad breath")
#
# Optional env (defaults shown):
#   PYTHON=python3  SOURCE_MODEL=llama_3.0_8b  TARGET_MODEL=llama_3.1_8b_hf
#   CONCEPT_TYPE=fears  LABEL=soft  REP_TOK=max_attn_per_layer
#   MAX_PROMPTS=200  DATASIZE=single  SEED=0  RIDGE=1e-2
#   TRIPLE_JSONL=data/transfer_runs/triple_full.jsonl
#   EVAL_CSV=data/transfer_runs/triple_full_eval.csv
#   EVAL_HTML=data/transfer_runs/triple_full_report.html
#   SKIP_TRIPLE=1       # only build per-concept artifacts, no triple batch
#   SKIP_EVAL=1         # no CSV/HTML
#   OPENAI_API_KEY=...  # if set and SKIP_EVAL unset, runs GPT + HTML report
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
SOURCE_MODEL="${SOURCE_MODEL:-llama_3.0_8b}"
TARGET_MODEL="${TARGET_MODEL:-llama_3.1_8b_hf}"
CONCEPT_TYPE="${CONCEPT_TYPE:-fears}"
LABEL="${LABEL:-soft}"
REP_TOK="${REP_TOK:-max_attn_per_layer}"
MAX_PROMPTS="${MAX_PROMPTS:-200}"
DATASIZE="${DATASIZE:-single}"
SEED="${SEED:-0}"
RIDGE="${RIDGE:-1e-2}"
CONCEPTS="${CONCEPTS:-fire}"

TRIPLE_JSONL="${TRIPLE_JSONL:-data/transfer_runs/triple_full.jsonl}"
EVAL_CSV="${EVAL_CSV:-data/transfer_runs/triple_full_eval.csv}"
EVAL_HTML="${EVAL_HTML:-data/transfer_runs/triple_full_report.html}"

IFS=',' read -ra CONCEPT_ARR <<< "$CONCEPTS"
for i in "${!CONCEPT_ARR[@]}"; do
  CONCEPT_ARR[i]="$(echo "${CONCEPT_ARR[i]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
done

CONC_CSV="$CONCEPTS"

echo "=== Concepts: $CONC_CSV ==="
echo "=== source=$SOURCE_MODEL target=$TARGET_MODEL rep_tok=$REP_TOK ==="

for c in "${CONCEPT_ARR[@]}"; do
  [[ -z "$c" ]] && continue
  echo ""
  echo "############################################"
  echo "### CONCEPT: $c"
  echo "############################################"

  if [[ "$REP_TOK" == "max_attn_per_layer" ]]; then
    echo "--- 0_visualize_attn (skips if .npy exists) ---"
    "$PYTHON" 0_visualize_attn.py -m "$SOURCE_MODEL" -c "$CONCEPT_TYPE" -l "$LABEL" --only-concept "$c"
  fi

  echo "--- 1_get_directions ---"
  "$PYTHON" 1_get_directions.py -m "$SOURCE_MODEL" -c "$CONCEPT_TYPE" -t "$REP_TOK" -cm rfm -v 1 -l "$LABEL" \
    --only-concept "$c"

  echo "--- collect source ---"
  "$PYTHON" transfer/collect_paired_activations.py -m "$SOURCE_MODEL" -c "$CONCEPT_TYPE" \
    --concept "$c" --max_prompts "$MAX_PROMPTS" -t -1 --datasize "$DATASIZE" --seed "$SEED"

  echo "--- collect target ---"
  "$PYTHON" transfer/collect_paired_activations.py -m "$TARGET_MODEL" -c "$CONCEPT_TYPE" \
    --concept "$c" --max_prompts "$MAX_PROMPTS" -t -1 --datasize "$DATASIZE" --seed "$SEED"

  safe="${c// /_}"
  safe="${safe//\//_}"
  SRC_NPZ="data/paired_activations/acts_${SOURCE_MODEL}_${CONCEPT_TYPE}_${safe}.npz"
  TGT_NPZ="data/paired_activations/acts_${TARGET_MODEL}_${CONCEPT_TYPE}_${safe}.npz"

  echo "--- merge ---"
  "$PYTHON" transfer/merge_and_fit_mapping.py \
    --src_npz "$SRC_NPZ" --tgt_npz "$TGT_NPZ" \
    --source_model "$SOURCE_MODEL" --target_model "$TARGET_MODEL" \
    -c "$CONCEPT_TYPE" --concept "$c" --ridge "$RIDGE"
done

if [[ -n "${SKIP_TRIPLE:-}" ]]; then
  echo "SKIP_TRIPLE set — done after per-concept build."
  exit 0
fi

echo ""
echo "=== batch_triple_compare (baseline + native + transfer) ==="
"$PYTHON" transfer/batch_triple_compare.py \
  --source_model "$SOURCE_MODEL" \
  --target_model "$TARGET_MODEL" \
  -c "$CONCEPT_TYPE" \
  -t "$REP_TOK" \
  -l "$LABEL" \
  --concepts "$CONC_CSV" \
  --coefs "${COEFS:-0.55,0.65,0.75,0.85}" \
  --versions "${VERSIONS:-1,2,3,4,5}" \
  ${PROMPTS_FILE:+--prompts_file "$PROMPTS_FILE"} \
  --overwrite \
  --out_jsonl "$TRIPLE_JSONL"

if [[ -n "${SKIP_EVAL:-}" ]]; then
  echo "SKIP_EVAL set — JSONL at $TRIPLE_JSONL"
  exit 0
fi

echo ""
echo "=== evaluate_jsonl ==="
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  "$PYTHON" transfer/evaluate_jsonl.py \
    --in_jsonl "$TRIPLE_JSONL" \
    --out_csv "$EVAL_CSV" \
    --out_report "$EVAL_HTML" \
    --concept_type "$CONCEPT_TYPE" \
    --mode both
else
  echo "(no OPENAI_API_KEY — metrics CSV only; run again with key for GPT + HTML)"
  "$PYTHON" transfer/evaluate_jsonl.py \
    --in_jsonl "$TRIPLE_JSONL" \
    --out_csv "$EVAL_CSV" \
    --mode metrics
fi

echo ""
echo "Done."
echo "  JSONL: $TRIPLE_JSONL"
echo "  CSV:   $EVAL_CSV"
[[ -n "${OPENAI_API_KEY:-}" ]] && echo "  HTML:  $EVAL_HTML"
