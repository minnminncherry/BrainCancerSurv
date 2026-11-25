#!/bin/sh
# Usage: sh genomic.sh CANCER MODEL LOSS [extra training options]
# Override mutation labels: MUTATION_FILE=/path/to/mutations.csv sh genomic.sh GBM resmlp nll_surv
# Without mutation supervision: MUTATION_FILE='' sh genomic.sh GBM resmlp nll_surv
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CANCER="$(printf '%s' "${1:?Provide GBM, LGG, or gbm_lgg}" | tr '[:upper:]' '[:lower:]')"
MODEL="${2:?Provide the model}"
LOSS="${3:?Provide the loss function}"
shift 3

PYTHON="${PYTHON_BIN:-python}"
LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clinical_${CANCER}.csv"
GENOMIC_DIR="$REPO_ROOT/dataset_csv/genomic_data"
GENOMIC_FILE="filter_normalized_zscore_${CANCER}.csv"
MUTATION_FILE="${MUTATION_FILE-$REPO_ROOT/dataset_csv/genomic_data/filter_gene_${CANCER}_mutation.csv}"
TYPE_OF_PATH="hallmark"

export PYTHONUNBUFFERED=1
printf '[%s] START genomic.sh | cancer=%s | model=%s | loss=%s | GPU=%s\n' \
    "$(date -Iseconds)" "$CANCER" "$MODEL" "$LOSS" "${CUDA_VISIBLE_DEVICES:-0}"
trap 'status=$?; printf "[%s] END genomic.sh | exit=%s\n" "$(date -Iseconds)" "$status"' 0

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" "$REPO_ROOT/main.py" \
    --data_factory SurvivalGenomicDataset \
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --type_of_pathway "$TYPE_OF_PATH" \
    --seed 42 \
    --genomic_dir "$GENOMIC_DIR" \
    --genomic_file_name "$GENOMIC_FILE" \
    --mutation_file "$MUTATION_FILE" \
    --modality "$MODEL" \
    --n_classes 4 \
    --loss_func "$LOSS" \
    --resmlp_hidden_dim 128 \
    --resmlp_num_blocks 5 \
    --encoder_dropout 0.4 \
    --opt "adam" \
    --lr 2e-4 \
    --batch_size 20 \
    --num_workers 2 \
    --epoch 30 \
    "$@"
