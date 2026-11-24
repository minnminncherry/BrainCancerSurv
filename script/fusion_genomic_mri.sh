#!/bin/sh
# Usage: sh fusion_genomic_mri.sh CANCER MODEL LOSS [extra training options]
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CANCER="$(printf '%s' "${1:?Provide GBM, LGG, or gbm_lgg}" | tr '[:upper:]' '[:lower:]')"
MODEL="${2:?Provide the model}"
LOSS="${3:?Provide the loss function}"
shift 3
COHORT="$(printf '%s' "$CANCER" | tr '[:lower:]' '[:upper:]')"

PYTHON="${PYTHON_BIN:-python}"
LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clinical_${CANCER}.csv"
GENOMIC_DIR="$REPO_ROOT/dataset_csv/genomic_data"
GENOMIC_FILE="normalized_zscore_${CANCER}.csv"
TYPE_OF_PATH="hallmark"
MRI_METADATA="$REPO_ROOT/dataset_csv/mri_data/${CANCER}_mri_metadata.csv"
MRI_DIR="${MRI_DIR:-/media/licongcong/Data 1/TCGA-${COHORT}/MRI-images}"

export PYTHONUNBUFFERED=1
printf '[%s] START fusion_genomic_mri.sh | cancer=%s | model=%s | loss=%s | GPU=%s\n' \
    "$(date -Iseconds)" "$CANCER" "$MODEL" "$LOSS" "0"
trap 'status=$?; printf "[%s] END fusion_genomic_mri.sh | exit=%s\n" "$(date -Iseconds)" "$status"' 0

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO_ROOT/main.py" \
    --data_factory SurvivalMultimodalMRIGeneDataset \
    --label_file "$LABEL_FILE" \
    --label_col CDE_survival_time \
    --n_bins 4 \
    --n_classes 4 \
    --n_folds 5 \
    --type_of_pathway hallmark \
    --seed 42 \
    --modality "$MODEL" \
    --loss_func "$LOSS" \
    --genomic_file_name "$GENOMIC_FILE" \
    --genomic_dir "$GENOMIC_DIR" \
    --mri_file_name "$MRI_METADATA" \
    --mri_dir "$MRI_DIR" \
    --crop_size 224 224 224 \
    --opt adam \
    --lr 3e-4 \
    --reg 1e-3 \
    --encoder_dropout 0.3 \
    --resmlp_hidden_dim 32 \
    --resmlp_num_blocks 1 \
    --num_workers 0 \
    --batch_size 1 \
    --epoch 20 \
    "$@"
