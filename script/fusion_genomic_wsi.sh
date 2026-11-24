#!/bin/sh
# Usage: sh fusion_genomic_wsi.sh CANCER MODEL LOSS [extra training options]
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CANCER="$(printf '%s' "${1:?Provide GBM or LGG}" | tr '[:upper:]' '[:lower:]')"
MODEL="${2:?Provide the fusion model}"
LOSS="${3:?Provide the loss function}"
shift 3
COHORT="$(printf '%s' "$CANCER" | tr '[:lower:]' '[:upper:]')"

PYTHON="${PYTHON_BIN:-python}"
LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clinical_${CANCER}.csv"
GENOMIC_DIR="$REPO_ROOT/dataset_csv/genomic_data"
GENOMIC_FILE="normalized_zscore_${CANCER}.csv"
SLIDE_FILE="$REPO_ROOT/dataset_csv/wsi_data/wsi_metadata_${CANCER}.csv"
PT_DIR="/media/licongcong/Data 1/TCGA-${COHORT}/Feature_wsi"
SLIDE_DIR="/media/licongcong/Data 1/TCGA-${COHORT}/collected_wsi"

export PYTHONUNBUFFERED=1
printf '[%s] START fusion_genomic_wsi.sh | cancer=%s | model=%s | loss=%s | GPU=%s\n' \
    "$(date -Iseconds)" "$CANCER" "$MODEL" "$LOSS" "0"
trap 'status=$?; printf "[%s] END fusion_genomic_wsi.sh | exit=%s\n" "$(date -Iseconds)" "$status"' 0

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO_ROOT/main.py" \
    --data_factory SurvivalMultimodalWSIGeneDataset \
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
    --slide_file_name "$SLIDE_FILE" \
    --pt_dir "$PT_DIR" \
    --slide_dir "$SLIDE_DIR" \
    --wsi_feature_dim 1024 \
    --opt adam \
    --lr 3e-4 \
    --reg 1e-3 \
    --encoder_dropout 0.4 \
    --resmlp_hidden_dim 128 \
    --resmlp_num_blocks 1 \
    --num_workers 0 \
    --batch_size 4 \
    --epoch 20 \
    "$@"
