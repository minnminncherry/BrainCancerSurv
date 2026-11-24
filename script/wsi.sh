#!/bin/sh
# Usage: sh wsi.sh CANCER MODEL LOSS [extra training options]
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
DATA_DIR="/media/licongcong/Data 1/TCGA-${COHORT}"
[ "$CANCER" != gbm_lgg ] || DATA_DIR="/media/licongcong/Data 1/TCGA_LGG_GBM"
SLIDE_FILE="$REPO_ROOT/dataset_csv/wsi_data/wsi_metadata_${CANCER}.csv"
PT_DIR="$DATA_DIR/Feature_wsi"
SLIDE_DIR="$DATA_DIR/collected_wsi"

export PYTHONUNBUFFERED=1
printf '[%s] START wsi.sh | cancer=%s | model=%s | loss=%s | GPU=%s\n' \
    "$(date -Iseconds)" "$CANCER" "$MODEL" "$LOSS" "0"
trap 'status=$?; printf "[%s] END wsi.sh | exit=%s\n" "$(date -Iseconds)" "$status"' 0

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO_ROOT/main.py" \
    --data_factory SurvivalWSIDataset\
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --slide_file_name "$SLIDE_FILE" \
    --slide_dir "$SLIDE_DIR" \
    --seed 42 \
    --pt_dir "$PT_DIR" \
    --modality "$MODEL" \
    --n_classes 4 \
    --loss_func "$LOSS" \
    --genomic_file_name "$CANCER" \
    --opt "adam"\
    --lr 5e-4 \
    --batch_size 1 \
    --num_workers 0 \
    --epoch 30 \
    --wsi_feature_dim 1024 \
    "$@"
