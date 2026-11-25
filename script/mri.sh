#!/bin/sh
# Usage: sh mri.sh CANCER MODEL LOSS [extra training options]
# Add mutation supervision with --mutation_file PATH --mutation_genes IDH1 TP53
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
MRI_METADATA="$REPO_ROOT/dataset_csv/mri_data/${CANCER}_mri_metadata.csv"
TUMOR_VOLUME_FILE="${TUMOR_VOLUME_FILE-$REPO_ROOT/dataset_csv/mri_data/${CANCER}_tumor_volumes.csv}"
MRI_DIR="${MRI_DIR:-/media/licongcong/Data 1/TCGA-${COHORT}/MRI-images}"
MUTATION_FILE="${MUTATION_FILE-$REPO_ROOT/dataset_csv/genomic_data/filter_gene_${CANCER}_mutation.csv}"

# Require CUDA; disable only cuDNN to avoid the earlier convolution errors.
export PYTHONUNBUFFERED=1
printf '[%s] START mri.sh | cancer=%s | model=%s | loss=%s | GPU=%s\n' \
    "$(date -Iseconds)" "$CANCER" "$MODEL" "$LOSS" "${CUDA_VISIBLE_DEVICES:-0}"
trap 'status=$?; printf "[%s] END mri.sh | exit=%s\n" "$(date -Iseconds)" "$status"' 0

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" -c '
import runpy, sys, torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and your PyTorch environment.")
print("Using CUDA:", torch.cuda.get_device_name(0), flush=True)
torch.backends.cudnn.enabled = False
runpy.run_path(sys.argv.pop(1), run_name="__main__")
' "$REPO_ROOT/main.py" \
    --data_factory SurvivalMRIDataset\
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --mri_file_name "$MRI_METADATA" \
    --tumor_volume_file "$TUMOR_VOLUME_FILE" \
    --mri_dir "$MRI_DIR" \
    --seed 42 \
    --n_folds 4 \
    --modality "$MODEL" \
    --n_classes 4 \
    --loss_func "$LOSS" \
    --genomic_file_name "$CANCER" \
    --mutation_file "$MUTATION_FILE" \
    --opt "sgd"\
    --lr 5e-4 \
    --encoder_dropout 0.3 \
    --reg 5e-4 \
    --batch_size 5 \
    --num_workers 0 \
    --epoch 20 \
    --crop_size 224 224 224 \
    "$@"
