#!/bin/sh
# Usage: sh fusion_genomic_wsi_mri.sh GBM mutation_guided_cross_attention nll_surv
# MRI and WSI embedding outputs: [batch, EMBEDDING_DIM], default 64.
# Set MUTATION_FILE for cohorts whose mutation CSV is stored elsewhere.
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
CANCER="$(printf '%s' "${1:?Provide GBM or LGG}" | tr '[:upper:]' '[:lower:]')"
MODEL="${2:?Provide the fusion model}"
LOSS="${3:?Provide the loss function}"
shift 3
case "$CANCER" in gbm|lgg) ;; *) echo "Choose GBM or LGG." >&2; exit 2 ;; esac
COHORT="$(printf '%s' "$CANCER" | tr '[:lower:]' '[:upper:]')"
export PYTHONUNBUFFERED=1
printf '[%s] START fusion_genomic_wsi_mri.sh | cancer=%s | model=%s | loss=%s | GPU=%s\n' \
    "$(date -Iseconds)" "$CANCER" "$MODEL" "$LOSS" "0"
trap 'status=$?; printf "[%s] END fusion_genomic_wsi_mri.sh | exit=%s\n" "$(date -Iseconds)" "$status"' 0

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN:-python}" "$SCRIPT_DIR/train_genomic_wsi_mri.py" \
    --label_file "$REPO_ROOT/dataset_csv/metadata/clinical_${CANCER}.csv" \
    --genomic_dir "$REPO_ROOT/dataset_csv/genomic_data" \
    --genomic_file_name "normalized_zscore_${CANCER}.csv" \
    --slide_file_name "$REPO_ROOT/dataset_csv/wsi_data/wsi_metadata_${CANCER}.csv" \
    --pt_dir "${PT_DIR:-/media/licongcong/Data 1/TCGA-${COHORT}/Feature_wsi}" \
    --mri_file_name "$REPO_ROOT/dataset_csv/mri_data/${CANCER}_mri_metadata.csv" \
    --mri_dir "${MRI_DIR:-/media/licongcong/Data 1/TCGA-${COHORT}/MRI-images}" \
    --label_col CDE_survival_time --n_bins 4 --n_classes 4 --n_folds 5 \
    --modality "$MODEL" --loss_func "$LOSS" --seed 42 \
    --mutation_file "${MUTATION_FILE:-$REPO_ROOT/dataset_csv/genomic_data/filter_gene_${CANCER}_mutation.csv}" \
    --mutation_genes IDH1 TP53 EGFR PTEN \
    --embedding_dim "${EMBEDDING_DIM:-64}" \
    --wsi_feature_dim 1024 --resmlp_hidden_dim 64 --resmlp_num_blocks 1 \
    --attention_heads 4 --mri_hidden_dim 128 --mri_num_layers 2 --mri_patch_size 32 \
    --mri_mutation_weight 1.0 --wsi_mutation_weight 1.0 \
    --encoder_dropout 0.3 --reg 1e-3 --opt adam --lr 3e-4 \
    --batch_size 1 --num_workers 0 --epoch 20 --crop_size 224 224 224 \
    "$@"
