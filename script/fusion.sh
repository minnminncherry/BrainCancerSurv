#!/bin/sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clinical_gbm.csv"
GENOMIC_DIR="$REPO_ROOT/dataset_csv/genomic_data"
SLIDE_FILE_PATH="$REPO_ROOT/dataset_csv/wsi_data/wsi_metadata_gbm.csv"
PT_DIR="/media/licongcong/Data 1/TCGA-GBM/Feature_wsi"
TYPE_OF_PATH="hallmark"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/main.py" \
    --data_factory SurvivalMultimodalDataset \
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --type_of_pathway "$TYPE_OF_PATH" \
    --seed 42 \
    --genomic_dir "$GENOMIC_DIR" \
    --genomic_file_name "normalized_zscore_gbm.csv" \
    --slide_file_name "$SLIDE_FILE_PATH" \
    --pt_dir "$PT_DIR" \
    --modality "multimodal_late_fusion" \
    --n_classes 4 \
    --loss_func "cross_entropy" \
    --opt "adam" \
    --lr 1e-4 \
    --batch_size 4 \
    --num_workers 0 \
    --epoch 10 \
    --wsi_feature_dim 1024 \
    --resmlp_hidden_dim 256
