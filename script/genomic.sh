#!/bin/sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

GENOMIC_DIR="$REPO_ROOT/dataset_csv/genomic_data"
LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clinical_lgg.csv"
TYPE_OF_PATH="hallmark" 

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/main.py" \
    --data_factory SurvivalGenomicDataset \
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --type_of_pathway "$TYPE_OF_PATH" \
    --seed 42 \
    --genomic_dir "$GENOMIC_DIR" \
    --genomic_file_name "normalized_zscore_lgg.csv" \
    --modality "mlp" \
    --n_classes 4 \
    --loss_func "cross_entropy" \
    --resmlp_hidden_dim 256 \
    --opt "adam"\
    --lr 1e-4 \
    --batch_size 20 \
    --num_workers 2 \
    --epoch 30
