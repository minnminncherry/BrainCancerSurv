#!/bin/sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

GENOMIC_DIR="$REPO_ROOT/dataset_csv/genomic_data"
LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clean_clincal_data.csv"
TYPE_OF_PATH="hallmark" # what type of pathways? 
MODEL="omics" # what type of model do you want to train? snn for model_SNNOmics.py, omics for model_Omics.py, mlp_per_path for model_MLPPerPath.py, mlp for model_MLP.py

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/main.py" \
    --data_factory SurvivalGenomicDataset \
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --type_of_pathway "$TYPE_OF_PATH" \
    --seed 42 \
    --genomic_dir "$GENOMIC_DIR" \
    --genomic_file_name "normalized_zscore_gbm.csv" \
    --modality "gen2vec" \
    --n_classes 4 \
    --loss_func "cross_entropy" \
    --opt "adam"\
    --lr 1e-3 \
    --batch_size 32 \
    --num_workers 2 \
    --epoch 10
