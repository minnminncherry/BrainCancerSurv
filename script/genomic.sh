#!/bin/sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

GENOMIC_DIR="$REPO_ROOT/dataset_csv/genomic_data"
LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clean_clinic_data.csv"
TYPE_OF_PATH="hallmark" # what type of pathways? 
MODEL="omics" # what type of model do you want to train? snn for model_SNNOmics.py, omics for model_Omics.py, mlp_per_path for model_MLPPerPath.py, mlp for model_MLP.py

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/main.py" \
    --data_factory SurvivalGenomicDataset \
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 2 \
    --type_of_pathway "$TYPE_OF_PATH" \
    --seed 42 \
    --genomic_dir "$GENOMIC_DIR" \
    --genomic_file_name "gbm.csv" \
    --modality "mlp" \
    --n_classes 2 \
    --loss_func "cross_entropy" \
    --opt "adam"\
    --lr 0.1 \
    --batch_size 32 \
    --worker 2 \
    --epoch 10 \
