SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clinical_lgg.csv"
MRI_METADATA_FILE_PATH="${MRI_METADATA_FILE_PATH:-$REPO_ROOT/dataset_csv/mri_data/lgg_mri_metadata.csv}"
MRI_DIR="${MRI_DIR:-/media/licongcong/Data 1/TCGA-LGG/MRI-images}"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/main.py" \
    --data_factory SurvivalMRIDataset\
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --mri_file_name "$MRI_METADATA_FILE_PATH" \
    --mri_dir "$MRI_DIR" \
    --seed 42 \
    --n_folds 4 \
    --modality "ViT" \
    --n_classes 4 \
    --loss_func "sigmoid" \
    --genomic_file_name "lgg" \
    --opt "sgd"\
    --lr 5e-4 \
    --encoder_dropout 0.3 \
    --reg 5e-4 \
    --batch_size 20 \
    --num_workers 0 \
    --epoch 20 \
    --crop_size 224 224 224
