SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clinical_gbm.csv"
MRI_METADATA_FILE_PATH="$REPO_ROOT/dataset_csv/mri_data/gbm_mri_metadata.csv"
MRI_DIR="/media/licongcong/Data 1/TCGA-GBM/CT-images/tcga_gbm"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/main.py" \
    --data_factory SurvivalMRIDataset\
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --mri_file_name "$MRI_METADATA_FILE_PATH" \
    --mri_dir "$MRI_DIR" \
    --seed 42 \
    --modality "ViT" \
    --n_classes 4 \
    --loss_func "cross_entropy" \
    --opt "adam"\
    --lr 5e-4 \
    --batch_size 1 \
    --num_workers 0 \
    --epoch 30 \
    --wsi_feature_dim 1024
