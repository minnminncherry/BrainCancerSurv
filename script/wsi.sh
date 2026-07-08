SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

WSI_DIR="/media/licongcong/Data 1/TCGA-GBM/CLAM_result/patches"
LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clinical_gbm.csv"
SLIDE_FILE_PATH="$REPO_ROOT/dataset_csv/wsi_data/wsi_metadata_gbm.csv"
SLIDE_DIR="/media/licongcong/Data 1/TCGA-GBM/collected_wsi"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/main.py" \
    --data_factory SurvivalWSIDataset\
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --slide_file_name "$SLIDE_FILE_PATH" \
    --slide_dir "$SLIDE_DIR" \
    --seed 42 \
    --h5_dir "$WSI_DIR" \
    --modality "mil" \
    --encoder_model_name "resnet50" \
    --n_classes 4 \
    --loss_func "cross_entropy" \
    --opt "adam"\
    --lr 1e-4 \
    --batch_size 100 \
    --num_workers 2 \
    --epoch 10 \
    --wsi_feature_dim 224