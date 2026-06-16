SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

WSI_DIR="/media/licongcong/Data 1/TCGA-GBM/WSI_pt_folder"
LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clinical_gbm.csv"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/main.py" \
    --data_factory SurvivalWSIDataset\
    --label_file "$LABEL_FILE" \
    --label_col "CDE_survival_time" \
    --n_bins 4 \
    --type_of_pathway "$TYPE_OF_PATH" \
    --seed 42 \
    --pt_file_path "$WSI_DIR" \
    --modality "mil" \
    --n_classes 4 \
    --loss_func "cross_entropy" \
    --opt "adam"\
    --lr 1e-4 \
    --batch_size 20 \
    --num_workers 2 \
    --epoch 10