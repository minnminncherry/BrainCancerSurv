SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

GENOMIC_DIR="$REPO_ROOT/dataset_csv/genomic_data"
LABEL_FILE="$REPO_ROOT/dataset_csv/metadata/clean_clinic_data.csv"
TYPE_OF_PATH="hallmark" # what type of pathways? 
MODEL="omics"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/main.py" \
  --label_file "$LABEL_FILE" \
  --genomic_dir "$GENOMIC_DIR" \
  --genomic_file_name normalized_minmax_gbm_reduce_column.csv \
  --label_col CDE_survival_time \
  --modality xgboost \
  --n_classes 4 \
  --lr 0.01
