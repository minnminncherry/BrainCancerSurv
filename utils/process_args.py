import argparse


def _process_args():
    parser = argparse.ArgumentParser(description="Process command-line arguments for genomic data processing.")
    parser.add_argument("--data_factory", type=str, default="SurvivalGenomicDataset", help="Dataset factory class name.")
    parser.add_argument('--study', type=str, default="", help='Name of the study (e.g., "TCGA-GBM").')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the true label CSV file.')
    parser.add_argument('--genomic_dir', type=str, required=True, help='Directory containing genomic data files.')
    parser.add_argument('--genomic_file_name', type=str, default="gbm.csv", help='Name of the genomic data file (e.g., "gbm.csv").')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for genomic data processing.')
    parser.add_argument('--label_col', type=str, default='survival_months', help='Column name in label file that contains survival months.')
    parser.add_argument('--n_bins', type=int, default=5, help='Number of bins to discretize survival months into.')
    parser.add_argument("--n_classes", type=int, default=4, help="Number of target classes. 2 uses vital status, >2 uses discretized survival time.")
    parser.add_argument("--modality", type=str, default="mlp", help="Input modality/model family.")
    parser.add_argument("--opt", type=str, default="adam", help="Optimizer name.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--loss_func', type=str, default='', help='Declear the loss function')
    parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--encoder_dropout", type=float, default=0.25)
    parser.add_argument("--num_workers", "--worker", dest="num_workers", type=int, default=0, help="Parallel data loading workers.")
    # Support both names used in scripts.
    parser.add_argument("--type_of_pathway", "--type_of_path", dest="type_of_pathway", type=str, default="hallmark")
    return parser.parse_args()