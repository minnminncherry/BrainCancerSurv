import argparse

def _process_args():
    parser = argparse.ArgumentParser(description="Process command-line arguments for genomic data processing.")
    parser.add_argument('--study', type=str, required=True, help='Name of the study (e.g., "TCGA-GBM").')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the true label CSV file.')
    parser.add_argument('--genomic_dir', type=str, required=True, help='Directory containing genomic data files.')
    parser.add_argument('--genomic_file_name', type=str, required=True, help='Name of the genomic data file (e.g., "cleaned_genomic_data.csv").')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for genomic data processing.')
    parser.add_argument('--label_col', type=str, default='survival_months', help='Column name in label file that contains survival months.')
    parser.add_argument('--n_bins', type=int, default=5, help='Number of bins to discretize survival months into.')
    parser.add_argument('--is_omic', action='store_true', help='Whether the genomic data is omic.')
    parser.add_argument('--type_of_pathway', type=str, default='hallmark', help='Type of pathway to use (e.g., "hallmark").')
    return parser.parse_args()

    if not (args.task == 'survival'):
        print("Task and folder do not match")
        exit()
    return args