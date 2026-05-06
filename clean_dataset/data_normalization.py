import pandas as pd
import numpy as np

class DataNormalization_genomic_data:
    def __init__(self, genomic_data_path, target_column, cancer_type):
        self.genomic_data_path = genomic_data_path
        self.target_column = target_column
        self.cancer_type = cancer_type
        self.preprocessor = None
        self.genomic_data = pd.read_csv(self.genomic_data_path)
    
    def __normalize_genomic_data(self,df_pivot_genomic_data: pd.DataFrame,method: str = "zscore", eps: float = 1e-8, output_path: str = "./") -> tuple[pd.DataFrame, dict]:
        """Normalize numeric genomic features and return (normalized_df, params)."""
        numeric_df = df_pivot_genomic_data.select_dtypes(include=["number"]).copy()
        if numeric_df.shape[1] == 0:
            raise ValueError("No numeric columns found to normalize.")

        method = method.lower().strip()
        if method == "zscore":
            mean = numeric_df.mean(axis=0)
            std = numeric_df.std(axis=0, ddof=0).replace(0, np.nan).fillna(1.0)
            normalized = (numeric_df - mean) / (std + eps)
            params = {"method": method, "mean": mean, "std": std, "eps": eps}
            output_file_name = f"{output_path}/normalized_{params['mean']},{params['std']}_{self.cancer_type}.csv"
            
        elif method == "minmax":
            min_v = numeric_df.min(axis=0)
            max_v = numeric_df.max(axis=0)
            denom = (max_v - min_v).replace(0, np.nan).fillna(1.0)
            normalized = (numeric_df - min_v) / (denom + eps)
            params = {"method": method, "min": min_v, "max": max_v, "eps": eps}
            output_file_name = f"{output_path}/normalized_{params['min']},{params['max']}_{self.cancer_type}.csv"
        else:
            raise ValueError(f"Unsupported normalization method: {method}. Use 'zscore' or 'minmax'.")

        normalized_df = df_pivot_genomic_data.copy()
        normalized_df[numeric_df.columns] = normalized
        self.__save_normalized_genomic_data(normalized_df, output_file_name)
        return normalized_df, params
    
    def __save_normalized_genomic_data(self, normalize_df: pd.DataFrame, output_file_name: str):
        """Save the normalized genomic data to a CSV file."""
        normalize_df.to_csv(output_file_name, index=False)