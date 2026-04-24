import os
import pandas as pd
import numpy as np

class SurvivalGenomicDataset:
    def __init__(self, label_file, genomic_dir, genomic_file_name, seed, label_col, is_omic, n_bins, type_of_pathway):
        self.label_file = label_file
        self.genomic_dir = genomic_dir
        self.genomic_file_name = genomic_file_name
        self.seed = seed
        self.label_col = label_col
        self.n_bins = n_bins
        self.is_omic = is_omic
        self.type_of_pathway = type_of_pathway

        self.__setup_genomic_data()

        self.__setup_metadata_and_labels()

    def __setup_genomic_data(self):
        if self.is_omic:
            self.genomic_data = os.path.join(self.genomic_dir, self.genomic_file_name)

        if not os.path.exists(self.genomic_data):
            raise FileNotFoundError(f"Genomic file not found: {self.genomic_data}")
        
    def __setup_metadata_and_labels(self):
        self.label_data = pd.read_csv(self.label_file)
        if self.label_col not in self.label_data.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in label file.")

        survival_months = pd.to_numeric(self.label_data[self.label_col], errors="coerce")
        if survival_months.isna().any():
            bad_count = int(survival_months.isna().sum())
            keep_mask = ~survival_months.isna()
            # Drop rows with missing/invalid survival months so downstream splitting/training works.
            self.label_data = self.label_data.loc[keep_mask].reset_index(drop=True)
            survival_months = pd.to_numeric(self.label_data[self.label_col], errors="coerce")
            print(
                f"[SurvivalGenomicDataset] Dropped {bad_count} row(s) with NaN/invalid "
                f"'{self.label_col}' before discretizing."
            )

        survival_bins = self.__discretize_suvival_month(survival_months)

        # Use discretized bins as labels (classification-friendly), while preserving the
        # original survival months and bins in metadata for analysis/debugging.
        self.labels = survival_bins
        self.metadata = self.label_data.drop(columns=[self.label_col]).copy()
        self.metadata[self.label_col] = survival_months.values
        self.metadata[f"{self.label_col}_bin"] = survival_bins

    def __discretize_suvival_month(self, survival_months):
        bins = pd.qcut(survival_months, q=self.n_bins, labels=False, duplicates="drop")
        if getattr(bins, "isna", None) is not None and bins.isna().any():
            bad_count = int(bins.isna().sum())
            raise ValueError(
                f"Discretization produced {bad_count} NaN bin(s). "
                "This usually means there are too few unique survival values for the requested n_bins."
            )
        return bins.astype("int64").values
    
    def __return_splits(self, args, fold_indices):
        # This function can be implemented to return the appropriate splits of the dataset based on the provided indices.
        # For example, it could return training and testing datasets for cross-validation.
        train_split, scalar = self._get_split_from_df(args, split_key="train", fold_indices=fold_indices, scalar=True)
        test_split, _ = self._get_split_from_df(args, split_key="test",fold_indices=fold_indices, scalar=False)
        return train_split, test_split, scalar

    # Public API (main.py calls this name).
    def return_splits(self, args, fold_indices):
        return self.__return_splits(args, fold_indices)
    
    def _get_split_from_df(self, args, split_key, fold_indices, scalar=False):
        # This function can be implemented to extract the specified split (train/test) from the dataset based on the provided indices.
        # It can also handle any necessary preprocessing or scaling of the data if scalar=True.
        if split_key not in {"train", "test"}:
            raise ValueError("split_key must be 'train' or 'test'")

        fold_indices = np.asarray(fold_indices, dtype=np.int64)

        genomic_df = pd.read_csv(self.genomic_data)
        if genomic_df.shape[0] == 0:
            raise ValueError(f"Genomic CSV is empty: {self.genomic_data}")

        genomic_id_col = None
        for candidate in ("patient_id", "_PATIENT", "sampleID", "bcr_patient_barcode"):
            if candidate in genomic_df.columns:
                genomic_id_col = candidate
                break
        if genomic_id_col is None:
            genomic_id_col = genomic_df.columns[0]

        label_df = self.metadata.copy()
        label_df["__label"] = self.labels
        label_df["__row_index"] = np.arange(len(label_df), dtype=np.int64)

        label_id_col = None
        # Prefer `_PATIENT` because our example label CSV uses it and genomic uses `patient_id`.
        for candidate in ("patient_id", "_PATIENT", "sampleID", "bcr_patient_barcode"):
            if candidate in label_df.columns:
                label_id_col = candidate
                break
        if label_id_col is None:
            raise ValueError(
                "Cannot find an ID column in label metadata. Expected one of: "
                "patient_id, _PATIENT, sampleID, bcr_patient_barcode."
            )

        # Common TCGA case: label has `_PATIENT` (e.g., TCGA-02-0003) while genomic has `patient_id`.
        merged = label_df.merge(
            genomic_df,
            left_on=label_id_col,
            right_on=genomic_id_col,
            how="inner",
            suffixes=("_label", "_genomic"),
        )
        if merged.shape[0] == 0:
            raise ValueError(
                f"No rows matched between label '{label_id_col}' and genomic '{genomic_id_col}'."
            )

        is_test = merged["__row_index"].isin(fold_indices)
        split_df = merged[~is_test].copy() if split_key == "train" else merged[is_test].copy()

        # Feature columns: everything from genomic_df except the ID column.
        feature_cols = [c for c in genomic_df.columns if c != genomic_id_col]
        if not feature_cols:
            raise ValueError("No feature columns found in genomic CSV (only ID column present).")

        x = split_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if np.isnan(x).any():
            bad = int(np.isnan(x).sum())
            raise ValueError(
                f"Found {bad} NaN values in genomic features after numeric conversion. "
                "Please clean the genomic CSV (non-numeric values/missing entries)."
            )

        y = split_df["__label"].to_numpy(dtype=np.int64)

        fitted_scaler = None
        if scalar:
            mean = x.mean(axis=0, dtype=np.float64)
            std = x.std(axis=0, dtype=np.float64)
            std[std == 0] = 1.0
            fitted_scaler = {"mean": mean, "std": std, "feature_cols": feature_cols}
            self.scaler_ = fitted_scaler

        scaler_to_use = getattr(self, "scaler_", None)
        if scaler_to_use is not None:
            if scaler_to_use.get("feature_cols") != feature_cols:
                raise ValueError("Feature columns do not match fitted scaler feature columns.")
            x = (x - scaler_to_use["mean"].astype(np.float32)) / scaler_to_use["std"].astype(np.float32)

        return {"x": x, "y": y, "df": split_df}, fitted_scaler
