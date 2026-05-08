import os
import pandas as pd
import numpy as np
import torch

class SurvivalGenomicDataset:
    ID_CANDIDATES = ("patient_id", "_PATIENT", "sampleID", "bcr_patient_barcode")
    CENSOR_CANDIDATES = ("censorship", "censor", "event", "status")

    def __init__(
        self,
        label_file,
        genomic_dir,
        genomic_file_name,
        seed,
        label_col,
        n_bins,
        type_of_pathway,
        n_classes=4,
        modality="mlp",
        opt="adam",
        lr=1e-3,
        batch_size=32,
        num_workers=0,
    ):
        self.label_file = label_file
        self.genomic_dir = genomic_dir
        self.genomic_file_name = genomic_file_name
        self.seed = seed
        self.label_col = label_col
        self.n_bins = n_bins
        self.n_classes = int(n_classes)
        self.type_of_pathway = type_of_pathway
        self.modality = modality
        self.opt = opt
        self.lr = lr
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

        self.__setup_genomic_data()

        self.__setup_metadata_and_labels()

    def __setup_genomic_data(self):
        self.genomic_data = os.path.join(self.genomic_dir, self.genomic_file_name)

        if not os.path.exists(self.genomic_data):
            raise FileNotFoundError(f"Genomic file not found: {self.genomic_data}")

        # Cache genomic table once; __getitem__ is called repeatedly.
        self.genomic_df = pd.read_csv(self.genomic_data)
        self.genomic_id_col = next(
            (c for c in self.ID_CANDIDATES if c in self.genomic_df.columns),
            self.genomic_df.columns[0],
        )
        self.genomic_feature_cols = [c for c in self.genomic_df.columns if c != self.genomic_id_col]
        if not self.genomic_feature_cols:
            raise ValueError("No feature columns found in genomic CSV (only ID column present).")
        
    def __setup_metadata_and_labels(self):
        self.label_data = pd.read_csv(self.label_file)
        if self.label_col not in self.label_data.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in label file.")

        # Simple single path: always use `label_col` and discretize into `n_classes`.
        label_values = pd.to_numeric(self.label_data[self.label_col], errors="coerce")
        keep_mask = label_values.notna()
        if (~keep_mask).any():
            self.label_data = self.label_data.loc[keep_mask].reset_index(drop=True)
            label_values = label_values.loc[keep_mask].reset_index(drop=True)

        class_bins = self.__discretize_suvival_month(label_values)
        self.labels = class_bins
        self.metadata = self.label_data.drop(columns=[self.label_col]).copy()
        self.metadata[self.label_col] = label_values.values
        self.metadata[f"{self.label_col}_bin"] = class_bins

    def __discretize_suvival_month(self, survival_months):
        n_quantiles = max(2, int(self.n_classes))
        # qcut function = Split data so each group has the SAME number of samples
        # Bin 0: [10, 20]     -> 10-20 months
        # Bin 1: [30, 40]    -> 30-40 months
        # Bin 2: [50, 60]    -> 50-60 months        
        # Bin 3: [70, 80]    -> 70-80 months
        bins = pd.qcut(survival_months, q=n_quantiles, labels=False, duplicates="drop")
        if getattr(bins, "isna", None) is not None and bins.isna().any():
            bad_count = int(bins.isna().sum())
            raise ValueError(
                f"Discretization produced {bad_count} NaN bin(s). "
                "This usually means there are too few unique values for the requested n_classes."
            )
        return bins.astype("int64").values
    
    def __return_splits(self, args, fold_indices):
        # This function can be implemented to return the appropriate splits of the dataset based on the provided indices.
        # For example, it could return training and testing datasets for cross-validation.
        train_split, scalar = self._get_split_from_df(args, split_key="train", fold_indices=fold_indices, scalar=True)
        test_split, _ = self._get_split_from_df(args, split_key="test",fold_indices=fold_indices, scalar=False)

        # Save merged split data under project `result/` folder.
        result_dir = os.path.join(os.getcwd(), "../result")
        os.makedirs(result_dir, exist_ok=True)
        train_split["df"].to_csv(os.path.join(result_dir, "train_merged_split.csv"), index=False)
        test_split["df"].to_csv(os.path.join(result_dir, "test_merged_split.csv"), index=False)
        print('Done!')
        print("Training on {} samples".format(len(train_split["df"])))
        print("Testing on {} samples".format(len(test_split["df"])))
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

        # Keep only essential columns before merge to avoid carrying unnecessary metadata.
        feature_cols = [c for c in genomic_df.columns if c != genomic_id_col]
        if not feature_cols:
            raise ValueError("No feature columns found in genomic CSV (only ID column present).")
        genomic_min = genomic_df[[genomic_id_col] + feature_cols].copy()

        # Minimal label-side fields used for split/target.
        label_keep_cols = [label_id_col, "__label", "__row_index"]
        if self.label_col in label_df.columns and self.label_col not in label_keep_cols:
            label_keep_cols.append(self.label_col)
        label_bin_col = f"{self.label_col}_bin"
        if label_bin_col in label_df.columns and label_bin_col not in label_keep_cols:
            label_keep_cols.append(label_bin_col)
        label_min = label_df[label_keep_cols].copy()

        merged = label_min.merge(
            genomic_min,
            left_on=label_id_col,
            right_on=genomic_id_col,
            how="inner",
            suffixes=("_label", "_genomic"),
        )

        if merged.shape[0] == 0:
            raise ValueError(
                f"No rows matched between label '{label_id_col}' and genomic '{genomic_id_col}'."
            )
        # save the merged dataframe to a csv file in the result folder
        result_dir = os.path.join(os.getcwd(), "../result")
        os.makedirs(result_dir, exist_ok=True)
        pd.DataFrame(merged).to_csv(os.path.join(result_dir, "merge_df.csv"), index=False)

        # Drop duplicated right-side ID after merge if different names were used.
        if genomic_id_col != label_id_col and genomic_id_col in merged.columns:
            merged = merged.drop(columns=[genomic_id_col])

        is_test = merged["__row_index"].isin(fold_indices)
        split_df = merged[~is_test].copy() if split_key == "train" else merged[is_test].copy()
        
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
            # normalization of the data
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
    
    def data_return_item(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError(f"Index out of range: {idx}")

        row = self.metadata.iloc[idx]
        label = int(self.labels[idx])
        event_source_col = "CDE_survival_time" if "CDE_survival_time" in self.metadata.columns else self.label_col
        event_time = float(pd.to_numeric(row[event_source_col], errors="coerce")) if event_source_col in self.metadata.columns else 0.0

        id_col = next((c for c in self.ID_CANDIDATES if c in self.metadata.columns), None)
        if id_col is None:
            raise ValueError("No patient ID column found in metadata.")
        case_id = row[id_col]

        df_small = self.genomic_df[self.genomic_df[self.genomic_id_col] == case_id]
        if df_small.empty:
            raise ValueError(f"No genomic row found for case_id '{case_id}'.")

        feature_df = df_small[self.genomic_feature_cols].reindex(sorted(self.genomic_feature_cols), axis=1)
        feature_values = feature_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if np.isnan(feature_values).any():
            raise ValueError(f"NaN values found in genomic features for case_id '{case_id}'.")

        c_col = next((c_name for c_name in self.CENSOR_CANDIDATES if c_name in self.metadata.columns), None)
        c = float(row[c_col]) if c_col is not None else 0.0

        omics_tensor = torch.from_numpy(feature_values[0])
        clinical_data = row.to_dict()
        return (torch.zeros((1, 1), dtype=torch.float32), omics_tensor, label, event_time, c, clinical_data)

    def __getitem__(self, idx):
        return self.data_return_item(idx)
