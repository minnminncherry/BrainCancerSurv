import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def _get_result_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))


class SurvivalMultimodalSplitDataset(Dataset):
    def __init__(
        self,
        patch_data,
        genomic_features,
        y,
        df,
        feature_dim=1024,
        pt_dir=None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.y = np.asarray(y, dtype=np.int64)
        self.patch_data = patch_data
        self.genomic_features = np.asarray(genomic_features, dtype=np.float32)
        self.feature_dim = int(feature_dim)
        self.pt_dir = pt_dir

    def __len__(self):
        return len(self.y)

    def _resize_feature_dim(self, data):
        if data.dim() == 1:
            data = data.unsqueeze(0)
        current_dim = data.shape[-1]
        if current_dim == self.feature_dim:
            return data
        if current_dim > self.feature_dim:
            return data[:, : self.feature_dim]

        pad = torch.zeros(
            data.shape[0],
            self.feature_dim - current_dim,
            dtype=data.dtype,
            device=data.device,
        )
        return torch.cat([data, pad], dim=1)

    def _load_pt_features(self, patch_entry):
        if isinstance(patch_entry, dict):
            pt_file = patch_entry.get("pt_file")
        elif isinstance(patch_entry, str):
            pt_file = patch_entry
        else:
            raise TypeError(f"Unsupported patch entry type: {type(patch_entry)}")

        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"WSI feature PT file not found: {pt_file}")

        load_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            features = torch.load(pt_file, map_location=load_device, weights_only=True)
        except TypeError:
            features = torch.load(pt_file, map_location=load_device)

        if isinstance(features, dict):
            features = next(
                (value for value in features.values() if isinstance(value, torch.Tensor)),
                None,
            )

        if features is None:
            raise ValueError(f"No tensor features found in PT file: {pt_file}")

        if features.dim() > 2:
            features = features.reshape(features.shape[0], -1)

        return self._resize_feature_dim(features.float())

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(self.y):
            raise IndexError(f"Index out of range: {idx}")

        row = self.df.iloc[idx]
        label = int(self.y[idx])
        event_time = pd.to_numeric(row.get("CDE_survival_time", row.get("survival_months", 0.0)), errors="coerce")
        event_time = 0.0 if pd.isna(event_time) else float(event_time)
        censorship = pd.to_numeric(row.get("censorship", 0.0), errors="coerce")
        censorship = 0.0 if pd.isna(censorship) else float(censorship)
        clinical_data = row.to_dict()

        patch_entry = self.patch_data[idx]
        if isinstance(patch_entry, dict):
            wsi_features = self._load_pt_features(patch_entry)
        else:
            raise TypeError(f"Unsupported patch entry type: {type(patch_entry)}")

        if wsi_features is None:
            clinical_data["n_patches"] = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            wsi_features = torch.empty((0, self.feature_dim), dtype=torch.float32, device=device)
        else:
            clinical_data["n_patches"] = wsi_features.shape[0]

        omics_tensor = torch.from_numpy(self.genomic_features[idx])
        return wsi_features, omics_tensor, label, event_time, censorship, clinical_data


class SurvivalMultimodalDataset:
    ID_CANDIDATES = ("patient_id", "_PATIENT", "sampleID", "bcr_patient_barcode")
    CENSOR_CANDIDATES = ("censorship", "censor", "event", "status")

    def __init__(
        self,
        label_file,
        genomic_dir,
        genomic_file_name,
        slide_file_name,
        pt_dir,
        seed,
        label_col,
        n_bins,
        n_classes=4,
        wsi_feature_dim=1024,
        modality="multimodal_late_fusion",
        opt="adam",
        lr=1e-3,
        batch_size=32,
        num_workers=0,
        type_of_pathway="hallmark",
    ):
        self.label_file = label_file
        self.genomic_dir = genomic_dir
        self.genomic_file_name = genomic_file_name
        self.slide_file_name = slide_file_name
        self.pt_dir = pt_dir
        self.seed = seed
        self.label_col = label_col
        self.n_bins = n_bins
        self.n_classes = int(n_classes)
        self.wsi_feature_dim = int(wsi_feature_dim)
        self.modality = modality
        self.opt = opt
        self.lr = lr
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.type_of_pathway = type_of_pathway

        self.__setup_genomic_data()
        self.__setup_metadata_and_labels()

    def __setup_genomic_data(self):
        self.genomic_data = os.path.join(self.genomic_dir, self.genomic_file_name)
        if not os.path.exists(self.genomic_data):
            raise FileNotFoundError(f"Genomic file not found: {self.genomic_data}")

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

        label_values = pd.to_numeric(self.label_data[self.label_col], errors="coerce")
        keep_mask = label_values.notna()
        if (~keep_mask).any():
            self.label_data = self.label_data.loc[keep_mask].reset_index(drop=True)
            label_values = label_values.loc[keep_mask].reset_index(drop=True)

        self.labels = self.__discretize_survival_month(label_values)
        self.metadata = self.label_data.drop(columns=[self.label_col]).copy()
        self.metadata[self.label_col] = label_values.values
        self.metadata[f"{self.label_col}_bin"] = self.labels
        self.metadata["__label"] = self.labels
        self.metadata["censorship"] = self.__build_censorship(self.metadata)

        self.label_id_col = next(
            (c for c in self.ID_CANDIDATES if c in self.metadata.columns),
            None,
        )
        if self.label_id_col is None:
            raise ValueError(
                "Cannot find an ID column in label metadata. Expected one of: "
                "patient_id, _PATIENT, sampleID, bcr_patient_barcode."
            )

        self.slide_data = pd.read_csv(self.slide_file_name)
        slide_info = (
            self.slide_data
            .groupby("patient_id")
            .agg(
                svs_filename=("svs_filename", "first"),
                n_slides=("patient_id", "size"),
            )
            .reset_index()
        )
        self.metadata = self.metadata.merge(
            slide_info,
            left_on=self.label_id_col,
            right_on="patient_id",
            how="left",
        ).drop(columns=["patient_id"])

        self.metadata["slide_ids"] = self.metadata.get("slide_ids", "")
        self.metadata["n_slides"] = self.metadata["n_slides"].fillna(0).astype(int)

    def __build_censorship(self, metadata):
        status_col = next((c for c in ("CDE_vital_status", "vital_status") if c in metadata.columns), None)
        if status_col is None:
            return pd.Series(np.zeros(len(metadata), dtype=np.float32), index=metadata.index)

        status = metadata[status_col].astype(str).str.strip().str.upper()
        censorship = pd.Series(np.nan, index=metadata.index, dtype="float32")
        censorship[status.isin({"LIVING", "ALIVE"})] = 1
        censorship[status.isin({"DECEASED", "DEAD"})] = 0

        if "days_to_death" in metadata.columns:
            has_death_day = pd.to_numeric(metadata["days_to_death"], errors="coerce").notna()
            censorship[has_death_day] = 0
        if "days_to_last_followup" in metadata.columns:
            has_followup = pd.to_numeric(metadata["days_to_last_followup"], errors="coerce").notna()
            censorship[censorship.isna() & has_followup] = 1

        return censorship.fillna(0.0)

    def __discretize_survival_month(self, survival_months):
        n_quantiles = max(2, int(self.n_classes))
        bins, bin_edges = pd.qcut(
            survival_months,
            q=n_quantiles,
            labels=False,
            duplicates="drop",
            retbins=True,
        )
        self.survival_bin_edges = np.asarray(bin_edges, dtype=np.float32)
        if getattr(bins, "isna", None) is not None and bins.isna().any():
            bad_count = int(bins.isna().sum())
            raise ValueError(
                f"Discretization produced {bad_count} NaN bin(s). "
                "This usually means there are too few unique values for the requested n_classes."
            )
        return bins.astype("int64").values.ravel()

    def __return_splits(self, split_key, fold_indices=None, scalar=False):
        data = self.metadata[self.metadata["n_slides"] > 0].reset_index(drop=True)
        if fold_indices is None:
            fold_indices = []
        fold_indices = np.asarray(fold_indices, dtype=np.int64)

        is_test = data.index.isin(fold_indices)
        if split_key == "train":
            split_df = data.loc[~is_test].reset_index(drop=True)
        elif split_key == "test":
            split_df = data.loc[is_test].reset_index(drop=True)
        else:
            split_df = data.reset_index(drop=True)

        split_df = split_df.merge(
            self.genomic_df,
            left_on=self.label_id_col,
            right_on=self.genomic_id_col,
            how="inner",
            suffixes=("_label", "_genomic"),
        )

        if split_df.shape[0] == 0:
            raise ValueError(
                f"No rows matched between label '{self.label_id_col}' and genomic '{self.genomic_id_col}'."
            )

        feature_cols = [c for c in self.genomic_df.columns if c != self.genomic_id_col]
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

        patch_data = []
        if self.pt_dir:
            for _, row in split_df.iterrows():
                patient_id = row[self.label_id_col]
                sample_slides = self.slide_data[self.slide_data["patient_id"] == patient_id]
                if sample_slides.empty:
                    raise FileNotFoundError(f"No WSI slide metadata found for patient: {patient_id}")
                svs_file = sample_slides.iloc[0]["svs_filename"]
                pt_file = os.path.join(self.pt_dir, os.path.splitext(svs_file)[0] + ".pt")
                patch_data.append({"pt_file": pt_file})
        else:
            raise ValueError("SurvivalMultimodalDataset requires pt_dir for multimodal training.")

        return (
            SurvivalMultimodalSplitDataset(
                patch_data,
                x,
                y,
                split_df,
                feature_dim=self.wsi_feature_dim,
                pt_dir=self.pt_dir,
            ),
            fitted_scaler,
        )

    def return_splits(self, args, fold_indices):
        train_split, scalar = self.__return_splits("train", fold_indices, scalar=True)
        test_split, _ = self.__return_splits("test", fold_indices, scalar=False)

        result_dir = _get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        train_split.df.to_csv(os.path.join(result_dir, "train_merged_split.csv"), index=False)
        test_split.df.to_csv(os.path.join(result_dir, "test_merged_split.csv"), index=False)
        print("Done!")
        print(f"Training on {len(train_split)} samples")
        print(f"Testing on {len(test_split)} samples")
        return train_split, test_split, scalar
