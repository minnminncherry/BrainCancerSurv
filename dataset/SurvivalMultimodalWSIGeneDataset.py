import os
import pandas as pd
import numpy as np
import torch
from dataset.SurvivalWSIDataset import SurvivalSplitWSIDataset, CLINICAL_COLUMNS


def _get_result_dir():
    return os.path.abspath(os.environ.get("RESULT_DIR") or os.path.join(os.path.dirname(__file__), "..", "result"))


class SurvivalMultimodalWSIGeneSplitDataset(SurvivalSplitWSIDataset):
    """Reuse WSI feature loading and add the patient's expression vector."""

    def __init__(self, patch_data, genomic_features, y, df, feature_dim=1024, pt_dir=None):
        super().__init__(patch_data, y, df, feature_dim, pt_dir)
        self.genomic_features = np.asarray(genomic_features, dtype=np.float32)

    def _load_pt_features(self, patch_entry):
        if isinstance(patch_entry, str):
            patch_entry = {"pt_file": patch_entry}
        if not isinstance(patch_entry, dict):
            raise TypeError(f"Unsupported patch entry type: {type(patch_entry)}")
        if not os.path.exists(patch_entry["pt_file"]):
            raise FileNotFoundError(f"WSI feature PT file not found: {patch_entry['pt_file']}")
        return super()._load_pt_features(patch_entry)

    def __getitem__(self, idx):
        features, label, time, censor, clinical = super().__getitem__(idx)
        genes = torch.from_numpy(self.genomic_features[int(idx)])
        return features, genes, label, time, censor, clinical


class SurvivalMultimodalWSIGeneDataset:
    ID_CANDIDATES = ("patient_id", "_PATIENT", "sampleID", "bcr_patient_barcode")
    CENSOR_CANDIDATES = ("censorship", "censor", "event", "status")

    def __init__(self, label_file, genomic_dir, genomic_file_name, slide_file_name,
                 pt_dir, seed, label_col, n_bins, n_classes=4, wsi_feature_dim=1024,
                 modality="multimodal_late_fusion", opt="adam", lr=1e-3,
                 batch_size=32, num_workers=0, type_of_pathway="hallmark"):
        self.label_file, self.slide_file_name = label_file, slide_file_name
        self.genomic_dir, self.genomic_file_name = genomic_dir, genomic_file_name
        self.pt_dir, self.seed, self.label_col = pt_dir, seed, label_col
        self.n_bins, self.n_classes = n_bins, int(n_classes)
        self.wsi_feature_dim = int(wsi_feature_dim)
        self.modality, self.opt, self.lr = modality, opt, lr
        self.batch_size, self.num_workers = int(batch_size), int(num_workers)
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
        self.slide_data = pd.read_csv(self.slide_file_name)

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

        def join_values(values):
            return ";".join(values.dropna().astype(str))

        slide_info = (
            self.slide_data
            .groupby("patient_id")
            .agg(
                slide_ids=("svs_filename", join_values),
                n_slides=("patient_id", "size"),
            )
            .reset_index()
        )

        self.metadata = self.metadata.merge(
            slide_info,
            left_on=self.label_id_col,
            right_on="patient_id",
            how="left",
        )
        if self.label_id_col != "patient_id":
            self.metadata = self.metadata.drop(columns=["patient_id"])

        self.metadata["slide_ids"] = self.metadata["slide_ids"].fillna("")
        self.metadata["n_slides"] = self.metadata["n_slides"].fillna(0).astype(int)
        # Finalize eligible unique patients before main.py constructs fold indices.
        self.metadata = self.metadata[
            (self.metadata["n_slides"] > 0)
            & self.metadata[self.label_id_col].isin(self.genomic_df[self.genomic_id_col])
            & self.metadata["censorship"].notna()
        ].drop_duplicates(self.label_id_col).reset_index(drop=True)
        self.labels = self.metadata["__label"].to_numpy(dtype=np.int64)
        if self.genomic_df[self.genomic_id_col].duplicated().any():
            raise ValueError("Resolve duplicate genomic patient IDs before multimodal training.")


    def __build_censorship(self, metadata):
        status_col = next((c for c in ("CDE_vital_status", "vital_status") if c in metadata.columns), None)
        if status_col is None:
            raise ValueError("Clinical vital status is required to derive censoring.")

        status = metadata[status_col].astype(str).str.strip().str.upper()
        censorship = pd.Series(np.nan, index=metadata.index, dtype="float32")
        censorship[status.isin({"LIVING", "ALIVE"})] = 1
        censorship[status.isin({"DECEASED", "DEAD"})] = 0

        if "days_to_death" in metadata.columns:
            has_death_day = pd.to_numeric(metadata["days_to_death"], errors="coerce").notna()
            censorship[has_death_day] = 0

        return censorship

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

        feature_cols = self.genomic_feature_cols
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


        split_df = split_df.drop(columns=CLINICAL_COLUMNS, errors="ignore")

        if not self.pt_dir:
            raise ValueError("SurvivalMultimodalWSIGeneDataset requires pt_dir for multimodal training.")
        first_slides = self.slide_data.drop_duplicates("patient_id").set_index("patient_id")
        patch_data = []
        for patient in split_df[self.label_id_col]:
            if patient not in first_slides.index:
                raise FileNotFoundError(f"No WSI slide metadata found for patient: {patient}")
            slide = first_slides.loc[patient]
            filename = slide.get("pt_filename")
            if pd.isna(filename):
                filename = os.path.splitext(slide["svs_filename"])[0] + ".pt"
            patch_data.append({"pt_file": os.path.join(self.pt_dir, filename)})
        return SurvivalMultimodalWSIGeneSplitDataset(
            patch_data, x, y, split_df, self.wsi_feature_dim, self.pt_dir,
        ), fitted_scaler

    def return_splits(self, args, fold_indices):
        if str(args.loss_func).lower() in {"nll_surv", "nllsurvloss", "nll_survival", "nll_loss", "lossclass"}:
            training = self.metadata.loc[~self.metadata.index.isin(fold_indices)]
            observed = training.loc[training["censorship"] == 0, self.label_col]
            _, edges = pd.qcut(observed, q=self.n_classes, retbins=True, duplicates="drop")
            if len(edges) != self.n_classes + 1:
                raise ValueError("Not enough unique observed training survival times for n_classes.")
            edges[0], edges[-1] = -np.inf, np.inf
            self.survival_bin_edges = edges
            self.labels = pd.cut(self.metadata[self.label_col], bins=edges, labels=False, include_lowest=True).to_numpy(dtype=np.int64)
            self.metadata["__label"] = self.labels
            self.metadata[f"{self.label_col}_bin"] = self.labels
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
