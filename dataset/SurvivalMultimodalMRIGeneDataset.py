"""Patient-matched MRI, gene expression, and clinical survival data."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.SurvivalMRIDataset import SurvivalMRIDataset


class SurvivalMultimodalMRIGeneSplitDataset(Dataset):
    def __init__(self, df, genomic_features, load_volume):
        self.df = df.reset_index(drop=True).copy()
        self.genomic_features = np.asarray(genomic_features, dtype=np.float32)
        self.y = self.df["__label"].to_numpy(dtype=np.int64)
        self.load_volume = load_volume

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        row = self.df.iloc[idx]
        return (
            self.load_volume(row["image_path"]),
            torch.from_numpy(self.genomic_features[idx]),
            int(self.y[idx]),
            float(row["event_time"]),
            float(row["censorship"]),
            row.to_dict(),
        )


class SurvivalMultimodalMRIGeneDataset(SurvivalMRIDataset):
    """One MRI scan + gene vector per sample; split by patient, not by scan."""

    ID_CANDIDATES = ("patient_id", "_PATIENT", "sampleID", "bcr_patient_barcode")

    def __init__(self, label_file, genomic_dir, genomic_file_name, mri_file_name,
                 seed, label_col, n_bins, n_classes=4, mri_dir=None,
                 crop_size=None, modality="multimodal_mri_late_fusion",
                 opt="adam", lr=1e-3, batch_size=1, num_workers=0,
                 type_of_pathway="hallmark"):
        self.label_col, self.n_classes = label_col, int(n_classes)
        self.seed, self.n_bins = seed, n_bins
        self.mri_dir, self.crop_size = mri_dir, crop_size
        self.modality, self.type_of_pathway = modality, type_of_pathway
        self.opt, self.lr = opt, lr
        self.batch_size, self.num_workers = batch_size, num_workers

        self._load_genes(os.path.join(genomic_dir, genomic_file_name))
        clinical = self._load_clinical(label_file)
        self._match_mri(clinical, mri_file_name)

    def _load_genes(self, path):
        self.genomic_df = self._read_patients(path).set_index("patient_id").drop(columns="cohort", errors="ignore")
        if self.genomic_df.index.duplicated().any():
            raise ValueError("Genomic patient IDs must be unique.")
        self.genomic_df = self.genomic_df.apply(pd.to_numeric, errors="raise")
        self.genomic_feature_cols = self.genomic_df.columns.tolist()
        if not self.genomic_feature_cols or not np.isfinite(self.genomic_df.to_numpy(dtype=np.float32)).all():
            raise ValueError("Expected finite numeric genomic features.")

    def _load_clinical(self, label_file):
        # Clinical survival time and censoring: 0 = death, 1 = censored.
        clinical = self._read_patients(label_file)
        clinical[self.label_col] = pd.to_numeric(clinical[self.label_col], errors="coerce")
        status_col = "CDE_vital_status" if "CDE_vital_status" in clinical else "vital_status"
        clinical["censorship"] = clinical[status_col].astype(str).str.strip().str.upper().map(
            {"LIVING": 1., "ALIVE": 1., "DECEASED": 0., "DEAD": 0.}
        )
        if "days_to_death" in clinical:
            death = pd.to_numeric(clinical["days_to_death"], errors="coerce").notna()
            clinical.loc[death, "censorship"] = 0.
        clinical = clinical.loc[
            np.isfinite(clinical[self.label_col]) & (clinical[self.label_col] >= 0)
            & clinical["censorship"].notna()
        ].drop_duplicates("patient_id")

        return clinical

    def _match_mri(self, clinical, mri_file_name):
        mri = pd.read_csv(mri_file_name)
        mri["patient_id"] = mri["patient_id"].astype(str).str.strip()
        # Keep MRI scans and remove segmentation masks.
        is_mask = mri["file_name"].str.contains("GlisterBoots", case=False, na=False)
        if "is_mask" in mri:
            is_mask |= mri["is_mask"].astype(str).str.lower().isin({"1", "1.0", "true"})
        mri = mri.loc[~is_mask].copy()
        mri["image_path"] = [
            os.path.join(self.mri_dir, row.patient_id, row.file_name) if self.mri_dir
            else os.path.join(row.folder_path, row.file_name)
            for row in mri.itertuples()
        ]
        self.mri_data = mri[["patient_id", "image_path"]].drop_duplicates("image_path")
        matched_ids = self.genomic_df.index.intersection(self.mri_data["patient_id"])
        self.metadata = clinical.loc[clinical["patient_id"].isin(matched_ids),
                                     ["patient_id", self.label_col, "censorship"]].reset_index(drop=True)
        if self.metadata.empty:
            raise ValueError("No patients matched clinical, genomic, and MRI data.")
        self.mri_data = self.mri_data.loc[
            self.mri_data["patient_id"].isin(self.metadata["patient_id"])
        ].reset_index(drop=True)
        print(f"MRI–gene dataset: {len(self.metadata)} patients, {len(self.mri_data)} scans")

    @classmethod
    def _read_patients(cls, path):
        frame = pd.read_csv(path)
        id_col = next(c for c in cls.ID_CANDIDATES if c in frame)
        frame = frame.dropna(subset=[id_col]).copy()
        frame["patient_id"] = frame[id_col].astype(str).str.strip()
        return frame.drop(columns=[id_col]) if id_col != "patient_id" else frame

    def return_splits(self, args=None, fold_indices=None):
        indices = np.asarray([] if fold_indices is None else fold_indices)
        if (indices.ndim != 1 or not indices.size
                or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any() or (indices >= len(self.metadata)).any()):
            raise ValueError("Provide valid validation patient indices in fold_indices.")
        validation = self.metadata.index.isin(indices)
        train = self.metadata.loc[~validation].copy()
        val = self.metadata.loc[validation].copy()
        if train.empty:
            raise ValueError("The training split has no patients.")

        # NLL intervals use observed training deaths; validation never sets the edges.
        times = train[self.label_col]
        if str(getattr(args, "loss_func", "nll_surv")).lower() in {
            "nll_surv", "nll_survival", "nll_loss", "nllsurvloss", "lossclass",
        }:
            times = train.loc[train["censorship"] == 0, self.label_col]
        if times.empty:
            raise ValueError("No training survival times available to fit bins.")
        _, edges = pd.qcut(times, q=self.n_classes, retbins=True, duplicates="drop")
        if len(edges) != self.n_classes + 1:
            raise ValueError("Too few unique training survival times for n_classes.")
        edges[0], edges[-1] = -np.inf, np.inf
        self.survival_bin_edges = edges

        # Fit once per training patient, so patients with more scans get no extra weight.
        x_train = self.genomic_df.loc[train["patient_id"]].to_numpy(dtype=np.float64)
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
        std[std < 1e-8] = 1.
        self.scaler_ = {"mean": mean, "std": std, "feature_cols": self.genomic_feature_cols}
        splits = []
        for name, patients in (("train", train), ("validation", val)):
            frame = patients.merge(self.mri_data, on="patient_id", how="inner")
            frame["__label"] = pd.cut(frame[self.label_col], edges, labels=False).astype(int)
            frame[f"{self.label_col}_bin"] = frame["__label"]
            frame["event_time"] = frame[self.label_col]
            x = self.genomic_df.loc[frame["patient_id"]].to_numpy(dtype=np.float64)
            x = ((x - mean) / std).astype(np.float32)
            splits.append(SurvivalMultimodalMRIGeneSplitDataset(frame, x, self.load_mri_series))
            print(f"{name}: {len(patients)} patients, {len(frame)} scans")
        return splits[0], splits[1], self.scaler_
