"""Match clinical outcomes, expression, WSI features, and MRI by patient ID."""
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.SurvivalMultimodalMRIGeneDataset import SurvivalMultimodalMRIGeneDataset


class SurvivalMultimodalWSIMRIGeneSplitDataset(Dataset):
    def __init__(self, mri_split, load_wsi):
        self.mri_split = mri_split
        self.df, self.y = mri_split.df, mri_split.y
        self.load_wsi = load_wsi

    def __len__(self):
        return len(self.mri_split)

    def __getitem__(self, idx):
        mri, genes, label, time, censor, clinical = self.mri_split[idx]
        wsi = self.load_wsi(clinical["patient_id"])
        return mri, wsi, genes, label, time, censor, clinical


class SurvivalMultimodalWSIMRIGeneDataset(SurvivalMultimodalMRIGeneDataset):
    """Reuse MRI loading, patient splits, survival bins, and train-only scaling."""

    def __init__(self, slide_file_name, pt_dir, wsi_feature_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.wsi_feature_dim = int(wsi_feature_dim)
        slides = pd.read_csv(slide_file_name).dropna(subset=["patient_id"])
        slides["patient_id"] = slides["patient_id"].astype(str).str.strip()
        if "pt_filename" in slides:
            names = slides["pt_filename"].copy()
        else:
            names = pd.Series(index=slides.index, dtype=object)
        if "svs_filename" in slides:
            names = names.fillna(slides["svs_filename"].map(
                lambda value: os.path.splitext(str(value))[0] + ".pt"
            ))
        if names.isna().any():
            raise ValueError("WSI metadata needs pt_filename or svs_filename.")
        slides["pt_path"] = names.map(lambda name: os.path.join(pt_dir, str(name)))
        # Match before fold indices are created; keep only patients with all modalities.
        self.wsi_paths = slides.groupby("patient_id")["pt_path"].agg(lambda paths: list(dict.fromkeys(paths)))
        self.metadata = self.metadata.loc[
            self.metadata.patient_id.isin(self.wsi_paths.index)
        ].reset_index(drop=True)
        if self.metadata.empty:
            raise ValueError("No patients have matching clinical, genomic, WSI, and MRI data.")
        self.mri_data = self.mri_data.loc[
            self.mri_data.patient_id.isin(self.metadata.patient_id)
        ].reset_index(drop=True)
        print(f"Gene–WSI–MRI: {len(self.metadata)} patients, {len(self.mri_data)} scans")

    def load_wsi(self, patient_id):
        bags = []
        for path in self.wsi_paths[patient_id]:
            features = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(features, dict):
                features = features.get("features")
            if not isinstance(features, torch.Tensor) or features.ndim != 2:
                raise ValueError(f"Expected a WSI feature tensor [patches, features]: {path}")
            if features.size(0) == 0 or features.size(1) != self.wsi_feature_dim:
                raise ValueError(f"Empty WSI bag or incorrect WSI feature dimension: {path}")
            if not torch.isfinite(features).all():
                raise ValueError(f"Nonfinite WSI features: {path}")
            bags.append(features.float())
        return torch.cat(bags, dim=0)

    def return_splits(self, args=None, fold_indices=None):
        train, val, scaler = super().return_splits(args, fold_indices)
        return (SurvivalMultimodalWSIMRIGeneSplitDataset(train, self.load_wsi),
                SurvivalMultimodalWSIMRIGeneSplitDataset(val, self.load_wsi), scaler)
