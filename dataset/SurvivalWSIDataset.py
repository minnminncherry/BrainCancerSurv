import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import openslide
import h5py
from utils.core_utils import encoder

def _get_result_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))

class SurvivalSplitWSIDataset(Dataset):
    """Dataset for WSI split samples.

    Each item returned follows the same format as the genomic dataset:
    `(img, x, label, event_time, censorship, clinical_data)`.
    `x` can be a NumPy array / list of tensors (one-per-sample) or an array/list
    of file path strings (possibly semi-colon separated) pointing to serialized
    `.pt` files containing embeddings/tensors.
    """

    def __init__(self, x, y, df, feature_dim=512, encoder_model_name="resnet50", encoder_batch_size=32):
        print(f"Initializing SurvivalSplitWSIDataset with {len(y)} samples and feature dimension {feature_dim}.")
        self.df = df.reset_index(drop=True).copy()
        self.y = np.asarray(y, dtype=np.int64)
        self.patch_data = list(x)
        self.feature_dim = int(feature_dim)
        self.encoder_model_name = encoder_model_name
        self.encoder_batch_size = int(encoder_batch_size)
        self.feature_encoder = None
        self.img_transform = None

    def __len__(self):
        return len(self.y)

    def _get_encoder(self):
        if self.feature_encoder is None or self.img_transform is None:
            self.feature_encoder, self.img_transform = encoder(self.encoder_model_name)
        return self.feature_encoder, self.img_transform

    def _resize_feature_dim(self, data):
        if data.dim() == 1:
            data = data.unsqueeze(0)
        current_dim = data.shape[-1]
        if current_dim == self.feature_dim:
            return data
        if current_dim > self.feature_dim:
            return data[:, :self.feature_dim]

        pad = torch.zeros(data.shape[0], self.feature_dim - current_dim, dtype=data.dtype)
        return torch.cat([data, pad], dim=1)

    def _features_from_h5(self, patch_entry):
        print(f"Loading features from h5 file: {patch_entry.get('h5_file', 'N/A')} and slide file: {patch_entry.get('slide_file', 'N/A')}")
        h5_file = patch_entry["h5_file"]
        slide_file = patch_entry["slide_file"]
        print(f"Loading features from h5 file: {h5_file}")

        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Patch coordinate h5 file not found: {h5_file}")
        if not os.path.exists(slide_file):
            raise FileNotFoundError(f"WSI slide file not found: {slide_file}")

        with h5py.File(h5_file, "r") as file:
            coords = np.asarray(file["coords"][:], dtype=np.int64)
            patch_level = int(file["coords"].attrs.get("patch_level", 0))
            patch_size = int(file["coords"].attrs.get("patch_size", patch_entry.get("patch_size", 224)))

        print(f"Extracting features from slide: {slide_file} at level {patch_level} with patch size {patch_size} and lenght coordinatio point: {len(coords)}")

        feature_encoder, img_transform = self._get_encoder()
        slide_img = openslide.OpenSlide(slide_file)
        features = []
        batch = []
        counter = 0

        try:
            with torch.no_grad():
                for x, y in coords:
                    patch_img = slide_img.read_region(
                        (int(x), int(y)),
                        patch_level,
                        (patch_size, patch_size),
                    ).convert("RGB")
                    batch.append(img_transform(patch_img))
                    counter += 1
                    if counter % 1000 == 0:
                        print(f"Extracted features for {counter} patches from slide: {slide_file}")
                    if len(batch) == self.encoder_batch_size:
                        batch_tensor = torch.stack(batch, dim=0)
                        features.append(feature_encoder(batch_tensor).cpu())
                        batch = []

                if batch:
                    batch_tensor = torch.stack(batch, dim=0)
                    features.append(feature_encoder(batch_tensor).cpu())
        finally:
            slide_img.close()

        features = torch.cat(features, dim=0)
        if features.dim() > 2:
            features = features.reshape(features.shape[0], -1)
        return self._resize_feature_dim(features.float()), coords.tolist()

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
            if "features" in patch_entry:
                wsi_features = patch_entry["features"].float()
                patch_coords = patch_entry.get("coords", [])
            else:
                wsi_features, patch_coords = self._features_from_h5(patch_entry)
        else:
            raise TypeError(f"Unsupported WSI patch entry type: {type(patch_entry)}")

        clinical_data["patch_coords"] = patch_coords
        clinical_data["n_patches"] = int(wsi_features.shape[0])
        return wsi_features, patch_coords, label, event_time, censorship, clinical_data


class SurvivalWSIDataset():

    ID_CANDIDATES = ("patient_id", "_PATIENT", "sampleID", "bcr_patient_barcode")
    CENSOR_CANDIDATES = ("censorship", "censor", "event", "status")

    def __init__(
        self,
        label_file,
        slide_file_name,
        seed,
        label_col,
        n_bins,
        n_classes=4,
        h5_dir=None,
        slide_dir=None,
        wsi_feature_dim=512,
        encoder_model_name="resnet50",
        modality="mil",
        opt="adam",
        lr=1e-3,
        batch_size=32,
        num_workers=0,
    ):
        self.label_file = label_file
        self.slide_file_name = slide_file_name
        self.seed = seed
        self.label_col = label_col
        self.n_bins = n_bins
        self.n_classes = int(n_classes)
        self.h5_dir = h5_dir
        self.slide_dir = slide_dir
        self.wsi_feature_dim = int(wsi_feature_dim)
        self.encoder_model_name = encoder_model_name
        self.modality = modality
        self.opt = opt
        self.lr = lr
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

        self.__setup_metadata_and_labels()
    
    def __setup_metadata_and_labels(self):
        self.label_data = pd.read_csv(self.label_file)
        self.slide_data = pd.read_csv(self.slide_file_name)

        label_values = pd.to_numeric(self.label_data[self.label_col], errors="coerce")
        self.label_data = (
            self.label_data.loc[label_values.notna()]
            .drop_duplicates(subset=["_PATIENT"])
            .reset_index(drop=True)
        )
        label_values = pd.to_numeric(self.label_data[self.label_col], errors="coerce")

        self.labels = self.__discretize_survival_month(label_values)
        self.metadata = self.label_data.drop(columns=[self.label_col]).copy()
        self.metadata[self.label_col] = label_values.values
        self.metadata[f"{self.label_col}_bin"] = self.labels
        self.metadata["censorship"] = self.__build_censorship(self.metadata)

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
            left_on="_PATIENT",
            right_on="patient_id",
            how="left",
        ).drop(columns=["patient_id"])

        self.metadata["slide_ids"] = self.metadata["slide_ids"].fillna("")
        self.metadata["n_slides"] = self.metadata["n_slides"].fillna(0).astype(int)

    def __build_censorship(self, metadata):
        """
        Return 1.0 for censored/living patients and 0.0 for observed death events.
        """
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
        bins, bin_edges = pd.qcut(
            survival_months,
            q=self.n_classes,
            labels=False,
            duplicates="drop",
            retbins=True,
        )
        self.survival_bin_edges = np.asarray(bin_edges, dtype=np.float32)
        return bins.astype("int64").values.ravel()
    
    def __return_splits(self, split_key, fold_indices=None):
        """
        Return simple patch data and survival labels for train or test.

        x = slide patch coordinates or identifiers
        y = survival bin label
        """
        data = self.metadata[self.metadata["n_slides"] > 0].reset_index(drop=True)
        bin_col = f"{self.label_col}_bin"

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

        y = split_df[bin_col].astype("int64").values

        if self.h5_dir and self.slide_dir:
            patch_data = []
            for _, row in split_df.iterrows():
                patient_id = row["_PATIENT"]
                sample_slides = self.slide_data.loc[self.slide_data["patient_id"] == patient_id]
                if sample_slides.empty:
                    raise FileNotFoundError(f"No WSI slide metadata found for patient: {patient_id}")

                svs_file = sample_slides.iloc[0]["svs_filename"]
                h5_file = os.path.join(self.h5_dir, os.path.splitext(svs_file)[0] + ".h5")
                slide_file = os.path.join(self.slide_dir, svs_file)
                patch_data.append({
                    "h5_file": h5_file,
                    "slide_file": slide_file,
                })
        else:
            raise ValueError("SurvivalWSIDataset requires both h5_dir and slide_dir for MIL training.")

        return SurvivalSplitWSIDataset(
            patch_data,
            y,
            split_df,
            feature_dim=self.wsi_feature_dim,
            encoder_model_name=self.encoder_model_name,
        )

    def return_splits(self, args, fold_indices):
        train_split = self.__return_splits("train", fold_indices)
        test_split = self.__return_splits("test", fold_indices)

        result_dir = _get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        train_split.df.to_csv(os.path.join(result_dir, "train_merged_split.csv"), index=False)
        test_split.df.to_csv(os.path.join(result_dir, "test_merged_split.csv"), index=False)
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Testing on {} samples".format(len(test_split)))
        return train_split, test_split, None
