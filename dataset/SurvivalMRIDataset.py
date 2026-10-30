import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset

def _get_result_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))

class SurvivalMRISplitDataset(Dataset):
    def __init__(self, x, y, df, load_volume, transform=None):
        self.mri_paths = list(x)
        self.y = list(y)
        self.df = df.reset_index(drop=True).copy()
        self.load_volume = load_volume
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.df.iloc[int(idx)]
        volume = self.load_volume(self.mri_paths[int(idx)])
        if self.transform is not None:
            volume = self.transform(volume)
        return (
            volume,
            int(self.y[int(idx)]),
            float(row["event_time"]),
            float(row["censorship"]),
            row.to_dict(),
        )


class SurvivalMRIDataset:

    D_CANDIDATES = ("patient_id", "_PATIENT", "bcr_patient_barcode")

    def __init__(
        self,
        label_file,
        mri_file_name,
        seed,
        label_col,
        n_bins,
        n_classes=4,
        h5_dir=None,
        mri_dir=None,
        modality="ViT",
        opt="adam",
        lr=1e-3,
        batch_size=32,
        num_workers=0,
        crop_size=None,
        ):
        self.label_file = label_file
        self.mri_file_name = mri_file_name
        self.seed = seed
        self.label_col = label_col
        self.n_bins = n_bins
        self.n_classes = int(n_classes)
        self.h5_dir = h5_dir
        self.mri_dir = mri_dir
        self.modality = modality
        self.opt = opt
        self.lr = lr
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.crop_size = crop_size
    
        self.__setup_metadata_and_labels()


    def __setup_metadata_and_labels(self):
        self.label_data = pd.read_csv(self.label_file)
        self.mri_data = pd.read_csv(self.mri_file_name)
        self.label_id_col = next(
            col for col in self.D_CANDIDATES if col in self.label_data.columns
        )

        self.mri_data["image_path"] = [
            os.path.join(self.mri_dir, patient, name) if self.mri_dir else os.path.join(folder, name)
            for patient, folder, name in zip(
                self.mri_data["patient_id"], self.mri_data["folder_path"], self.mri_data["file_name"]
            )
        ]
        self.mri_data = self.mri_data.drop_duplicates("image_path")

        self.label_data[self.label_col] = pd.to_numeric(
            self.label_data[self.label_col], errors="coerce"
        )
        self.label_data = self.label_data.dropna(subset=[self.label_col])
        self.label_data = self.label_data.drop_duplicates(self.label_id_col)

        # Keep one metadata row per patient so all scans stay in the same fold.
        patients = self.mri_data[["patient_id"]].drop_duplicates()
        print(f"Loaded {len(patients)} and {len(self.mri_data[['patient_id']])} patients with MRI metadata.")
        print(f"Loaded {len(self.label_data)} patients with clinical labels.")
        self.metadata = self.label_data.merge(
            patients, left_on=self.label_id_col, right_on="patient_id", how="inner"
        ).reset_index(drop=True)

        print(f"Loaded {len(self.metadata)} patients with MRI and clinical labels.")

        if self.metadata.empty:
            raise ValueError("No patients matched the clinical labels and NIfTI metadata.")
        
        self.metadata["censorship"] = self.__build_censorship(self.metadata)
        self.metadata = self.metadata[["patient_id", self.label_col, "censorship"]].copy()
        self.mri_data = self.mri_data[["patient_id", "image_path"]].copy()
        print(f"MRI dataset: {len(self.metadata)} patients, {len(self.mri_data)} images")

    def __build_censorship(self, metadata):
        """Return 1 for censored/living patients and 0 for observed deaths."""
        status = metadata["CDE_vital_status"].astype(str).str.upper()
        censorship = status.map(
            {"LIVING": 1.0, "ALIVE": 1.0, "DECEASED": 0.0, "DEAD": 0.0}
        )

        if "days_to_death" in metadata.columns:
            death_recorded = pd.to_numeric(metadata["days_to_death"], errors="coerce").notna()
            censorship = censorship.mask(death_recorded, 0.0)

        return censorship.fillna(0.0).astype("float32")

    def load_mri_series(self, file_path):
        """Load one NIfTI image as a float tensor (1, D, H, W)."""
        image = nib.as_closest_canonical(nib.load(file_path))
        volume = image.get_fdata(dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI image: {file_path}")
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        volume = volume.transpose(2, 1, 0)  # NIfTI X/Y/Z to tensor D/H/W.
        volume = self.normalize_mri_volume(volume)
        if self.crop_size is not None:
            volume = self.crop_mri_volume(volume, self.crop_size)
        return torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)

    def normalize_mri_volume(self, volume):
        """Z-score nonzero foreground voxels and keep background at zero."""
        volume = np.nan_to_num(
            np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        foreground = volume != 0  # Foreground approximation, not skull stripping.
        values = volume[foreground]
        if not values.size or values.std() < 1e-6:
            raise ValueError("MRI foreground is empty or has constant intensity.")
        normalized = np.zeros_like(volume)
        normalized[foreground] = (values - values.mean()) / values.std()
        return normalized

    def crop_mri_volume(self, volume, crop_size):
        """Center-crop a 3D volume to crop_size (slices, height, width)."""
        if volume.ndim != 3 or len(crop_size) != 3:
            raise ValueError("Volume and crop size must both have three dimensions.")
        if any(not isinstance(crop, (int, np.integer)) or crop <= 0 for crop in crop_size):
            raise ValueError("Crop dimensions must be positive integers.")
        # Pad small volumes so every cropped sample has the requested shape.
        padding = [max(0, crop - dim) for dim, crop in zip(volume.shape, crop_size)]
        volume = np.pad(volume, [(pad // 2, pad - pad // 2) for pad in padding])
        start = [(dim - crop) // 2 for dim, crop in zip(volume.shape, crop_size)]
        return volume[tuple(slice(s, s + crop) for s, crop in zip(start, crop_size))]

    def return_splits(self, args=None, fold_indices=None):
        """Split patients and fit survival quantiles on training patients only."""
        test_mask = self.metadata.index.isin([] if fold_indices is None else fold_indices)
        train_df = self.metadata.loc[~test_mask].copy()
        test_df = self.metadata.loc[test_mask].copy()
        if train_df.empty:
            raise ValueError("The training split has no patients.")

        _, edges = pd.qcut(
            train_df[self.label_col], q=self.n_classes,
            labels=False, duplicates="drop", retbins=True,
        )
        if len(edges) != self.n_classes + 1:
            raise ValueError("Too few unique training survival times for n_classes.")
        # Allow validation survival times outside the observed training range.
        edges[0], edges[-1] = -np.inf, np.inf
        self.survival_bin_edges = edges
        result_dir = _get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        splits = []
        for name, patient_df in (("train", train_df), ("test", test_df)):
            patient_df[f"{self.label_col}_bin"] = pd.cut(
                patient_df[self.label_col], bins=edges, labels=False, include_lowest=True
            ).astype(int)
            split_df = patient_df.merge(self.mri_data, on="patient_id", how="inner")
            split_df["event_time"] = split_df[self.label_col]
            splits.append(SurvivalMRISplitDataset(
                split_df["image_path"], split_df[f"{self.label_col}_bin"],
                split_df, self.load_mri_series,
            ))
            split_df.to_csv(os.path.join(result_dir, f"mri_{name}_merged_split.csv"), index=False)
            print(f"{name}: {len(patient_df)} patients, {len(split_df)} images")
        return splits[0], splits[1], None
