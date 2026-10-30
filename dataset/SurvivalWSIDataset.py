import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

def _get_result_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))

class SurvivalSplitWSIDataset(Dataset):
    """Dataset for WSI split samples.

    Each item loads a CLAM-style `.pt` bag of patch features and returns:
    `(wsi_features, label, event_time, censorship, clinical_data)`.
    """

    def __init__(self, x, y, df, feature_dim=1024, pt_dir=None):
        print(f"Initializing SurvivalSplitWSIDataset with {len(y)} samples and feature dimension {feature_dim}.")
        self.df = df.reset_index(drop=True).copy()
        self.y = np.asarray(y, dtype=np.int64)
        self.patch_data = list(x)
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
            return data[:, :self.feature_dim]

        pad = torch.zeros(
            data.shape[0],
            self.feature_dim - current_dim,
            dtype=data.dtype,
            device=data.device,
        )
        return torch.cat([data, pad], dim=1)

    def _load_pt_features(self, patch_entry):
        pt_file = patch_entry["pt_file"]

        if not os.path.exists(pt_file):
            print(f"WSI feature PT file not found: {pt_file}")
            return None

        load_device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            print(f"Loading WSI features from PT file: {pt_file} on device: {load_device}")
            features = torch.load(pt_file, map_location=load_device, weights_only=True)
            print(f"Feature shape : {features.shape}")
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

        print(f"Fetching item at index: {idx}")
        
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
            else:
                wsi_features = self._load_pt_features(patch_entry)
                # print(f"Final wsi feature shape : {wsi_features.shape}")
        else:
            raise TypeError(f"Unsupported WSI patch entry type: {type(patch_entry)}")

        if wsi_features is None:
            clinical_data["n_patches"] = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            wsi_features = torch.empty((0, self.feature_dim), dtype=torch.float32, device=device)
        else:
            clinical_data["n_patches"] = wsi_features.shape[0]
        # print("Shape of WSI features:", wsi_features)

        return wsi_features, label, event_time, censorship, clinical_data


class SurvivalWSIDataset():

    ID_CANDIDATES = ("patient_id", "_PATIENT", "bcr_patient_barcode")
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
        pt_dir=None,
        wsi_feature_dim=1024,
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
        self.pt_dir = pt_dir
        self.wsi_feature_dim = int(wsi_feature_dim)
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

        self.metadata = self.metadata.merge(
            self.slide_data,
            left_on="_PATIENT",
            right_on="patient_id",
            how="inner",
        ).drop(columns=["patient_id", "svs_filename", "h5_filename", "status"])

        # print(f"Metadata head after merging with slide data:\n{self.metadata.head()}, columns: {self.metadata.columns.tolist()}")

        # self.metadata["slide_ids"] = self.metadata["slide_ids"].fillna("")
        # self.metadata["n_slides"] = self.metadata["n_slides"].fillna(0).astype(int)

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

        x = PT file paths containing pre-computed WSI patch features
        y = survival bin label
        """
        # removing the patient info if there is no slide data available for that patient
        # data = self.metadata[self.metadata["n_slides"] > 0].reset_index(drop=True)
        bin_col = f"{self.label_col}_bin"

        if fold_indices is None:
            fold_indices = []
        fold_indices = np.asarray(fold_indices, dtype=np.int64)

        is_test = self.metadata.index.isin(fold_indices)

        if split_key == "train":
            split_df = self.metadata.loc[~is_test].reset_index(drop=True)
        elif split_key == "test":
            split_df = self.metadata.loc[is_test].reset_index(drop=True)
        else:
            split_df = self.metadata.reset_index(drop=True)

        print(split_df[bin_col], type(split_df[bin_col]))
 
        split_df = split_df.drop(columns=['sampleID', 'CDE_DxAge', 'CDE_survival_days', 'CDE_vital_status','_INTEGRATION','bcr_patient_barcode','_primary_site', 'bcr_patient_uuid',
                                          'days_to_death', 'days_to_last_followup','days_to_new_tumor_event_after_initial_treatment', 'gender','vital_status',
                                           'year_of_initial_pathologic_diagnosis'], errors='ignore')

        y = split_df[bin_col].astype("int64").values

        if self.pt_dir:
            patch_data = []
            for _, row in split_df.iterrows():
                patient_id = row["_PATIENT"]
                sample_slides = self.slide_data.loc[self.slide_data["patient_id"] == patient_id]
                if sample_slides.empty:
                    raise FileNotFoundError(f"No WSI slide metadata found for patient: {patient_id}")

                svs_file = sample_slides.iloc[0]["svs_filename"]
                pt_file = os.path.join(self.pt_dir, os.path.splitext(svs_file)[0] + ".pt")
                patch_data.append({
                    "pt_file": pt_file,
                })
                # print(f"Processing patient: {patient_id}")
        else:
            raise ValueError("SurvivalWSIDataset requires pt_dir for MIL training.")

        result = SurvivalSplitWSIDataset(
            patch_data,
            y,
            split_df,
            feature_dim=self.wsi_feature_dim,
            pt_dir=self.pt_dir,
        )

        print(f"Returning split with {len(result)} samples, head of result.df:\n{result.df.columns}")
        
        return result

    def return_splits(self, args, fold_indices):
        train_split = self.__return_splits("train", fold_indices)
        test_split = self.__return_splits("test", fold_indices)
        # one_sample_train_split = train_split[0]
        # print(f"One sample train split: {one_sample_tra_, x_batch, y_batch, event_time_batch, censor_batch, clinical_data_listin_split}")

        result_dir = _get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        train_split.df.to_csv(os.path.join(result_dir, "wsi_train_merged_split.csv"), index=False)
        test_split.df.to_csv(os.path.join(result_dir, "wsi_test_merged_split.csv"), index=False)
        print('Done!')
        print(f"Training on {len(train_split)} samples")
        print(f"Testing on {len(test_split)} samples")
        return train_split, test_split, None
