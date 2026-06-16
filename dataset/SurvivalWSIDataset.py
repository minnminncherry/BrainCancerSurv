import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class SurvivalSplitWSIDataset(Dataset):
    """Dataset for WSI split samples.

    Each item returned is a tuple `(wsi_img_tensor, label, event_time, clinical_data)`.
    `x` can be a NumPy array / list of tensors (one-per-sample) or an array/list
    of file path strings (possibly semi-colon separated) pointing to serialized
    `.pt` files containing embeddings/tensors.
    """

    def __init__(self, x, y, df):
        self.df = df.reset_index(drop=True).copy()
        self.y = np.asarray(y, dtype=np.int64)

        if isinstance(x, np.ndarray):
            self.embs = torch.from_numpy(x.astype(np.float32))
            self.paths = None
        else:
            # treat as list/array of path strings or tensor-like objects
            # if elements are tensors/arrays, convert to tensor stack
            first = x[0]

            if isinstance(first, (np.ndarray, torch.Tensor)):
                arr = np.asarray(x, dtype=np.float32)
                self.embs = torch.from_numpy(arr)
                self.paths = None
            else:
                self.paths = [str(v) if not pd.isna(v) else "" for v in list(x)]
                self.embs = None

    def __len__(self):
        return len(self.y)

    def _load_path(self, path_str):
        paths = [p.strip() for p in str(path_str).split(";") if p.strip()]
        path = next((p for p in paths if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(f"WSI embedding file not found: {path_str}")

        data = torch.load(path)
        if isinstance(data, dict):
            data = next(
                (v for v in data.values() if isinstance(v, (torch.Tensor, np.ndarray))),
                None,
            )

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        return data.float()

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(self.y):
            raise IndexError(f"Index out of range: {idx}")

        row = self.df.iloc[idx]
        label = int(self.y[idx])
        event_time = float(
            pd.to_numeric(row.get("CDE_survival_time", row.get("survival_months", 0.0)), errors="coerce")
            or 0.0
        )
        clinical_data = row.to_dict()

        if self.embs is not None:
            wsi_img_tensor = self.embs[idx]
            if not isinstance(wsi_img_tensor, torch.Tensor):
                wsi_img_tensor = torch.from_numpy(np.asarray(wsi_img_tensor, dtype=np.float32))
            wsi_img_tensor = wsi_img_tensor.float()
        else:
            path_str = self.paths[idx]
            wsi_img_tensor = self._load_path(path_str)

        return (wsi_img_tensor, label, event_time, clinical_data)


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
        pt_file_path,
        n_classes=4,
        modality="mlp",
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
        self.pt_file_path = pt_file_path
        self.modality = modality
        self.opt = opt
        self.lr = lr
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

        self.__setup_metadata_and_labels()
        self.slide_cls_id_prep()
    
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
                slide_pt=("pt_filename", join_values),
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
        self.metadata["slide_pt"] = self.metadata["slide_pt"].fillna("")
        self.metadata["n_slides"] = self.metadata["n_slides"].fillna(0).astype(int)
        self.metadata["slide_pt_path"] = self.metadata["slide_pt"].apply(self.__build_slide_pt_path)

    def __build_slide_pt_path(self, slide_pt):
        pt_files = [pt for pt in str(slide_pt).split(";") if pt]
        return ";".join(os.path.join(self.pt_file_path, pt) for pt in pt_files)
    
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
    
    def slide_cls_id_prep(self):
        """
        Identify how many WSI slides belong to each survival label.
        """
        label_df = self.metadata[
            ["_PATIENT", self.label_col, f"{self.label_col}_bin", "censorship"]
        ].drop_duplicates(subset=["_PATIENT"]).copy()
        label_df = label_df.rename(columns={f"{self.label_col}_bin": "label"})

        self.slide_label_data = self.slide_data.merge(
            label_df,
            left_on="patient_id",
            right_on="_PATIENT",
            how="inner",
        ).reset_index(drop=True)
        self.unmatched_slide_data = self.slide_data.loc[
            ~self.slide_data["patient_id"].isin(label_df["_PATIENT"])
        ].reset_index(drop=True)

        self.slide_cls_ids = [[] for _ in range(self.n_classes)]
        for i in range(self.n_classes):
            self.slide_cls_ids[i] = np.where(
                self.slide_label_data["label"].to_numpy(dtype=np.int64) == i
            )[0]

        self.slide_label_summary = (
            self.slide_label_data.groupby("label")
            .agg(n_slides=("label", "size"), n_patients=("patient_id", "nunique"))
            .reindex(range(self.n_classes), fill_value=0)
            .rename_axis("label")
            .reset_index()
        )

        print(f"Matched {len(self.slide_label_data)} / {len(self.slide_data)} slides.")
        print("WSI slide label summary:")
        print(self.slide_label_summary.to_string(index=False))

    
    def __return_splits(self, split_key, fold_indices=None):
        """
        Return simple x, y data for train or test.

        x = slide pt path
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

        print(split_df.head(3))

        x = split_df["slide_pt_path"].values
        y = split_df[bin_col].astype("int64").values
        return SurvivalSplitWSIDataset(x, y, split_df)

    def return_splits(self, args, fold_indices):
        return self.__return_splits(args, fold_indices)
