import os
import pandas as pd
from torch.utils.data import Dataset

def _get_result_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))

class SurvivalMRISplitDataset(Dataset):
        def __init__(self, x, y, df, transform=None):
            self.mri_paths = list(x)
            self.y = list(y)
            self.df = df.reset_index(drop=True).copy()
            self.transform = transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            row = self.df.iloc[int(idx)]
            return (
                self.mri_paths[int(idx)],
                int(self.y[int(idx)]),
                float(row["CDE_survival_time"]),
                float(row["censorship"]),
                row.to_dict(),
            )


class SurvivalMRIDataset:

    D_CANDIDATES = ("patient_id", "_PATIENT", "bcr_patient_barcode")
    CENSOR_CANDIDATES = ("censorship", "censor", "event", "status")

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
    
            self.__setup_metadata_and_labels()


    def __setup_metadata_and_labels(self):
        self.label_data = pd.read_csv(self.label_file)
        self.mri_data = pd.read_csv(self.mri_file_name)
        self.label_id_col = "_PATIENT"

        self.label_data[self.label_col] = pd.to_numeric(
            self.label_data[self.label_col], errors="coerce"
        )
        self.label_data = self.label_data.dropna(subset=[self.label_col])
        self.label_data = self.label_data.drop_duplicates(self.label_id_col)

        self.labels, self.survival_bin_edges = pd.qcut(
            self.label_data[self.label_col],
            q=self.n_classes,
            labels=False,
            duplicates="drop",
            retbins=True,
        )
        self.labels = self.labels.astype("int64").to_numpy()

        self.metadata = self.label_data.copy()
        self.metadata[f"{self.label_col}_bin"] = self.labels
        self.metadata["__label"] = self.labels
        self.metadata["censorship"] = self.__build_censorship(self.metadata)

        self.mri_data["image_path"] = self.mri_data.apply(
            lambda row: os.path.join(str(row["folder_path"]), str(row["image_name"])),
            axis=1,
        )

        mri_info = (
            self.mri_data.groupby("patient_id")
            .agg(mri_paths=("image_path", list), n_images=("image_path", "size"))
            .reset_index()
        )

        self.metadata = self.metadata.merge(
            mri_info,
            left_on=self.label_id_col,
            right_on="patient_id",
            how="inner",
        ).drop(columns="patient_id").reset_index(drop=True)

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

    def read_dicom_slice(self, dicom_file_path):
        # Implement the logic to read a DICOM slice from the given file path
        # You can use libraries like pydicom to read DICOM files
        pass

    def create_3D_volume(self, dicom_file_paths):
        # Implement the logic to create a 3D volume from a list of DICOM file paths
        # You can use libraries like numpy to stack the slices into a 3D array
        pass

    def print_shape_of_3D_volume(self, volume):
        # Implement the logic to print the shape of the 3D volume
        # You can use the shape attribute of numpy arrays to get the dimensions
        pass

    def __return_splits(self, split_key, fold_indices=None):
        if fold_indices is None:
            fold_indices = []

        is_test = self.metadata.index.isin(fold_indices)
        if split_key == "train":
            split_df = self.metadata.loc[~is_test].reset_index(drop=True)
        elif split_key == "test":
            split_df = self.metadata.loc[is_test].reset_index(drop=True)
        else:
            raise ValueError("split_key must be 'train' or 'test'")

        x = split_df["mri_paths"].tolist()
        y = split_df[f"{self.label_col}_bin"].astype("int64").tolist()

        return SurvivalMRISplitDataset(x, y, split_df)

    def return_splits(self, args=None, fold_indices=None):
        train_split = self.__return_splits("train", fold_indices)
        test_split = self.__return_splits("test", fold_indices)
        # one_sample_train_split = train_split[0]
        # print(f"One sample train split: {one_sample_tra_, x_batch, y_batch, event_time_batch, censor_batch, clinical_data_listin_split}")

        result_dir = _get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        train_split.df.to_csv(os.path.join(result_dir, "mri_train_merged_split.csv"), index=False)
        test_split.df.to_csv(os.path.join(result_dir, "mri_test_merged_split.csv"), index=False)
        print('Done!')
        print(f"Training on {len(train_split)} samples")
        print(f"Testing on {len(test_split)} samples")
        return train_split, test_split, None
