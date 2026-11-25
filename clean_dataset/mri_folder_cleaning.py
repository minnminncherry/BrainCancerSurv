import pymysql
import yaml
import pandas as pd
import os
from pathlib import Path
import nibabel as nib
import numpy as np

class MRIDataCleaner:
    @staticmethod
    def calculate_tumor_volumes(metadata_csv, output_csv, cancer_type=None, mri_dir=None):
        """Measure all positive mask labels in native image space, once per scan.

        This is the whole labeled region, potentially including edema; per-label
        volumes are also saved without assuming biological label definitions.
        """
        metadata_csv = Path(metadata_csv)
        if metadata_csv.is_dir():
            metadata_csv = metadata_csv / f'{cancer_type}_mri_metadata.csv'
        frame = pd.read_csv(metadata_csv)
        output_csv = Path(output_csv)
        if output_csv.resolve() == metadata_csv.resolve():
            raise ValueError('Output must differ from the source metadata CSV.')
        mask_rows = frame.file_name.str.contains('glistrboost|glisterboots', case=False, na=False)
        if 'is_mask' in frame:
            mask_rows |= frame.is_mask.astype(str).str.lower().isin({'1', '1.0', 'true'})
        frame = frame.assign(_is_mask=mask_rows)
        records = []

        def resolve(row):
            path = Path(row.folder_path) / row.file_name
            if not path.is_file() and mri_dir:
                path = Path(mri_dir) / str(row.patient_id) / row.file_name
            if not path.is_file():
                raise FileNotFoundError(f'MRI/mask file not found: {path}')
            return path

        for (patient, date), scan in frame.groupby(['patient_id', 'scan_date'], sort=True):
            masks = scan.loc[scan._is_mask].drop_duplicates(['folder_path', 'file_name']).copy()
            if masks.empty:
                raise ValueError(f'No tumor mask for {patient}, {date}.')
            masks['_corrected'] = masks.file_name.str.contains('ManuallyCorrected', case=False, na=False)
            selected = masks.sort_values(['_corrected', 'file_name'], ascending=[False, True]).iloc[0]
            mask_path = resolve(selected)
            mask = nib.load(mask_path)
            images = scan.loc[~scan._is_mask]
            if images.empty:
                raise ValueError(f'No MRI scan for mask {mask_path}.')
            image = nib.load(resolve(images.iloc[0]))
            if len(mask.shape) != 3 or mask.shape != image.shape or not np.allclose(mask.affine, image.affine):
                raise ValueError(f'Mask and MRI geometry differ: {mask_path}')
            if mask.header.get_xyzt_units()[0] != 'mm':
                raise ValueError(f'Expected millimetre spatial units: {mask_path}')
            voxel_mm3 = float(abs(np.linalg.det(mask.affine[:3, :3])))
            if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
                raise ValueError(f'Invalid voxel dimensions: {mask_path}')
            labels, counts = np.unique(np.asanyarray(mask.dataobj), return_counts=True)
            if not np.isfinite(labels).all() or (labels < 0).any() or not (labels == np.floor(labels)).all():
                raise ValueError(f'Expected nonnegative integer segmentation labels: {mask_path}')
            positive = labels > 0
            voxel_count = int(counts[positive].sum())
            record = dict(patient_id=patient, scan_date=date, mask_path=str(mask_path),
                mask_type='manually_corrected' if selected['_corrected'] else 'automatic',
                region='all_positive_mask_labels', included_labels=','.join(str(int(v)) for v in labels[positive]),
                voxel_count=voxel_count, voxel_volume_mm3=voxel_mm3, volume_ml=voxel_count*voxel_mm3/1000)
            record.update({f'volume_label_{int(label)}_ml': int(count)*voxel_mm3/1000
                           for label, count in zip(labels[positive], counts[positive])})
            records.append(record)
        if not records:
            raise ValueError('No patient scans found in MRI metadata.')
        result = pd.DataFrame(records)
        label_columns = [c for c in result if c.startswith('volume_label_')]
        result[label_columns] = result[label_columns].fillna(0)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f'Saved {len(result)} scan volumes to: {output_csv}')
        return result

    def __init__(self, file):
        self.conn = None
        data = yaml.safe_load(file)

    def connect_db(self, database_name):
        """Establish database connection"""
        try:
            self.conn = pymysql.connect(
                host='localhost',
                user='mmc',
                password='root',
                db=database_name
            )
            print("Database connection established")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def disconnect_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def  load_filename_to_db(self, folder_path, table_name):
        """Insert NIfTI metadata from patient folders into the MRI table."""
        if not self.conn:
            print("No database connection. call connect_db() first.")
            return

        truncate_query = f"TRUNCATE TABLE {table_name};"
        with self.conn.cursor() as cursor:
            cursor.execute(truncate_query)
        self.conn.commit()
        print(f"Table {table_name} truncated successfully.")

        rows = []
        for patient_id in os.listdir(folder_path):
            patient_folder = os.path.join(folder_path, patient_id)
            if not os.path.isdir(patient_folder):
                continue

            for root, _, files in os.walk(patient_folder):
                for file_name in files:
                    if not file_name.lower().endswith((".nii", ".nii.gz")):
                        continue

                    name = file_name[:-7] if file_name.lower().endswith(".nii.gz") else file_name[:-4]
                    parts = name.split("_")
                    if len(parts) < 3:
                        continue

                    scan_date = parts[1]
                    modality = parts[2].upper()
                    is_mask = (
                        "glistrboost" in file_name.lower()
                        or "glisterboots" in file_name.lower()
                    )
                    rows.append((patient_id, scan_date, modality, file_name, root, is_mask))

        if not rows:
            print("No matching MRI files found.")
            return

        sql = (
            f"INSERT INTO `{table_name}` "
            "(patient_id, scan_date, modality, file_name, folder_path, is_mask) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(sql, rows)
            self.conn.commit()
            print(f"MRI filenames loaded successfully. Total rows: {len(rows)}")
        except Exception as e:
            self.conn.rollback()
            print(f"Error loading filenames into database: {e}")

    def generate_csv_from_db(self, table_name, output_csv_path):
        """Export scans and masks; retain is_mask for training/volume selection."""
        if not self.conn:
            print("No database connection. call connect_db() first.")
            return

        try:
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, self.conn)
            df.to_csv(output_csv_path, index=False)
            print(f"CSV file generated successfully at: {output_csv_path}")
        except Exception as e:
            print(f"Error generating CSV from database: {e}")
