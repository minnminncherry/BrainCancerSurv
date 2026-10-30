import pymysql
import yaml
import pandas as pd
import os
import re

class MRIDataCleaner:
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
        """Generate a CSV file from the database table"""
        if not self.conn:
            print("No database connection. call connect_db() first.")
            return

        try:
            query = f"SELECT * FROM {table_name} where is_mask = 0;"
            df = pd.read_sql(query, self.conn)
            df.to_csv(output_csv_path, index=False)
            print(f"CSV file generated successfully at: {output_csv_path}")
        except Exception as e:
            print(f"Error generating CSV from database: {e}")
