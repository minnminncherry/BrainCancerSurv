import pymysql
import yaml
import pandas as pd
import os

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
    
    def load_filename_to_db(self, folder_path, table_name):
        """Load MRI filenames from the folder into the database table"""
        if not self.conn:
            print("No database connection. call connect_db() first.")
            return

        try:
            with self.conn.cursor() as cursor:
                truncate_qry = f"TRUNCATE TABLE {table_name};"
                cursor.execute(truncate_qry)

                for filename in os.listdir(folder_path):
                    root_name = filename
                    patient_folder = os.path.join(folder_path, filename)
                    file_count = 0
                    print ("Processing file: ", root_name)
                    for root, dirs, files in os.walk(patient_folder):
                        # print(f"Found {len(files)} files in {root} and dirname is {dirs}")
                        for file in files:
                            if file.lower().endswith(".dcm"):
                                # print(f"Found DICOM file: {file} in {root}")
                                file_count += 1
                    cursor.execute(
                        f"INSERT INTO {table_name} (patient_id, img_count) VALUES (%s, %s)",
                        (root_name, file_count)
                    )
            self.conn.commit()
            print("Filenames loaded into database successfully.")
        except Exception as e:
            print(f"Error loading filenames into database: {e}")