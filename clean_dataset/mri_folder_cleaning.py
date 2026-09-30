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

                sql = (
                    f"INSERT INTO {table_name} "
                    "(patient_id, study_id, series_id, image_name, folder_path) "
                    "VALUES (%s, %s, %s, %s, %s)"
                )

                inserted_count = 0
                for patient_id in os.listdir(folder_path):
                    patient_folder = os.path.join(folder_path, patient_id)

                    print("Processing patient:", patient_id)
                    for root, _, files in os.walk(patient_folder):
                        image_files = [f for f in files if not f.startswith(".")]
                        if not image_files:
                            continue

                        relative_parts = os.path.relpath(root, patient_folder).split(os.sep)
                        study_id = relative_parts[0] if relative_parts[0] != "." else ""
                        series_id = os.path.basename(root)

                        for image_name in image_files:
                            # print(f"Inserting: {patient_id}, {study_id}, {series_id}, {image_name}, {os.path.join(root, image_name)}")
                            cursor.execute(sql, (patient_id, study_id, series_id, image_name, root))
                            inserted_count += 1

                self.conn.commit()
            print(f"Filenames loaded into database successfully. Total rows: {inserted_count}")
        except Exception as e:
            print(f"Error loading filenames into database: {e}")
            self.conn.rollback()

    def generate_csv_from_db(self, table_name, output_csv_path):
        """Generate a CSV file from the database table"""
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
