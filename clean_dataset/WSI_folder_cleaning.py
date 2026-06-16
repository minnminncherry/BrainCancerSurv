from __future__ import annotations
import json

import pandas as pd
from pathlib import Path
import os

import pymysql
import openslide
from PIL import Image
import numpy as np
import torch
import xml.etree.ElementTree as ET

class WSIDataCleaner:
    def __init__(self):
        self.conn = None
    
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

    def WSI_format_change_pt(self, output_file_path: str | Path, input_path: str | Path, table_name: str = "map_wsi_filepath"):
        # resize the images to a reasonable size (e.g. 256x256) and save as .pt tensors for faster loading in downstream tasks. Also create a metadata CSV linking case IDs to the new .pt file paths.
        input_path = Path(input_path)
        output_file_path = Path(output_file_path)
        output_file_path.mkdir(parents=True, exist_ok=True)

        # Define valid WSI file extensions
        valid_extensions = '.svs'
        data = []

        # truncate table
        if self.conn:
            try:
                with self.conn.cursor() as cursor:
                    truncate_qry = f"TRUNCATE TABLE {table_name};"
                    cursor.execute(truncate_qry)
                    self.conn.commit()
                    print(f"Truncated table {table_name} before inserting new metadata")
            except Exception as e:
                print(f"Error truncating table {table_name}: {e}")
                self.conn.rollback()

        # List all files in the input directory
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith(valid_extensions):
                    svs_path = Path(root) / file

                    # Convert SVS to a downsampled image tensor and save as .pt for faster downstream loading.
                    if openslide is None or torch is None or np is None:
                        raise  ImportError("Missing dependencies for SVS->PT conversion. Install 'openslide-python', 'Pillow', 'numpy', and 'torch'.")

                    slide = openslide.OpenSlide(str(svs_path))
                    # Create a reasonable thumbnail to keep memory usage small (adjust size if needed)
                    thumbnail = slide.get_thumbnail((256, 256)).convert('RGB')
                    arr = np.array(thumbnail, dtype=np.uint8)
                    # Convert HWC -> CHW and to float32 normalized [0,1]
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)

                    pt_filename = file.rsplit('.', 1)[0] + '.pt'
                    pt_path = output_file_path / pt_filename
                    # Save tensor directly to the output directory
                    torch.save(tensor, str(pt_path))
                    print(f"Converted {svs_path} to {pt_path}")
                    data.append({
                        'case_id': file,  # Assuming case_id can be derived from filename
                        'pt_filename': pt_filename
                    })
                    if self.conn:
                        sql = f"INSERT INTO {table_name} (svs_filename, svs_filepath, pt_filename, pt_filepath) VALUES (%s, %s, %s, %s)"
                        values = [file, str(svs_path), pt_filename, str(pt_path)]

                        # insert the wsi metadata into the database
                        try:
                            with self.conn.cursor() as cursor:
                                cursor.execute(sql, values)
                                self.conn.commit()
                                print(f"Inserted metadata for {file} into database")
                        except Exception as e:
                            print(f"Error inserting metadata for {file} into database: {e}")
                            self.conn.rollback()
    
    def get_wsi_metadata_json_data(self, json_file_path):
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
        
        extracted_data = []

        for item in json_data:
            # Get the first associated entity if it exists
            associated_entity = item.get('associated_entities', [{}])[0] if item.get('associated_entities') else {}
            
            extracted_item = {
                'submitter_id': item.get('submitter_id'),
                'entity_submitter_id': associated_entity.get('entity_submitter_id'),
                'file_name': item.get('file_name'),
                'case_id': associated_entity.get('case_id'),
                'entity_id': associated_entity.get('entity_id'),
                'experimental_strategy': item.get('experimental_strategy'),
                'case_submitter_id': associated_entity.get('case_submitter_id')
            }
            extracted_data.append(extracted_item)
            print(f"Extracted: {extracted_item}")
        
        return extracted_data

    def insert_wsi_json_data_to_db(self, json_data, table_name):
        """Insert extracted JSON data into database table"""
        if not self.conn:
            print("No database connection. Call connect_db() first.")
            return
        
        try:
            with self.conn.cursor() as cursor:
                # Truncate table first
                truncate_qry = f"TRUNCATE TABLE {table_name};"
                cursor.execute(truncate_qry)
                
                # Insert data
                sql = f"INSERT INTO {table_name} (submitter_id, entity_submitter_id, file_name, case_id, entity_id, experimental_strategy, case_submitter_id) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                
                values = [(item['submitter_id'], item['entity_submitter_id'], 
                          item['file_name'], item['case_id'], item['entity_id'], item['experimental_strategy'], item['case_submitter_id']) 
                         for item in json_data]
                
                cursor.executemany(sql, values)
                self.conn.commit()
                
                print(f"Successfully inserted {len(json_data)} records into {table_name}")
                
        except Exception as e:
            print(f"Error inserting data into database: {e}")
            self.conn.rollback()

    def generate_wsi_metadata_csv(self, output_csv_path):
        """Generate a CSV file from metadata records"""
        sql = '''select b.file_name as svs_filename, a.pt_filename, coalesce(case_submitter_id, SUBSTRING_INDEX(entity_submitter_id, '-', 3)) as patient_id,
            CASE WHEN entity_submitter_id like '%11A%' then 'normal'
            else 'tumor' end as status
            from map_wsi_filepath a 
            left join wsi_metadata_json_data b on a.svs_filename = b.file_name
            where b.experimental_strategy like '%Diagnostic Slide%' '''
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
                df = pd.DataFrame(results, columns=['svs_filename', 'pt_filename', 'patient_id', 'status'])
                df.to_csv(os.path.join(output_csv_path), index=False)
                print(f"Generated WSI metadata CSV at {output_csv_path}")
        except Exception as e:
            print(f"Error generating WSI metadata CSV: {e}")