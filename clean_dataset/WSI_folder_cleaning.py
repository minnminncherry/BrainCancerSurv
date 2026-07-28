from __future__ import annotations
import json

import pandas as pd
from pathlib import Path
import os

import pymysql
import yaml
import openslide
from PIL import Image
import numpy as np
import torch
import xml.etree.ElementTree as ET
import cv2
import shutil

class WSIDataCleaner:
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
        sql = ''' select b.file_name as svs_filename, c.h5_filename, a.pt_filename, coalesce(case_submitter_id, SUBSTRING_INDEX(entity_submitter_id, '-', 3)) as patient_id, 
            CASE WHEN entity_submitter_id like '%11A%' then 'normal'
            else 'tumor' end as status
            from map_wsi_filepath a 
            left join wsi_metadata_json_data b on a.svs_filename = b.file_name
            inner join raw_wsi_h5_filenames c on SUBSTRING_INDEX(c.h5_filename, '.h5', 1) = SUBSTRING_INDEX(b.file_name, '.svs', 1)
                            '''
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
                df = pd.DataFrame(results, columns=['svs_filename', 'h5_filename', 'pt_filename', 'patient_id', 'status'])
                df.to_csv(os.path.join(output_csv_path), index=False)
                print(f"Generated WSI metadata CSV at {output_csv_path}")
        except Exception as e:
            print(f"Error generating WSI metadata CSV: {e}")

    def copy_wsi_files_to_one_folder(self, source_dir, dest_dir, csv_file_path):
        """Copy all WSI files from source directory to destination directory"""
        source_dir = Path(source_dir)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        row = []

        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.svs'):
                    src_file_path = Path(root) / file
                    dest_file_path = dest_dir / file
                    row.append({'slide_id': str(file), 'process': 1})
                    try:
                        shutil.copy2(src_file_path, dest_file_path)
                        print(f"Copied {src_file_path} to {dest_file_path}")
                    except Exception as e:
                        print(f"Error copying {src_file_path} to {dest_file_path}: {e}")
        
        df = pd.DataFrame(row) 
        df.to_csv(os.path.join(csv_file_path, 'wsi_copy_log.csv'), index=False)

    
    def insert_h5_filename_DB(self, h5_dir, table_name):
        """Insert h5 filenames into database table"""
        if not self.conn:
            print("No database connection. Call connect_db() first.")
            return
        
        try:
            with self.conn.cursor() as cursor:
                # Truncate table first
                truncate_qry = f"TRUNCATE TABLE {table_name};"
                cursor.execute(truncate_qry)
                
                # Insert data
                sql = f"INSERT INTO {table_name} (h5_filename) VALUES (%s)"
                
                h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
                values = [(f,) for f in h5_files]
                
                cursor.executemany(sql, values)
                self.conn.commit()
                
                print(f"Successfully inserted {len(h5_files)} h5 filenames into {table_name}")
                
        except Exception as e:
            print(f"Error inserting h5 filenames into database: {e}")
            self.conn.rollback()
        
    def insert_svs_filename_DB(self, svs_dir, pt_dir, table_name):
        """Insert svs filenames into database table"""
        if not self.conn:
            print("No database connection. Call connect_db() first.")
            return
        
        try:
            with self.conn.cursor() as cursor:
                # Truncate table first
                truncate_qry = f"TRUNCATE TABLE {table_name};"
                cursor.execute(truncate_qry)
                count = 0
                # Insert data
                sql = f"INSERT INTO {table_name} (svs_filename, svs_filepath, pt_filename, pt_filepath) VALUES (%s, %s, %s, %s)"

                for f in os.listdir(svs_dir):
                    count+=1
                    if f.endswith('.svs'):
                        svs_filepath = os.path.join(svs_dir, f)
                        f_suf = f.removesuffix(".svs")
                        print(f"Processing {f} with suffix {f_suf}")
                        pt_filename = None
                        for f_pt in os.listdir(pt_dir):
                            if f_pt.startswith(f_suf):
                                pt_filename = f_pt
                                break
                        if pt_filename:
                            pt_filepath = os.path.join(pt_dir, pt_filename)
                        pt_filepath = os.path.join(pt_dir, pt_filename) if pt_filename else None
                        cursor.execute(sql, (f, svs_filepath, pt_filename, pt_filepath))
                
        except Exception as e:
            print(f"Error inserting svs filenames into database: {e}")
            self.conn.rollback()

        self.conn.commit()
        print("Total count : ",count)
