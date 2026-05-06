import json
import os
import pymysql
import pandas as pd
import numpy as np
import sys
import yaml 
import shutil

class GenomicDataCleaner:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.conn = None
    
    def connect_db(self):
        """Establish database connection"""
        try:
            self.conn = pymysql.connect(
                host='localhost',
                user='mmc',
                password='root',
                db='tcga_gbm'
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

    def get_json_data(self, json_file_path):
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
                'entity_id': associated_entity.get('entity_id')
            }
            extracted_data.append(extracted_item)
            print(f"Extracted: {extracted_item}")
        
        return extracted_data
    
    def insert_json_data_to_db(self, json_data, table_name):
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
                sql = f"INSERT INTO {table_name} (submitter_id, entity_submitter_id, file_name, case_id, entity_id) VALUES (%s, %s, %s, %s, %s)"
                
                values = [(item['submitter_id'], item['entity_submitter_id'], 
                          item['file_name'], item['case_id'], item['entity_id']) 
                         for item in json_data]
                
                cursor.executemany(sql, values)
                self.conn.commit()
                
                print(f"Successfully inserted {len(json_data)} records into {table_name}")
                
        except Exception as e:
            print(f"Error inserting data into database: {e}")
            self.conn.rollback()

    def save_genomic_data_to_db(self, gene_file_path, table_name, truncate=True):
        """Read genomic TSV data and insert selected fields into a database table."""
        if not self.conn:
            print("No database connection. Call connect_db() first.")
            return

        df = pd.read_csv(gene_file_path, sep='\t', dtype=str, comment='#')
        df = df[["gene_id", "gene_name", "gene_type","tpm_unstranded", "fpkm_uq_unstranded"]]

        print(f"Columns found in {gene_file_path}: {list(df.columns)}")
        print(f"First few rows:\n{df.head()}")
        if 'gene_id' not in df.columns:
            print(f"'gene_id' column not found in {gene_file_path}. Available columns: {list(df.columns)}")
            return

        df = df[~df['gene_id'].str.startswith('[N]')].copy()
        df['file_name'] = os.path.basename(gene_file_path)

        if 'tpm_unstranded' not in df.columns or 'fpkm_uq_unstranded' not in df.columns:
            print(f"Required expression columns missing in {gene_file_path}")
            return

        df['tpm_unstranded'] = pd.to_numeric(df['tpm_unstranded'], errors='coerce')
        df['fpkm_uq_unstranded'] = pd.to_numeric(df['fpkm_uq_unstranded'], errors='coerce')

        # Drop rows with NaN values in expression columns
        df = df.dropna(subset=['tpm_unstranded', 'fpkm_uq_unstranded'])

        values = list(df[['file_name', 'gene_id', 'gene_name', 'gene_type', 'tpm_unstranded', 'fpkm_uq_unstranded']].itertuples(index=False, name=None))

        try:
            with self.conn.cursor() as cursor:
                if truncate:
                    truncate_qry = f"TRUNCATE TABLE {table_name};"
                    cursor.execute(truncate_qry)

                sql = f"INSERT INTO {table_name} (file_name, gene_id, gene_name, gene_type, tpm_unstranded, fpkm_uq_unstranded) VALUES (%s, %s, %s, %s, %s, %s)"

                cursor.executemany(sql, values)
                self.conn.commit()
                print(f"Successfully inserted {len(values)} records into {table_name}")
        except Exception as e:
            print(f"Error inserting TSV data into database: {e}")
            self.conn.rollback()
        
    def generate_clean_genomic_data(self, out_file_path):
        """Generate cleaned genomic data and save to output directory"""
        # This function can be implemented to read from the database, perform any necessary cleaning or transformation, and save the cleaned data to the output directory.
        date = pd.Timestamp.now().strftime("%Y-%m-%d")
        sql = '''
                SELECT 
            a.gene_name, 
            SUBSTRING_INDEX(c.entity_submitter_id, '-', 3) AS patient_id, 
            SUM(a.fpkm_uq_unstranded) AS fpkm_value 
        FROM raw_genomic_data a 
        INNER JOIN raw_file_name_genomic_data b 
            ON a.file_name = b.file_name 
        INNER JOIN genomic_metadata_json_data c 
            ON b.file_name = c.file_name 
        WHERE 
            a.gene_id IS NOT NULL
            AND a.gene_name IS NOT NULL
            AND a.gene_name != ''
            AND a.gene_id NOT LIKE '%PAR%'
        GROUP BY 
            a.gene_name, patient_id 
        ORDER BY 
            a.gene_name, patient_id;'''

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
                df_cleaned = pd.DataFrame(result, columns=['gene_name', 'patient_id', 'fpkm_value'])
                output_csv_path = os.path.join(out_file_path, f"cleaned_genomic_data_{date}.csv")
                df_cleaned.to_csv(output_csv_path, index=False)
                print(f"Cleaned genomic data saved to {output_csv_path}")
        except Exception as e:
            print(f"Error inserting TSV data into database: {e}")
            self.conn.rollback()

    def transform_genomic_data(self, outfile_path, hallmarks_data_path, output_file_path,fill_type='mean', cancer_type='gbm'):
        genomic_data = pd.read_csv(outfile_path)
        genomic_data.rename(columns={0: 'gene_names', 1:'patient_id', 2:'expression'}, inplace=True)
        hallmark_df = pd.read_csv(hallmarks_data_path)
        hallmark_gene_df = hallmark_df.iloc[:, [0]]
        df_filter = genomic_data[
            genomic_data["gene_names"].isin(hallmark_gene_df["gene_name"])
        ]

        df_pivot = df_filter.pivot_table(
            index="patient_id",
            columns="gene_names",
            values="expression",
            aggfunc="sum",
            fill_value=np.mean if fill_type == 'mean' else 0,
        )

        df_pivot.to_csv(output_file_path+f"/{cancer_type}_{fill_type}.csv", index=False)

def main(args1, args2):

    yaml_file_path = os.path.join(os.getcwd(), "../config.yaml")
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    input_dir = data['gbm_src_file_paths']['genomic_data_dir']
    output_dir = data['gbm_clean_file_paths']['genomic_data_dir']
    output_genomic_dir = data['mysql_file_paths']
    final_genomic_data_file_name = data['gbm_clean_file_paths']['final_genomic_data_file_name']
    json_file_path = os.path.join(input_dir, data['gbm_src_file_paths']['genomic_metadata_file_name'])

    gen_cls = GenomicDataCleaner(input_dir, output_dir)
    gen_cls.connect_db()

    # Connect to database
    if(args1 == 'INSERT_METADATA_JSON_DATA'):
        if gen_cls.conn:
            # Extract data from JSON
            extracted_data = gen_cls.get_json_data(json_file_path)
            
            # Insert data into database
            gen_cls.insert_json_data_to_db(extracted_data, "genomic_metadata_json_data")  
        else:
            print("Failed to connect to database")
    
    elif(args1 == 'INSERT_GENOMIC_TSV_DATA'):
        if gen_cls.conn:
            first = True
            for(root, dirs, files) in os.walk(input_dir):
                for filename in files:
                    if filename.endswith(".tsv"):
                        gene_file_path = os.path.join(root, filename)
                        gen_cls.save_genomic_data_to_db(gene_file_path, "raw_genomic_data", truncate=first)
                        first = False

    elif(args1 == 'GENERATE_CLEAN_GENOMIC_DATA'):
        destination_path = data['gbm_clean_file_paths']['genomic_data_dir']
        if gen_cls.conn:
            gen_cls.generate_clean_genomic_data(output_genomic_dir)
            shutil.move(output_genomic_dir, destination_path)
        else:
            print("Failed to connect to database")

    elif(args1 == 'GENERATE_TRANSFORMED_GENOMIC_DATA'):
        hallmarks_data_path = os.path.join(input_dir, data['pathway_clean_file_path']['hallmark_pathway_matrix_output_file'])
        if args2 == 'gbm' and gen_cls.conn:
            gen_cls.transform_genomic_data(output_genomic_dir, hallmarks_data_path, final_genomic_data_file_name, fill_type='mean', cancer_type='gbm')
        elif args2 == 'lgg' and gen_cls.conn:
             gen_cls.transform_genomic_data(output_genomic_dir, hallmarks_data_path, final_genomic_data_file_name, fill_type='mean', cancer_type='lgg')
        # Handle other cancer types if needed
    else:
        print("Invalid arguments provided")

    # Disconnect from database
    gen_cls.disconnect_db()

if __name__ == "__main__":
    args1, args2 = sys.argv[1:]

    # args1 for instruction and args2 for cancer type
    main(args1, args2)