import os
import glob
import pandas as pd
import pymysql
import gzip
import shutil

class GenomicMutation:
    def __init__(self, input_dir, output_dir):
            self.input_dir = input_dir
            self.output_dir = output_dir
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

    def clean_mutation_file(self, input_file_path, table_name):

        with self.conn.cursor() as cursor:
            # Truncate table first
            truncate_qry = f"TRUNCATE TABLE {table_name};"
            cursor.execute(truncate_qry)

            if os.path.isdir(input_file_path):
                maf_files = glob.glob(os.path.join(input_file_path, "**", "*.maf"), recursive=True)
            else:
                maf_files = [input_file_path]

            mutation_rows = []
            for maf_file in maf_files:
                maf = pd.read_csv(maf_file, sep="\t", comment="#", low_memory=False)
                if maf.empty:
                    continue

                patient_ids = maf["Tumor_Sample_Barcode"].str.split("-").str[:3].str.join("-")
                mutation_rows.append(
                    pd.DataFrame({"patient_id": patient_ids, "gene": maf["Hugo_Symbol"]})
                )

            if not mutation_rows:
                raise ValueError(f"No mutation records found in: {input_file_path}")

            mutations = pd.concat(mutation_rows).dropna().drop_duplicates()
            cleaned_data = (
                mutations.assign(mutated=1)
                .pivot(index="patient_id", columns="gene", values="mutated")
                .fillna(0)
                .astype(int)
                .reset_index()
            )
            cleaned_data.columns.name = None


        return cleaned_data

    def extract_zip_file(self, base_path, file_exten=".gz", table_name="raw_file_mutation_data"):

        extension = file_exten if file_exten.startswith(".") else f".{file_exten}"
        os.makedirs(self.output_dir, exist_ok=True)
        extracted_files = []

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {table_name}")
                insert_query = (
                    f"INSERT INTO {table_name} "
                    "(folder_name, input_zpfile_path, output_file_path, "
                    "input_zpfile_name, output_file_name) "
                    "VALUES (%s, %s, %s, %s, %s)"
                )

                for root, _, files in os.walk(base_path):
                    for file in files:
                        if not file.lower().endswith(extension.lower()):
                            continue

                        input_path = os.path.join(root, file)
                        output_name = file[:-len(extension)]
                        output_path = os.path.join(self.output_dir, output_name)

                        with gzip.open(input_path, "rb") as source:
                            with open(output_path, "wb") as destination:
                                shutil.copyfileobj(source, destination)

                        cursor.execute(insert_query,(os.path.basename(root), input_path, output_path, file, output_name))
                        extracted_files.append(output_path)
                        print(f"Extracted: {input_path} -> {output_path}")

            self.conn.commit()
            return extracted_files
        except Exception:
            self.conn.rollback()
            raise

    def create_gene_mutation_file(self, output_file_path, output_file_name):
        functional_variants = [
            "Missense_Mutation",
            "Nonsense_Mutation",
            "Frame_Shift_Del",
            "Frame_Shift_Ins",
            "In_Frame_Del",
            "In_Frame_Ins",
            "Splice_Site",
            "Translation_Start_Site",
            "Nonstop_Mutation",
        ]
        maf_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.output_dir)
            for file in files if file.lower().endswith(".maf")
        ]
        required_columns = {"Tumor_Sample_Barcode", "Hugo_Symbol", "Mutation_Status", "Variant_Classification"}

        mutation_frames = []
        for maf_file in maf_files:
            maf = pd.read_csv(maf_file, sep="\t", comment="#", low_memory=False)
            if maf.empty:
                continue

            if not required_columns.issubset(maf.columns):
                raise ValueError(f"Missing required columns in MAF file: {maf_file}")

            mut = maf[
                (maf["Mutation_Status"] == "Somatic") &
                (maf["Variant_Classification"].isin(functional_variants))
            ]
            if mut.empty:
                continue

            patient_ids = (mut["Tumor_Sample_Barcode"].astype(str).str.split("-").str[:3].str.join("-"))
            mutation_frames.append(
                pd.DataFrame({"patient_id": patient_ids, "gene": mut["Hugo_Symbol"]})
            )

        if not mutation_frames:
            raise ValueError(f"No mutation records found in: {self.output_dir}")

        mutations = pd.concat(mutation_frames, ignore_index=True).dropna().drop_duplicates()
        cleaned_data = (mutations.assign(mutated=1)
            .pivot(index="patient_id", columns="gene", values="mutated")
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        cleaned_data.columns.name = None
        
        output_path = os.path.join(output_file_path, output_file_name)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cleaned_data.to_csv(output_path, index=False)
        return cleaned_data
