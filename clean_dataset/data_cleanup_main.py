
import sys
import pandas as pd
from data_normalization import DataNormalization_genomic_data
from clean_genomic_data import GenomicDataCleaner
import os
import yaml
import shutil

def main(args1, args2, args3):

    date = pd.Timestamp.now().strftime("%Y-%m-%d")
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
            print(f"Copying cleaned genomic data from {output_genomic_dir} to {destination_path}")
            shutil.copy(output_genomic_dir+f"cleaned_genomic_data_{date}.csv", destination_path+f"/raw_{args2}.csv")
        else:
            print("Failed to connect to database")

    elif(args1 == 'GENERATE_TRANSFORMED_GENOMIC_DATA'):
        hallmarks_data_path = os.path.join(data['pathway_clean_file_path']['hallmark_pathway_matrix_output_file'])
        if args2 == 'gbm' and gen_cls.conn:
            gen_cls.transform_genomic_data(output_genomic_dir+f"cleaned_genomic_data_{date}.csv", hallmarks_data_path, final_genomic_data_file_name, fill_type='mean', cancer_type='gbm')
        elif args2 == 'lgg' and gen_cls.conn:
             gen_cls.transform_genomic_data(output_genomic_dir+f"cleaned_genomic_data_{date}.csv", hallmarks_data_path, final_genomic_data_file_name, fill_type='mean', cancer_type='lgg')

    elif(args1 == 'NORMALIZE_GENOMIC_DATA'):

        normal_cls = DataNormalization_genomic_data(os.path.join(final_genomic_data_file_name, args2)+".csv", target_column='case_id', cancer_type=args2)
        # df_pivot_genomic_data: pd.DataFrame,method: str = "zscore", eps: float = 1e-8, output_path: str = "./"
        if args2 == 'gbm' and args3 in ['zscore']:
            normal_cls.normalize_genomic_data(method=args3, output_path=final_genomic_data_file_name)
        elif args2 == 'lgg' and args3 in ['zscore']:
            normal_cls.normalize_genomic_data(method=args3, output_path=final_genomic_data_file_name)
        elif args2 == 'gbm' and args3 in ['minmax']:
            normal_cls.normalize_genomic_data(method=args3, output_path=final_genomic_data_file_name)
        elif args2 == 'lgg' and args3 in ['minmax']:
            normal_cls.normalize_genomic_data(method=args3, output_path=final_genomic_data_file_name)
        
    else:
        print("Invalid arguments provided")

    # Disconnect from database
    gen_cls.disconnect_db()

if __name__ == "__main__":

    # args1 is for instruction and args2 for cancer type and args3 for normalization method
    args1, args2, args3 = sys.argv[1:]

    # args1 for instruction and args2 for cancer type
    main(args1, args2, args3)