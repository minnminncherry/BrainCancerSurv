
import sys
import pandas as pd
from data_normalization import DataNormalization_genomic_data
from clean_genomic_data import GenomicDataCleaner
from WSI_folder_cleaning import WSIDataCleaner
from mri_folder_cleaning import MRIDataCleaner
import os
import yaml
import shutil

def main(args1, args2=None, args3=None):

    db_name = None
    date = pd.Timestamp.now().strftime("%Y-%m-%d")
    script_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    yaml_file_path = os.path.join(project_root, "config.yaml")
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    if (args2 == 'gbm'):
        input_dir = data['gbm_src_file_paths']['genomic_data_dir']
        output_dir = data['gbm_clean_file_paths']['genomic_data_dir']
        output_genomic_dir = data['mysql_file_paths']
        final_genomic_data_file_name = data['gbm_clean_file_paths']['final_genomic_data_file_name']
        json_file_path = os.path.join(input_dir, data['gbm_src_file_paths']['genomic_metadata_file_name'])
        db_name = data['database']['database_name_gbm']
        wsi_input_dir = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['gbm_wsi_input_file_path']))
        wsi_h5_dir = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['gbm_wsi_h5_file_path']))
        wsi_metadata_file_path = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['gbm_wsi_metadata_file_path']))
        wsi_final_file_path = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['gbm_wsi_final_file_path']))
        wsi_new_file_path = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['gbm_wsi_new_file_path']))
        mri_input_dir = os.path.abspath(os.path.join(script_dir, data['MRI_file_path']['gbm_mri_input_file_path']))

    elif (args2 == 'lgg'):
        input_dir = data['lgg_src_file_paths']['genomic_data_dir']
        output_dir = data['lgg_clean_file_paths']['genomic_data_dir']
        output_genomic_dir = data['mysql_file_paths']
        final_genomic_data_file_name = data['lgg_clean_file_paths']['final_genomic_data_file_name']
        db_name = data['database']['database_name_lgg']
        json_file_path = os.path.join(input_dir, data['lgg_src_file_paths']['genomic_metadata_file_name'])
        wsi_input_dir = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['lgg_wsi_input_file_path']))
        wsi_h5_dir = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['lgg_wsi_h5_file_path']))
        wsi_metadata_file_path = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['lgg_wsi_metadata_file_path']))
        wsi_final_file_path = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['lgg_wsi_final_file_path']))
        wsi_new_file_path = os.path.abspath(os.path.join(script_dir, data['WSI_file_path']['lgg_wsi_new_file_path']))
    else:
        print("Invalid cancer type provided. Please specify 'gbm' or 'lgg'.")
        return
    
    gen_cls = GenomicDataCleaner(input_dir, output_dir)
    wsi_cls = WSIDataCleaner(wsi_final_file_path)
    mri_cls = MRIDataCleaner(mri_input_dir)

    if args1 != 'TEXT_TO_CSV':
        gen_cls.connect_db(db_name)
        wsi_cls.connect_db(db_name)
        mri_cls.connect_db(db_name)

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
    
    elif(args1 == "INSERT_FILE_NAME_TO_DB"):
        if gen_cls.conn:
                genomic_records = gen_cls.collect_file_records(input_dir, ".tsv")
                gen_cls.insert_file_records(genomic_records)
        else:
            print("Failed to connect to database")

    elif(args1 == 'GENERATE_CLEAN_GENOMIC_DATA'):
        destination_path = None
        if gen_cls.conn:
            if args2 == 'gbm':
                destination_path = data['gbm_clean_file_paths']['genomic_data_dir']
                if args3 == 'tpm':
                    gen_cls.generate_clean_genomic_data(output_genomic_dir, unstranded_column='tpm_unstranded')
                elif args3 == 'fpkm':
                    gen_cls.generate_clean_genomic_data(output_genomic_dir, unstranded_column='fpkm_uq_unstranded')

            elif args2 == 'lgg':
                destination_path = data['lgg_clean_file_paths']['genomic_data_dir']
                if args3 == 'tpm':
                    gen_cls.generate_clean_genomic_data(output_genomic_dir, unstranded_column='tpm_unstranded') 
                elif args3 == 'fpkm':
                    gen_cls.generate_clean_genomic_data(output_genomic_dir, unstranded_column='fpkm_uq_unstranded')

            print(f"Copying cleaned genomic data from {output_genomic_dir} to {destination_path}")
            shutil.copy(output_genomic_dir+f"cleaned_genomic_data_{date}.csv", destination_path+f"/cleaned_genomic_data_{date}.csv")
        else:
            print("Failed to connect to database")
    
    elif(args1 == 'GENERATE_TRANSFORMED_GENOMIC_DATA'):
        hallmarks_data_path = os.path.join(data['pathway_clean_file_path']['hallmark_pathway_matrix_output_file'])
        if args2 == 'gbm' and args3 == 'tpm' and gen_cls.conn:
            gen_cls.transform_genomic_data(output_genomic_dir+f"cleaned_genomic_data_{date}.csv", hallmarks_data_path, final_genomic_data_file_name, unstranded_column='tpm_unstranded', cancer_type='gbm')
        elif args2 == 'lgg' and args3 == 'tpm' and gen_cls.conn:
             gen_cls.transform_genomic_data(output_genomic_dir+f"cleaned_genomic_data_{date}.csv", hallmarks_data_path, final_genomic_data_file_name, unstranded_column='tpm_unstranded', cancer_type='lgg')
        elif args2 == 'gbm' and args3 == 'fpkm' and gen_cls.conn:
            gen_cls.transform_genomic_data(output_genomic_dir+f"cleaned_genomic_data_{date}.csv", hallmarks_data_path, final_genomic_data_file_name, unstranded_column='fpkm_uq_unstranded', cancer_type='gbm')
        elif args2 == 'lgg' and args3 == 'fpkm' and gen_cls.conn:
             gen_cls.transform_genomic_data(output_genomic_dir+f"cleaned_genomic_data_{date}.csv", hallmarks_data_path, final_genomic_data_file_name, unstranded_column='fpkm_uq_unstranded', cancer_type='lgg')
    
        
    elif(args1 == 'NORMALIZE_GENOMIC_DATA'):

        normal_cls = DataNormalization_genomic_data(os.path.join(final_genomic_data_file_name, args2+".csv"), target_column='case_id', cancer_type=args2)
        # df_pivot_genomic_data: pd.DataFrame,method: str = "zscore", eps: float = 1e-8, output_path: str = "./"
        if args2 == 'gbm' and args3 in ['zscore']:
            normal_cls.normalize_genomic_data(method=args3, output_path=final_genomic_data_file_name)
        elif args2 == 'lgg' and args3 in ['zscore']:
            normal_cls.normalize_genomic_data(method=args3, output_path=final_genomic_data_file_name)
        elif args2 == 'gbm' and args3 in ['minmax']:
            normal_cls.normalize_genomic_data(method=args3, output_path=final_genomic_data_file_name)
        elif args2 == 'lgg' and args3 in ['minmax']:
            normal_cls.normalize_genomic_data(method=args3, output_path=final_genomic_data_file_name)
    
    elif(args1 == 'TEXT_TO_CSV'):
        os.makedirs(output_dir, exist_ok=True)
        clinical_dir = os.path.dirname(input_dir.rstrip(os.sep))

        if args2 == 'lgg':
            text_file_path = os.path.join(clinical_dir, "LGG_survival.txt")
            csv_file_path = os.path.join(output_dir, "LGG_survival.csv")
            gen_cls.change_plain_text_file_to_csv(text_file_path, csv_file_path)
        elif args2 == 'gbm':
            text_file_path = os.path.join(clinical_dir, "TCGA.GBM.sampleMap_GBM_clinicalMatrix")
            csv_file_path = os.path.join(output_dir, "GBM_clinicalMatrix.csv")
            gen_cls.change_plain_text_file_to_csv(text_file_path, csv_file_path)
        else:
            print("Invalid cancer type provided for TEXT_TO_CSV. Please specify 'gbm' or 'lgg'.")

    elif(args1 == 'COPY_WSI_FILES_ONE_FOLDER'):
        if args2 == 'gbm':
            wsi_cls.copy_wsi_files_to_one_folder(wsi_input_dir, wsi_new_file_path, wsi_final_file_path)
        elif args2 == 'lgg':
            wsi_cls.copy_wsi_files_to_one_folder(wsi_input_dir, wsi_new_file_path, wsi_final_file_path)

    elif(args1 == 'INSERT_WSI_METADATA_TO_DB'):
        if wsi_cls.conn:
            json_data = wsi_cls.get_wsi_metadata_json_data(wsi_metadata_file_path)
            wsi_cls.insert_wsi_json_data_to_db(json_data, "wsi_metadata_json_data")

    elif(args1 == 'GENERATE_WSI_METADATA_CSV'):
        if wsi_cls.conn:
            if args2 == 'gbm':
                wsi_cls.generate_wsi_metadata_csv(os.path.join(wsi_final_file_path, "wsi_metadata_gbm.csv"))
            elif args2 == 'lgg':
                wsi_cls.generate_wsi_metadata_csv(os.path.join(wsi_final_file_path, "wsi_metadata_lgg.csv"))

    elif(args1 == 'INSERT_H5_FILE_NAMES_TO_DB'):
        if wsi_cls.conn:
            wsi_cls.insert_h5_filename_DB(wsi_h5_dir, "raw_wsi_h5_filenames")
    
    elif(args1 == 'INSERT_MRI_FILE_NAMES_TO_DB'):
        if mri_cls.conn:
            mri_cls.load_filename_to_db(mri_input_dir, "raw_patient_info_MRI")

    else:
        print("Invalid arguments provided")

    # Disconnect from database
    gen_cls.disconnect_db()

if __name__ == "__main__":

    # args1 is for instruction and args2 for cancer type and args3 for normalization method
    args = sys.argv[1:]

    args1 = args[0] if len(args) > 0 else None
    args2 = args[1] if len(args) > 1 else None
    args3 = args[2] if len(args) > 2 else None

    # args1 for instruction and args2 for cancer type
    main(args1, args2, args3)
