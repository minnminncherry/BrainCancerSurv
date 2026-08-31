import pandas as pd
import os
import numpy as np
from inspect import signature
from dataset.SurvivalGenomicDataset import SurvivalGenomicDataset
from dataset.SurvivalWSIDataset import SurvivalWSIDataset
from dataset.SurvivalMultimodalDataset import SurvivalMultimodalDataset
from utils.process_args import _process_args
from utils.core_utils import _train_val, save_final_fold_summary

def main(args, dataset_class):

    # prepartaion for 5 fold cv study
    folds = 5
    model_paths = []
    fold_metrics = []

    for i in range(folds):
        
        train_split, test_split, scalar  = dataset_class.return_splits(
            args,
            fold_indices=list(range(i, len(dataset_class.metadata), folds))
        )

        print("sample of training split:", type(train_split))
        results_dict, metrics, best_model_path, eval_results = _train_val(args, train_split, test_split, i)
        (
            train_cindex,
            total_loss,
            val_cindex,
            val_loss,
            training_time_seconds,
            training_time_per_sample_seconds,
            inference_time_seconds,
            inference_time_per_sample_seconds,
        ) = metrics
        model_paths.append(best_model_path)
        fold_metrics.append(
            {
                "fold": i,
                "train_cindex": train_cindex,
                "train_loss": total_loss,
                "final_val_cindex": val_cindex,
                "final_val_loss": val_loss,
                "training_time_seconds": training_time_seconds,
                "training_time_per_sample_seconds": training_time_per_sample_seconds,
                "inference_time_seconds": inference_time_seconds,
                "inference_time_per_sample_seconds": inference_time_per_sample_seconds,
                "best_model_path": best_model_path,
            }
        )

    save_final_fold_summary(fold_metrics, dataset_class.modality, args.genomic_file_name)
    if fold_metrics:
        avg_val_cindex = float(np.mean([metric["final_val_cindex"] for metric in fold_metrics]))
        avg_val_loss = float(np.mean([metric["final_val_loss"] for metric in fold_metrics]))
        avg_training_time = float(np.mean([metric["training_time_seconds"] for metric in fold_metrics]))
        avg_inference_time = float(np.mean([metric["inference_time_seconds"] for metric in fold_metrics]))
        print(f"\nAverage final val_cindex across {folds} folds: {avg_val_cindex:.4f}")
        print(f"Average final val_loss across {folds} folds: {avg_val_loss:.4f}")
        print(f"Average training time across {folds} folds: {avg_training_time:.4f}s")
        print(f"Average inference time across {folds} folds: {avg_inference_time:.4f}s")

    print(f"\nTraining completed! All models saved to result/model_checkpoints/")
    return model_paths

if __name__ == "__main__":

    DATASET_FACTORY = {
    "SurvivalGenomicDataset": SurvivalGenomicDataset,
    "SurvivalWSIDataset": SurvivalWSIDataset,
    "SurvivalMultimodalDataset": SurvivalMultimodalDataset,
    }

    args = _process_args()
    dataset_class = DATASET_FACTORY[args.data_factory]
    dataset_args = {
        "label_file": args.label_file,
        "genomic_dir": args.genomic_dir,
        "genomic_file_name": args.genomic_file_name,
        "slide_file_name": args.slide_file_name,
        "slide_dir": args.slide_dir,
        "h5_dir": args.h5_dir,
        "pt_dir": args.pt_dir,
        "wsi_feature_dim": args.wsi_feature_dim,
        "seed": args.seed,
        "label_col": args.label_col,
        "n_bins": args.n_bins,
        "n_classes": args.n_classes,
        "type_of_pathway": args.type_of_pathway,
        "modality": args.modality,
        "opt": args.opt,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    valid_args = signature(dataset_class.__init__).parameters
    dataset_args = {key: value for key, value in dataset_args.items() if key in valid_args}
    data_factory = dataset_class(**dataset_args)
    args.data_factory = data_factory
    
    # elif(data_fac == 'SurvivalWSIDataset'):
    #     data_factory = args.data


    # args.data_factory = data_factory
    # # print(args.data_factory.genomic_feature_cols)
    # # print(len(args.data_factory.genomic_feature_cols))
    # print(f"Data factory initialized with {len(args.data_factory.metadata)} samples and {len(args.data_factory.genomic_feature_cols)} genomic features.")

    model_paths = main(args, data_factory)
