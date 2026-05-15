import pandas as pd
import os
import numpy as np
from dataset.SurvivalGenomicDataset import SurvivalGenomicDataset
from utils.process_args import _process_args
from utils.core_utils import _train_val, save_final_fold_summary

def main(args):

    # prepartaion for 5 fold cv study
    folds = 5
    model_paths = []
    fold_metrics = []

    for i in range(folds):
        
        train_split, test_split, scalar  = args.data_factory.return_splits(
            args,
            fold_indices=list(range(i, len(args.data_factory.metadata), folds))
        )
    
        results_dict, (total_acc, total_loss, val_acc, val_loss), best_model_path, eval_results = _train_val(args, train_split, test_split, i)
        model_paths.append(best_model_path)
        fold_metrics.append(
            {
                "fold": i,
                "final_val_acc": val_acc,
                "final_val_loss": val_loss,
                "best_model_path": best_model_path,
            }
        )
        # print("train_split: ", train_split)
        # print("test_split: ", test_split)
        # print("scalar: ", scalar)

    save_final_fold_summary(fold_metrics)
    if fold_metrics:
        avg_val_acc = float(np.mean([metric["final_val_acc"] for metric in fold_metrics]))
        avg_val_loss = float(np.mean([metric["final_val_loss"] for metric in fold_metrics]))
        print(f"\nAverage final val_acc across {folds} folds: {avg_val_acc:.4f}")
        print(f"Average final val_loss across {folds} folds: {avg_val_loss:.4f}")

    print(f"\nTraining completed! All models saved to result/model_checkpoints/")
    return model_paths

if __name__ == "__main__":
    args = _process_args()
    data_factory = SurvivalGenomicDataset(
        label_file=args.label_file,
        genomic_dir=args.genomic_dir,
        genomic_file_name=args.genomic_file_name,
        seed=args.seed,
        label_col=args.label_col,
        n_bins=args.n_bins,
        n_classes=args.n_classes,
        type_of_pathway=args.type_of_pathway,
        modality=args.modality,
        opt=args.opt,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    args.data_factory = data_factory
    model_paths = main(args)
