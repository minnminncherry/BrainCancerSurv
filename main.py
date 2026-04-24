import pandas as pd
import os
import numpy as np
from dataset.cross_validation import fold_arange
from dataset.SurvivalGenomicDataset import SurvivalGenomicDataset
from utils.process_args import _process_args

def main(args):

    # prepartaion for 5 fold cv study
    folds = 5

    for i in range(folds):
        
        dataset = args.data_factory.return_splits(
            args,
            fold_arange(n_samples=len(args.data_factory.labels))

        )
