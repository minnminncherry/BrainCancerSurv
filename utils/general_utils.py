import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """Iterate over a fixed list of indices in the provided order."""

    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _get_split_loader(args, split_dataset, training=False, testing=False, weighted=False, batch_size=1):
    r"""
    Take a dataset and make a dataloader from it using a custom collate function.

    Args:
        - args : argspace.Namespace
        - split_dataset : SurvivalDataset
        - training : Boolean
        - testing : Boolean
        - weighted : Boolean
        - batch_size : Int
    
    Returns:
        - loader : Pytorch Dataloader
    """
    kwargs = {"num_workers": 8} if torch.cuda.is_available() else {}

    if isinstance(split_dataset, dict):
        x_tensor = torch.tensor(split_dataset["x"], dtype=torch.float32)
        y_tensor = torch.tensor(split_dataset["y"], dtype=torch.long)
        split_dataset = TensorDataset(x_tensor, y_tensor)
        collate_fn = None
    elif args.modality in ["mlp", "kmeans"]:
        collate_fn = _collate_genomic
    else:
        raise NotImplementedError(f"Modality {args.modality} not implemented")

    # Use all data for both training and testing (not just 10% of test)
    loader = DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=training,  # Shuffle training data, not test
        collate_fn=collate_fn,
        drop_last=False,
        **kwargs,
    )
    
    return loader 

def _collate_genomic(batch):
    r"""
    Collate function for the unimodal omics models
    
    Args:
        - batch
    
    Returns:
        - img : torch.Tensor
        - omics : torch.Tensor
        - label : torch.LongTensor
        - event_time : torch.FloatTensor
        - c : torch.FloatTensor
        - clinical_data_list : List
    """
    if len(batch[0]) == 2:  # TensorDataset case: (omics, label)
        omics = torch.stack([item[0] for item in batch], dim=0)
        label = torch.LongTensor([item[1] for item in batch])
        img = torch.ones(len(batch), 1)
        event_time = torch.zeros(len(batch))
        c = torch.zeros(len(batch))
        clinical_data_list = [{}] * len(batch)
    else:  # Full dataset case: (img, omics, label, event_time, c, clinical_data)
        img = torch.ones(len(batch), 1)
        omics = torch.stack([item[1] for item in batch], dim=0)
        label = torch.LongTensor([item[2] for item in batch])
        event_time = torch.FloatTensor([item[3] for item in batch])
        c = torch.FloatTensor([item[4] for item in batch])
        clinical_data_list = [item[5] for item in batch]

    # print("clinical data: ", clinical_data_list)
    return img, omics, label, event_time, c, clinical_data_list
