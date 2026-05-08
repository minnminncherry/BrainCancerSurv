import torch
from torch.utils.data import DataLoader
import numpy as np

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
    kwargs = {'num_workers': 8} if torch.cuda.is_available() else {}
    
    if args.modality in ["omics", "snn", "mlp_per_path"]:
        collate_fn = _collate_genomic
    else:
        raise NotImplementedError(f"Modality {args.modality} not implemented")

    ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset) * 0.1), replace=False)
    loader = DataLoader(split_dataset, batch_size=batch_size, sampler=SubsetSequentialSampler(ids), 
                       collate_fn=collate_fn, drop_last=False, **kwargs)
    
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
    img = torch.ones([1, 1])
    omics = torch.stack([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    return [img, omics, label, event_time, c, clinical_data_list]