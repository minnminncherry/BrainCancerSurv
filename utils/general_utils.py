import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


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
    kwargs = {"num_workers": args.num_workers} if torch.cuda.is_available() else {}
    
    if args.modality in ["mlp", "snn", "gen2vec", "resmlp", "resnet_mlp"]:
        collate_fn = _collate_genomic
    elif args.modality in ["mil","transmil"]:
        collate_fn = _collate_wsi
        batch_size = args.batch_size
    elif args.modality in ["multimodal_late_fusion"]:
        collate_fn = _collate_multimodal
        batch_size = args.batch_size
    else:
        raise NotImplementedError(f"Modality {args.modality} not implemented")

    loader = DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=training,
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
    # img = torch.ones(len(batch), 1)
    omics = torch.stack([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    clinical_data_list = [item[5] for item in batch]
    return omics, label, event_time, c, clinical_data_list


def _collate_wsi(batch):
    """
    Pad variable-length WSI bags to the longest bag in the batch.
    """
    img = pad_sequence(
        [item[0].float() for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    x = img
    label = torch.LongTensor([item[1] for item in batch])
    event_time = torch.FloatTensor([item[2] for item in batch])
    c = torch.FloatTensor([item[3] for item in batch])
    clinical_data_list = [item[4] for item in batch]
    return x, label, event_time, c, clinical_data_list

def _collate_multimodal(batch):
    """
    Collate function for multimodal datasets.
    """
    img = pad_sequence(
        [item[0].float() for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    omics = torch.stack([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    clinical_data_list = [item[5] for item in batch]
    return img, omics, label, event_time, c, clinical_data_list
