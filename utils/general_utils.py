"""Build modality-specific data loaders and collate patient samples."""
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def _get_split_loader(args, split_dataset, training=False, testing=False, weighted=False, batch_size=1):
    """Keep optional legacy arguments so existing callers remain compatible."""
    collators = {
        "mlp": _collate_genomic, "snn": _collate_genomic,
        "gen2vec": _collate_genomic, "resmlp": _collate_genomic,
        "resnet_mlp": _collate_genomic,
        "ViT": _collate_mri, "3dcnn": _collate_mri,
        "mil": _collate_wsi, "transmil": _collate_wsi,
        "multimodal_late_fusion": _collate_multimodal,
        "multimodal_cross_attention": _collate_multimodal,
        "multimodal_mri_late_fusion": _collate_mri_gene,
        "multimodal_mri_cross_attention": _collate_mri_gene,
        "multimodal_wsi_mri_late_fusion": _collate_wsi_mri_gene,
        "multimodal_wsi_mri_cross_attention": _collate_wsi_mri_gene,
        "mutation_guided_late_fusion": _collate_wsi_mri_gene,
        "mutation_guided_cross_attention": _collate_wsi_mri_gene,
    }
    if args.modality not in collators:
        raise NotImplementedError(f"Modality {args.modality} not implemented")
    collate = collators[args.modality]
    if collate is not _collate_genomic:
        batch_size = args.batch_size
    workers = args.num_workers if torch.cuda.is_available() else 0
    return DataLoader(split_dataset, batch_size=batch_size, shuffle=training,
                      collate_fn=collate, num_workers=workers, drop_last=False)


def _collate_targets(batch):
    """Every sample ends with label, event time, censoring, and clinical metadata."""
    return (
        torch.LongTensor([item[-4] for item in batch]),
        torch.FloatTensor([item[-3] for item in batch]),
        torch.FloatTensor([item[-2] for item in batch]),
        [item[-1] for item in batch],
    )


def _collate_genomic(batch):
    return torch.stack([item[0] for item in batch]), *_collate_targets(batch)


def _collate_mri(batch):
    """Pad MRI volumes to the largest depth, height, and width in the batch."""
    shape = [max(item[0].shape[axis] for item in batch) for axis in range(1, 4)]
    volumes = torch.zeros((len(batch), 1, *shape), dtype=torch.float32)
    for i, item in enumerate(batch):
        depth, height, width = item[0].shape[1:]
        volumes[i, :, :depth, :height, :width] = item[0]
    return volumes, *_collate_targets(batch)


def _collate_wsi(batch):
    patches = pad_sequence([item[0].float() for item in batch], batch_first=True)
    return patches, *_collate_targets(batch)


def _collate_mri_gene(batch):
    volumes, *targets = _collate_mri(batch)
    genes = torch.stack([item[1] for item in batch])
    return volumes, genes, *targets


def _collate_multimodal(batch):
    patches = pad_sequence([item[0].float() for item in batch], batch_first=True)
    genes = torch.stack([item[1] for item in batch])
    return patches, genes, *_collate_targets(batch)


def _collate_wsi_mri_gene(batch):
    volumes, *targets = _collate_mri(batch)
    patches = pad_sequence([item[1] for item in batch], batch_first=True)
    genes = torch.stack([item[2] for item in batch])
    return volumes, patches, genes, *targets
