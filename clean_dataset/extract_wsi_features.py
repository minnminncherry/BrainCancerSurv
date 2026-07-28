"""
Extract features from WSI patches using coordinates from H5 files.
Saves pre-computed features as PT files for efficient dataset loading.

Usage:
    python extract_wsi_features.py --h5_dir <path_to_h5_dir> --slide_dir <path_to_slides> --pt_dir <path_to_pt_output> --encoder resnet50 --feature_dim 1024
"""

import os
import sys
import argparse
import h5py
import numpy as np
import torch
import openslide
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.core_utils import encoder

CUDA_RETRY_ERRORS = (
    "CUDNN_STATUS_NOT_INITIALIZED",
    "CUDA out of memory",
    "cuDNN error",
)


def is_cuda_retry_error(error):
    error_message = str(error)
    return any(error_text in error_message for error_text in CUDA_RETRY_ERRORS)


def encode_patch_batch(batch, encoder_model, device):
    batch_tensor = torch.stack(batch, dim=0).to(device, non_blocking=True)

    try:
        batch_features = encoder_model(batch_tensor)
        return batch_features.detach().cpu()
    except RuntimeError as error:
        if device.type != "cuda" or not is_cuda_retry_error(error):
            raise

        torch.cuda.empty_cache()

        if "CUDNN_STATUS_NOT_INITIALIZED" in str(error) and torch.backends.cudnn.enabled:
            print("cuDNN failed to initialize. Retrying this batch with cuDNN disabled.")
            torch.backends.cudnn.enabled = False
            return encode_patch_batch(batch, encoder_model, device)

        if len(batch) == 1:
            raise

        middle = len(batch) // 2
        print(f"CUDA failed for batch size {len(batch)}. Retrying as {middle} + {len(batch) - middle}.")
        first_features = encode_patch_batch(batch[:middle], encoder_model, device)
        second_features = encode_patch_batch(batch[middle:], encoder_model, device)
        return torch.cat([first_features, second_features], dim=0)


def _resize_feature_dim(features, output_dim):
    """Resize feature tensor to the desired output dimension by truncation or zero padding."""
    current_dim = features.shape[-1]
    if current_dim == output_dim:
        return features
    if current_dim > output_dim:
        return features[:, :output_dim]

    pad = torch.zeros(features.shape[0], output_dim - current_dim, dtype=features.dtype)
    return torch.cat([features, pad], dim=1)


def extract_features_from_wsi(h5_file, slide_file, encoder_model, img_transform, device, batch_size=32, output_dim=None):
    """
    Extract features from a WSI slide using patch coordinates from h5 file.
    
    Args:
        h5_file: Path to h5 file containing patch coordinates
        slide_file: Path to original WSI slide (.svs file)
        encoder_model: Pre-trained encoder model
        img_transform: Image transformation pipeline
        device: Torch device for extraction
        batch_size: Batch size for feature extraction
        output_dim: Optional output feature dimension to resize features to
        
    Returns:
        features: Tensor of shape (n_patches, feature_dim)
        coords: List of patch coordinates
    """
    
    print(f"Loading patch coordinates from: {h5_file}")
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"H5 file not found: {h5_file}")
    
    with h5py.File(h5_file, "r") as f:
        coords = np.asarray(f["coords"][:], dtype=np.int64)
        patch_level = int(f["coords"].attrs.get("patch_level", 0))
        patch_size = int(f["coords"].attrs.get("patch_size", 224))
    
    print(f"Loading slides from: {slide_file}")
    if not os.path.exists(slide_file):
        raise FileNotFoundError(f"Slide file not found: {slide_file}")
    
    slide_img = openslide.OpenSlide(slide_file)
    
    print(f"Extracting {len(coords)} patches at level {patch_level} with size {patch_size}")
    features = []
    batch = []
    counter = 0
    
    try:
        with torch.inference_mode():
            for x, y in tqdm(coords, desc="Extracting patches"):
                # Read patch from original slide
                patch_img = slide_img.read_region(
                    (int(x), int(y)),
                    patch_level,
                    (patch_size, patch_size),
                ).convert("RGB")
                
                # Transform to tensor
                patch_tensor = img_transform(patch_img)
                batch.append(patch_tensor)
                counter += 1
                
                # Encode batch when it reaches batch_size
                if len(batch) == batch_size:
                    features.append(encode_patch_batch(batch, encoder_model, device))
                    batch = []
            
            # Encode remaining patches
            if batch:
                features.append(encode_patch_batch(batch, encoder_model, device))
    finally:
        slide_img.close()
    
    # Concatenate all batch features
    features = torch.cat(features, dim=0)
    
    # Flatten if needed
    if features.dim() > 2:
        features = features.reshape(features.shape[0], -1)

    if output_dim is not None:
        features = _resize_feature_dim(features, output_dim)
        print(f"Resized features to output dimension: {output_dim}, features shape: {features.shape}")
    
    print(f"Extracted features shape: {features.shape} and length of features: {len(features)}")
    return features.float(), coords.tolist()


def get_device(device_name="auto", disable_cudnn=False):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is False.")
        torch.cuda.set_device(device.index or 0)
        torch.backends.cudnn.enabled = not disable_cudnn
        torch.backends.cudnn.benchmark = not disable_cudnn
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        print("Using CPU")

    return device


def main(args):
    """Main preprocessing pipeline."""
    
    h5_dir = Path(args.h5_dir)
    slide_dir = Path(args.slide_dir)
    pt_dir = Path(args.pt_dir)
    device = get_device(args.device, disable_cudnn=args.disable_cudnn)
    
    if not h5_dir.exists():
        raise ValueError(f"H5 directory does not exist: {h5_dir}")
    if not slide_dir.exists():
        raise ValueError(f"Slide directory does not exist: {slide_dir}")
    
    # Create PT output directory if it doesn't exist
    pt_dir.mkdir(parents=True, exist_ok=True)
    print(f"PT files will be saved to: {pt_dir}")
    
    print(f"Loading encoder: {args.encoder}")
    encoder_model, img_transform = encoder(args.encoder)
    encoder_model = encoder_model.to(device)
    encoder_model.eval()
    
    # Find all h5 files
    h5_files = sorted(h5_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} H5 files to process")
    
    for h5_file in tqdm(h5_files, desc="Processing slides"):
        # Corresponding PT file path (in pt_dir)
        pt_file = pt_dir / f"{h5_file.stem}.pt"
        
        if pt_file.exists():
            print(f"PT file already exists, skipping: {pt_file}")
            continue
        
        # Find corresponding slide file
        slide_name = h5_file.stem  # Remove .h5 extension
        slide_file = slide_dir / f"{slide_name}.svs"
        
        if not slide_file.exists():
            print(f"Warning: Slide file not found for {slide_name}, skipping")
            continue
        
        try:
            # Extract features and save
            features, coords = extract_features_from_wsi(
                str(h5_file),
                str(slide_file),
                encoder_model,
                img_transform,
                device=device,
                batch_size=args.batch_size,
                output_dim=args.feature_dim,
            )
            
            # Save features as PT file
            torch.save(features, str(pt_file))
            print(f"Saved features to: {pt_file}\n")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
        except Exception as e:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Error processing {h5_file}: {e}\n")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract WSI features from patch coordinates and save as PT files"
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
        help="Directory containing H5 files with patch coordinates"
    )
    parser.add_argument(
        "--slide_dir",
        type=str,
        required=True,
        help="Directory containing original WSI slide files (.svs)"
    )
    parser.add_argument(
        "--pt_dir",
        type=str,
        required=True,
        help="Directory to save extracted features as PT files"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet50",
        help="Encoder model name (default: resnet50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction (default: 32)"
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=2048,
        help="Optional output feature dimension. If set, extracted features are truncated or padded to this size."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for feature extraction: auto, cuda, or cpu (default: auto)"
    )
    parser.add_argument(
        "--disable_cudnn",
        action="store_true",
        help="Disable cuDNN if CUDA gives CUDNN_STATUS_NOT_INITIALIZED"
    )
    
    args = parser.parse_args()
    main(args)
