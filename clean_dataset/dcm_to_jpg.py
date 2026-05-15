import argparse
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
import pydicom


def _normalize_to_uint8(pixel_array: np.ndarray) -> np.ndarray:
    pixel_array = pixel_array.astype(np.float32)

    min_val = float(pixel_array.min())
    max_val = float(pixel_array.max())
    if max_val <= min_val:
        return np.zeros(pixel_array.shape, dtype=np.uint8)

    scaled = (pixel_array - min_val) / (max_val - min_val)
    scaled = np.clip(scaled * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def convert_dcm_to_jpg(dcm_path: Path, output_path: Path) -> None:
    try:
        print(f"Reading DICOM file: {dcm_path}")
        dicom = pydicom.dcmread(str(dcm_path))

        if not hasattr(dicom, 'pixel_array'):
            raise ValueError(f"No pixel data found in {dcm_path}")

        pixel_array = dicom.pixel_array
        print(f"Pixel array shape: {pixel_array.shape}, dtype: {pixel_array.dtype}")

        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            print("Inverting MONOCHROME1 image")
            pixel_array = pixel_array.max() - pixel_array

        image_array = _normalize_to_uint8(pixel_array)
        image = Image.fromarray(image_array)

        if image.mode != "L":
            print(f"Converting image mode from {image.mode} to L")
            image = image.convert("L")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=95)
        print(f"Successfully converted: {dcm_path} -> {output_path}")

    except Exception as e:
        print(f"Error converting {dcm_path}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


def convert_directory(input_dir: Path, output_dir: Path) -> None:
    dcm_files = list(input_dir.rglob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found under: {input_dir}")

    print(f"Found {len(dcm_files)} DCM files to convert")

    success_count = 0
    error_count = 0

    for dcm_file in dcm_files:
        try:
            relative_path = dcm_file.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".jpg")
            convert_dcm_to_jpg(dcm_file, output_path)
            success_count += 1
        except Exception as e:
            print(f"Failed to convert {dcm_file}: {e}")
            error_count += 1

    print(f"Conversion complete: {success_count} successful, {error_count} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DICOM (.dcm) files to JPG.")
    parser.add_argument("--input_dir", required=True, help="Directory containing .dcm files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save .jpg files.")
    args = parser.parse_args()

    convert_directory(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
