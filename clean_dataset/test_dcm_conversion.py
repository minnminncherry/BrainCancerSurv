#!/usr/bin/env python3
"""
Test script for DCM to JPG conversion
This script helps diagnose issues with DICOM file conversion
"""

import sys
from pathlib import Path
import traceback

try:
    import pydicom
    from PIL import Image
    import numpy as np
    print("✓ All required packages are installed")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    sys.exit(1)


def test_dicom_file(dcm_path: Path):
    """Test if a DICOM file can be read and converted"""
    try:
        print(f"\nTesting DICOM file: {dcm_path}")

        # Check if file exists
        if not dcm_path.exists():
            print(f"✗ File does not exist: {dcm_path}")
            return False

        # Try to read DICOM
        dicom = pydicom.dcmread(str(dcm_path))
        print(f"✓ Successfully read DICOM file")
        print(f"  - SOP Class: {getattr(dicom, 'SOPClassUID', 'Unknown')}")
        print(f"  - Modality: {getattr(dicom, 'Modality', 'Unknown')}")
        print(f"  - Photometric Interpretation: {getattr(dicom, 'PhotometricInterpretation', 'Unknown')}")

        # Check for pixel data
        if not hasattr(dicom, 'pixel_array'):
            print("✗ No pixel data found in DICOM file")
            return False

        pixel_array = dicom.pixel_array
        print(f"✓ Pixel data found: shape={pixel_array.shape}, dtype={pixel_array.dtype}")
        print(f"  - Value range: {pixel_array.min()} to {pixel_array.max()}")

        # Test conversion
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array
            print("✓ Applied MONOCHROME1 inversion")

        # Normalize to uint8
        pixel_array = pixel_array.astype(np.float32)
        min_val = float(pixel_array.min())
        max_val = float(pixel_array.max())

        if max_val <= min_val:
            normalized = np.zeros(pixel_array.shape, dtype=np.uint8)
        else:
            scaled = (pixel_array - min_val) / (max_val - min_val)
            scaled = np.clip(scaled * 255.0, 0, 255)
            normalized = scaled.astype(np.uint8)

        print(f"✓ Normalized to uint8: range {normalized.min()} to {normalized.max()}")

        # Create PIL image
        image = Image.fromarray(normalized)
        if image.mode != "L":
            image = image.convert("L")
            print(f"✓ Converted to grayscale mode")

        print(f"✓ PIL Image created: size={image.size}, mode={image.mode}")

        # Test save
        test_output = dcm_path.parent / f"test_{dcm_path.stem}.jpg"
        image.save(test_output, format="JPEG", quality=95)
        print(f"✓ Successfully saved test image: {test_output}")

        # Clean up test file
        if test_output.exists():
            test_output.unlink()
            print("✓ Cleaned up test file")

        return True

    except Exception as e:
        print(f"✗ Error processing {dcm_path}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_dcm_conversion.py <dicom_file_or_directory>")
        print("\nExamples:")
        print("  python test_dcm_conversion.py /path/to/file.dcm")
        print("  python test_dcm_conversion.py /path/to/dicom/directory")
        sys.exit(1)

    test_path = Path(sys.argv[1])

    if test_path.is_file():
        if test_path.suffix.lower() == '.dcm':
            success = test_dicom_file(test_path)
        else:
            print(f"✗ Not a .dcm file: {test_path}")
            success = False
    elif test_path.is_dir():
        dcm_files = list(test_path.rglob("*.dcm"))
        if not dcm_files:
            print(f"✗ No .dcm files found in directory: {test_path}")
            success = False
        else:
            print(f"Found {len(dcm_files)} DCM files")
            success_count = 0
            for dcm_file in dcm_files[:5]:  # Test first 5 files
                if test_dicom_file(dcm_file):
                    success_count += 1
            success = success_count > 0
            print(f"\nTested {min(5, len(dcm_files))} files: {success_count} successful")
    else:
        print(f"✗ Path does not exist: {test_path}")
        success = False

    if success:
        print("\n✓ DCM to JPG conversion should work!")
        print("You can now run the conversion script:")
        print("python clean_dataset/dcm_to_jpg.py --input_dir /path/to/dicom/files --output_dir /path/to/output")
    else:
        print("\n✗ DCM to JPG conversion has issues.")
        print("Please check:")
        print("1. Are your DICOM files valid?")
        print("2. Do they contain pixel data?")
        print("3. Are the file paths correct?")


if __name__ == "__main__":
    main()