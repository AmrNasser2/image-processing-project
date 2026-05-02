from dataclasses import dataclass

import numpy as np
import pydicom


@dataclass
class DicomReadResult:
    image: np.ndarray
    metadata: dict


def normalize_to_uint8(image):
    image = image.astype(np.float32)

    image = image - np.min(image)

    max_value = np.max(image)
    if max_value > 0:
        image = image / max_value * 255.0

    return image.astype(np.uint8)


def get_first_value(value):
    """
    DICOM tags such as WindowCenter and WindowWidth may contain multiple values.
    This helper safely returns the first one.
    """
    if isinstance(value, pydicom.multival.MultiValue):
        return value[0]
    return value


def apply_dicom_window(pixel_array, ds):
    """
    Converts raw DICOM pixel data into an 8-bit displayable image.

    Steps:
    1. Convert to float.
    2. Apply RescaleSlope and RescaleIntercept if available.
    3. Apply DICOM window center/width if available.
    4. Normalize to 0-255.
    """
    image = pixel_array.astype(np.float32)

    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    image = image * slope + intercept

    center = ds.get("WindowCenter", None)
    width = ds.get("WindowWidth", None)

    if center is None or width is None:
        return normalize_to_uint8(image)

    center = float(get_first_value(center))
    width = float(get_first_value(width))

    if width <= 0:
        return normalize_to_uint8(image)

    low = center - width / 2.0
    high = center + width / 2.0

    image = np.clip(image, low, high)
    image = (image - low) / (high - low) * 255.0

    return image.astype(np.uint8)


def extract_metadata(ds, image):
    """
    Extracts the metadata required for Phase 1.
    """
    transfer_syntax = "Unknown"

    if hasattr(ds, "file_meta") and "TransferSyntaxUID" in ds.file_meta:
        try:
            transfer_syntax = ds.file_meta.TransferSyntaxUID.name
        except Exception:
            transfer_syntax = str(ds.file_meta.TransferSyntaxUID)

    metadata = {
        "Width": str(ds.get("Columns", image.shape[1])),
        "Height": str(ds.get("Rows", image.shape[0])),
        "Bit depth": str(ds.get("BitsAllocated", "Unknown")),
        "Modality": str(ds.get("Modality", "Unknown")),
        "Patient Name": str(ds.get("PatientName", "Unknown")),
        "Patient Age": str(ds.get("PatientAge", "Unknown")),
        "Body Part Examined": str(ds.get("BodyPartExamined", "Unknown")),
        "Manufacturer": str(ds.get("Manufacturer", "Unknown")),
        "Study Description": str(ds.get("StudyDescription", "Unknown")),
        "Series Description": str(ds.get("SeriesDescription", "Unknown")),
        "Transfer Syntax": transfer_syntax,
    }

    return metadata


def read_dicom(path):
    """
    Reads a DICOM image from path.

    Works with:
    - .dcm files
    - .dicom files
    - extensionless DICOM files such as I3, I4, I5
    - compressed DICOM files if pylibjpeg decoders are installed
    """
    ds = pydicom.dcmread(path, force=True)

    if not hasattr(ds, "PixelData"):
        raise ValueError("This DICOM file has no pixel data.")

    try:
        pixels = ds.pixel_array

    except Exception as exc:
        raise ValueError(
            "Could not decode DICOM pixel data.\n\n"
            "If this is a compressed DICOM file, install the required decoders:\n"
            "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n\n"
            f"Original error:\n{exc}"
        )

    # If the DICOM has multiple frames, use the first frame for Phase 1.
    if pixels.ndim == 3:
        pixels = pixels[0]

    # If image is RGB, keep it displayable.
    # If grayscale, window and normalize it.
    if pixels.ndim == 2:
        image = apply_dicom_window(pixels, ds)

    elif pixels.ndim == 3 and pixels.shape[-1] in {3, 4}:
        image = pixels.astype(np.float32)
        image = image - image.min()
        if image.max() > 0:
            image = image / image.max() * 255.0
        image = image.astype(np.uint8)

        if image.shape[-1] == 4:
            image = image[:, :, :3]

    else:
        raise ValueError(f"Unsupported DICOM pixel shape: {pixels.shape}")

    metadata = extract_metadata(ds, image)

    return DicomReadResult(image=image, metadata=metadata)