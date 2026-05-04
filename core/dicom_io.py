import struct
from dataclasses import dataclass

import numpy as np

try:
    import pydicom
except ModuleNotFoundError:
    pydicom = None


@dataclass
class DicomReadResult:
    image: np.ndarray
    metadata: dict


TEXT_TAGS = {
    (0x0008, 0x0060): "Modality",
    (0x0010, 0x0010): "Patient Name",
    (0x0010, 0x1010): "Patient Age",
    (0x0018, 0x0015): "Body Part Examined",
    (0x0008, 0x0070): "Manufacturer",
    (0x0008, 0x1030): "Study Description",
    (0x0008, 0x103E): "Series Description",
}

NUMERIC_TAGS = {
    (0x0028, 0x0002): "Samples Per Pixel",
    (0x0028, 0x0010): "Height",
    (0x0028, 0x0011): "Width",
    (0x0028, 0x0100): "Bit depth",
    (0x0028, 0x0101): "Bits Stored",
    (0x0028, 0x0103): "Pixel Representation",
}

LONG_VR = {"OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UR", "UT", "UN"}


def normalize_to_uint8(image):
    image = image.astype(np.float32)
    image = image - np.min(image)
    max_value = np.max(image)
    if max_value > 0:
        image = image / max_value * 255.0
    return image.astype(np.uint8)


def _clean_text(raw):
    return raw.decode("latin1", errors="ignore").strip("\0 ").replace("^", " ")


def _read_number(raw):
    if len(raw) == 2:
        return struct.unpack("<H", raw)[0]
    if len(raw) == 4:
        return struct.unpack("<I", raw)[0]
    text = _clean_text(raw)
    try:
        return int(float(text))
    except ValueError:
        return text


def _parse_elements(data, start, explicit_vr=True, stop_at_pixel=False):
    pos = start
    values = {}
    pixel_data = None

    while pos + 8 <= len(data):
        group, elem = struct.unpack("<HH", data[pos:pos + 4])
        pos += 4
        tag = (group, elem)

        if explicit_vr:
            vr = data[pos:pos + 2].decode("ascii", errors="ignore")
            pos += 2
            if vr in LONG_VR:
                pos += 2
                if pos + 4 > len(data):
                    break
                length = struct.unpack("<I", data[pos:pos + 4])[0]
                pos += 4
            else:
                if pos + 2 > len(data):
                    break
                length = struct.unpack("<H", data[pos:pos + 2])[0]
                pos += 2
        else:
            vr = ""
            if pos + 4 > len(data):
                break
            length = struct.unpack("<I", data[pos:pos + 4])[0]
            pos += 4

        if length == 0xFFFFFFFF:
            break
        if pos + length > len(data):
            break

        raw = data[pos:pos + length]
        pos += length

        if tag == (0x7FE0, 0x0010):
            pixel_data = raw
            if stop_at_pixel:
                break
        elif tag in TEXT_TAGS:
            values[TEXT_TAGS[tag]] = _clean_text(raw)
        elif tag in NUMERIC_TAGS:
            values[NUMERIC_TAGS[tag]] = _read_number(raw)
        elif tag == (0x0002, 0x0010):
            values["Transfer Syntax UID"] = _clean_text(raw)
        elif tag == (0x0028, 0x0004):
            values["Photometric Interpretation"] = _clean_text(raw)
        elif tag == (0x0028, 0x1052):
            values["Rescale Intercept"] = _clean_text(raw)
        elif tag == (0x0028, 0x1053):
            values["Rescale Slope"] = _clean_text(raw)
        elif tag == (0x0028, 0x1050):
            values["Window Center"] = _clean_text(raw)
        elif tag == (0x0028, 0x1051):
            values["Window Width"] = _clean_text(raw)

    return values, pixel_data


def _window_image(pixel_array, metadata):
    image = pixel_array.astype(np.float32)

    try:
        slope = float(str(metadata.get("Rescale Slope", "1")).split("\\")[0])
        intercept = float(str(metadata.get("Rescale Intercept", "0")).split("\\")[0])
        image = image * slope + intercept
    except ValueError:
        pass

    try:
        center = float(str(metadata.get("Window Center")).split("\\")[0])
        width = float(str(metadata.get("Window Width")).split("\\")[0])
    except (TypeError, ValueError):
        return normalize_to_uint8(image)

    if width <= 0:
        return normalize_to_uint8(image)

    low = center - width / 2.0
    high = center + width / 2.0
    image = np.clip(image, low, high)
    return ((image - low) / (high - low) * 255.0).astype(np.uint8)


def _read_dicom_without_pydicom(path):
    with open(path, "rb") as f:
        data = f.read()

    start = 132 if len(data) > 132 and data[128:132] == b"DICM" else 0
    metadata, pixel_data = _parse_elements(data, start, explicit_vr=True, stop_at_pixel=True)

    transfer = metadata.get("Transfer Syntax UID", "1.2.840.10008.1.2.1")
    if transfer.startswith("1.2.840.10008.1.2.4"):
        raise ValueError(
            "Compressed DICOM files need pydicom plus decoder packages. "
            "Install pydicom for broader DICOM support."
        )

    if pixel_data is None and transfer == "1.2.840.10008.1.2":
        metadata, pixel_data = _parse_elements(data, start, explicit_vr=False, stop_at_pixel=True)

    if pixel_data is None:
        raise ValueError("DICOM pixel data was not found.")

    width = int(metadata.get("Width", 0))
    height = int(metadata.get("Height", 0))
    bits = int(metadata.get("Bit depth", 8))
    samples = int(metadata.get("Samples Per Pixel", 1))
    signed = int(metadata.get("Pixel Representation", 0)) == 1

    if width <= 0 or height <= 0:
        raise ValueError("DICOM width/height metadata is missing.")

    if bits <= 8:
        dtype = np.int8 if signed else np.uint8
    elif bits <= 16:
        dtype = np.int16 if signed else np.uint16
    else:
        raise ValueError("Only 8-bit and 16-bit uncompressed DICOM images are supported without pydicom.")

    count = width * height * samples
    expected_bytes = count * np.dtype(dtype).itemsize
    pixels = np.frombuffer(pixel_data[:expected_bytes], dtype=dtype, count=count)
    if pixels.size != count:
        raise ValueError("DICOM pixel data is shorter than expected.")

    if samples > 1:
        pixels = pixels.reshape((height, width, samples))
        image = normalize_to_uint8(pixels)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
    else:
        pixels = pixels.reshape((height, width))
        image = _window_image(pixels, metadata)
        if metadata.get("Photometric Interpretation") == "MONOCHROME1":
            image = 255 - image

    metadata.setdefault("Width", width)
    metadata.setdefault("Height", height)
    metadata.setdefault("Bit depth", bits)
    metadata.setdefault("Modality", "Unknown")
    metadata.setdefault("Patient Name", "Unknown")
    metadata.setdefault("Patient Age", "Unknown")
    metadata.setdefault("Body Part Examined", "Unknown")
    metadata["Transfer Syntax"] = transfer

    return DicomReadResult(image=image, metadata=metadata)


def _get_first_value(value):
    if pydicom is not None and isinstance(value, pydicom.multival.MultiValue):
        return value[0]
    return value


def _read_dicom_with_pydicom(path):
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

    if pixels.ndim == 3 and pixels.shape[-1] not in {3, 4}:
        pixels = pixels[0]

    metadata = {
        "Width": int(ds.get("Columns", pixels.shape[1])),
        "Height": int(ds.get("Rows", pixels.shape[0])),
        "Bit depth": int(ds.get("BitsAllocated", 8)),
        "Modality": str(ds.get("Modality", "Unknown")),
        "Patient Name": str(ds.get("PatientName", "Unknown")),
        "Patient Age": str(ds.get("PatientAge", "Unknown")),
        "Body Part Examined": str(ds.get("BodyPartExamined", "Unknown")),
        "Manufacturer": str(ds.get("Manufacturer", "Unknown")),
        "Study Description": str(ds.get("StudyDescription", "Unknown")),
        "Series Description": str(ds.get("SeriesDescription", "Unknown")),
    }

    if hasattr(ds, "file_meta") and "TransferSyntaxUID" in ds.file_meta:
        try:
            metadata["Transfer Syntax"] = ds.file_meta.TransferSyntaxUID.name
        except Exception:
            metadata["Transfer Syntax"] = str(ds.file_meta.TransferSyntaxUID)

    if pixels.ndim == 2:
        temp_metadata = {
            "Rescale Slope": ds.get("RescaleSlope", 1),
            "Rescale Intercept": ds.get("RescaleIntercept", 0),
            "Window Center": _get_first_value(ds.get("WindowCenter", None)),
            "Window Width": _get_first_value(ds.get("WindowWidth", None)),
        }
        image = _window_image(pixels, temp_metadata)
    elif pixels.ndim == 3 and pixels.shape[-1] in {3, 4}:
        image = normalize_to_uint8(pixels)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
    else:
        raise ValueError(f"Unsupported DICOM pixel shape: {pixels.shape}")

    return DicomReadResult(image=image, metadata=metadata)


def read_dicom(path):
    if pydicom is not None:
        return _read_dicom_with_pydicom(path)
    return _read_dicom_without_pydicom(path)

