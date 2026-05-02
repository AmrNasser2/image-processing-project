import struct
from dataclasses import dataclass

import numpy as np

from core.utils import normalize_to_uint8


TEXT_TAGS = {
    (0x0008, 0x0060): "Modality",
    (0x0010, 0x0010): "Patient Name",
    (0x0010, 0x1010): "Patient Age",
    (0x0018, 0x0015): "Body Part Examined",
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


@dataclass
class DicomResult:
    image: np.ndarray
    metadata: dict


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

    return values, pixel_data, pos


def read_dicom(path):
    with open(path, "rb") as f:
        data = f.read()

    start = 132 if len(data) > 132 and data[128:132] == b"DICM" else 0
    metadata, pixel_data, _ = _parse_elements(data, start, explicit_vr=True, stop_at_pixel=True)

    transfer = metadata.get("Transfer Syntax UID", "1.2.840.10008.1.2.1")
    if transfer.startswith("1.2.840.10008.1.2.4"):
        raise ValueError("Compressed DICOM transfer syntaxes are not supported in this Phase 1 reader")

    if pixel_data is None and transfer == "1.2.840.10008.1.2":
        metadata, pixel_data, _ = _parse_elements(data, start, explicit_vr=False, stop_at_pixel=True)

    if pixel_data is None:
        raise ValueError("DICOM pixel data was not found")

    width = int(metadata.get("Width", 0))
    height = int(metadata.get("Height", 0))
    bits = int(metadata.get("Bit depth", 8))
    samples = int(metadata.get("Samples Per Pixel", 1))
    signed = int(metadata.get("Pixel Representation", 0)) == 1
    if width <= 0 or height <= 0:
        raise ValueError("DICOM width/height metadata is missing")

    if bits <= 8:
        dtype = np.int8 if signed else np.uint8
    elif bits <= 16:
        dtype = np.int16 if signed else np.uint16
    else:
        raise ValueError("Only 8-bit and 16-bit DICOM images are supported")

    count = width * height * samples
    arr = np.frombuffer(pixel_data[:count * np.dtype(dtype).itemsize], dtype=dtype, count=count)
    if arr.size != count:
        raise ValueError("DICOM pixel data is shorter than expected")
    if samples > 1:
        arr = arr.reshape((height, width, samples))
    else:
        arr = arr.reshape((height, width))

    try:
        slope = float(str(metadata.get("Rescale Slope", "1")).split("\\")[0])
        intercept = float(str(metadata.get("Rescale Intercept", "0")).split("\\")[0])
        arr = arr.astype(np.float64) * slope + intercept
    except ValueError:
        arr = arr.astype(np.float64)

    metadata.setdefault("Width", width)
    metadata.setdefault("Height", height)
    metadata.setdefault("Bit depth", bits)
    image = normalize_to_uint8(arr)
    if metadata.get("Photometric Interpretation") == "MONOCHROME1":
        image = 255 - image
    return DicomResult(image=image, metadata=metadata)

