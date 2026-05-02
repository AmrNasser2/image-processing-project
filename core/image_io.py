import os

import numpy as np
from PIL import Image

from core.dicom_io import read_dicom
from core.utils import display_ready


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".dcm", ".dicom"}:
        result = read_dicom(path)
        return result.image, result.metadata

    with Image.open(path) as img:
        bit_depth = 8
        if img.mode in {"I;16", "I;16L", "I;16B"}:
            bit_depth = 16
        converted = img.convert("RGB") if img.mode not in {"L", "RGB"} else img.copy()
        arr = np.asarray(converted, dtype=np.uint8)
        metadata = {
            "Width": arr.shape[1],
            "Height": arr.shape[0],
            "Bit depth": bit_depth,
            "Format": img.format or ext.upper().strip("."),
            "Mode": img.mode,
        }
        return arr, metadata


def save_image(path, image):
    arr = display_ready(image)
    im = Image.fromarray(arr)
    if im.mode not in {"L", "RGB"}:
        im = im.convert("RGB")
    im.save(path)

