import os
import shutil
import struct
import uuid

import numpy as np
from PIL import Image

from core.filters import average_filter, edge_filter, gaussian_filter, median_filter
from core.histogram import local_histogram_equalization
from core.image_io import load_image, save_image
from core.interpolation import resize, rotate, shear
from core.pipeline import ImagePipeline


def make_workspace_tmp():
    path = os.path.join(os.getcwd(), f"test_tmp_{uuid.uuid4().hex}")
    os.makedirs(path)
    return path


def test_algorithms_smoke():
    img = np.array(
        [
            [0, 10, 20, 30],
            [40, 50, 60, 70],
            [80, 90, 100, 110],
            [120, 130, 140, 255],
        ],
        dtype=np.uint8,
    )
    assert resize(img, 2, "nearest").shape == (8, 8)
    assert resize(img, 0.5, "bilinear").shape == (2, 2)
    assert average_filter(img, 3).shape == img.shape
    assert gaussian_filter(img, 3, 1.0).shape == img.shape
    assert median_filter(img, 3).shape == img.shape
    assert edge_filter(img, "sobel", "magnitude").shape == img.shape
    assert local_histogram_equalization(img, 2).shape == img.shape
    assert rotate(img, 10).ndim == 2
    assert shear(img, 0.1, 0.0).ndim == 2


def test_pipeline_undo_reset():
    img = np.zeros((3, 3), dtype=np.uint8)
    pipe = ImagePipeline()
    pipe.load(img)
    pipe.apply(np.ones((3, 3), dtype=np.uint8), "ones")
    assert pipe.current[0, 0] == 1
    assert pipe.undo()
    assert pipe.current[0, 0] == 0
    pipe.apply(np.ones((3, 3), dtype=np.uint8), "ones")
    assert pipe.reset()
    assert pipe.current[0, 0] == 0


def test_standard_image_io_smoke():
    img = np.arange(27, dtype=np.uint8).reshape((3, 3, 3))
    tmp = make_workspace_tmp()
    try:
        bmp_path = f"{tmp}/sample.bmp"
        out_path = f"{tmp}/processed.png"
        Image.fromarray(img).save(bmp_path)

        loaded, metadata = load_image(bmp_path)
        assert loaded.shape == img.shape
        assert metadata["Width"] == 3
        assert metadata["Height"] == 3
        assert metadata["Bit depth"] == 8

        save_image(out_path, loaded)
        reloaded, out_metadata = load_image(out_path)
        assert reloaded.shape == img.shape
        assert out_metadata["Format"] == "PNG"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_uncompressed_dicom_io_smoke():
    def element(group, tag, vr, raw):
        if len(raw) % 2:
            raw += b"\0"
        header = struct.pack("<HH", group, tag) + vr.encode("ascii")
        if vr in {"OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UR", "UT", "UN"}:
            return header + b"\0\0" + struct.pack("<I", len(raw)) + raw
        return header + struct.pack("<H", len(raw)) + raw

    pixels = np.arange(16, dtype=np.uint8).reshape((4, 4))
    data = (
        (b"\0" * 128) + b"DICM"
        + element(0x0002, 0x0010, "UI", b"1.2.840.10008.1.2.1\0")
        + element(0x0028, 0x0002, "US", struct.pack("<H", 1))
        + element(0x0028, 0x0004, "CS", b"MONOCHROME2")
        + element(0x0028, 0x0010, "US", struct.pack("<H", 4))
        + element(0x0028, 0x0011, "US", struct.pack("<H", 4))
        + element(0x0028, 0x0100, "US", struct.pack("<H", 8))
        + element(0x0028, 0x0101, "US", struct.pack("<H", 8))
        + element(0x0028, 0x0103, "US", struct.pack("<H", 0))
        + element(0x7FE0, 0x0010, "OB", pixels.tobytes())
    )

    tmp = make_workspace_tmp()
    try:
        path = f"{tmp}/sample.dcm"
        with open(path, "wb") as f:
            f.write(data)
        loaded, metadata = load_image(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert loaded.shape == (4, 4)
    assert metadata["Width"] == 4
    assert metadata["Height"] == 4
    assert metadata["Bit depth"] == 8

