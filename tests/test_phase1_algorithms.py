import numpy as np

from core.filters import average_filter, edge_filter, median_filter
from core.histogram import local_histogram_equalization
from core.interpolation import resize, rotate, shear
from core.pipeline import ImagePipeline


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

