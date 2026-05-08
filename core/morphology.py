import numpy as np

from core.utils import to_grayscale_float


def threshold_binary(image, threshold):
    if threshold < 0 or threshold > 255:
        raise ValueError("Threshold must be between 0 and 255")

    gray = to_grayscale_float(image)
    out = np.zeros(gray.shape, dtype=np.uint8)
    out[gray >= threshold] = 255
    return out


def structuring_element(size, shape="square"):
    if size < 3 or size % 2 == 0:
        raise ValueError("Structuring element size must be odd and at least 3")

    if shape == "cross":
        se = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        se[center, :] = 1
        se[:, center] = 1
        return se

    return np.ones((size, size), dtype=np.uint8)


def _to_binary_bool(image):
    gray = to_grayscale_float(image)
    return gray >= 128


def erode(image, size=3, shape="square"):
    src = _to_binary_bool(image)
    se = structuring_element(size, shape).astype(bool)
    pad = size // 2
    padded = np.pad(src, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)
    out = np.zeros_like(src, dtype=bool)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            region = padded[y:y + size, x:x + size]
            out[y, x] = np.all(region[se])

    return (out.astype(np.uint8) * 255)


def dilate(image, size=3, shape="square"):
    src = _to_binary_bool(image)
    se = structuring_element(size, shape).astype(bool)
    pad = size // 2
    padded = np.pad(src, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)
    out = np.zeros_like(src, dtype=bool)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            region = padded[y:y + size, x:x + size]
            out[y, x] = np.any(region[se])

    return (out.astype(np.uint8) * 255)


def opening(image, size=3, shape="square"):
    return dilate(erode(image, size, shape), size, shape)


def closing(image, size=3, shape="square"):
    return erode(dilate(image, size, shape), size, shape)


def boundary_extraction(image, size=3, shape="square"):
    binary = threshold_binary(image, 128)
    eroded = erode(binary, size, shape)
    boundary = binary.astype(np.int16) - eroded.astype(np.int16)
    return np.clip(boundary, 0, 255).astype(np.uint8)
