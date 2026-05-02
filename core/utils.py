import numpy as np


def to_grayscale_float(image):
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float64)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    raise ValueError("Unsupported image shape")


def normalize_to_uint8(image):
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.uint8)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) * 255.0 / (hi - lo)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def clip_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def display_ready(image):
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    return normalize_to_uint8(arr)

