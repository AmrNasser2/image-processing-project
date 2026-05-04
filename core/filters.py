import math
import numpy as np

from core.convolution import convolve2d, pad_edge
from core.utils import clip_uint8, normalize_to_uint8, to_grayscale_float


def average_kernel(size):
    if size < 3 or size % 2 == 0:
        raise ValueError("Kernel size must be odd and at least 3")
    return np.ones((size, size), dtype=np.float64) / (size * size)


def gaussian_kernel(size, variance):
    if size < 3 or size % 2 == 0:
        raise ValueError("Kernel size must be odd and at least 3")
    if variance <= 0:
        raise ValueError("Variance must be positive")
    radius = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            kernel[y + radius, x + radius] = math.exp(-(x * x + y * y) / (2.0 * variance))

    domain = 2.0 * math.pi * variance
    return kernel / domain


def average_filter(image, size):
    return clip_uint8(convolve2d(image, average_kernel(size)))


def gaussian_filter(image, size, variance):
    return clip_uint8(convolve2d(image, gaussian_kernel(size, variance)))


def edge_filter(image, operator="sobel", direction="magnitude"):
    gray = to_grayscale_float(image)
    if operator == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
    else:
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

    gx = convolve2d(gray, kx)
    gy = convolve2d(gray, ky)
    if direction == "horizontal":
        return normalize_to_uint8(np.abs(gx))
    if direction == "vertical":
        return normalize_to_uint8(np.abs(gy))
    return normalize_to_uint8(np.sqrt(gx * gx + gy * gy))


def median_filter(image, size):
    if size < 3 or size % 2 == 0:
        raise ValueError("Kernel size must be odd and at least 3")
    src = np.asarray(image)
    pad = size // 2
    padded = pad_edge(src, pad, pad)
    out = np.zeros_like(src)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            region = padded[y:y + size, x:x + size]
            if src.ndim == 2:
                values = sorted(float(v) for v in region.reshape(-1))
                out[y, x] = values[len(values) // 2]
            else:
                for c in range(src.shape[2]):
                    values = sorted(float(v) for v in region[..., c].reshape(-1))
                    out[y, x, c] = values[len(values) // 2]
    return out.astype(src.dtype)

