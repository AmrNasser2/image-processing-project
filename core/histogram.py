import numpy as np

from core.utils import normalize_to_uint8, to_grayscale_float


def equalize_block(block):
    src = np.asarray(block, dtype=np.uint8)
    hist = [0] * 256
    for value in src.reshape(-1):
        hist[int(value)] += 1

    cdf = [0] * 256
    running = 0
    cdf_min = None
    for i, count in enumerate(hist):
        running += count
        cdf[i] = running
        if cdf_min is None and running > 0:
            cdf_min = running

    total = src.size
    out = np.zeros(src.shape, dtype=np.uint8)
    if cdf_min is None or total == cdf_min:
        return out

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            value = int(src[y, x])
            out[y, x] = round((cdf[value] - cdf_min) * 255 / (total - cdf_min))
    return out


def local_histogram_equalization(image, block_size):
    if block_size < 2:
        raise ValueError("Block size must be at least 2")
    gray = normalize_to_uint8(to_grayscale_float(image))
    h, w = gray.shape
    out = np.zeros_like(gray)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y2 = min(y + block_size, h)
            x2 = min(x + block_size, w)
            out[y:y2, x:x2] = equalize_block(gray[y:y2, x:x2])
    return out

