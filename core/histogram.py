import numpy as np

from core.utils import normalize_to_uint8, to_grayscale_float


def equalize_block(block, clip_limit=0.02):
    src = np.asarray(block, dtype=np.uint8)
    total = src.size

    hist = np.zeros(256, dtype=np.float64)
    for value in src.reshape(-1):
        hist[int(value)] += 1

    # Clip histogram to reduce noise amplification
    max_count = max(1, int(clip_limit * total))
    excess = 0.0

    for i in range(256):
        if hist[i] > max_count:
            excess += hist[i] - max_count
            hist[i] = max_count

    hist += excess / 256.0

    cdf = np.cumsum(hist)
    nonzero = cdf[cdf > 0]

    if len(nonzero) == 0:
        return src.copy()

    cdf_min = nonzero[0]

    if total == cdf_min:
        return src.copy()

    lookup = np.round((cdf - cdf_min) * 255 / (total - cdf_min))
    lookup = np.clip(lookup, 0, 255).astype(np.uint8)

    return lookup[src]


def local_histogram_equalization(image, block_size):
    if block_size < 2:
        raise ValueError("Block size must be at least 2")

    gray = normalize_to_uint8(to_grayscale_float(image))
    h, w = gray.shape

    # Accumulate overlapping enhanced blocks
    acc = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    step = max(1, block_size // 2)

    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = max(0, y - block_size // 2)
            x1 = max(0, x - block_size // 2)
            y2 = min(h, y1 + block_size)
            x2 = min(w, x1 + block_size)

            block = gray[y1:y2, x1:x2]
            enhanced = equalize_block(block, clip_limit=0.02)

            # Smooth weighting reduces block borders
            bh, bw = enhanced.shape
            wy = np.hanning(bh) if bh > 1 else np.ones(1)
            wx = np.hanning(bw) if bw > 1 else np.ones(1)
            window = np.outer(wy, wx)

            if window.max() == 0:
                window = np.ones_like(enhanced, dtype=np.float64)

            acc[y1:y2, x1:x2] += enhanced.astype(np.float64) * window
            weight[y1:y2, x1:x2] += window

    enhanced_full = acc / np.maximum(weight, 1e-8)

    # Blend with original image so the result does not look too harsh
    blend = 0.40
    result = blend * enhanced_full + (1.0 - blend) * gray.astype(np.float64)

    return np.clip(result, 0, 255).astype(np.uint8)