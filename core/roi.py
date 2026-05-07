import numpy as np

from core.utils import normalize_to_uint8, to_grayscale_float


def roi_statistics(image, box):
    x0, y0, x1, y1 = box
    gray = normalize_to_uint8(to_grayscale_float(image))
    h, w = gray.shape

    left = max(0, min(x0, x1))
    right = min(w, max(x0, x1))
    top = max(0, min(y0, y1))
    bottom = min(h, max(y0, y1))

    if right <= left or bottom <= top:
        raise ValueError("ROI must have a positive width and height")

    region = gray[top:bottom, left:right]
    hist = np.zeros(256, dtype=np.float64)
    total = 0
    running_sum = 0.0

    for value in region.reshape(-1):
        ivalue = int(value)
        hist[ivalue] += 1
        running_sum += ivalue
        total += 1

    mean = running_sum / max(total, 1)
    variance = 0.0
    for value in region.reshape(-1):
        variance += (float(value) - mean) ** 2
    variance /= max(total, 1)

    return {
        "box": (left, top, right, bottom),
        "width": right - left,
        "height": bottom - top,
        "mean": mean,
        "variance": variance,
        "histogram": hist,
    }
