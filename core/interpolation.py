import math
import numpy as np


def _sample_nearest(image, y, x):
    h, w = image.shape[:2]
    yy = int(round(y))
    xx = int(round(x))
    yy = min(max(yy, 0), h - 1)
    xx = min(max(xx, 0), w - 1)
    return image[yy, xx]


def _sample_bilinear(image, y, x):
    h, w = image.shape[:2]

    y = min(max(float(y), 0.0), h - 1.0)
    x = min(max(float(x), 0.0), w - 1.0)

    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    y1 = min(y0 + 1, h - 1)
    x1 = min(x0 + 1, w - 1)

    dx = x - x0
    dy = y - y0

    v00 = image[y0, x0].astype(np.float64)
    v10 = image[y0, x1].astype(np.float64)
    v01 = image[y1, x0].astype(np.float64)
    v11 = image[y1, x1].astype(np.float64)

    d = v00
    a = v10 - v00
    b = v01 - v00
    c = v11 - v10 - v01 + v00

    return a * dx + b * dy + c * dx * dy + d


def resize(image, scale, method="bilinear"):
    if scale <= 0:
        raise ValueError("Scale must be positive")
    src = np.asarray(image)
    h, w = src.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    out_shape = (new_h, new_w) if src.ndim == 2 else (new_h, new_w, src.shape[2])
    out = np.zeros(out_shape, dtype=np.float64)

    for y in range(new_h):
        src_y = y / scale
        for x in range(new_w):
            src_x = x / scale
            if method == "nearest":
                out[y, x] = _sample_nearest(src, src_y, src_x)
            else:
                out[y, x] = _sample_bilinear(src, src_y, src_x)
    return np.clip(out, 0, 255).astype(src.dtype if np.issubdtype(src.dtype, np.integer) else np.uint8)
