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
    wy = y - y0
    wx = x - x0

    top = (1.0 - wx) * image[y0, x0].astype(np.float64) + wx * image[y0, x1].astype(np.float64)
    bottom = (1.0 - wx) * image[y1, x0].astype(np.float64) + wx * image[y1, x1].astype(np.float64)
    return (1.0 - wy) * top + wy * bottom


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


def rotate(image, angle_degrees):
    src = np.asarray(image)
    h, w = src.shape[:2]
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (-w / 2, h / 2), (w / 2, h / 2)]
    rotated = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in corners]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    new_w = max(1, int(math.ceil(max(xs) - min(xs))))
    new_h = max(1, int(math.ceil(max(ys) - min(ys))))
    out_shape = (new_h, new_w) if src.ndim == 2 else (new_h, new_w, src.shape[2])
    out = np.zeros(out_shape, dtype=np.float64)

    cy_src = (h - 1) / 2.0
    cx_src = (w - 1) / 2.0
    cy_dst = (new_h - 1) / 2.0
    cx_dst = (new_w - 1) / 2.0

    for y in range(new_h):
        yy = y - cy_dst
        for x in range(new_w):
            xx = x - cx_dst
            src_x = xx * cos_a + yy * sin_a + cx_src
            src_y = -xx * sin_a + yy * cos_a + cy_src
            if 0 <= src_x <= w - 1 and 0 <= src_y <= h - 1:
                out[y, x] = _sample_bilinear(src, src_y, src_x)
    return np.clip(out, 0, 255).astype(src.dtype if np.issubdtype(src.dtype, np.integer) else np.uint8)


def shear(image, shear_x=0.0, shear_y=0.0):
    src = np.asarray(image)
    h, w = src.shape[:2]
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    mapped = [(x + shear_x * y, y + shear_y * x) for x, y in corners]
    min_x = min(p[0] for p in mapped)
    max_x = max(p[0] for p in mapped)
    min_y = min(p[1] for p in mapped)
    max_y = max(p[1] for p in mapped)
    new_w = max(1, int(math.ceil(max_x - min_x + 1)))
    new_h = max(1, int(math.ceil(max_y - min_y + 1)))
    out_shape = (new_h, new_w) if src.ndim == 2 else (new_h, new_w, src.shape[2])
    out = np.zeros(out_shape, dtype=np.float64)

    det = 1.0 - shear_x * shear_y
    if abs(det) < 1e-8:
        raise ValueError("Invalid shear factors")

    for y in range(new_h):
        world_y = y + min_y
        for x in range(new_w):
            world_x = x + min_x
            src_x = (world_x - shear_x * world_y) / det
            src_y = (world_y - shear_y * world_x) / det
            if 0 <= src_x <= w - 1 and 0 <= src_y <= h - 1:
                out[y, x] = _sample_bilinear(src, src_y, src_x)
    return np.clip(out, 0, 255).astype(src.dtype if np.issubdtype(src.dtype, np.integer) else np.uint8)

