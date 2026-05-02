import numpy as np


def pad_edge(image, pad_y, pad_x):
    src = np.asarray(image)
    if src.ndim == 2:
        out = np.zeros((src.shape[0] + 2 * pad_y, src.shape[1] + 2 * pad_x), dtype=src.dtype)
    else:
        out = np.zeros((src.shape[0] + 2 * pad_y, src.shape[1] + 2 * pad_x, src.shape[2]), dtype=src.dtype)
    out[pad_y:pad_y + src.shape[0], pad_x:pad_x + src.shape[1]] = src

    for y in range(pad_y):
        out[y, pad_x:pad_x + src.shape[1]] = src[0]
        out[-y - 1, pad_x:pad_x + src.shape[1]] = src[-1]
    for x in range(pad_x):
        out[:, x] = out[:, pad_x]
        out[:, -x - 1] = out[:, pad_x + src.shape[1] - 1]
    return out


def convolve2d(image, kernel):
    src = np.asarray(image, dtype=np.float64)
    ker = np.asarray(kernel, dtype=np.float64)
    if ker.ndim != 2 or ker.shape[0] % 2 == 0 or ker.shape[1] % 2 == 0:
        raise ValueError("Kernel must have odd height and width")
    pad_y = ker.shape[0] // 2
    pad_x = ker.shape[1] // 2
    padded = pad_edge(src, pad_y, pad_x)
    out = np.zeros_like(src, dtype=np.float64)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            region = padded[y:y + ker.shape[0], x:x + ker.shape[1]]
            if src.ndim == 2:
                out[y, x] = np.sum(region * ker)
            else:
                for c in range(src.shape[2]):
                    out[y, x, c] = np.sum(region[..., c] * ker)
    return out

