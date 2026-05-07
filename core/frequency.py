import numpy as np

from core.utils import clip_uint8, normalize_to_uint8, to_grayscale_float


def fourier_spectrum(image):
    gray = to_grayscale_float(image)
    shifted = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(shifted))
    return normalize_to_uint8(magnitude), shifted


def notch_reject_mask(shape, center, radius=10, kind="gaussian", order=2):
    h, w = shape
    cy, cx = center
    mirror_y = h - cy - 1
    mirror_x = w - cx - 1

    yy, xx = np.indices((h, w))
    mask = np.ones((h, w), dtype=np.float64)

    for y0, x0 in ((cy, cx), (mirror_y, mirror_x)):
        distance = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2)

        if kind == "ideal":
            notch = (distance > radius).astype(np.float64)
        elif kind == "butterworth":
            safe_distance = np.maximum(distance, 1e-8)
            notch = 1.0 / (1.0 + (radius / safe_distance) ** (2 * max(1, order)))
        else:
            notch = 1.0 - np.exp(-(distance ** 2) / (2.0 * max(radius, 1e-8) ** 2))

        mask *= notch

    return mask


def apply_notch_reject(image, center, radius=10, kind="gaussian", order=2):
    gray = to_grayscale_float(image)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    mask = notch_reject_mask(gray.shape, center, radius, kind, order)
    filtered = spectrum * mask
    restored = np.fft.ifft2(np.fft.ifftshift(filtered))
    return clip_uint8(np.real(restored))


def frequency_cross_correlation(image, template):
    src = to_grayscale_float(image)
    tpl = to_grayscale_float(template)
    th, tw = tpl.shape

    if th < 2 or tw < 2:
        raise ValueError("Template is too small")
    if th > src.shape[0] or tw > src.shape[1]:
        raise ValueError("Template must be smaller than the image")

    tpl = tpl - np.mean(tpl)
    padded = np.zeros_like(src, dtype=np.float64)
    padded[:th, :tw] = tpl

    corr = np.fft.ifft2(np.fft.fft2(src) * np.conj(np.fft.fft2(padded)))
    corr = np.real(corr)

    valid_h = src.shape[0] - th + 1
    valid_w = src.shape[1] - tw + 1
    valid = corr[:valid_h, :valid_w]
    y, x = np.unravel_index(np.argmax(valid), valid.shape)
    return int(y), int(x), normalize_to_uint8(valid)
