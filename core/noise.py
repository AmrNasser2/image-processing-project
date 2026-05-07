import numpy as np

from core.utils import clip_uint8


def add_gaussian_noise(image, mean=0.0, variance=100.0):
    if variance < 0:
        raise ValueError("Noise variance must not be negative")

    src = np.asarray(image, dtype=np.float64)
    noise = np.random.normal(mean, variance ** 0.5, src.shape)
    return clip_uint8(src + noise)


def add_uniform_noise(image, low=-20.0, high=20.0):
    if high <= low:
        raise ValueError("Uniform noise max must be greater than min")

    src = np.asarray(image, dtype=np.float64)
    noise = np.random.uniform(low, high, src.shape)
    return clip_uint8(src + noise)
