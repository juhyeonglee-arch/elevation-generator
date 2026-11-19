"""작은 영상 연산(numpy만): scipy.ndimage 대신 쓴다. 가장자리 처리는 scipy 기본값(reflect)과 같다."""
import numpy as np


def maximum_filter3(a):
    """3×3 최댓값."""
    p = np.pad(a, 1, mode="symmetric")
    r = np.maximum(np.maximum(p[:, :-2], p[:, 1:-1]), p[:, 2:])
    return np.maximum(np.maximum(r[:-2], r[1:-1]), r[2:])


def minimum_filter1d(a, size):
    """1차원 이동 최솟값(창 size, 가운데 맞춤)."""
    a = np.asarray(a)
    size = int(size)
    if size <= 1 or a.size == 0:
        return a.copy()
    left = size // 2
    p = np.pad(a, (left, size - 1 - left), mode="symmetric")
    return np.lib.stride_tricks.sliding_window_view(p, size).min(axis=-1)


def _erode(m):
    p = np.pad(m, 1, mode="constant", constant_values=False)
    return p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def _dilate(m):
    p = np.pad(m, 1, mode="constant", constant_values=False)
    return p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def binary_opening(mask, iterations=1):
    """십자 모양으로 깎았다가(iterations번) 다시 불리기: 작은 점·가는 조각 없애기."""
    m = np.asarray(mask, bool)
    for _ in range(int(iterations)):
        m = _erode(m)
    for _ in range(int(iterations)):
        m = _dilate(m)
    return m


def gaussian_filter(a, sigma, truncate=4.0):
    """가우스 흐림(가로·세로 따로)."""
    a = np.asarray(a)
    r = int(truncate * float(sigma) + 0.5)
    if r < 1:
        return a.copy()
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k = k / k.sum()
    if a.ndim == 2 and a.dtype == np.float32 and a.size:
        from . import raster
        out = raster.gauss_blur(a, k)
        if out is not None:
            return out
    k = k.astype(a.dtype if a.dtype.kind == "f" else np.float64)
    out = a.astype(k.dtype, copy=False)
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]; pad[ax] = (r, r)
        p = np.pad(out, pad, mode="symmetric")
        n = out.shape[ax]
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            acc += w * (p[i:i + n] if ax == 0 else p[:, i:i + n])
        out = acc
    return out
