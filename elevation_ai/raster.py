"""래스터라이저: 정사영 깊이·삼각형 번호 버퍼.

작은 C 라이브러리(raster_win64.dll, 소스는 native/raster.c)를 ctypes로 부른다.
라이브러리를 못 읽으면 numba(깔려 있을 때) → 순수 numpy(느림) 순으로 대신 쓴다."""
import ctypes
import os
import sys

import numpy as np

BACKEND = None
_fn = None
_gauss = None


def _load():
    global BACKEND, _fn
    here = os.path.dirname(os.path.abspath(__file__))
    name = "raster_win64.dll" if sys.platform == "win32" else "raster_linux64.so"
    try:
        lib = ctypes.CDLL(os.path.join(here, name))
        f = lib.rasterize
        P = ctypes.c_void_p
        f.argtypes = [P, P, P, P, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, P, P]
        f.restype = None
        g = lib.gauss_blur
        g.argtypes = [P, P, P, ctypes.c_int32, ctypes.c_int32, P, ctypes.c_int32]
        g.restype = None
        _fn, BACKEND = f, "c"
        global _gauss
        _gauss = g
        return
    except (OSError, AttributeError):
        pass
    try:
        from numba import njit
        _fn, BACKEND = njit(cache=True)(_rasterize_py), "numba"
    except Exception:
        _fn, BACKEND = _rasterize_np, "numpy"


def rasterize(sx, sy, sz, tris, W, H):
    """sx,sy: 픽셀 좌표(실수), sz: 깊이(작을수록 가까움). tris (T,3). → (깊이 float32[H,W], 삼각형 번호 int32[H,W])"""
    if _fn is None:
        _load()
    W, H = int(W), int(H)
    if BACKEND != "c":
        return _fn(np.asarray(sx, np.float64), np.asarray(sy, np.float64), np.asarray(sz, np.float64), np.asarray(tris), W, H)
    sx = np.ascontiguousarray(sx, np.float64); sy = np.ascontiguousarray(sy, np.float64); sz = np.ascontiguousarray(sz, np.float64)
    tr = np.ascontiguousarray(tris, np.int32)
    zb = np.full((H, W), np.inf, np.float32)
    ib = np.full((H, W), -1, np.int32)
    if len(tr) and W > 0 and H > 0:
        _fn(sx.ctypes.data, sy.ctypes.data, sz.ctypes.data, tr.ctypes.data, len(tr), W, H, zb.ctypes.data, ib.ctypes.data)
    return zb, ib


def _rasterize_py(sx, sy, sz, tris, W, H):
    zb = np.full((H, W), np.inf, np.float32)
    ib = np.full((H, W), -1, np.int32)
    for t in range(tris.shape[0]):
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        x0, y0, z0 = sx[a], sy[a], sz[a]
        x1, y1, z1 = sx[b], sy[b], sz[b]
        x2, y2, z2 = sx[c], sy[c], sz[c]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        minx = max(int(np.floor(min(x0, x1, x2))), 0)
        maxx = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        miny = max(int(np.floor(min(y0, y1, y2))), 0)
        maxy = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        for py in range(miny, maxy + 1):
            cy = py + 0.5
            for px in range(minx, maxx + 1):
                cx = px + 0.5
                w0 = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / den
                w1 = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / den
                w2 = 1.0 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z < zb[py, px]:
                    zb[py, px] = z
                    ib[py, px] = t
    return zb, ib


def _rasterize_np(sx, sy, sz, tris, W, H):
    """마지막 대안: 삼각형마다 numpy로(느리지만 결과는 같다)."""
    zb = np.full((H, W), np.inf, np.float32)
    ib = np.full((H, W), -1, np.int32)
    for t in range(tris.shape[0]):
        a, b, c = tris[t]
        x0, y0, z0 = sx[a], sy[a], sz[a]
        x1, y1, z1 = sx[b], sy[b], sz[b]
        x2, y2, z2 = sx[c], sy[c], sz[c]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if not abs(den) >= 1e-12:
            continue
        minx = max(int(np.floor(min(x0, x1, x2))), 0); maxx = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        miny = max(int(np.floor(min(y0, y1, y2))), 0); maxy = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        if maxx < minx or maxy < miny:
            continue
        cx = np.arange(minx, maxx + 1) + 0.5
        cy = (np.arange(miny, maxy + 1) + 0.5)[:, None]
        w0 = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / den
        w1 = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / den
        w2 = 1.0 - w0 - w1
        z = w0 * z0 + w1 * z1 + w2 * z2
        zs = zb[miny:maxy + 1, minx:maxx + 1]; is_ = ib[miny:maxy + 1, minx:maxx + 1]
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (z < zs)
        zs[m] = z[m]; is_[m] = t
    return zb, ib


def gauss_blur(a, k):
    """float32 2차원 배열을 커널 k(길이 2r+1)로 가로·세로 흐림. C 라이브러리가 없으면 None."""
    if _fn is None:
        _load()
    if _gauss is None:
        return None
    a = np.ascontiguousarray(a, np.float32)
    k = np.ascontiguousarray(k, np.float64)
    H, W = a.shape
    r = (len(k) - 1) // 2
    dst = np.empty_like(a); tmp = np.empty(H * W + W + 2 * r, np.float32)
    _gauss(a.ctypes.data, dst.ctypes.data, tmp.ctypes.data, H, W, k.ctypes.data, r)
    return dst
