"""모델 살펴보기: 레벨 후보, 건물 벽 방향, 평면 미리보기 그림."""
import numpy as np

from .elevation import merge_parts, tri_normals
from .raster import rasterize


def building_parts(parts, cfg):
    """이 건물만: 넣은 레이어 중 지형·주변 건물·수목을 뺀 것(층·GL·벽 방향 계산용)."""
    out = []
    for p in parts:
        c = cfg.get(p[2], {})
        if c.get("include", True) and not c.get("terrain") and not c.get("context") and c.get("material") != "수목":
            out.append(p)
    return out


def level_candidates(parts, cfg=None, min_frac=0.025, merge=0.45):
    """위를 향한 수평면의 높이를 면적으로 모아 바닥 후보를 만든다.

    → [{z, area, layer}] (낮은 순). 지형·주변 건물·수목 레이어와 뺀 레이어는 제외.
    """
    cfg = cfg or {}
    use = building_parts(parts, cfg)
    if not use:
        return []
    V, F, P = merge_parts(use)
    N, A = tri_normals(V, F)
    up = N[:, 2] > 0.98
    if not up.any():
        return []
    z = np.round(V[F[up, 0], 2], 2); a = A[up]; pl = P[up]
    order = np.argsort(z)
    z, a, pl = z[order], a[order], pl[order]
    uz, inv = np.unique(z, return_inverse=True)
    area = np.bincount(inv, weights=a)
    tot = area.sum()
    # 가까운 높이 묶기(바닥판 두께·마감 차이) → 묶음에서 면적이 가장 큰 높이
    groups = []
    for zz, ar in zip(uz, area):
        if groups and zz - groups[-1]["zmax"] < merge:
            g = groups[-1]; g["zmax"] = zz; g["area"] += ar
            if ar > g["best_a"]:
                g["best_a"] = ar; g["z"] = zz
        else:
            groups.append(dict(z=zz, zmax=zz, area=ar, best_a=ar))
    out = []
    for g in groups:
        if g["area"] < tot * min_frac:
            continue
        sel = (z >= g["z"] - 0.005) & (z <= g["z"] + 0.005)
        lay = np.bincount(pl[sel], weights=a[sel]) if sel.any() else np.zeros(1)
        layer = use[int(np.argmax(lay))][2] if sel.any() else ""
        out.append(dict(z=round(float(g["z"]), 3), area=round(float(g["area"]), 1), layer=layer))
    return out


def default_level_names(cands, gl, top):
    """GL 위 첫 바닥 = 1F, 그 위로 2F…, 맨 위가 건물 꼭대기 근처면 RF, GL 아래는 B1, B2…"""
    above = [c for c in cands if c["z"] > gl - 0.05]
    below = [c for c in cands if c["z"] <= gl - 0.05]
    names = {}
    for i, c in enumerate(sorted(below, key=lambda c: -c["z"])):
        names[c["z"]] = f"B{i + 1}"
    for i, c in enumerate(sorted(above, key=lambda c: c["z"])):
        names[c["z"]] = f"{i + 1}F"
    if above:
        hi = max(above, key=lambda c: c["z"])
        if len(above) >= 2 and top - hi["z"] < 2.5:
            names[hi["z"]] = "RF"
    return [dict(c, name=names[c["z"]], show=True) for c in cands]


def wall_angle_arrays(N, A):
    """법선·면적 → 가장 많은 벽 방향(−45~45°, +Y 기준 시계방향)."""
    vert = np.abs(N[:, 2]) < 0.1
    if not vert.any():
        return 0.0
    ang = np.degrees(np.arctan2(N[vert, 0], N[vert, 1])) % 90.0
    h, edges = np.histogram(ang, bins=180, range=(0, 90), weights=A[vert])
    h = h + np.roll(h, 1) + np.roll(h, -1)
    a = (edges[np.argmax(h)] + 0.25) % 90.0
    return round(float(a if a < 45 else a - 90), 1)


def wall_angle(parts, cfg=None):
    """수직 벽 법선의 방향을 면적으로 모아, 가장 많은 벽 방향(+Y 기준 시계방향)."""
    cfg = cfg or {}
    use = building_parts(parts, cfg)
    if not use:
        return 0.0
    V, F, P = merge_parts(use)
    N, A = tri_normals(V, F)
    return wall_angle_arrays(N, A)


def summary(parts, cfg=None):
    """화면용: 층 후보(기본 이름 포함), 기본 GL(건물 맨 아래), 건물 꼭대기, 벽 방향."""
    cfg = cfg or {}
    bld = building_parts(parts, cfg)
    if not bld:
        return dict(levels=[], gl=0.0, top=0.0, wall=0.0)
    zs = np.concatenate([p[0][:, 2] for p in bld])
    gl, top = float(zs.min()), float(zs.max())
    cands = level_candidates(parts, cfg)
    return dict(levels=default_level_names(cands, gl, top), gl=round(gl, 3), top=round(top, 3), wall=wall_angle(parts, cfg))


def plan_png(parts, path, cfg=None, size=760):
    """위에서 본 회색조 평면(높을수록 진하게) + 범위 [xmin, ymin, xmax, ymax]."""
    from PIL import Image
    cfg = cfg or {}
    use = [p for p in parts if cfg.get(p[2], {}).get("include", True)] or parts
    V, F, P = merge_parts(use)
    x0, y0 = V[:, 0].min(), V[:, 1].min(); x1, y1 = V[:, 0].max(), V[:, 1].max()
    s = max(x1 - x0, y1 - y0) / size
    W = int((x1 - x0) / s) + 2; H = int((y1 - y0) / s) + 2
    zb, ib = rasterize((V[:, 0] - x0) / s, (y1 - V[:, 1]) / s, (-V[:, 2]).astype(np.float64), F, W, H)
    hgt = np.where(np.isfinite(zb), -zb, np.nan)
    lo, hi = np.nanmin(hgt), np.nanmax(hgt)
    g = np.where(np.isnan(hgt), 255, 235 - (hgt - lo) / max(hi - lo, 1e-6) * 150).astype(np.uint8)
    Image.fromarray(g).save(path)
    return [float(x0), float(y0), float(x1), float(y1)], [W, H]
