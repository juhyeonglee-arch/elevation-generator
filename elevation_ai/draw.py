"""입면 결과 → PNG/PDF(matplotlib), SVG(그룹 = 레이어), DXF(캐드 레이어·해치·치수), 도곽 시트.

도면 좌표는 종이 mm. 입면 그림(선·재료·그림자·텍스처)은 각 형식이 직접 그리고,
레벨·치수·축선·재료 번호·지반선·제목은 annotations()가 만든 '도형 목록'을 세 형식이 같이 쓴다.
"""
import base64
import datetime
import io
import math
import os
import re
from xml.sax.saxutils import escape

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MPoly, PathPatch, Circle
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
for f in ("Pretendard-Regular.ttf", "Pretendard-Medium.ttf", "Pretendard-SemiBold.ttf"):
    p = os.path.join(HERE, "fonts", f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams["font.family"] = "Pretendard"
plt.rcParams["hatch.linewidth"] = 0.25
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# 선 굵기(도면 mm): 0 윤곽, 1 근경, 2 중경, 3 원경, 4 먼 곳, 5 유리 뒤
WEIGHTS_MM = [0.50, 0.30, 0.20, 0.14, 0.09, 0.09]
WEIGHT_NAMES = ["윤곽선", "근경", "중경", "원경", "먼곳", "유리뒤"]
TONES = ["#111111", "#3a3a3a", "#6b6b6b", "#9a9a9a"]
GLASS_BEHIND = "#8a94a3"
GROUND_MM = 0.8
EARTH_COLLAGE = "#e7e0d6"

# 재료 표현: (채움색, 해치(matplotlib), DXF 해치 이름, DXF 배율)
MAT_STYLE = {
    "유리": ("#dfe8f3", "////", "ANSI31", 0.6),
    "폴리카보네이트": ("#eef1f5", "||||", "LINE", 0.25),
    "목재": ("#f1eee8", "----", "LINE", 0.3),
    "벽돌": ("#efe9e4", "-|-", "BRICK", 0.5),
    "콘크리트블록": ("#ededed", "..", "AR-CONC", 0.02),
    "콘크리트": ("#f0f0f0", "...", "AR-CONC", 0.03),
    "금속": ("#f6f6f6", None, None, 1),
    "기타": ("#ffffff", None, None, 1),
    "수목": ("#ffffff", None, None, 1),
}
SHADOW = ("#5f6670", 0.42)
INK = "#111111"
BLUE = "#2f6fdb"
GREY = "#555555"
MARGIN = dict(left=34, right=58, bottom=30, top=24)  # 종이 mm


# ── 좌표 ─────────────────────────────────────────────────────────────
class Paper:
    """입면 하나의 종이 좌표. 원점 = 버퍼 왼쪽 아래."""

    def __init__(self, r, scale):
        self.r, self.scale = r, scale
        self.k = r["px"] * 1000.0 / scale       # 픽셀 → mm
        self.W, self.H = r["W"] * self.k, r["H"] * self.k
        self.vbot = r["vtop"] - r["H"] * r["px"]

    def px(self, x, y):
        return x * self.k, self.H - y * self.k

    def m(self, u, z):
        s = 1000.0 / self.scale
        return (u - self.r["u0"]) * s, (z - self.vbot) * s

    def mx(self, u):
        return (u - self.r["u0"]) * 1000.0 / self.scale

    def my(self, z):
        return (z - self.vbot) * 1000.0 / self.scale


def nice_bar(scale):
    """막대 축척 길이(m): 종이에서 30~60 mm 쯤."""
    target = 45 * scale / 1000.0
    for L in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if L >= target * 0.6:
            return L
    return 500


def fmt_h(mm):
    return "±0" if abs(mm) < 1 else (f"+{mm:,.0f}" if mm > 0 else f"−{-mm:,.0f}")


# ── 주석(레벨·치수·축선·재료 번호·지반) → 도형 목록 ─────────────────────
def annotations(r, scale, ann):
    """ann = {levels:[(이름,z)], gl, dims, grids, tags:{재료:(번호,이름)}, title, legend}
    → [(종류, …, 레이어)] (종이 mm, 원점 = 버퍼 왼쪽 아래)."""
    P = Paper(r, scale)
    out = []
    gl = ann.get("gl", r["gl"])
    levels = sorted(ann.get("levels", []), key=lambda t: t[1])
    xl = P.mx(r["umin"]); xr = P.mx(r["umax"])
    gy = P.my(gl)
    collage = ann.get("style") == "collage"
    earth = "earth-collage" if collage else "earth"
    gw = 0.35 if collage else GROUND_MM
    # 지반: 지형 단면이 있으면 그 선, 없으면 GL 수평선
    if r.get("ground"):
        segs = np.array(r["ground"])
        uu = np.r_[segs[:, 0], segs[:, 2]]; zz = np.r_[segs[:, 1], segs[:, 3]]
        o = np.argsort(uu); uu, zz = uu[o], zz[o]
        xs = np.linspace(uu.min(), uu.max(), 400)
        # 같은 u에서 가장 높은 지면
        zi = np.full_like(xs, -1e9)
        for a, b, c, d in segs:
            lo, hi = min(a, c), max(a, c)
            m = (xs >= lo) & (xs <= hi)
            if hi - lo < 1e-6:
                continue
            zi[m] = np.maximum(zi[m], b + (d - b) * (xs[m] - a) / (c - a))
        ok = zi > -1e8
        pts = [P.m(x, z) for x, z in zip(xs[ok], zi[ok])]
        if len(pts) > 1:
            bottom = min(p[1] for p in pts) - 10
            if collage:
                bottom = min(p[1] for p in pts) - 14
            out.append(("poly", pts + [(pts[-1][0], bottom), (pts[0][0], bottom)], earth, "지반"))
            out.append(("line", pts, dict(w=gw, c=INK), "지반"))
    else:
        if collage:
            out.append(("poly", [(xl - 8, gy), (xr + 8, gy), (xr + 8, gy - 14), (xl - 8, gy - 14)], earth, "지반"))
        out.append(("line", [(xl - 8, gy), (xr + 8, gy)], dict(w=gw, c=INK), "지반"))
    # 레벨
    xlev = xr + 10
    last_ty = -1e9
    for name, z in [("GL", gl)] + [(n, z) for n, z in levels if abs(z - gl) > 0.05]:
        y = P.my(z)
        ty = max(y + 0.3, last_ty + 3.2)  # 가까운 레벨 글자가 겹치지 않게
        last_ty = ty
        hh = (z - gl) * 1000
        if collage:  # 작은 검은 삼각형 + 미터 두 자리(+ 해발), 긴 레벨선 없이
            el = ann.get("el_base")
            mtxt = "±0.00" if abs(hh) < 1 else f"{hh / 1000:+.2f}"
            lab = (f"{name}  " if name != "GL" else "") + mtxt + (f" = EL {el + hh / 1000:.2f}" if el is not None else "")
            out.append(("poly", [(xr + 3, y), (xr + 1.9, y + 1.5), (xr + 4.1, y + 1.5)], "levelmark-ink", "레벨"))
            out.append(("line", [(xr + 0.8, y), (xr + 5.2, y)], dict(w=0.1, c=INK), "레벨"))
            out.append(("text", xr + 5.6, ty, lab, 2.2, "left", "bottom", "레벨"))
            continue
        # 레벨선은 건물 바깥에만(창호 위를 가로지르지 않게)
        out.append(("line", [(xl - 4, y), (xl - 1, y)], dict(w=0.13, c=BLUE), "레벨"))
        out.append(("line", [(xr + 1.5, y), (xlev + 3, y)], dict(w=0.13, c=BLUE, dash=(6, 3)), "레벨"))
        out.append(("poly", [(xlev + 5, y), (xlev + 3.6, y + 2.2), (xlev + 6.4, y + 2.2)],
                    "levelmark-gl" if name == "GL" else "levelmark", "레벨"))
        lab = f"{name}  {fmt_h(hh)}" if name in ("GL",) else f"{name}  FL {fmt_h(hh)}" if name[-1] == "F" and name != "RF" else f"{name}  {fmt_h(hh)}"
        out.append(("text", xlev + 8, ty, lab, 2.6, "left", "bottom", "레벨"))
    if r["top"] - (levels[-1][1] if levels else gl) > 0.3 and collage:
        y = P.my(r["top"]); hh = (r["top"] - gl) * 1000; el = ann.get("el_base")
        out.append(("poly", [(xr + 3, y), (xr + 1.9, y + 1.5), (xr + 4.1, y + 1.5)], "levelmark-ink", "레벨"))
        out.append(("text", xr + 5.6, max(y + 0.3, last_ty + 3.2), f"{hh / 1000:+.2f}" + (f" = EL {el + hh / 1000:.2f}" if el is not None else ""),
                    2.2, "left", "bottom", "레벨"))
    elif r["top"] - (levels[-1][1] if levels else gl) > 0.3:
        y = P.my(r["top"])
        out.append(("line", [(xr - 2, y), (xlev + 3, y)], dict(w=0.13, c=GREY, dash=(2, 2)), "레벨"))
        out.append(("text", xlev + 8, y + 0.3, f"최고높이  {fmt_h((r['top'] - gl) * 1000)}", 2.6, "left", "bottom", "레벨"))
    # 높이 치수(왼쪽): 안쪽 = 층별, 바깥 = 전체
    if ann.get("dims", True):
        chain = [gl] + [z for n, z in levels if z > gl + 0.05]
        if r["top"] - chain[-1] > 0.3:
            chain.append(r["top"])
        xd1, xd2 = xl - 9, xl - 17
        def vdim(x, z0, z1, layer="치수"):
            y0, y1 = P.my(z0), P.my(z1)
            out.append(("line", [(x, y0), (x, y1)], dict(w=0.13, c=INK), layer))
            for y in (y0, y1):
                out.append(("line", [(x - 1, y - 1), (x + 1, y + 1)], dict(w=0.25, c=INK), layer))
                out.append(("line", [(x - 1.5, y), (x + 3.5, y)], dict(w=0.09, c=INK), layer))
            ym = (y0 + y1) / 2 if y1 - y0 >= 8 else y1 + 4.5  # 짧은 구간은 글자를 위로 빼기
            out.append(("text", x - 1.2, ym, f"{(z1 - z0) * 1000:,.0f}", 2.2, "center", "bottom", layer, 90))
            out.append(("dimspec", (x + 3.5, y0), (x + 3.5, y1), (x, y0), 90, layer))
        if len(chain) > 2:
            for a, b in zip(chain[:-1], chain[1:]):
                vdim(xd1, a, b)
        vdim(xd2 if len(chain) > 2 else xd1, chain[0], chain[-1])
    # 축선
    grids = ann.get("grid_list") or []
    if ann.get("grids", True) and grids:
        ytop = P.my(r["top"]) + 8
        ybot = gy - 2
        for g in grids:
            x = P.mx(g["u"])
            out.append(("line", [(x, ybot), (x, ytop)], dict(w=0.09, c=GREY, dash=(8, 1.5, 1, 1.5)), "축선"))
            out.append(("circle", x, ytop + 3.2, 3.2, "축선"))
            out.append(("text", x, ytop + 3.2, g["name"], 2.4, "center", "center", "축선"))
        if len(grids) > 1:
            yd = gy - 7
            xs = [P.mx(g["u"]) for g in grids]
            out.append(("line", [(xs[0], yd), (xs[-1], yd)], dict(w=0.13, c=INK), "치수"))
            for i, x in enumerate(xs):
                out.append(("line", [(x - 1, yd - 1), (x + 1, yd + 1)], dict(w=0.25, c=INK), "치수"))
                out.append(("line", [(x, yd - 1.5), (x, yd + 3)], dict(w=0.09, c=INK), "치수"))
                if i:
                    span = (grids[i]["u"] - grids[i - 1]["u"]) * 1000
                    out.append(("text", (xs[i - 1] + x) / 2, yd + 0.5, f"{span:,.0f}", 2.2, "center", "bottom", "치수"))
                    out.append(("dimspec", (xs[i - 1], yd + 3), (x, yd + 3), (x, yd), 0, "치수"))
    # 재료 번호: 면 안에 자리가 있으면 그 자리, 없으면 점 + 지시선 + 건물 위 번호
    pts = ann.get("tag_pts", [])
    up = sorted([t for t in pts if len(t) > 3 and t[3] < 2.8], key=lambda t: t[0])
    ytag = P.my(r["top"]) + 4.2
    lastx = -1e9
    for t in pts:
        if t in up:
            continue
        out.append(("tag", t[0], t[1], str(t[2]), "재료표"))
    for t in up:
        x = max(t[0], lastx + 5.6); lastx = x
        # 지시선이 너무 길면(높은 건물 아래쪽 면) 그 면 바로 위에 번호를 둔다
        yt = ytag if len(t) < 5 or ytag - t[4] < 30 else t[4] + 5.5
        out.append(("line", [(t[0], t[1]), (t[0], yt - 4.5), (x, yt - 2.3)], dict(w=0.09, c=INK), "재료표"))
        out.append(("dot", t[0], t[1], "재료표"))
        out.append(("tag", x, yt, str(t[2]), "재료표"))
    return out


def tag_points(r, scale, tags):
    """재료마다 번호 놓을 점 → [(x, y, 번호, 여유 mm)]. 넓은 면 몇 개 중 안쪽 여유가 가장 큰 곳."""
    from shapely.ops import polylabel
    P = Paper(r, scale)
    cand = {}
    for ln, it in r["layer_polys"].items():
        cat = it["cat"]
        if cat in tags:
            cand.setdefault(cat, []).extend(it["polys"])
    pts = []
    for cat, ps in cand.items():
        best = None
        for p in sorted(ps, key=lambda p: -p.area)[:6]:
            if p.area * P.k * P.k < 4:
                continue
            try:
                c = polylabel(p, tolerance=1)
            except Exception:
                c = p.representative_point()
            clear = p.exterior.distance(c)
            for h in p.interiors:
                clear = min(clear, h.distance(c))
            if best is None or clear > best[0]:
                best = (clear, c, p.bounds[1])
        if best is None:
            continue
        x, y = P.px(best[1].x, best[1].y)
        pts.append((x, y, tags[cat][0], best[0] * P.k, P.H - best[2] * P.k))  # 마지막 = 그 면 윗변(종이 y)
    return pts


# ── matplotlib (PNG·PDF·시트) ────────────────────────────────────────
def _poly_path(poly, f, ox=0, oy=0):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        c = np.array([f(x, y) for x, y in ring.coords]) + (ox, oy)
        verts.extend(c); codes.extend([Path.MOVETO] + [Path.LINETO] * (len(c) - 2) + [Path.CLOSEPOLY])
    return Path(np.array(verts), codes)


def line_style(w, tone, scale):
    mul = 1.0 if scale <= 100 else 0.75
    if w == 5:
        return WEIGHTS_MM[5] * mul, GLASS_BEHIND
    return WEIGHTS_MM[w] * mul, TONES[tone]


def draw_elev(ax, r, scale, ox, oy, ann, show_mat=True, show_shadow=True):
    """입면 하나를 종이 좌표 (ox, oy)에 그린다."""
    if r.get("collage") is not None:
        from .collage import draw_collage
        return draw_collage(ax, r, scale, ox, oy, ann, r["collage"])
    P = Paper(r, scale)
    f = P.px
    pt = 72 / 25.4
    modes = r.get("modes", {})
    if show_mat:
        for ln, it in r["layer_polys"].items():
            if modes.get(ln, "hatch") != "hatch":
                continue
            fc, hatch, *_ = MAT_STYLE.get(it["cat"], MAT_STYLE["기타"])
            for p in it["polys"]:
                ax.add_patch(PathPatch(_poly_path(p, f, ox, oy), facecolor=fc, edgecolor="#9aa3ad" if hatch else "none",
                                       hatch=hatch, lw=0, zorder=1))
        comp = composite_tex(r)
        if comp is not None:
            img, x0, y0 = comp
            hh, ww = img.shape[:2]
            xa, ya = f(x0, y0)
            ax.imshow(img, extent=(ox + xa, ox + xa + ww * P.k, oy + ya - hh * P.k, oy + ya), zorder=1.5,
                      interpolation="nearest")
    if show_shadow:
        for p in r["shadow"]:
            ax.add_patch(PathPatch(_poly_path(p, f, ox, oy), facecolor=SHADOW[0], alpha=SHADOW[1], lw=0, zorder=2))
    groups = {}
    for pts, w, l, tn in r["lines"]:
        groups.setdefault((w, tn), []).append(np.c_[pts[:, 0] * P.k + ox, P.H - pts[:, 1] * P.k + oy])
    for (w, tn), ls in sorted(groups.items(), key=lambda kv: -kv[0][0]):
        lw, col = line_style(w, tn, scale)
        ax.add_collection(LineCollection(ls, colors=col, linewidths=lw * pt, capstyle="round", joinstyle="round",
                                         zorder=3 + (5 - w) * 0.1))
    for a in annotations(r, scale, ann):
        _mpl_prim(ax, a, ox, oy, pt)


def composite_tex(r):
    """레이어별 텍스처를 한 장으로 겹침(PNG·PDF용, 큰 투명 그림 여러 장을 그리는 시간 절약)."""
    T = r.get("tex") or {}
    if not T:
        return None
    x0 = min(t["x"] for t in T.values()); y0 = min(t["y"] for t in T.values())
    x1 = max(t["x"] + t["rgba"].shape[1] for t in T.values()); y1 = max(t["y"] + t["rgba"].shape[0] for t in T.values())
    acc = np.zeros((y1 - y0, x1 - x0, 3), np.float32); al = np.zeros((y1 - y0, x1 - x0), np.float32)
    for t in T.values():
        h, w = t["rgba"].shape[:2]
        sl = (slice(t["y"] - y0, t["y"] - y0 + h), slice(t["x"] - x0, t["x"] - x0 + w))
        a = t["rgba"][..., 3].astype(np.float32) / 255
        acc[sl] = acc[sl] * (1 - a[..., None]) + t["rgba"][..., :3] * a[..., None]
        al[sl] = al[sl] * (1 - a) + a
    out = np.dstack([np.where(al[..., None] > 0, acc / np.maximum(al[..., None], 1e-6), 0), al * 255]).astype(np.uint8)
    return out, x0, y0


def _mpl_prim(ax, a, ox, oy, pt):
    kind = a[0]
    if kind == "line":
        pts = np.array(a[1]) + (ox, oy); st = a[2]
        ls = (0, st["dash"]) if st.get("dash") else "-"
        ax.plot(pts[:, 0], pts[:, 1], color=st["c"], lw=st["w"] * pt, ls=ls, solid_capstyle="butt", zorder=8)
    elif kind == "poly":
        pts = np.array(a[1]) + (ox, oy); sty = a[2]
        if sty == "earth":
            ax.add_patch(MPoly(pts, closed=True, facecolor="#e9e9e9", edgecolor="#999", hatch="////", lw=0, zorder=6))
        elif sty == "earth-collage":
            ax.add_patch(MPoly(pts, closed=True, facecolor=EARTH_COLLAGE, edgecolor="none", lw=0, zorder=0.5))
        else:
            if sty == "levelmark-ink":
                ax.add_patch(MPoly(pts, closed=True, facecolor=INK, edgecolor="none", lw=0, zorder=9))
                return
            ax.add_patch(MPoly(pts, closed=True, facecolor=BLUE if sty == "levelmark-gl" else "white", edgecolor=BLUE,
                               lw=0.18 * pt, zorder=9))
    elif kind == "text":
        _, x, y, s, size, ha, va, layer, *rot = a
        ax.text(x + ox, y + oy, s, fontsize=size * pt, color=INK, ha=ha, va={"bottom": "bottom", "center": "center"}[va],
                rotation=rot[0] if rot else 0, zorder=9)
    elif kind == "circle":
        _, x, y, rr, layer = a
        ax.add_patch(Circle((x + ox, y + oy), rr, facecolor="white", edgecolor=GREY, lw=0.13 * pt, zorder=9))
    elif kind == "dot":
        _, x, y, layer = a
        ax.add_patch(Circle((x + ox, y + oy), 0.45, facecolor=INK, lw=0, zorder=10))
    elif kind == "tag":
        _, x, y, s, layer = a
        ax.add_patch(Circle((x + ox, y + oy), 2.3, facecolor="white", edgecolor=INK, lw=0.18 * pt, zorder=10))
        ax.text(x + ox, y + oy, s, fontsize=2.4 * pt, ha="center", va="center", zorder=11, weight="semibold")


def _text_w(s, size):
    return sum(size * (1.0 if ord(c) > 0x2000 else 0.58) for c in s)


def bottom_layout(r, scale, ann):
    """아래 띠(제목·재료 범례·막대 축척) 배치. 상자 좌표(왼쪽 아래 원점, mm).
    → mb(그림 원점 높이), 범례 (x, 맨 윗줄 y), 막대 (x, y)."""
    P = Paper(r, scale)
    n = len(ann.get("legend", []))
    ox = MARGIN["left"]
    tw = _text_w(ann.get("title") or r["name"], 6)
    bl = nice_bar(scale) * 1000 / scale
    bar = [ox + P.W - bl, 6]
    lx = max(ox + tw + 12, ox + P.W * 0.45)
    if not n:
        if bar[0] < ox + tw + 8:  # 좁은 입면: 막대를 제목 위로
            bar = [ox, 24]
            return MARGIN["bottom"] + 11, None, bar
        return MARGIN["bottom"], None, bar
    if lx + 34 <= bar[0] - 6:
        mb = MARGIN["bottom"] + max(0, n - 2) * 4.5
        return mb, (lx, 4 + 4.5 * (n - 1) + 3), bar
    # 좁은 입면: 제목 → 막대 → 범례 순으로 위로 쌓는다
    bar = [ox, 24]
    ly = 33 + 4.5 * (n - 1)
    return ly + 4 + 14, (ox, ly), bar


def elev_box(r, scale, ann):
    P = Paper(r, scale)
    mb = bottom_layout(r, scale, ann)[0]
    # 버퍼 위로 올라가는 레벨·축선 동그라미까지 들어가게
    zs = [r["top"]] + [z for _, z in ann.get("levels", [])]
    over = max(0.0, P.my(max(zs)) - P.H) + (12 if ann.get("grid_list") and ann.get("grids", True) else 0)
    return P.W + MARGIN["left"] + MARGIN["right"], P.H + mb + MARGIN["top"] + over, mb


def scale_line(scale, ann):
    sub = ann.get("subtitle")
    return f"SCALE 1:{scale}" + (f"   {sub}" if sub else "")


def _title_block(ax, x, y, r, scale, ann, pt):
    ax.text(x, y + 9, ann.get("title") or r["name"], fontsize=6 * pt, weight="semibold", color=INK, ha="left", va="bottom")
    ax.text(x, y + 4.5, scale_line(scale, ann), fontsize=2.8 * pt, color=GREY, ha="left", va="bottom")


def _scale_bar(ax, x, y, scale, pt):
    L = nice_bar(scale); seg = L * 1000 / scale / 5
    for i in range(5):
        ax.add_patch(MPoly([[x + i * seg, y], [x + (i + 1) * seg, y], [x + (i + 1) * seg, y + 1.4], [x + i * seg, y + 1.4]],
                           facecolor=INK if i % 2 == 0 else "white", edgecolor=INK, lw=0.15 * pt))
    ax.text(x, y + 2.5, "0", fontsize=2.2 * pt, ha="center"); ax.text(x + 5 * seg, y + 2.5, f"{L}m", fontsize=2.2 * pt, ha="center")
    return 5 * seg


def _legend(ax, x, y, ann, pt):
    for i, (num, name, fc) in enumerate(ann.get("legend", [])):
        yy = y - i * 4.5
        ax.add_patch(Circle((x + 2.3, yy), 2.0, facecolor="white", edgecolor=INK, lw=0.15 * pt))
        ax.text(x + 2.3, yy, str(num), fontsize=2.2 * pt, ha="center", va="center", weight="semibold")
        ax.text(x + 6, yy, name, fontsize=2.5 * pt, ha="left", va="center", color=INK)


def render_mpl(r, scale, out_paths, ann, dpi=200, show_mat=True, show_shadow=True):
    BW, BH, mb = elev_box(r, scale, ann)
    fig = plt.figure(figsize=(BW / 25.4, BH / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, BW); ax.set_ylim(0, BH); ax.set_aspect("equal"); ax.axis("off")
    ox, oy = MARGIN["left"], mb
    draw_elev(ax, r, scale, ox, oy, ann, show_mat, show_shadow)
    pt = 72 / 25.4
    _title_block(ax, MARGIN["left"], 4, r, scale, ann, pt)
    _, leg, bar = bottom_layout(r, scale, ann)
    _scale_bar(ax, bar[0], bar[1], scale, pt)
    if leg:
        _legend(ax, leg[0], leg[1], ann, pt)
    ax.set_xlim(0, BW); ax.set_ylim(0, BH)
    for p in out_paths:
        fig.savefig(p, dpi=dpi, facecolor="white")
    plt.close(fig)


PAPERS = {"A3": (420, 297), "A2": (594, 420), "A1": (841, 594), "A0": (1189, 841)}


def render_sheet(results, scale, path, ann_of, paper="A1", project="", log=print):
    """여러 입면을 한 장(가로)에 배치 + 표제란. 안 들어가면 큰 종이로 올린다."""
    boxes = [elev_box(r, scale, ann_of(r)) for r in results]

    def fits(bx, PW, PH):
        uw, uh = PW - 20, PH - 20 - 32
        for c in (1, 2):
            rows = [bx[i:i + c] for i in range(0, len(bx), c)]
            if max(sum(b[0] for b in row) for row in rows) <= uw and sum(max(b[1] for b in row) for row in rows) <= uh:
                return c
        return 0

    def rescaled(new):  # 축척을 바꿨을 때의 상자 크기(여백은 그대로, 그림만 줄어듦)
        f = scale / new
        mw = MARGIN["left"] + MARGIN["right"]
        return [((w - mw) * f + mw, (h - b - MARGIN["top"]) * f + b + MARGIN["top"], b) for w, h, b in boxes]
    order = list(PAPERS)
    start = order.index(paper) if paper in PAPERS else 2
    for pname in order[start:]:
        PW, PH = PAPERS[pname]
        usable_w, usable_h = PW - 20, PH - 20 - 32
        for cols in (1, 2):
            rows = [boxes[i:i + cols] for i in range(0, len(boxes), cols)]
            wneed = max(sum(b[0] for b in row) for row in rows)
            hneed = sum(max(b[1] for b in row) for row in rows)
            if wneed <= usable_w and hneed <= usable_h:
                break
        else:
            continue
        break
    else:
        pname = "A0"; PW, PH = PAPERS["A0"]; cols = 1
        log(f"입면이 A0에도 다 안 들어가요. 1:{scale}보다 작은 축척을 고르세요.")
    suggest = None
    if pname != paper and paper in PAPERS:
        for new in (150, 200, 250, 300, 400, 500, 600, 750, 1000):
            if new > scale and fits(rescaled(new), *PAPERS[paper]):
                suggest = new
                break
        log(f"시트가 {paper}에 안 들어가서 {pname}로 만들었어요" + (f". {paper}에 넣으려면 축척을 1:{suggest} 정도로 바꿔 보세요" if suggest else ""))
    fig = plt.figure(figsize=(PW / 25.4, PH / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, PW); ax.set_ylim(0, PH); ax.set_aspect("equal"); ax.axis("off")
    pt = 72 / 25.4
    ax.add_patch(MPoly([[10, 10], [PW - 10, 10], [PW - 10, PH - 10], [10, PH - 10]], closed=True, fill=False, edgecolor=INK, lw=0.5 * pt))
    rows = [list(range(k, min(k + cols, len(results)))) for k in range(0, len(results), cols)]
    hneed = sum(max(boxes[j][1] for j in row) for row in rows)
    y = PH - 10 - max(0.0, (PH - 20 - 32 - hneed) / 2)  # 위아래 가운데
    for row in rows:
        rh = max(boxes[j][1] for j in row)
        x = 10 + max(0.0, (PW - 20 - sum(boxes[j][0] for j in row)) / 2)  # 좌우 가운데
        y -= rh
        for j in row:
            r = results[j]; BW, BH, mb = boxes[j]; ann = ann_of(r)
            draw_elev(ax, r, scale, x + MARGIN["left"], y + (rh - BH) + mb, ann)
            ax.text(x + MARGIN["left"], y + (rh - BH) + 6, ann.get("title") or r["name"], fontsize=5 * pt, weight="semibold")
            ax.text(x + MARGIN["left"], y + (rh - BH) + 2.5, scale_line(scale, ann), fontsize=2.6 * pt, color=GREY)
            if ann.get("legend") and j == 0:
                pass
            x += BW
    # 표제란
    tb_w, tb_h = 150, 32
    x0, y0 = PW - 10 - tb_w, 10
    ax.add_patch(MPoly([[x0, y0], [x0 + tb_w, y0], [x0 + tb_w, y0 + tb_h], [x0, y0 + tb_h]], closed=True, facecolor="white", edgecolor=INK, lw=0.35 * pt))
    ax.plot([x0, x0 + tb_w], [y0 + 16, y0 + 16], color=INK, lw=0.18 * pt)
    ax.text(x0 + 4, y0 + 24, project or "프로젝트", fontsize=5 * pt, weight="semibold", va="center")
    ax.text(x0 + 4, y0 + 8, f"입면도   SCALE 1:{scale}   {pname}", fontsize=3 * pt, va="center")
    ax.text(x0 + tb_w - 4, y0 + 8, datetime.date.today().isoformat(), fontsize=3 * pt, va="center", ha="right", color=GREY)
    bl = _scale_bar(ax, x0 - 10 - nice_bar(scale) * 1000 / scale, y0 + 4, scale, pt)
    used = set()
    for r in results:
        used |= {l[0] for l in ann_of(r).get("legend", [])}
    leg = [l for l in (ann_of(results[0]).get("legend_all") or []) if l[0] in used]
    if leg:
        _legend(ax, x0 - 10 - bl - 60, y0 + 4 + 4.5 * (len(leg) - 1), {"legend": leg}, pt)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return pname, suggest


# ── SVG ──────────────────────────────────────────────────────────────
def _sid(s):
    return escape(re.sub(r"\s+", "_", s)).replace('"', "")


def _png_b64(rgba):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(b, "PNG", compress_level=6)
    return base64.b64encode(b.getvalue()).decode()


def export_svg(r, scale, path, ann):
    P = Paper(r, scale)
    BW, BH, mb = elev_box(r, scale, ann)
    ox, oy = MARGIN["left"], mb
    X = lambda x: x + ox
    Y = lambda y: BH - (y + oy)
    f = P.px
    o = ['<?xml version="1.0" encoding="UTF-8"?>',
         f'<svg xmlns="http://www.w3.org/2000/svg" width="{BW:.2f}mm" height="{BH:.2f}mm" viewBox="0 0 {BW:.2f} {BH:.2f}">',
         '<defs>',
         '<pattern id="h-glass" patternUnits="userSpaceOnUse" width="1.6" height="1.6" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="1.6" stroke="#8fa3bb" stroke-width="0.07"/></pattern>',
         '<pattern id="h-conc" patternUnits="userSpaceOnUse" width="1.2" height="1.2"><circle cx="0.3" cy="0.3" r="0.06" fill="#888"/><circle cx="0.9" cy="0.85" r="0.05" fill="#888"/></pattern>',
         '<pattern id="h-wood" patternUnits="userSpaceOnUse" width="2" height="0.8"><line x1="0" y1="0.4" x2="2" y2="0.4" stroke="#a99" stroke-width="0.06"/></pattern>',
         '<pattern id="h-poly" patternUnits="userSpaceOnUse" width="0.6" height="2"><line x1="0.3" y1="0" x2="0.3" y2="2" stroke="#99a" stroke-width="0.05"/></pattern>',
         '<pattern id="h-brick" patternUnits="userSpaceOnUse" width="2.2" height="1.3"><path d="M0 0.65H2.2M0 1.3H2.2M0.55 0V0.65M1.65 0.65V1.3" stroke="#a88" stroke-width="0.06" fill="none"/></pattern>',
         '<pattern id="h-earth" patternUnits="userSpaceOnUse" width="1.5" height="1.5" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="1.5" stroke="#999" stroke-width="0.08"/></pattern>',
         '</defs>', f'<rect id="바탕" width="{BW:.2f}" height="{BH:.2f}" fill="#fff"/>']
    pat = {"유리": "h-glass", "콘크리트": "h-conc", "콘크리트블록": "h-conc", "목재": "h-wood", "폴리카보네이트": "h-poly", "벽돌": "h-brick"}

    def pathd(p):
        s = []
        for ring in [p.exterior, *p.interiors]:
            c = [f(x, y) for x, y in ring.coords]
            s.append("M" + " L".join(f"{X(a):.2f} {Y(b):.2f}" for a, b in c) + "Z")
        return " ".join(s)
    LP = r["layer_polys"]
    modes = r.get("modes", {})
    comp = r.get("collage")
    lstyle = line_style
    if comp is not None:
        from . import collage as CL
        defs, body = CL.svg_layers(r, scale, comp, X, Y, pathd)
        o[o.index("</defs>")] = "".join(defs) + "</defs>"
        o += body
        lstyle = CL.line_style
    else:
        o.append('<g id="재료_채움">')
        for ln, it in LP.items():
            if modes.get(ln, "hatch") != "hatch":
                continue
            fc = MAT_STYLE.get(it["cat"], MAT_STYLE["기타"])[0]
            o.append(f'<g id="채움_{_sid(ln)}">' + "".join(f'<path d="{pathd(p)}" fill="{fc}" fill-rule="evenodd"/>' for p in it["polys"]) + "</g>")
        o.append('</g><g id="재료_해치">')
        for ln, it in LP.items():
            if it["cat"] in pat and modes.get(ln, "hatch") == "hatch":
                o.append(f'<g id="해치_{_sid(ln)}">' + "".join(f'<path d="{pathd(p)}" fill="url(#{pat[it["cat"]]})" fill-rule="evenodd"/>' for p in it["polys"]) + "</g>")
        o.append('</g><g id="재료_텍스처">')
        for ln, t in r.get("tex", {}).items():
            hh, ww = t["rgba"].shape[:2]
            xa, ya = f(t["x"], t["y"])
            o.append(f'<g id="텍스처_{_sid(ln)}"><image x="{X(xa):.3f}" y="{Y(ya):.3f}" width="{ww * P.k:.3f}" height="{hh * P.k:.3f}" '
                     f'preserveAspectRatio="none" href="data:image/png;base64,{_png_b64(t["rgba"])}"/></g>')
        o.append('</g><g id="그림자">' + "".join(
            f'<path d="{pathd(p)}" fill="{SHADOW[0]}" fill-opacity="{SHADOW[1]}" fill-rule="evenodd"/>' for p in r["shadow"]) + "</g>")
    o.append('<g id="선" fill="none" stroke-linecap="round" stroke-linejoin="round">')
    by_layer = {}
    for pts, w, l, tn in r["lines"]:
        by_layer.setdefault(l, {}).setdefault((w, tn), []).append(pts)
    for li, ln in enumerate(r["layer_names"]):
        if li not in by_layer:
            continue
        o.append(f'<g id="{_sid(ln)}">')
        for (w, tn), pl in sorted(by_layer[li].items()):
            lw, col = lstyle(w, tn, scale)
            o.append(f'<g id="{_sid(ln)}_{WEIGHT_NAMES[w]}_{lw:.2f}mm{"_" + str(tn) if tn else ""}" stroke="{col}" stroke-width="{lw:.3f}">')
            o.append("".join('<polyline points="' + " ".join(f"{X(x * P.k):.2f},{Y(P.H - y * P.k):.2f}" for x, y in pts) + '"/>' for pts in pl))
            o.append("</g>")
        o.append("</g>")
    o.append("</g>")
    if comp is not None:
        o.append(CL.svg_sprites(r, scale, comp, X, Y))
    # 주석
    groups = {}
    for a in annotations(r, scale, ann):
        if a[0] == "dimspec":
            continue
        groups.setdefault(a[-1] if a[0] != "text" else a[7], []).append(a)
    for gname, items in groups.items():
        o.append(f'<g id="{gname}" font-family="Pretendard">')
        for a in items:
            o.append(_svg_prim(a, X, Y))
        o.append("</g>")
    o.append(f'<g id="제목" font-family="Pretendard"><text x="{MARGIN["left"]:.2f}" y="{BH - 13:.2f}" font-size="6" font-weight="600">{escape(ann.get("title") or r["name"])}</text>'
             f'<text x="{MARGIN["left"]:.2f}" y="{BH - 8.5:.2f}" font-size="2.8" fill="{GREY}">{escape(scale_line(scale, ann))}</text></g>')
    _, leg, bar = bottom_layout(r, scale, ann)
    if leg:
        lx, ly = leg
        o.append('<g id="재료_범례" font-family="Pretendard">')
        for i, (num, name, fc) in enumerate(ann["legend"]):
            yy = BH - (ly - i * 4.5)
            o.append(f'<circle cx="{lx + 2.3:.2f}" cy="{yy:.2f}" r="2" fill="#fff" stroke="{INK}" stroke-width="0.15"/>'
                     f'<text x="{lx + 2.3:.2f}" y="{yy + 0.8:.2f}" font-size="2.2" text-anchor="middle" font-weight="600">{num}</text>'
                     f'<text x="{lx + 6:.2f}" y="{yy + 0.9:.2f}" font-size="2.5">{escape(name)}</text>')
        o.append("</g>")
    L = nice_bar(scale); seg = L * 1000 / scale / 5; bx, by = bar
    o.append('<g id="막대축척">' + "".join(
        f'<rect x="{bx + i * seg:.2f}" y="{BH - by - 1.4:.2f}" width="{seg:.2f}" height="1.4" fill="{INK if i % 2 == 0 else "#fff"}" stroke="{INK}" stroke-width="0.15"/>' for i in range(5))
        + f'<text x="{bx:.2f}" y="{BH - by - 2.6:.2f}" font-size="2.2" text-anchor="middle" font-family="Pretendard">0</text>'
        + f'<text x="{bx + 5 * seg:.2f}" y="{BH - by - 2.6:.2f}" font-size="2.2" text-anchor="middle" font-family="Pretendard">{L}m</text></g>')
    o.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(o))


def _svg_prim(a, X, Y):
    k = a[0]
    if k == "line":
        st = a[2]
        dash = f' stroke-dasharray="{" ".join(str(v) for v in st["dash"])}"' if st.get("dash") else ""
        return f'<polyline points="{" ".join(f"{X(x):.2f},{Y(y):.2f}" for x, y in a[1])}" fill="none" stroke="{st["c"]}" stroke-width="{st["w"]}"{dash}/>'
    if k == "poly":
        pts = " ".join(f"{X(x):.2f},{Y(y):.2f}" for x, y in a[1])
        if a[2] == "earth":
            return f'<polygon points="{pts}" fill="url(#h-earth)" stroke="none"/>'
        if a[2] == "earth-collage":
            return f'<polygon points="{pts}" fill="{EARTH_COLLAGE}" stroke="none"/>'
        if a[2] == "levelmark-ink":
            return f'<polygon points="{pts}" fill="{INK}" stroke="none"/>'
        return f'<polygon points="{pts}" fill="{BLUE if a[2] == "levelmark-gl" else "#fff"}" stroke="{BLUE}" stroke-width="0.18"/>'
    if k == "text":
        _, x, y, s, size, ha, va, layer, *rot = a
        anchor = {"left": "start", "center": "middle"}[ha]
        dy = 0.35 * size if va == "center" else 0
        tr = f' transform="rotate({-rot[0]} {X(x):.2f} {Y(y):.2f})"' if rot else ""
        return f'<text x="{X(x):.2f}" y="{Y(y) + dy:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{INK}"{tr}>{escape(s)}</text>'
    if k == "circle":
        return f'<circle cx="{X(a[1]):.2f}" cy="{Y(a[2]):.2f}" r="{a[3]}" fill="#fff" stroke="{GREY}" stroke-width="0.13"/>'
    if k == "dot":
        return f'<circle cx="{X(a[1]):.2f}" cy="{Y(a[2]):.2f}" r="0.45" fill="{INK}"/>'
    if k == "tag":
        return (f'<circle cx="{X(a[1]):.2f}" cy="{Y(a[2]):.2f}" r="2.3" fill="#fff" stroke="{INK}" stroke-width="0.18"/>'
                f'<text x="{X(a[1]):.2f}" y="{Y(a[2]) + 0.85:.2f}" font-size="2.4" text-anchor="middle" font-weight="600">{a[3]}</text>')
    return ""


# ── DXF (실제 크기 mm) ───────────────────────────────────────────────
def _dxf_name(s):
    return re.sub(r'[<>/\\":;?*|=`]', "_", s).strip() or "0"


def export_dxf(r, scale, path, ann):
    import ezdxf
    doc = ezdxf.new("R2018", setup=True); doc.units = 4
    doc.header["$LWDISPLAY"] = 1
    # 실제 크기(mm) 도면: 점선 한 마디(1.5 mm 기준 패턴)가 종이에서 약 3 mm가 되게
    doc.header["$LTSCALE"] = 2.0 * scale
    doc.header["$PSLTSCALE"] = 0
    if "KOREAN" not in doc.styles:
        st = doc.styles.add("KOREAN", font="malgun.ttf")
        st.set_extended_font_data(family="맑은 고딕")
    msp = doc.modelspace()
    P = Paper(r, scale)
    px = r["px"]
    M = lambda x, y: ((x * px + r["u0"]) * 1000, (r["vtop"] - y * px) * 1000)
    S = lambda x, y: ((x + 0) * scale + (P.r["u0"] * 1000), (y * scale) + P.vbot * 1000)  # 종이 mm → 실제 mm
    lw = [50, 30, 20, 13, 9, 9]
    names = [_dxf_name(n) for n in r["layer_names"]]
    for n in set(names):
        if n not in doc.layers:
            doc.layers.add(n, color=7, lineweight=-3)
    aci_tone = [7, 250, 252, 253]
    for pts, w, l, tn in r["lines"]:
        att = {"layer": names[l], "lineweight": lw[w]}
        if w == 5:
            att["color"] = 8
        elif tn:
            att["color"] = aci_tone[tn]
        msp.add_lwpolyline([M(x, y) for x, y in pts], dxfattribs=att)
    for n, cl, lwt in (("지반", 7, 80), ("그림자", 8, 0), ("레벨", 5, 13), ("치수", 7, 13), ("축선", 8, 9), ("재료표", 7, 13)):
        if n not in doc.layers:
            doc.layers.add(n, color=cl, lineweight=lwt)

    def hatch(polys, layer, pattern=None, sc=1.0, color=8, solid_tr=None):
        for p in polys:
            h = msp.add_hatch(color=color, dxfattribs={"layer": layer})
            if pattern:
                h.set_pattern_fill(pattern, scale=sc * 10)
            if solid_tr is not None:
                h.transparency = solid_tr
            h.paths.add_polyline_path([M(x, y) for x, y in p.exterior.coords], is_closed=True, flags=1)
            for ring in p.interiors:
                h.paths.add_polyline_path([M(x, y) for x, y in ring.coords], is_closed=True)
    modes = r.get("modes", {})
    for ln, it in r["layer_polys"].items():
        _, _, patn, sc = MAT_STYLE.get(it["cat"], MAT_STYLE["기타"])
        if patn and modes.get(ln, "hatch") == "hatch":
            hn = "해치_" + _dxf_name(ln)
            if hn not in doc.layers:
                doc.layers.add(hn, color=8, lineweight=5)
            hatch(it["polys"], hn, patn, sc, color=8)
    if r.get("tex"):
        from PIL import Image
        tdir = os.path.splitext(path)[0] + "_텍스처"
        os.makedirs(tdir, exist_ok=True)
        for ln, t in r["tex"].items():
            hh, ww = t["rgba"].shape[:2]
            fn = os.path.join(tdir, _dxf_name(ln) + ".png")
            Image.fromarray(t["rgba"], "RGBA").save(fn)
            lname = "텍스처_" + _dxf_name(ln)
            if lname not in doc.layers:
                doc.layers.add(lname, color=7)
            idef = doc.add_image_def(filename=os.path.relpath(fn, os.path.dirname(path)), size_in_pixel=(ww, hh))
            msp.add_image(idef, insert=M(t["x"], t["y"] + hh), size_in_units=(ww * px * 1000, hh * px * 1000), dxfattribs={"layer": lname})
    hatch(r["shadow"], "그림자", None, color=8, solid_tr=0.55)
    # 주석: 종이 mm → 실제 mm (×축척)
    if "입면치수" not in doc.dimstyles:
        doc.dimstyles.new("입면치수", dxfattribs=dict(dimtxt=2.2, dimtsz=1.0, dimasz=1.2, dimscale=scale, dimexo=1.0,
                                                    dimexe=1.5, dimgap=0.6, dimtad=1, dimdec=0, dimtih=0, dimtoh=0,
                                                    dimtxsty="KOREAN", dimlfac=1.0))
    for a in annotations(r, scale, ann):
        k = a[0]
        if k == "dimspec":  # 캐드에서 고칠 수 있는 진짜 치수 개체
            _, p0, p1, base, ang, layer = a
            d = msp.add_linear_dim(base=S(*base), p1=S(*p0), p2=S(*p1), angle=ang, dimstyle="입면치수",
                                   dxfattribs={"layer": layer})
            d.render()
            continue
        if (a[7] if k == "text" else a[-1]) == "치수" and k in ("line", "text"):
            continue  # 치수 개체가 대신 그린다
        if k == "line":
            att = {"layer": a[-1]}
            if a[2].get("dash"):
                att["linetype"] = "DASHED" if a[-1] == "레벨" else "CENTER"
            msp.add_lwpolyline([S(x, y) for x, y in a[1]], dxfattribs=att)
        elif k == "poly":
            if a[2] in ("earth", "earth-collage"):
                h = msp.add_hatch(color=8, dxfattribs={"layer": a[-1]})
                h.set_pattern_fill("EARTH", scale=scale / 10)
                h.paths.add_polyline_path([S(x, y) for x, y in a[1]], is_closed=True)
            else:
                msp.add_lwpolyline([S(x, y) for x, y in a[1]], close=True, dxfattribs={"layer": a[-1]})
        elif k == "text":
            _, x, y, s, size, ha, va, layer, *rot = a
            t = msp.add_text(s, height=size * scale, dxfattribs={"layer": layer, "style": "KOREAN", "rotation": rot[0] if rot else 0})
            align = {("left", "bottom"): "BOTTOM_LEFT", ("center", "bottom"): "BOTTOM_CENTER", ("center", "center"): "MIDDLE_CENTER",
                     ("left", "center"): "MIDDLE_LEFT"}[(ha, va)]
            t.set_placement(S(x, y), align=ezdxf.enums.TextEntityAlignment[align])
        elif k == "dot":
            h = msp.add_hatch(color=7, dxfattribs={"layer": a[-1]})
            h.paths.add_polyline_path([S(a[1] + 0.45 * math.cos(t), a[2] + 0.45 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 13)[:-1]], is_closed=True)
        elif k == "circle":
            msp.add_circle(S(a[1], a[2]), a[3] * scale, dxfattribs={"layer": a[-1]})
        elif k == "tag":
            msp.add_circle(S(a[1], a[2]), 2.3 * scale, dxfattribs={"layer": a[-1]})
            t = msp.add_text(a[3], height=2.4 * scale, dxfattribs={"layer": a[-1], "style": "KOREAN"})
            t.set_placement(S(a[1], a[2]), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    mb, leg, _ = bottom_layout(r, scale, ann)
    for i, (num, name, fc) in enumerate(ann.get("legend", []) if leg else []):
        x, y = leg[0] - MARGIN["left"], leg[1] - mb - 4.5 * i
        msp.add_circle(S(x + 2.3, y), 2.0 * scale, dxfattribs={"layer": "재료표"})
        t = msp.add_text(f"{num}", height=2.2 * scale, dxfattribs={"layer": "재료표", "style": "KOREAN"})
        t.set_placement(S(x + 2.3, y), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
        t = msp.add_text(name, height=2.5 * scale, dxfattribs={"layer": "재료표", "style": "KOREAN"})
        t.set_placement(S(x + 6, y), align=ezdxf.enums.TextEntityAlignment.MIDDLE_LEFT)
    # 배치(레이아웃): 종이 크기 = 입면 상자, 뷰포트 축척 1:scale
    try:
        BW, BH, mb = elev_box(r, scale, ann)
        lay = doc.layouts.new(f"입면_1-{scale}")
        lay.page_setup(size=(BW, BH), margins=(0, 0, 0, 0), units="mm")
        cx, cy = S(BW / 2 - MARGIN["left"], BH / 2 - mb)
        lay.add_viewport(center=(BW / 2, BH / 2), size=(BW, BH), view_center_point=(cx, cy), view_height=BH * scale)
    except Exception:
        pass
    doc.saveas(path)


def export_3dm(r, scale, path, ann):
    """라이노로 돌려보내기: 실제 크기 mm, 입면 = XY 평면. 레이어 이름·선 굵기(출력 굵기) 그대로, 면은 닫힌 선."""
    import rhino3dm as R
    m = R.File3dm()
    m.Settings.ModelUnitSystem = R.UnitSystem.Millimeters
    P = Paper(r, scale)
    px = r["px"]
    M = lambda x, y: R.Point3d((x * px + r["u0"]) * 1000, (r["vtop"] - y * px) * 1000, 0)
    S = lambda x, y: R.Point3d(x * scale + r["u0"] * 1000, y * scale + P.vbot * 1000, 0)
    idx = {}

    def layer(name, color=(0, 0, 0), parent=None):
        key = (parent, name)
        if key not in idx:
            L = R.Layer(); L.Name = name; L.Color = (*color, 255)
            if parent is not None:
                L.ParentLayerId = m.Layers.FindIndex(parent).Id
            idx[key] = m.Layers.Add(L)
        return idx[key]

    def poly(pts, li, weight=None, close=False):
        pl = R.Polyline([*pts, pts[0]] if close else pts)
        a = R.ObjectAttributes(); a.LayerIndex = li
        if weight is not None:
            a.PlotWeightSource = R.ObjectPlotWeightSource.PlotWeightFromObject; a.PlotWeight = weight
        m.Objects.AddPolyline(pl, a)
    top_lines = layer("선"); top_fill = layer("면"); top_ann = layer("주석")
    lw = [0.50, 0.30, 0.20, 0.13, 0.09, 0.09]
    for pts, w, l, tn in r["lines"]:
        poly([M(x, y) for x, y in pts], layer(r["layer_names"][l], parent=top_lines), lw[w])
    for ln, it in r["layer_polys"].items():
        li = layer(ln, (150, 150, 150), parent=top_fill)
        for p in it["polys"]:
            poly([M(x, y) for x, y in list(p.exterior.coords)[:-1]], li, close=True)
    li = layer("그림자", (90, 90, 90), parent=top_fill)
    for p in r["shadow"]:
        poly([M(x, y) for x, y in list(p.exterior.coords)[:-1]], li, close=True)
    for a in annotations(r, scale, ann):
        k = a[0]
        if k == "line":
            poly([S(*q) for q in a[1]], layer(a[-1], (47, 111, 219) if a[-1] == "레벨" else (0, 0, 0), parent=top_ann), a[2]["w"])
        elif k == "poly" and a[2] not in ("earth", "earth-collage"):
            poly([S(*q) for q in a[1]], layer(a[-1], parent=top_ann), 0.13, close=True)
        elif k == "text":
            att = R.ObjectAttributes(); att.LayerIndex = layer(a[7], parent=top_ann)
            m.Objects.AddTextDot(a[3], S(a[1], a[2]), att)
        elif k in ("tag", "circle"):
            att = R.ObjectAttributes(); att.LayerIndex = layer(a[-1], parent=top_ann)
            m.Objects.AddCircle(R.Circle(S(a[1], a[2]), (2.3 if k == "tag" else a[3]) * scale), att)
            if k == "tag":
                m.Objects.AddTextDot(a[3], S(a[1], a[2]), att)
    m.Write(path, 7)
