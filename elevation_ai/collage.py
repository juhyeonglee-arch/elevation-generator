"""콜라주 입면: 면마다 살아 있는 텍스처, 깊이에 따른 흰 막, 부드러운 그림자, 나무·사람 점경.

엔진(Elevation.run(keep_buffers=True))이 남긴 버퍼에서 그림 한 장을 만든다.
벡터 선·주석은 draw.py가 그 위에 그대로 그린다(선 색만 연한 회색 계열)."""
import math

import numpy as np
from PIL import Image
from . import imgops

from . import entourage

# 재료 바탕색(텍스처는 이 색 위에 결만 얹는다) — 바랜 유럽식 콜라주 톤
BASE = {"유리": (204, 212, 218), "금속": (214, 214, 212), "콘크리트": (226, 223, 217), "콘크리트블록": (221, 217, 210),
        "벽돌": (174, 124, 106), "목재": (208, 188, 156), "폴리카보네이트": (232, 234, 236), "기타": (234, 231, 225)}
# 텍스처 결의 세기(0 = 바탕색만, 1 = 텍스처 명암 그대로)
GRAIN = {"유리": 0.22, "금속": 0.45, "콘크리트": 0.55, "콘크리트블록": 0.55, "벽돌": 0.8, "목재": 0.7,
         "폴리카보네이트": 0.3, "기타": 0.4}
SHADE = np.array([0.84, 0.83, 0.86])     # 그림자 곱하기 색(살짝 푸른 회색)
TERRAIN_RGB = (238, 235, 228)            # 단면 뒤로 보이는 땅
LINE_TONES = ["#4a4a4a", "#7d7d7d", "#a5a5a5", "#c4c4c4"]
LINE_WEIGHTS = [0.25, 0.13, 0.09, 0.07, 0.05, 0.05]
from .elevation import is_tree as _is_tree

DEFAULTS = dict(veil_start=6.0, veil_end=45.0, veil_max=0.7, jitter=0.035, shadow_soft=0.12,
                trees="auto", people=True, people_every=9.0, seed=7)


def is_tree_layer(name, cfg=None):
    """레이어 표에서 재료를 '수목'으로 고른 레이어(설정이 없으면 이름으로 짐작)."""
    if cfg is not None and name in cfg:
        return cfg[name].get("material") == "수목"
    return _is_tree(name)


def line_style(w, tone, scale):
    mul = 1.0 if scale <= 100 else 0.8
    return LINE_WEIGHTS[min(w, 5)] * mul, LINE_TONES[min(tone, 3)] if w != 5 else LINE_TONES[3]


# ── 텍스처 ─────────────────────────────────────────────────────────
def _grain(img, w_m, h_m, px):
    """텍스처 그림 → (명암 결 배열, 한 장 크기 픽셀). 결 = 밝기 / 평균 밝기."""
    tw = max(2, int(round(w_m / px))); th = max(2, int(round(h_m / px)))
    a = np.asarray(img.convert("RGB").resize((tw, th), Image.LANCZOS)).astype(np.float32)
    lum = a.mean(2)
    g = a / max(float(lum.mean()), 1.0)
    return g, tw, th


def _tile(g, tw, th, uu, vv):
    ix = np.floor((uu * 1.0) % tw).astype(np.int64) % tw
    iy = th - 1 - np.floor(vv % th).astype(np.int64) % th  # 위로 갈수록 v가 커짐, 그림은 아래 줄부터
    return g[iy, ix]


def compose(r, scale, mat_tex=None, opts=None, parts=None, north=0.0, user_trees=None, log=print):
    """→ dict(base=RGBA uint8 (건물·땅 그림), sprites=RGBA uint8 (나무·사람), layer_masks={레이어: bool 배열})."""
    o = dict(DEFAULTS); o.update(opts or {})
    B = r["buf"]
    zb, part, lay = B["zb"], B["part"], B["layer"]
    H, W = zb.shape
    px, u0, vtop = r["px"], r["u0"], r["vtop"]
    cats = B["cat"]; terr = B["terrain"]
    valid = part >= 0
    rng = np.random.default_rng(o["seed"])
    nparts = len(cats)
    ys, xs = np.nonzero(valid)
    pp = part[ys, xs]
    uu = u0 + (xs + 0.5) * px
    vv = vtop - (ys + 0.5) * px
    cat_of_part = np.array([c if c in BASE else "기타" for c in cats])
    is_t = terr[pp]
    rgb = np.zeros((len(pp), 3), np.float32)
    # 부재마다 색을 아주 조금 다르게(같은 재료라도 판·동마다 다른 느낌)
    jit = 1.0 + rng.uniform(-o["jitter"], o["jitter"], (nparts, 1)) + rng.uniform(-o["jitter"] / 3, o["jitter"] / 3, (nparts, 3))
    for c in BASE:
        sel = (cat_of_part[pp] == c) & ~is_t
        if not sel.any():
            continue
        col = np.array(BASE[c], np.float32)
        t = (mat_tex or {}).get(c)
        if t is not None:
            g, tw, th = _grain(t["image"], t["w"], t["h"], px)
            k = GRAIN.get(c, 0.5) * float(t.get("opacity", 1.0))
            # 텍스처 시작점 = 부재의 왼쪽 아래 모서리(면마다 무늬가 이어 붙은 듯 어긋나지 않게)
            au = B["anchor_u"][pp[sel]]; av = B["anchor_v"][pp[sel]]
            gg = _tile(g, tw, th, (uu[sel] - au) / px, (vv[sel] - av) / px)
            rgb[sel] = col * (1.0 + k * (gg - 1.0))
        else:
            rgb[sel] = col
        if c == "유리":  # 유리: 위가 조금 밝은 하늘 반사
            h = (vv[sel] - B["anchor_v"][pp[sel]]) % 3.5 / 3.5
            rgb[sel] *= (0.94 + 0.08 * h)[:, None]
        rgb[sel] *= jit[pp[sel]]
    rgb[is_t] = TERRAIN_RGB
    img = np.full((H, W, 3), 255.0, np.float32)
    img[ys, xs] = rgb
    # 부드러운 그림자(곱하기)
    sh = B["shadow"].astype(np.float32)
    s = max(0.5, o["shadow_soft"] / px)
    sh = imgops.gaussian_filter(sh, s) * valid
    img = img * (1.0 - sh[..., None] * (1.0 - SHADE))
    # 깊이 흐림: 건물 맨 앞에서 멀수록 흰 막
    dz = np.where(valid, zb - B["bld_front"], 0.0)
    k = np.clip((dz - o["veil_start"]) / max(o["veil_end"] - o["veil_start"], 1e-6), 0, 1) * o["veil_max"]
    img = img * (1 - k[..., None]) + 255.0 * k[..., None]
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    base = np.dstack([np.clip(img, 0, 255).astype(np.uint8), alpha])
    sprites = np.zeros((H, W, 4), np.float32)
    n_tree = n_people = 0
    if o.get("trees") != "off":
        n_tree = _place_trees(sprites, r, B, o, parts, north, user_trees, rng)
    if o.get("people"):
        n_people = _place_people(sprites, r, B, o, rng)
    log(f"{r['name']}: 콜라주 — 나무 {n_tree}그루, 사람 {n_people}명")
    masks = {}
    for li, ln in enumerate(r["layer_names"]):
        m = lay == li
        if m.any():
            masks[ln] = m
    return dict(base=base, sprites=np.clip(sprites, 0, 255).astype(np.uint8), layer_masks=masks)


# ── 점경 ──────────────────────────────────────────────────────────
def _ground_at(r, u):
    """u 위치의 지면 높이(지형 단면이 있으면 그 선, 없으면 GL)."""
    g = r.get("ground") or []
    best = None
    for a, b, c, d in g:
        lo, hi = min(a, c), max(a, c)
        if lo <= u <= hi and hi - lo > 1e-6:
            z = b + (d - b) * (u - a) / (c - a)
            best = z if best is None else max(best, z)
    return r["gl"] if best is None else best


def _paste(dst, spr, x0, y_bottom, depth, zb, veil):
    """RGBA 그림을 (x0, 아래 끝 y) 에 붙인다. 그 자리 건물이 더 가까우면 가려진다. veil = 흰 막 세기."""
    h, w = spr.shape[:2]
    H, W = dst.shape[:2]
    y0 = int(round(y_bottom)) - h; x0 = int(round(x0))
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    s = spr[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    a = s[..., 3:4] / 255.0
    if depth is not None:
        vis = (zb[ya:yb, xa:xb] > depth)[..., None]
        a = a * vis
    col = s[..., :3] * (1 - veil) + 255.0 * veil
    a = a * (1 - 0.5 * veil)
    d = dst[ya:yb, xa:xb]
    da = d[..., 3:4] / 255.0
    out_a = a + da * (1 - a)
    d[..., :3] = np.where(out_a > 0, (col * a + d[..., :3] * da * (1 - a)) / np.maximum(out_a, 1e-6), 0)
    d[..., 3:4] = out_a * 255


def tree_instances(parts, north, right, d):
    """모델의 나무 레이어 → [(u, 아래 z, 높이, 폭, 깊이)]. 붙어 있는 조각(줄기·수관)은 한 그루로.
    parts는 jobs.make에서 이미 수목 레이어만 골라 넘긴다."""
    pts = []
    for V, F, name, mat in parts or []:
        V = np.asarray(V, float)
        if north:
            t = math.radians(north); c, s_ = math.cos(t), math.sin(t)
            V = np.c_[V[:, 0] * c - V[:, 1] * s_, V[:, 0] * s_ + V[:, 1] * c, V[:, 2]]
        pts.append((V[:, :2].mean(0), V[:, 2].min(), V[:, 2].max(), V))
    trees = []
    used = [False] * len(pts)
    for i, (xy, z0, z1, V) in enumerate(pts):
        if used[i]:
            continue
        group = [i]; used[i] = True
        for j in range(i + 1, len(pts)):
            if not used[j] and np.hypot(*(pts[j][0] - xy)) < 1.5:
                group.append(j); used[j] = True
        VV = np.vstack([pts[k][3] for k in group])
        uu = VV @ np.asarray(right); dd = VV @ np.asarray(d)
        trees.append((float((uu.min() + uu.max()) / 2), float(VV[:, 2].min()), float(VV[:, 2].max() - VV[:, 2].min()),
                      float(uu.max() - uu.min()), float(dd.mean())))
    return trees


def _place_trees(dst, r, B, o, parts, north, user_trees, rng):
    imgs = entourage.tree_images(user_trees)
    px, u0, vtop = r["px"], r["u0"], r["vtop"]
    zb = B["zb"]
    right = np.asarray(r["right"]); d = np.asarray(B["d"])
    trees = tree_instances(parts, north, right, d)
    if not trees and o.get("trees") == "auto":
        # 모델에 나무가 없으면: 건물 양 끝 바깥과 앞쪽 몇 곳에 알맞게
        span = r["umax"] - r["umin"]
        us = [r["umin"] - rng.uniform(2, 5), r["umax"] + rng.uniform(2, 5)]
        n = int(span // 14)
        for i in range(n):
            us.append(r["umin"] + span * (i + 0.5 + rng.uniform(-0.25, 0.25)) / max(n, 1))
        for i, u in enumerate(us):
            h = rng.uniform(7.0, 11.0)
            half = h * 0.4  # 그림 반쪽 폭쯤: 도면 밖으로 잘리지 않게
            u = float(np.clip(u, r["umin"] - 0.8 + half, r["umax"] + 0.8 - half))
            front = 2 <= i  # 건물 앞(가운데) 나무는 앞쪽, 양 끝은 건물 옆
            dep = B["bld_front"] - 3.0 if front else B["bld_front"] + rng.uniform(0, 6)
            trees.append((u, _ground_at(r, u), h, h * rng.uniform(0.6, 0.8), dep))
    trees.sort(key=lambda t: -t[4])  # 먼 것부터
    for i, (u, z0, h, w, dep) in enumerate(trees):
        if h < 1.0:
            continue
        # 모양이 비슷한 그림(가는 나무엔 가는 그림) 중에서 하나
        want = w / h if w > 0 else 0.7
        order = sorted(range(len(imgs)), key=lambda j: abs(math.log(imgs[j].width / imgs[j].height / max(want, 0.2))))
        im = imgs[order[int(rng.integers(min(3, len(order))))]]
        hp = max(4, int(h / px)); wp = max(4, int(hp * im.width / im.height * (want / (im.width / im.height)) ** 0.5))
        spr = np.asarray(im.resize((wp, hp), Image.LANCZOS))
        if rng.uniform() < 0.5:
            spr = spr[:, ::-1]
        dz = dep - B["bld_front"]
        veil = float(np.clip((dz - o["veil_start"]) / max(o["veil_end"] - o["veil_start"], 1e-6), 0, 1) * o["veil_max"])
        _paste(dst, spr, (u - u0) / px - wp / 2, (vtop - z0) / px, dep, zb, veil)
    return len(trees)


def _place_people(dst, r, B, o, rng):
    px, u0, vtop = r["px"], r["u0"], r["vtop"]
    u = r["umin"] + rng.uniform(0.5, o["people_every"])
    n = 0
    while u < r["umax"] - 0.5:
        h = rng.uniform(1.6, 1.8)
        spr = entourage.person_silhouette(h, px, seed=int(rng.integers(1e6)))
        z0 = _ground_at(r, u)
        _paste(dst, spr, (u - u0) / px - spr.shape[1] / 2, (vtop - z0) / px, None, B["zb"], 0.0)
        n += 1
        # 두세 명씩 모이기도 하게
        u += rng.uniform(0.7, 1.4) if rng.uniform() < 0.3 else rng.uniform(0.5, 1.6) * o["people_every"]
    return n


# ── 그리기 ────────────────────────────────────────────────────────
def draw_collage(ax, r, scale, ox, oy, ann, comp):
    """matplotlib: 땅 띠 → 건물 그림 → 선 → 점경 → 주석."""
    from matplotlib.collections import LineCollection
    from . import draw
    P = draw.Paper(r, scale)
    pt = 72 / 25.4
    ext = (ox, ox + r["W"] * P.k, oy, oy + r["H"] * P.k)
    ax.imshow(comp["base"], extent=ext, zorder=1, interpolation="bilinear")
    groups = {}
    for pts, w, l, tn in r["lines"]:
        groups.setdefault((w, tn), []).append(np.c_[pts[:, 0] * P.k + ox, P.H - pts[:, 1] * P.k + oy])
    for (w, tn), ls in sorted(groups.items(), key=lambda kv: -kv[0][0]):
        lw, col = line_style(w, tn, scale)
        ax.add_collection(LineCollection(ls, colors=col, linewidths=lw * pt, capstyle="round", joinstyle="round",
                                         zorder=3 + (5 - w) * 0.1))
    if comp["sprites"][..., 3].any():
        ax.imshow(comp["sprites"], extent=ext, zorder=4, interpolation="bilinear")
    for a in draw.annotations(r, scale, ann):
        draw._mpl_prim(ax, a, ox, oy, pt)


def _jpg_b64(rgb):
    import base64, io
    b = io.BytesIO(); Image.fromarray(np.ascontiguousarray(rgb[..., :3]), "RGB").save(b, "JPEG", quality=88)
    return base64.b64encode(b.getvalue()).decode()


def svg_layers(r, scale, comp, X, Y, pathd_px):
    """SVG 조각: 레이어마다 벡터 경계(clipPath)로 자른 텍스처 그림 + 선 + 점경 그림.
    X, Y = 종이 mm → SVG 좌표, pathd_px(shapely 다각형, 픽셀) → path d."""
    from . import draw
    P = draw.Paper(r, scale)
    o = []
    defs = []
    base = comp["base"]
    o.append('<g id="콜라주_재료">')
    LP = r["layer_polys"]
    # 지형 등 다각형이 없는 픽셀은 따로 한 장
    covered = np.zeros(base.shape[:2], bool)
    for i, (ln, it) in enumerate(LP.items()):
        m = comp["layer_masks"].get(ln)
        if m is None or not it["polys"]:
            continue
        covered |= m
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - 2), min(base.shape[0], ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(base.shape[1], xs.max() + 3)
        # 모양은 벡터 clipPath가 정하므로 그림은 투명 없이 JPEG(파일이 훨씬 작음)
        crop = base[y0:y1, x0:x1].copy()
        bg = crop[..., 3] == 0
        crop[bg, :3] = 255
        cid = f"clip_{i}"
        defs.append(f'<clipPath id="{cid}">' + "".join(f'<path d="{pathd_px(p)}" fill-rule="evenodd"/>' for p in it["polys"]) + "</clipPath>")
        xa, ya = P.px(x0, y0)
        o.append(f'<g id="{draw._sid(ln)}" clip-path="url(#{cid})"><image x="{X(xa):.3f}" y="{Y(ya):.3f}" '
                 f'width="{(x1 - x0) * P.k:.3f}" height="{(y1 - y0) * P.k:.3f}" preserveAspectRatio="none" '
                 f'href="data:image/jpeg;base64,{_jpg_b64(crop)}"/></g>')
    rest = (base[..., 3] > 0) & ~covered
    if rest.sum() > 50:
        ys, xs = np.nonzero(rest)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = base[y0:y1, x0:x1].copy(); crop[..., 3] = np.where(rest[y0:y1, x0:x1], 255, 0)
        xa, ya = P.px(x0, y0)
        o.append(f'<g id="뒤_땅"><image x="{X(xa):.3f}" y="{Y(ya):.3f}" width="{(x1 - x0) * P.k:.3f}" height="{(y1 - y0) * P.k:.3f}" '
                 f'preserveAspectRatio="none" href="data:image/png;base64,{draw._png_b64(crop)}"/></g>')
    o.append("</g>")
    return defs, o


def svg_sprites(r, scale, comp, X, Y):
    from . import draw
    P = draw.Paper(r, scale)
    s = comp["sprites"]
    if not s[..., 3].any():
        return ""
    ys, xs = np.nonzero(s[..., 3])
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    xa, ya = P.px(x0, y0)
    return (f'<g id="점경"><image x="{X(xa):.3f}" y="{Y(ya):.3f}" width="{(x1 - x0) * P.k:.3f}" height="{(y1 - y0) * P.k:.3f}" '
            f'preserveAspectRatio="none" href="data:image/png;base64,{draw._png_b64(s[y0:y1, x0:x1])}"/></g>')
