"""콜라주 입면용 점경: 기본 나무 그림(절차 생성)과 사람 실루엣, 사용자 PNG 폴더 읽기."""
import glob
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
BUILTIN = os.path.join(HERE, "textures", "점경")

# 색: 바랜 초록·회녹색(유럽식 콜라주 입면 톤), 봄꽃 변형
PALETTES = {
    "broad": [(118, 132, 92), (138, 150, 108), (100, 114, 80), (160, 168, 128), (86, 100, 72)],
    "grey": [(132, 140, 118), (154, 160, 140), (114, 122, 102), (176, 180, 162)],
    "blossom": [(222, 176, 182), (232, 196, 198), (206, 156, 164), (160, 168, 136), (240, 214, 214)],
}


def make_tree(kind="broad", seed=0, W=700, H=900, palette="broad"):
    """반투명한 수채 느낌 나무 한 그루(RGBA). kind: broad(넓은 활엽), tall(좁고 높은), round(둥근).
    수관 = 크기가 다른 잎 뭉치 여러 개 × 작은 붓 자국 수천 개, 사이사이 비치는 틈, 가지가 비쳐 보임."""
    rng = np.random.default_rng(seed)
    cx = W / 2
    base = H - 4
    trunk_h = H * (0.30 if kind != "tall" else 0.2)
    if kind == "tall":
        rx, ry = W * 0.20, H * 0.40
    elif kind == "round":
        rx, ry = W * 0.36, H * 0.30
    else:
        rx, ry = W * 0.44, H * 0.33
    cy = base - trunk_h - ry * 0.95
    # 가지(수관 안까지 뻗음)
    br = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(br)
    tw = W * 0.026

    def inside(x, y, k=0.92):
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= k * k

    def branch(x, y, ang, L, w, depth):
        x2, y2 = x + math.sin(ang) * L, y - math.cos(ang) * L
        for _ in range(6):  # 수관 밖으로 삐져나가지 않게 줄인다
            if depth == 0 or inside(x2, y2):
                break
            L *= 0.75; x2, y2 = x + math.sin(ang) * L, y - math.cos(ang) * L
        dr.line([(x, y), (x2, y2)], fill=(84, 78, 70, 235), width=max(1, int(w)))
        if depth < 4:
            for da in (rng.uniform(-0.75, -0.25), rng.uniform(0.25, 0.75)):
                branch(x2, y2, ang + da, L * rng.uniform(0.6, 0.78), w * 0.62, depth + 1)
    def trunk_then(x, y):
        for a0 in (rng.uniform(-0.6, -0.2), rng.uniform(-0.1, 0.1), rng.uniform(0.2, 0.6)):
            branch(x, y, a0, ry * rng.uniform(0.45, 0.6), tw * 0.65, 1)
    dr.line([(cx, base), (cx, base - trunk_h)], fill=(84, 78, 70, 235), width=int(tw))
    trunk_then(cx, base - trunk_h)
    # 잎 뭉치 중심들
    clumps = [(cx, cy, 0.75)]
    for k in range(int(rng.integers(8, 13))):
        a = 2 * math.pi * k / 10 + rng.uniform(-0.3, 0.3); rr = rng.uniform(0.35, 0.62)
        clumps.append((cx + math.cos(a) * rx * rr, cy + math.sin(a) * ry * rr, rng.uniform(0.38, 0.55)))
    pal = PALETTES[palette]
    can = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dc = ImageDraw.Draw(can)
    for (qx, qy, qs) in clumps:
        n = int(900 * qs)
        crx, cry = rx * qs, ry * qs * 0.85
        shade = rng.uniform(-14, 10)
        for _ in range(n):
            a = rng.uniform(0, 2 * math.pi); rr = rng.uniform(0, 1) ** 0.7
            x = qx + math.cos(a) * crx * rr; y = qy + math.sin(a) * cry * rr
            s = W * rng.uniform(0.006, 0.016)
            c = pal[rng.integers(len(pal))]
            lift = 26 * (qy - y) / cry - 10 * rr + shade  # 뭉치 위쪽이 밝게, 가장자리 조금 어둡게
            col = tuple(int(np.clip(v + lift, 0, 255)) for v in c)
            ang = rng.uniform(0, math.pi)
            pts = [(x + math.cos(ang + t) * s * (1.0 if k % 2 == 0 else 0.55), y + math.sin(ang + t) * s * (1.0 if k % 2 == 0 else 0.55))
                   for k, t in enumerate(np.linspace(0, 2 * math.pi, 7)[:-1])]
            dc.polygon(pts, fill=(*col, int(rng.uniform(90, 190))))
    can = can.filter(ImageFilter.GaussianBlur(W * 0.0015))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    img = Image.alpha_composite(img, br)
    img = Image.alpha_composite(img, can)
    over = br.copy(); oa = np.asarray(over).copy(); oa[..., 3] = (oa[..., 3] * 0.35).astype(np.uint8)
    img = Image.alpha_composite(img, Image.fromarray(oa))  # 잎 사이로 가지가 비침
    a = np.asarray(img).copy()
    a[..., 3] = (a[..., 3].astype(np.float32) * 0.9).astype(np.uint8)
    ys, xs = np.nonzero(a[..., 3] > 8)
    return Image.fromarray(a).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def ensure_builtin():
    os.makedirs(BUILTIN, exist_ok=True)
    specs = [("broad", 1, "broad"), ("broad", 2, "grey"), ("round", 3, "broad"), ("tall", 4, "broad"),
             ("broad", 5, "blossom"), ("round", 6, "grey")]
    files = []
    for k, sd, pal in specs:
        p = os.path.join(BUILTIN, f"나무_{k}_{pal}_{sd}.png")
        if not os.path.exists(p):
            make_tree(k, sd, palette=pal).save(p)
        files.append(p)
    return files


def tree_images(user_dir=None):
    """사용자 폴더(나무 PNG)가 있으면 그걸, 없으면 기본 그림."""
    files = []
    if user_dir and os.path.isdir(user_dir):
        files = sorted(glob.glob(os.path.join(user_dir, "*.png")))
    if not files:
        files = ensure_builtin()
    return [Image.open(f).convert("RGBA") for f in files]


def person_silhouette(height_m, px, seed=0, color=(128, 128, 128), alpha=0.72):
    """vector-assets의 사람 형상을 회색 실루엣 RGBA로. 돌려주는 그림은 발이 아래 끝."""
    from shapely.ops import unary_union
    from .assetgen import person
    rng = np.random.default_rng(seed)
    pose = ["stand", "walk", "walk", "phone", "stand", "walk"][seed % 6]
    sex = "f" if rng.uniform() < 0.5 else "m"
    hmm = height_m * 1000
    if rng.uniform() < 0.35:
        layers, _ = person.front(hmm, sex, bag=rng.uniform() < 0.3)
    else:
        layers, _ = person.side(hmm, sex, pose, bag=rng.uniform() < 0.3)
    g = unary_union([x[0] if isinstance(x, tuple) else x for x in layers])
    x0, y0, x1, y1 = g.bounds
    k = 1.0 / (px * 1000)  # mm → 픽셀
    W = max(2, int((x1 - x0) * k) + 2); H = max(2, int((y1 - y0) * k) + 2)
    im = Image.new("L", (W * 4, H * 4), 0); d = ImageDraw.Draw(im)
    for p in (g.geoms if hasattr(g, "geoms") else [g]):
        if p.geom_type != "Polygon":
            continue
        d.polygon([((x - x0) * k * 4, (y1 - y) * k * 4) for x, y in p.exterior.coords], fill=255)
    m = np.asarray(im.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255
    rgba = np.zeros((H, W, 4), np.uint8); rgba[..., :3] = color; rgba[..., 3] = (m * alpha * 255).astype(np.uint8)
    if rng.uniform() < 0.5:
        rgba = rgba[:, ::-1]
    return rgba
