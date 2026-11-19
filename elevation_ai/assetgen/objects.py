"""가구·나무 점경. 단위 mm."""
import math
import random
from shapely.geometry import LineString, Point
from shapely import affinity
from .geom import rect, capsule, ellipse, union, blob, scallop_circle


# ---------------- 가구 ----------------
def chair_side(seat_h=450, depth=480, back_h=850, facing=1):
    t = 25
    g = union(
        rect(0, seat_h - t, depth, seat_h, r=6),                    # 좌판
        rect(depth - 30, 0, depth - 5, seat_h, r=4),                 # 앞다리
        rect(10, 0, 35, back_h, r=4),                                # 뒷다리+등받이 기둥
        rect(5, seat_h + 180, 40, back_h, r=8),                      # 등받이
    )
    g = affinity.translate(g, -depth / 2, 0)
    if facing < 0:
        g = affinity.scale(g, -1, 1, origin=(0, 0))
    return [(g, "body")]


def chair_plan(w=460, d=480):
    g = union(rect(-d / 2, -w / 2, d / 2, w / 2, r=30))
    back = rect(-d / 2 - 10, -w / 2, -d / 2 + 30, w / 2, r=12)
    return [(g, "body"), (back, "body")]


def table_side(w=1200, h=720, top=30):
    g = union(rect(-w / 2, h - top, w / 2, h, r=4),
              rect(-w / 2 + 40, 0, -w / 2 + 80, h - top),
              rect(w / 2 - 80, 0, w / 2 - 40, h - top))
    return [(g, "body")]


def table_plan(w=1200, d=750, chairs=0):
    layers = []
    if chairs:
        n_side = max(1, chairs // 2)
        for i in range(n_side):
            x = -w / 2 + w * (i + 0.5) / n_side
            for sgn in (1, -1):
                c = chair_plan()
                c = [(affinity.rotate(gg, 90 * sgn, origin=(0, 0)), k) for gg, k in c]
                c = [(affinity.translate(gg, x, sgn * (d / 2 + 120)), k) for gg, k in c]
                layers += [(gg, "far") for gg, _ in c]
    layers.append((rect(-w / 2, -d / 2, w / 2, d / 2, r=8), "body"))
    return layers


def sofa_side(w=900, h=800, seat_h=420):
    g = union(rect(0, 80, w, seat_h, r=40), rect(0, 80, 200, h, r=60),
              rect(30, 0, 70, 90), rect(w - 70, 0, w - 30, 90))
    cushion = rect(200, seat_h - 10, w - 10, seat_h + 90, r=40)
    g = affinity.translate(union(g), -w / 2, 0)
    return [(g, "body"), (affinity.translate(cushion, -w / 2, 0), "body")]


def sofa_plan(w=2000, d=900):
    base = rect(-w / 2, -d / 2, w / 2, d / 2, r=60)
    back = rect(-w / 2, d / 2 - 200, w / 2, d / 2, r=60)
    arms = union(rect(-w / 2, -d / 2, -w / 2 + 180, d / 2, r=60), rect(w / 2 - 180, -d / 2, w / 2, d / 2, r=60))
    cus = []
    n = 3 if w >= 1800 else 2
    iw = (w - 360) / n
    for i in range(n):
        x0 = -w / 2 + 180 + i * iw
        cus.append((rect(x0 + 8, -d / 2 + 20, x0 + iw - 8, d / 2 - 210, r=40), "body"))
    return [(base, "body"), (back, "body"), (arms, "body")] + cus


def bed_plan(w=1500, l=2000):
    frame = rect(-w / 2, -l / 2, w / 2, l / 2, r=20)
    pillow = [(rect(-w / 2 + 60 + i * (w - 120) / 2 + 20, l / 2 - 450, -w / 2 + 60 + (i + 1) * (w - 120) / 2 - 20, l / 2 - 110, r=60), "body")
              for i in range(2 if w >= 1300 else 1)]
    blanket = rect(-w / 2 + 10, -l / 2 + 10, w / 2 - 10, l / 2 - 650, r=30)
    fold = LineString([(-w / 2 + 10, l / 2 - 800), (w / 2 - 10, l / 2 - 800)])
    return [(frame, "body")] + pillow + [(blanket, "body"), (fold, "line")]


# ---------------- 나무 ----------------
def tree_side(height=8000, crown_w=None, kind="broad", seed=1):
    """단면·입면 나무. kind: broad(활엽, 둥근 수관), conifer(침엽, 원뿔), columnar(좁고 높은), spread(옆으로 넓은)"""
    rnd = random.Random(seed)
    H = float(height)
    W = crown_w or {"broad": 0.65, "conifer": 0.4, "columnar": 0.3, "spread": 1.0}[kind] * H
    trunk_r = max(80, H * 0.018)
    layers = []
    if kind == "conifer":
        base_y = H * 0.18
        pts = []
        tiers = 7
        for i in range(tiers + 1):
            y = base_y + (H - base_y) * i / tiers
            hw = W / 2 * (1 - i / tiers) ** 0.95
            pts.append((y, hw))
        poly = [(0, H)]
        for i in range(tiers, 0, -1):
            y, hw = pts[i - 1]
            yt = pts[i][0]
            poly += [(hw, y), (hw * 0.55, y + (yt - y) * 0.15)]
        right = poly[:]
        left = [(-x, y) for x, y in reversed(right[1:])]
        from shapely.geometry import Polygon
        crown = Polygon(right + left).buffer(H * 0.006).buffer(-H * 0.004)
        trunk = rect(-trunk_r, 0, trunk_r, base_y + H * 0.1)
        layers += [(trunk, "body"), (crown, "body")]
        return layers
    if kind == "columnar":
        cy0, cy1 = H * 0.22, H
    elif kind == "spread":
        cy0, cy1 = H * 0.38, H
    else:
        cy0, cy1 = H * 0.32, H
    cy = (cy0 + cy1) / 2
    ry = (cy1 - cy0) / 2
    crown = blob(0, cy, W / 2 * 0.92, ry * 0.92, seed=seed, lobes=11)
    # 줄기 + 큰 가지
    trunk = capsule((0, 0), (0, cy0 + ry * 0.2), trunk_r * 1.15, trunk_r * 0.8)
    layers.append((trunk, "body"))
    layers.append((crown, "body"))
    # 수관 안쪽 가지 표현(얇은 선) — 수관 밖으로 안 나가게 잘라 냄
    branches = []
    fork = (0, cy0 + ry * 0.15)
    for i in range(5):
        a = -55 + i * 27 + rnd.uniform(-8, 8)
        L = ry * rnd.uniform(0.7, 1.05)
        end = (fork[0] + L * math.sin(math.radians(a)) * (W / (2 * ry)) * 0.8, fork[1] + L * math.cos(math.radians(a)))
        mid = ((fork[0] + end[0]) / 2 + rnd.uniform(-1, 1) * W * 0.03, (fork[1] + end[1]) / 2)
        branches.append(LineString([fork, mid, end]))
    inner = crown.buffer(-W * 0.06)
    for br in branches:
        layers.append((br.intersection(inner), "line"))
    return layers


def tree_plan(crown_d=5000, seed=1, kind="broad"):
    r = crown_d / 2
    if kind == "conifer":
        g = scallop_circle((0, 0), r, n=22, depth=0.12, seed=seed)
    else:
        g = scallop_circle((0, 0), r, n=13, depth=0.1, seed=seed)
    layers = [(g, "body"), (Point(0, 0).buffer(max(60, crown_d * 0.02), 24), "body")]
    rnd = random.Random(seed)
    for i in range(6):
        a = math.radians(i * 60 + rnd.uniform(-15, 15))
        L = r * rnd.uniform(0.45, 0.7)
        layers.append((LineString([(math.cos(a) * crown_d * 0.03, math.sin(a) * crown_d * 0.03), (math.cos(a) * L, math.sin(a) * L)]), "line"))
    return layers
