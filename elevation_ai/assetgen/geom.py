"""기본 도형 도구. 모든 단위는 실제 크기 mm, y 는 위쪽이 + (도면 좌표)."""
import math
import random
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity


def capsule(a, b, r1, r2=None):
    """두 점을 잇는 끝이 둥근 막대(굵기가 점점 변함)."""
    r2 = r1 if r2 is None else r2
    return Point(a).buffer(r1, 24).union(Point(b).buffer(r2, 24)).convex_hull


def ellipse(c, rx, ry, rot=0):
    e = affinity.scale(Point(c).buffer(1, 48), rx, ry)
    return affinity.rotate(e, rot, origin=c) if rot else e


def rect(x0, y0, x1, y1, r=0):
    p = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    if r > 0:
        p = p.buffer(-r).buffer(r, 12)
    return p


def polar(p, length, ang_deg):
    """ang: 0=아래, +는 앞쪽(오른쪽)으로 들림."""
    a = math.radians(ang_deg)
    return (p[0] + length * math.sin(a), p[1] - length * math.cos(a))


def union(*gs):
    return unary_union([g for g in gs if g is not None and not g.is_empty])


def blob(cx, cy, rx, ry, seed=0, lobes=9, rough=0.18, n=64):
    """수관처럼 울퉁불퉁한 덩어리 — 원 여러 개를 합친 뒤 부드럽게."""
    rnd = random.Random(seed)
    parts = [ellipse((cx, cy), rx * 0.72, ry * 0.72)]
    for i in range(lobes):
        t = 2 * math.pi * i / lobes + rnd.uniform(-0.25, 0.25)
        rr = rnd.uniform(0.28, 0.42)
        px = cx + math.cos(t) * rx * (0.62 + rnd.uniform(-rough, rough) * 0.5)
        py = cy + math.sin(t) * ry * (0.62 + rnd.uniform(-rough, rough) * 0.5)
        parts.append(ellipse((px, py), rx * rr, ry * rr))
    g = unary_union(parts)
    k = min(rx, ry) * 0.04
    return g.buffer(k, 16).buffer(-k, 16)


def scallop_circle(c, r, n=14, depth=0.08, seed=0):
    """평면 나무: 가장자리가 물결진 원."""
    rnd = random.Random(seed)
    parts = []
    for i in range(n):
        t = 2 * math.pi * i / n + rnd.uniform(-0.08, 0.08)
        rr = r * (1 - depth) + rnd.uniform(-1, 1) * r * depth * 0.4
        pr = 2 * math.pi * r / n * 0.62
        parts.append(Point(c[0] + math.cos(t) * (rr - pr * 0.6),
                           c[1] + math.sin(t) * (rr - pr * 0.6)).buffer(pr, 16))
    parts.append(Point(c).buffer(r * (1 - depth) - 2 * math.pi * r / n * 0.6, 32))
    return unary_union(parts)


def mirror_x(g, x0=0.0):
    return affinity.scale(g, -1, 1, origin=(x0, 0))


def polys(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
