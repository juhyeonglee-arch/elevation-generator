"""뒤→앞 면 목록 → 숨은선 지운 선 → SVG(축척 반영 실제 인쇄 크기) / DXF(1:1 mm)."""
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union, linemerge
from shapely import affinity
from .geom import polys

# 인쇄 기준 선 굵기(mm, 종이 위)
WEIGHTS = {"body": 0.18, "far": 0.13, "prop": 0.13, "line": 0.09}


def visible_lines(layers):
    """앞 면이 가리는 뒤 외곽선을 지움. 결과: [(LineString, kind)]"""
    out = []
    cover = None
    for g, kind in reversed(layers):
        if g is None or g.is_empty:
            continue
        if kind == "line":
            seg = g if cover is None else g.difference(cover)
        else:
            g = g.buffer(0)
            seg = g.boundary if cover is None else g.boundary.difference(cover)
        for ln in _lines(seg):
            out.append((ln, kind))
        if kind != "line":
            cover = g if cover is None else unary_union([cover, g])
    return list(reversed(out)), cover


def _lines(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    try:
        g = linemerge(g) if isinstance(g, MultiLineString) else g
    except Exception:
        pass
    if isinstance(g, LineString):
        return [g]
    return [x for x in getattr(g, "geoms", []) if isinstance(x, LineString) and x.length > 0.5]


def bounds(layers, pad=0):
    xs, ys = [], []
    for g, _ in layers:
        if g is None or g.is_empty:
            continue
        b = g.bounds
        xs += [b[0], b[2]]
        ys += [b[1], b[3]]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def _d(ln, closed_hint=False):
    pts = list(ln.coords)
    s = "M" + " L".join(f"{x:.1f},{-y:.1f}" for x, y in pts)
    if ln.is_ring:
        s += "Z"
    return s


def svg_group(layers, scale=50, fill="#ffffff", style="line", ground=False):
    """다른 SVG 안에 끼울 수 있는 <g> 문자열과 경계를 돌려준다 (실제 mm 좌표, y 뒤집음)."""
    lines, cover = visible_lines(layers)
    parts = []
    if style == "solid" and cover is not None:
        for p in polys(cover):
            parts.append(f'<path d="{_poly_d(p)}" fill="#3a3a3a" stroke="none" fill-rule="evenodd"/>')
    else:
        if fill and cover is not None:
            for p in polys(cover):
                parts.append(f'<path d="{_poly_d(p)}" fill="{fill}" stroke="none" fill-rule="evenodd"/>')
        for ln, kind in lines:
            w = WEIGHTS.get(kind, 0.13) * scale
            parts.append(f'<path d="{_d(ln)}" fill="none" stroke="#000" stroke-width="{w:.1f}" '
                         f'stroke-linecap="round" stroke-linejoin="round"/>')
    return "\n".join(parts), bounds(layers)


def _poly_d(p):
    s = "M" + " L".join(f"{x:.1f},{-y:.1f}" for x, y in p.exterior.coords) + "Z"
    for i in p.interiors:
        s += "M" + " L".join(f"{x:.1f},{-y:.1f}" for x, y in i.coords) + "Z"
    return s


def write_svg(path, layers, scale=50, style="line", title=""):
    g, (x0, y0, x1, y1) = svg_group(layers, scale, style=style)
    pad = 4 * scale  # 종이 위 4mm 여백
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    w, h = x1 - x0, y1 - y0
    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{w/scale:.2f}mm" height="{h/scale:.2f}mm" '
           f'viewBox="{x0:.1f} {-y1:.1f} {w:.1f} {h:.1f}">\n<title>{title} (1:{scale})</title>\n{g}\n</svg>\n')
    open(path, "w", encoding="utf-8").write(svg)


def write_dxf(path, layers, name="ASSET"):
    import ezdxf
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    lay = {"body": ("점경_외곽", 7), "far": ("점경_뒤", 8), "prop": ("점경_소품", 8), "line": ("점경_내부선", 9)}
    for k, (n, c) in lay.items():
        if n not in doc.layers:
            doc.layers.add(n, color=c)
    lines, _ = visible_lines(layers)
    for ln, kind in lines:
        msp.add_lwpolyline(list(ln.coords), close=ln.is_ring, dxfattribs={"layer": lay.get(kind, lay["body"])[0]})
    doc.saveas(path)
