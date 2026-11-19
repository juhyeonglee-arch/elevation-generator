"""3D 메시 → 입면 벡터(숨은선 제거, 깊이별 선 굵기, 그림자, 재료 영역, 레벨)."""
import math
import numpy as np
from . import imgops
import contourpy
from shapely.geometry import Polygon
from .raster import rasterize

# ── 재료 분류: 레이어/재료 이름 키워드 → 분류 ─────────────────────────────
# 수목(나무 심은 것)과 목재(나무 마감)를 가른다: '나무'는 둘 다 뜻해서, 마감 낱말이 붙으면 목재.
TREE_WORDS = ("나무", "수목", "식재", "조경수", "조경", "tree", "plant", "vegetation", "veg", "shrub", "관목")
WOOD_FINISH = ("루버", "louver", "판", "마감", "데크", "deck", "사이딩", "siding", "패널", "panel", "널", "목재",
               "wood", "timber", "clt", "합판", "집성", "문", "door", "창호", "프레임", "frame", "난간", "벽", "wall")


def is_tree(layer, mat=""):
    s = f"{layer} {mat}".lower()
    if not any(w in s for w in TREE_WORDS):
        return False
    return not any(w in s for w in WOOD_FINISH)


MATERIAL_RULES = [
    ("금속", ["frame", "mullion", "프레임"]),
    ("유리", ["glass", "유리", "window", "glazing", "창"]),
    ("폴리카보네이트", ["polycarbonate", "poly", "ploy", "폴리"]),
    ("목재", ["clt", "wood", "timber", "목재", "나무"]),
    ("벽돌", ["brick", "벽돌", "조적"]),
    ("콘크리트블록", ["pc block", "block", "블록"]),
    ("콘크리트", ["concrete", "콘크리트", "basement", "rc", "slab", "wall", "structure"]),
    ("금속", ["metal", "steel", "beam", "column", "frame", "bolt", "paint", "금속", "철", "panel"]),
]


def classify(layer, mat):
    if is_tree(layer, mat):
        return "수목"
    s = f"{layer} {mat}".lower()
    for cat, keys in MATERIAL_RULES:
        if any(k in s for k in keys):
            return cat
    return "기타"


VIEWS = {  # 바라보는 방향(보는 사람이 서 있는 쪽의 반대)
    "남측입면": (0.0, 1.0),
    "북측입면": (0.0, -1.0),
    "동측입면": (-1.0, 0.0),
    "서측입면": (1.0, 0.0),
}


def merge_parts(parts):
    V, F, P = [], [], []
    off = 0
    for i, (v, f, *_rest) in enumerate(parts):
        V.append(v); F.append(f + off); P.append(np.full(len(f), i, np.int32)); off += len(v)
    return np.vstack(V), np.vstack(F), np.concatenate(P)


def tri_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    l = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(l, 1e-12), l[:, 0] * 0.5


def feature_edges(V, F, P, N, view_dir, crease_deg=25.0):
    """경계선·꺾인선·윤곽선(실루엣) 모서리 → (E,2) 정점 번호."""
    T = len(F)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    tid = np.tile(np.arange(T), 3)
    e.sort(axis=1)
    # 같은 위치 정점끼리 묶기(메시가 용접 안 된 경우) → 위치 기준 키
    q = np.round(V / 0.002).astype(np.int64)
    _, vkey = np.unique(q, axis=0, return_inverse=True)
    vkey = vkey.ravel()
    ek = np.sort(vkey[e], axis=1)
    key = ek[:, 0] * (vkey.max() + 1) + ek[:, 1]
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    start = np.r_[0, np.nonzero(np.diff(key_s))[0] + 1]
    cnt = np.diff(np.r_[start, len(key_s)])
    first = order[start]
    keep = np.zeros(len(start), bool)
    # 1) 한 면만 붙은 모서리 = 면 경계
    keep |= cnt == 1
    # 2) 두 면: 꺾인 각 / 다른 부재 / 실루엣
    two = cnt == 2
    i2 = np.nonzero(two)[0]
    ta = tid[order[start[i2]]]; tb = tid[order[start[i2] + 1]]
    na, nb = N[ta], N[tb]
    cosang = np.einsum("ij,ij->i", na, nb)
    crease = cosang < math.cos(math.radians(crease_deg))
    other_part = P[ta] != P[tb]
    fa = na @ view_dir; fb = nb @ view_dir
    sil = (fa * fb) < 0
    keep[i2] = crease | other_part | sil
    keep |= cnt > 2
    E = e[first[keep]]
    return E, tid[first[keep]]


class EdgeTopo:
    """모서리 이웃 관계를 한 번만 계산해 두고, 면마다 실루엣만 다시 본다."""

    def __init__(self, V, F, P, N, crease_deg=25.0):
        T = len(F)
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        tid = np.tile(np.arange(T), 3)
        q = np.round(V / 0.002).astype(np.int64)
        _, vkey = np.unique(q, axis=0, return_inverse=True)
        vkey = vkey.ravel()
        ek = np.sort(vkey[e], axis=1)
        key = ek[:, 0] * (vkey.max() + 1) + ek[:, 1]
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        start = np.r_[0, np.nonzero(np.diff(key_s))[0] + 1]
        cnt = np.diff(np.r_[start, len(key_s)])
        first = order[start]
        self.E = e[first]
        self.ta = tid[first]
        self.tb = np.where(cnt >= 2, tid[order[np.minimum(start + 1, len(order) - 1)]], -1)
        two = cnt == 2
        na, nb = N[self.ta], N[np.maximum(self.tb, 0)]
        crease = np.einsum("ij,ij->i", na, nb) < math.cos(math.radians(crease_deg))
        other = P[self.ta] != P[np.maximum(self.tb, 0)]
        self.static = (cnt == 1) | (cnt > 2) | (two & (crease | other))
        self.two = two
        self.N = N

    def edges(self, view_dir, keep_tri):
        """keep_tri(원래 삼각형 마스크) → (E, 붙은 삼각형 번호)."""
        ka = keep_tri[self.ta]
        kb = np.where(self.tb >= 0, keep_tri[np.maximum(self.tb, 0)], False)
        fa = self.N[self.ta] @ view_dir; fb = self.N[np.maximum(self.tb, 0)] @ view_dir
        sil = self.two & ((fa * fb) < 0)
        both = ka & kb
        sel = (both & (self.static | sil)) | (ka ^ kb)
        t = np.where(ka, self.ta, np.maximum(self.tb, 0))
        return self.E[sel], t[sel]


class Cancelled(Exception):
    pass


GLASS_CATS = ("유리",)
TONES = ["#111111", "#3a3a3a", "#6b6b6b", "#9a9a9a"]  # 깊이별 선 색(앞 → 뒤)


def view_dir_from_bearing(b):
    """입면이 향하는 방위(북=0°, 시계방향) → 보는 방향 단위벡터(진북 좌표)."""
    t = math.radians(b)
    return (-math.sin(t), -math.cos(t))


BEARINGS = {"북측입면": 0.0, "동측입면": 90.0, "남측입면": 180.0, "서측입면": 270.0}


class Elevation:
    def __init__(self, parts, px=0.008, crease_deg=25.0, layer_cfg=None, north_deg=0.0):
        """layer_cfg = {레이어: {include, material, terrain}}. north_deg = 진북이 모델 +Y에서 시계방향 몇 도."""
        cfg = layer_cfg or {}
        parts = [p for p in parts if cfg.get(p[2], {}).get("include", True)]
        if not parts:
            raise ValueError("포함된 레이어가 없어요")
        self.parts = parts
        self.V, self.F, self.P = merge_parts(parts)
        if north_deg:
            t = math.radians(north_deg); c, s_ = math.cos(t), math.sin(t)
            x, y = self.V[:, 0].copy(), self.V[:, 1].copy()
            self.V[:, 0] = x * c - y * s_; self.V[:, 1] = x * s_ + y * c
        self.N, self.A = tri_normals(self.V, self.F)
        self.px = px
        self.crease = crease_deg
        self.cat = [cfg.get(p[2], {}).get("material") or classify(p[2], p[3]) for p in parts]
        self.terrain_part = np.array([bool(cfg.get(p[2], {}).get("terrain")) for p in parts])
        self.glass_part = np.array([c in GLASS_CATS for c in self.cat])
        self.lnames = sorted(set(p[2] for p in parts))
        lidx = {n: i for i, n in enumerate(self.lnames)}
        self.part_layer = np.array([lidx[p[2]] for p in parts], np.int32)
        self.zmin_all = float(self.V[:, 2].min()); self.zmax_all = float(self.V[:, 2].max())
        nt = ~self.terrain_part[self.P]
        self.bld_top = float(self.V[self.F[nt].ravel(), 2].max()) if nt.any() else self.zmax_all
        self._topo = None

    def topo(self):
        if self._topo is None:
            self._topo = EdgeTopo(self.V, self.F, self.P, self.N, self.crease)
        return self._topo

    def frame(self, d):
        d = np.array([d[0], d[1], 0.0]); d /= np.linalg.norm(d)
        right = np.array([d[1], -d[0], 0.0])
        return d, right

    def model_extent(self, bearing):
        """보는 방향 기준 깊이·가로 범위(m) — 화면의 자르기 입력용."""
        d, right = self.frame(view_dir_from_bearing(bearing))
        z = self.V @ d; u = self.V @ right
        return dict(depth=[float(z.min()), float(z.max())], width=[float(u.min()), float(u.max())])

    def run(self, name, bearing=None, sun=(45.0, 45.0), progress=print, textures=None, fade=None,
            clip=None, glass_through=False, depth_tone=True, px=None, max_pixels=40e6, keep_buffers=False):
        """clip = {near, far, left, right} (m, 모델 앞면·왼쪽 끝에서 잰 값, None이면 끝까지)."""
        if bearing is None:
            bearing = BEARINGS[name]
        d, right = self.frame(view_dir_from_bearing(bearing))
        V = self.V
        u_all = V @ right; z_all = V @ d
        zfront, ulo = float(z_all.min()), float(u_all.min())
        # ── 자르기: 삼각형 무게중심 기준 ──
        F, P = self.F, self.P
        clip = clip or {}
        cz = z_all[F].mean(1); cu = u_all[F].mean(1)
        keep = np.ones(len(F), bool)
        if clip.get("near") is not None:
            keep &= cz >= zfront + float(clip["near"])
        if clip.get("far") is not None:
            keep &= cz <= zfront + float(clip["far"])
        if clip.get("left") is not None:
            keep &= cu >= ulo + float(clip["left"])
        if clip.get("right") is not None:
            keep &= cu <= ulo + float(clip["right"])
        # 지형: 단면 위치 앞쪽은 지운다(건물을 가리지 않게)
        tpart = self.terrain_part[P]
        nt_keep = keep & ~tpart
        if not nt_keep.any():
            raise ValueError(f"{name}: 자른 범위 안에 건물이 없어요")
        cut_depth = float(cz[nt_keep].min()) if clip.get("near") is None else zfront + float(clip["near"])
        if tpart.any():
            keep &= ~(tpart & (cz < cut_depth - 0.05))
        F, P = F[keep], P[keep]
        TID = np.nonzero(keep)[0]  # 원래 삼각형 번호
        used = np.unique(F)
        u = u_all; v = V[:, 2]; z = z_all
        uu_used, vv_used = u[used], v[used]
        u0, vtop = float(uu_used.min()) - 1.0, float(vv_used.max()) + 1.0
        vbot = float(vv_used.min()) - 0.5
        if px is None:
            px = self.px
        W = int((uu_used.max() + 1.0 - u0) / px) + 1; H = int((vtop - vbot + 0.5) / px) + 1
        if W * H > max_pixels:  # 메모리 한도: 해상도를 낮춘다
            px *= math.sqrt(W * H / max_pixels)
            W = int((uu_used.max() + 1.0 - u0) / px) + 1; H = int((vtop - vbot + 0.5) / px) + 1
        self.px_used = px
        sx = (u - u0) / px
        sy = (vtop - v) / px
        progress(f"{name}: 깊이 버퍼 {W:,}×{H:,} (한 칸 {px * 1000:.0f} mm)")
        zf = z.astype(np.float64)
        zb, ib_local = rasterize(sx, sy, zf, F, W, H)
        ib = np.where(ib_local >= 0, ib_local, -1)
        # 유리 투명: 유리를 뺀 깊이 버퍼로 숨은선 판단
        zb_op = zb
        if glass_through and self.glass_part[P].any():
            op = ~self.glass_part[P]
            zb_op, _ = rasterize(sx, sy, zf, F[op], W, H)

        progress(f"{name}: 모서리 찾기")
        E, et = self.topo().edges(d, keep)
        progress(f"{name}: 숨은선 제거 ({len(E):,}개 모서리)")
        segs, wts, lays, tones = self._visible(E, sx, sy, z, zb, zb_op, self.part_layer[self.P[et]], px, zfront, depth_tone)
        progress(f"{name}: 선 합치기 ({len(segs):,}개)")
        lines = merge_lines(segs, wts, lays, tones)

        progress(f"{name}: 그림자")
        shadow = self._shadow(name, d, right, sun, zb, ib, u0, vtop, W, H, F, TID, px)

        progress(f"{name}: 재료 영역")
        valid = ib >= 0
        part_of_px = np.where(valid, P[np.maximum(ib, 0)], -1)
        layer_of_px = np.where(valid, self.part_layer[np.maximum(part_of_px, 0)], -1)
        layer_polys = {}
        for li, ln in enumerate(self.lnames):
            m = layer_of_px == li
            if m.sum() > 50:
                pi = next(i for i, p in enumerate(self.parts) if p[2] == ln)
                if self.terrain_part[pi]:
                    continue
                layer_polys[ln] = dict(cat=self.cat[pi], polys=self._polys(m))
        shadow_polys = self._polys(shadow, open_px=1)
        tex = {}
        if textures:
            progress(f"{name}: 텍스처 입히기")
            tex = self._textures(textures, fade or {}, zb, layer_of_px, u0, vtop, px)

        # 지형 단면
        ground = []
        if tpart.any():
            ground = terrain_section(V, self.F[self.terrain_part[self.P]], d, right, cut_depth)

        ntv = np.unique(F[~tpart[keep]])
        top_clip = float(V[ntv, 2].max()) if len(ntv) else self.bld_top
        # 축선 후보(기둥)
        grids = self._grids(F, P, u, v, z, right, valid, part_of_px)
        return dict(name=name, bearing=bearing, W=W, H=H, px=px, u0=u0, vtop=vtop,
                    lines=lines, layer_names=self.lnames, layer_polys=layer_polys, shadow=shadow_polys,
                    gl=self.zmin_all, top=top_clip, umin=float(uu_used.min()), umax=float(uu_used.max()),
                    vbot=vbot, tex=tex, ground=ground, grids=grids, right=right.tolist(),
                    **({"buf": self._buffers(zb, part_of_px, layer_of_px, shadow, u, v, d, right, zfront)} if keep_buffers else {}))

    def _buffers(self, zb, part_of_px, layer_of_px, shadow, u, v, d, right, zfront):
        """콜라주 그리기용 원본 버퍼 + 부재마다 텍스처 기준점(왼쪽 아래, m)."""
        tu, tv = u[self.F].min(1), v[self.F].min(1)
        au = np.full(len(self.parts), np.inf); av = np.full(len(self.parts), np.inf)
        np.minimum.at(au, self.P, tu); np.minimum.at(av, self.P, tv)
        bz = np.where(np.isfinite(zb), zb, np.inf)
        nt = part_of_px >= 0
        nt_front = float(bz[nt & ~self.terrain_part[np.maximum(part_of_px, 0)]].min()) if nt.any() else zfront
        return dict(zb=zb.astype(np.float32), part=part_of_px.astype(np.int32), layer=layer_of_px.astype(np.int16),
                    shadow=shadow, anchor_u=au, anchor_v=av, d=np.asarray(d, float).tolist(), zfront=zfront,
                    bld_front=nt_front, cat=list(self.cat), terrain=self.terrain_part.copy())

    def _grids(self, F, P, u, v, z, right, valid, part_of_px):
        """세로로 길고 가는 부재(기둥) 위치 → 축선.
        → {"all": 자른 범위 안 모든 기둥 줄(u), "vis": 이 면에서 보이는 줄(u)}. 이름은 all 기준으로 붙여야 면끼리 같다."""
        names = [p[2].lower() for p in self.parts]
        skip = ("가로등", "정류장", "출입구", "신호", "전봇대", "전주", "교통", "lamp", "light", "sign", "fence", "울타리")
        colp = np.array([("column" in n or "기둥" in n or n.split("::")[-1].startswith("col")) and not any(w in n for w in skip)
                         for n in names])
        if not colp.any():
            return {"all": [], "vis": []}
        seen = np.zeros(len(self.parts), bool)
        vis_parts = np.unique(part_of_px[valid])
        seen[vis_parts[vis_parts >= 0]] = True
        sel = colp[P]
        if not sel.any():
            return {"all": [], "vis": []}
        pp = P[sel]; ff = F[sel]
        o = np.argsort(pp, kind="stable"); pp, ff = pp[o], ff[o]
        st = np.r_[0, np.nonzero(np.diff(pp))[0] + 1]
        tu, tv = u[ff], v[ff]
        umin = np.minimum.reduceat(tu.min(1), st); umax = np.maximum.reduceat(tu.max(1), st)
        vmin = np.minimum.reduceat(tv.min(1), st); vmax = np.maximum.reduceat(tv.max(1), st)
        ok = ((vmax - vmin) > 2.5) & ((umax - umin) < 1.2)
        cols = [(float((a + b) / 2), bool(seen[k])) for a, b, k in zip(umin[ok], umax[ok], pp[st][ok])]

        def cluster(xs):
            groups = []
            for x in sorted(xs):
                if groups and x - groups[-1][-1] < 1.5:  # 붙어 있는 기둥·쌍기둥은 축선 하나
                    groups[-1].append(x)
                else:
                    groups.append([x])
            return [float(np.mean(g)) for g in groups]
        return {"all": cluster([c for c, _ in cols]), "vis": cluster([c for c, s_ in cols if s_])}

    def _textures(self, textures, fade, zb, layer_of_px, u0, vtop, px):
        self_px = px
        """textures = {레이어: {image: PIL.Image, w: m, h: m}} → {레이어: {rgba, x, y}} (버퍼 픽셀 위치)."""
        from PIL import Image
        fin = np.isfinite(zb)
        if fade.get("ref", "local") == "local":
            colmin = np.where(fin, zb, np.inf).min(axis=0)
            colmin = imgops.minimum_filter1d(np.where(np.isfinite(colmin), colmin, 1e9), size=max(3, int(3.0 / self_px)))
            ref = colmin[None, :]
        else:
            ref = np.percentile(zb[fin], 0.5) if fin.any() else 0.0
        st, en, mn = float(fade.get("start", 1.5)), float(fade.get("end", 15.0)), float(fade.get("min", 0.2))
        use_fade = fade.get("on", True)
        out = {}
        for li, ln in enumerate(self.lnames):
            t = textures.get(ln)
            if not t:
                continue
            m = layer_of_px == li
            if m.sum() < 50:
                continue
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            tw = max(2, int(round(t["w"] / self_px))); th = max(2, int(round(t["h"] / self_px)))
            img = t["image"].convert("RGB").resize((tw, th), Image.LANCZOS)
            arr = np.asarray(img)
            X = np.arange(x0, x1); Y = np.arange(y0, y1)
            uu = (X + 0.5) * self_px + u0
            vv = vtop - (Y + 0.5) * self_px
            ix = np.floor((uu / t["w"]) % 1.0 * tw).astype(int) % tw
            iy = np.floor((1.0 - (vv / t["h"]) % 1.0) * th).astype(int) % th
            rgb = arr[iy[:, None], ix[None, :]]
            mm = m[y0:y1, x0:x1]
            a = np.where(mm, 1.0, 0.0)
            if use_fade:
                rr = ref[:, x0:x1] if np.ndim(ref) else ref
                dd = zb[y0:y1, x0:x1] - rr
                k = np.clip((dd - st) / max(en - st, 1e-6), 0, 1)
                a = a * (1.0 - k * (1.0 - mn))
            a = a * float(t.get("opacity", 1.0))
            rgba = np.dstack([rgb, (a * 255).astype(np.uint8)])
            out[ln] = dict(rgba=rgba, x=int(x0), y=int(y0))
        return out

    def _visible(self, E, sx, sy, z, zb, zb_op, elay, px, zfront, depth_tone):
        """→ 선분, 굵기(0 윤곽…4 먼 곳, 5 유리 뒤), 레이어, 색 단계(0~3)."""
        H, W = zb.shape
        zmax3 = imgops.maximum_filter3(np.where(np.isinf(zb_op), 1e9, zb_op))
        glass = zb_op is not zb
        if glass:
            zmax3_all = imgops.maximum_filter3(np.where(np.isinf(zb), 1e9, zb))
        colmin = np.where(np.isfinite(zb), zb, np.inf).min(axis=0)
        colmin = imgops.minimum_filter1d(np.where(np.isfinite(colmin), colmin, 1e9), size=max(3, int(3.0 / px)))
        eps = 0.03 + 1.5 * px
        out = []
        CH = 400_000
        for c0 in range(0, len(E), CH):
            e = E[c0:c0 + CH]; el = elay[c0:c0 + CH]
            x0, y0, z0 = sx[e[:, 0]], sy[e[:, 0]], z[e[:, 0]]
            x1, y1, z1 = sx[e[:, 1]], sy[e[:, 1]], z[e[:, 1]]
            L = np.hypot(x1 - x0, y1 - y0)
            ok = L > 0.6
            x0, y0, z0, x1, y1, z1, L, el = [a[ok] for a in (x0, y0, z0, x1, y1, z1, L, el)]
            if not len(L):
                continue
            ns = np.maximum(2, np.ceil(L / 0.7).astype(int) + 1)
            idx = np.repeat(np.arange(len(L)), ns)
            k = np.arange(ns.sum()) - np.repeat(np.cumsum(ns) - ns, ns)
            t = k / (ns[idx] - 1)
            X = x0[idx] + (x1 - x0)[idx] * t; Y = y0[idx] + (y1 - y0)[idx] * t
            Z = z0[idx] + (z1 - z0)[idx] * t
            xi = np.clip(X.astype(int), 0, W - 1); yi = np.clip(Y.astype(int), 0, H - 1)
            vis = zmax3[yi, xi] >= Z - eps
            behind = (zmax3_all[yi, xi] < Z - eps) if glass else np.zeros_like(vis)
            state = vis.astype(np.int8) + (vis & behind).astype(np.int8)  # 0 숨음, 1 보임, 2 유리 뒤
            jump = (zmax3[yi, xi] - Z) > 1.0
            brk = np.r_[True, (idx[1:] != idx[:-1]) | (state[1:] != state[:-1])]
            g_start = np.nonzero(brk)[0]
            g_end = np.r_[g_start[1:], len(idx)] - 1
            gv = state[g_start] > 0
            gs, ge = g_start[gv], g_end[gv]
            keep = ge > gs
            gs, ge = gs[keep], ge[keep]
            seg = np.c_[X[gs], Y[gs], X[ge], Y[ge]]
            depth = np.minimum(Z[gs], Z[ge]) - np.minimum(colmin[xi[gs]], colmin[xi[ge]])
            cj = np.cumsum(jump)
            jfrac = (cj[ge] - cj[gs] + jump[gs]) / (ge - gs + 1)
            w = np.select([jfrac > 0.5, depth < 1.5, depth < 5, depth < 12], [0, 1, 2, 3], 4)
            w = np.where(state[gs] == 2, 5, w)
            gd = np.minimum(Z[gs], Z[ge]) - zfront
            tone = np.select([gd < 15, gd < 35, gd < 70], [0, 1, 2], 3) if depth_tone else np.zeros(len(gs), int)
            out.append((seg, w, el[idx[gs]], tone))
        if not out:
            return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
        segs = np.vstack([o[0] for o in out]); wts = np.concatenate([o[1] for o in out])
        lays = np.concatenate([o[2] for o in out]); tones = np.concatenate([o[3] for o in out])
        a = np.round(segs, 1)
        sw = (a[:, 0] > a[:, 2]) | ((a[:, 0] == a[:, 2]) & (a[:, 1] > a[:, 3]))
        a[sw] = a[sw][:, [2, 3, 0, 1]]
        o = np.lexsort((wts,))
        _, ui = np.unique(a[o], axis=0, return_index=True)
        ui = o[ui]
        return segs[ui], wts[ui], lays[ui], tones[ui]

    def _shadow(self, name, d, right, sun, zb, ib, u0, vtop, W, H, F, TID, px):
        if isinstance(sun, dict) and sun.get("mode") == "real":
            A, h = solar_position(sun.get("month", 6), sun.get("day", 21), sun.get("hour", 14.0), sun.get("lat", 37.56), sun.get("lon", 126.99))
            if h <= 1:
                m = np.zeros((H, W), bool); m[ib >= 0] = True  # 해가 진 시각: 전부 그늘
                return m
            A, h = math.radians(A), math.radians(h)
            toward_sun = np.array([math.sin(A) * math.cos(h), math.cos(A) * math.cos(h), math.sin(h)])
        else:
            az, alt = map(math.radians, sun if not isinstance(sun, dict) else (45.0, 45.0))
            # 보는 사람 기준: 왼쪽 앞 위에서 빛이 옴
            toward_sun = (-math.sin(az) * right - math.cos(az) * d) * math.cos(alt) + np.array([0, 0, math.sin(alt)])
        s = toward_sun / np.linalg.norm(toward_sun)
        L = -s  # 빛 진행 방향
        a1 = np.cross(L, [0, 0, 1.0]); a1 /= np.linalg.norm(a1)
        a2 = np.cross(a1, L)
        V = self.V
        pu, pv, pz = V @ a1, V @ a2, V @ L
        used = np.unique(F)
        spx = px * 1.5
        su0, sv0 = pu[used].min() - 0.5, pv[used].min() - 0.5
        SW = int((pu[used].max() - su0 + 0.5) / spx) + 1; SH = int((pv[used].max() - sv0 + 0.5) / spx) + 1
        if SW * SH > 60e6:
            spx *= math.sqrt(SW * SH / 60e6)
            SW = int((pu[used].max() - su0 + 0.5) / spx) + 1; SH = int((pv[used].max() - sv0 + 0.5) / spx) + 1
        szb, sib = rasterize((pu - su0) / spx, (pv - sv0) / spx, pz.astype(np.float64), F, SW, SH)
        ys, xs = np.nonzero(ib >= 0)
        uu = xs * px + u0 + px / 2; vv = vtop - (ys + 0.5) * px
        P = uu[:, None] * right + vv[:, None] * np.array([0, 0, 1.0]) + zb[ys, xs][:, None] * d
        qx = ((P @ a1) - su0) / spx; qy = ((P @ a2) - sv0) / spx
        qz = P @ L
        qxi = np.clip(qx.astype(int), 0, SW - 1); qyi = np.clip(qy.astype(int), 0, SH - 1)
        occl = szb[qyi, qxi] < qz - (0.04 + 3 * spx)
        # 법선 방향이 뒤집혀 있을 수 있음: 보는 쪽을 향하도록 맞춰서 판단
        nn = self.N[TID[ib[ys, xs]]]
        flip = (nn @ d) > 0
        nn[flip] *= -1
        back = (nn @ s) < 0.02
        m = np.zeros((H, W), bool)
        m[ys, xs] = occl | back
        return m

    def _polys(self, mask, open_px=1, simplify_px=0.8):
        # 칠해진 곳만 잘라서 본다(레이어마다 화면 전체를 훑지 않게). 결과 좌표는 원래 자리로 되돌린다.
        rows = np.flatnonzero(mask.any(axis=1))
        if not len(rows):
            return []
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        cols = np.flatnonzero(mask[y0:y1].any(axis=0))
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        mask = mask[y0:y1, x0:x1]
        if open_px:
            mask = imgops.binary_opening(mask, iterations=open_px)
        if not mask.any():
            return []
        cg = contourpy.contour_generator(z=np.pad(mask, 1).astype(np.float32), name="serial",
                                         fill_type=contourpy.FillType.OuterOffset)
        pts_list, offs_list = cg.filled(0.5, 1.5)
        shift = np.array([x0 - 1.0, y0 - 1.0])
        out = []
        for pts, offs in zip(pts_list, offs_list):
            rings = [pts[offs[k]:offs[k + 1]] + shift for k in range(len(offs) - 1)]
            if len(rings[0]) < 4:
                continue
            p = Polygon(rings[0], [r for r in rings[1:] if len(r) >= 4])
            if p.area < 6:
                continue
            if not p.is_valid:
                p = p.buffer(0)
            p = p.simplify(simplify_px)
            for g in (p.geoms if hasattr(p, "geoms") else [p]):
                if isinstance(g, Polygon) and g.area > 6:
                    out.append(g)
        return out


def solar_position(month, day, hour, lat=37.56, lon=126.99, tz=9.0):
    """태양 방위각(북=0, 시계방향)·고도(도). 한국 표준시 기준 간이 계산(오차 1° 안팎)."""
    import datetime
    n = datetime.date(2026, int(month), int(day)).timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1 + (hour - 12) / 24)
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) \
        + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    tst = hour * 60 + eqt + 4 * lon - 60 * tz
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1, min(1, cz)))
    alt = 90 - math.degrees(zen)
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(decl) * math.cos(la))) + 180
    return az % 360, alt


def terrain_section(V, Ft, d, right, depth):
    """지형 삼각형을 깊이 = depth 인 면으로 자른 선분들 → [(u0, v0, u1, v1)] (m)."""
    z = V @ d
    zt = z[Ft] - depth
    s = np.sign(zt)
    cross = ~((s[:, 0] == s[:, 1]) & (s[:, 1] == s[:, 2]))
    out = []
    for tri, zz in zip(Ft[cross], zt[cross]):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (zz[a] > 0) != (zz[b] > 0):
                t = zz[a] / (zz[a] - zz[b])
                p = V[tri[a]] + (V[tri[b]] - V[tri[a]]) * t
                pts.append((float(p @ right), float(p[2])))
        if len(pts) == 2:
            out.append((*pts[0], *pts[1]))
    return out


def merge_lines(segs, wts, lays, tones, ang_q=0.15, off_q=0.35, gap=0.9):
    """같은 직선 위 선분을 합치고(겹침·이어짐), 끝점이 맞는 선은 폴리라인으로 잇는다.

    → [(점 배열(n,2), 굵기, 레이어, 색 단계)] (픽셀 좌표).
    """
    if not len(segs):
        return []
    x0, y0, x1, y1 = segs.T
    dx, dy = x1 - x0, y1 - y0
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    ang = np.where(ang > 180 - ang_q / 2, ang - 180, ang)
    th = np.radians(np.round(ang / ang_q) * ang_q)
    nx, ny = -np.sin(th), np.cos(th)
    off = x0 * nx + y0 * ny
    key_a = np.round(ang / ang_q).astype(np.int64)
    key_o = np.round(off / off_q).astype(np.int64)
    tx, ty = np.cos(th), np.sin(th)
    t0 = x0 * tx + y0 * ty; t1 = x1 * tx + y1 * ty
    lo, hi = np.minimum(t0, t1), np.maximum(t0, t1)
    order = np.lexsort((lo, key_o, key_a, tones, lays, wts))
    merged = []
    cur = None
    for i in order:
        k = (wts[i], lays[i], tones[i], key_a[i], key_o[i])
        if cur is not None and cur[0] == k and lo[i] <= cur[2] + gap:
            cur[2] = max(cur[2], hi[i])
            continue
        if cur is not None:
            merged.append(cur)
        cur = [k, lo[i], hi[i], th[i], off[i]]
    merged.append(cur)
    segs2 = []
    for k, a, b, t, o in merged:
        nxv, nyv = -math.sin(t), math.cos(t)
        txv, tyv = math.cos(t), math.sin(t)
        p = (o * nxv + a * txv, o * nyv + a * tyv); q = (o * nxv + b * txv, o * nyv + b * tyv)
        segs2.append((k[0], k[1], k[2], p, q))
    # 끝점 잇기(같은 굵기·레이어·색)
    from collections import defaultdict
    groups = defaultdict(list)
    for w, l, tn, p, q in segs2:
        groups[(w, l, tn)].append((p, q))
    out = []
    for (w, l, tn), ss in groups.items():
        rk = lambda p: (round(p[0] * 2), round(p[1] * 2))
        adj = defaultdict(list)
        for i, (p, q) in enumerate(ss):
            adj[rk(p)].append(i); adj[rk(q)].append(i)
        used = np.zeros(len(ss), bool)
        for i in range(len(ss)):
            if used[i]:
                continue
            used[i] = True
            p, q = ss[i]
            chain = [p, q]
            for end in (1, 0):
                while True:
                    tip = chain[-1] if end else chain[0]
                    nb = [j for j in adj[rk(tip)] if not used[j]]
                    if len(adj[rk(tip)]) != 2 or not nb:
                        break
                    j = nb[0]; used[j] = True
                    a, b = ss[j]
                    nxt = b if rk(a) == rk(tip) else a
                    if end:
                        chain.append(nxt)
                    else:
                        chain.insert(0, nxt)
            out.append((np.array(chain), int(w), int(l), int(tn)))
    return out
