"""3dm 읽기: 레이어 목록·경고를 만들고, 형상을 삼각형 메시로 푼다(블록 포함).

결과는 캐시(npz)로 남겨서 같은 파일은 다시 읽지 않는다.
"""
import hashlib
import json
import os

import numpy as np
import rhino3dm as r

AUTO_SKIP = ("Make2D",)  # Make2D 결과 선은 항상 뺀다


def _mesh_arrays(mesh):
    if mesh is None:
        return None
    mesh.Faces.ConvertQuadsToTriangles()
    nv = len(mesh.Vertices)
    if nv == 0 or len(mesh.Faces) == 0:
        return None
    v = np.array([[p.X, p.Y, p.Z] for p in mesh.Vertices], float)
    f = np.array([mesh.Faces[i][:3] for i in range(len(mesh.Faces))], int)
    return v, f


def _geom_meshes(g, stats):
    t = type(g).__name__
    out = []
    if t == "Brep":
        miss = 0
        for i in range(len(g.Faces)):
            a = _mesh_arrays(g.Faces[i].GetMesh(r.MeshType.Any))
            if a:
                out.append(a)
            else:
                miss += 1
        if miss:
            stats["no_mesh"] += 1
    elif t == "Extrusion":
        a = _mesh_arrays(g.GetMesh(r.MeshType.Any))
        if a:
            out.append(a)
        else:
            stats["no_mesh"] += 1
    elif t == "Mesh":
        a = _mesh_arrays(g)
        if a:
            out.append(a)
    elif t == "SubD":
        stats["subd"] += 1
    return out


def _xf(x):
    return np.array([[x.M00, x.M01, x.M02, x.M03], [x.M10, x.M11, x.M12, x.M13],
                     [x.M20, x.M21, x.M22, x.M23], [x.M30, x.M31, x.M32, x.M33]])


def file_key(path):
    st = os.stat(path)
    return hashlib.md5(f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


def read(path, cache_dir=None, log=print):
    """→ (parts, info). parts = [(V(m), F, 레이어 전체 이름, 재료 이름)]. info = 레이어 목록·경고."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = file_key(path)
        cp = os.path.join(cache_dir, key + ".npz")
        ip = os.path.join(cache_dir, key + ".json")
        if os.path.exists(cp) and os.path.exists(ip):
            log("예전에 읽어 둔 모델을 불러와요")
            return load_cache(cp), json.load(open(ip, encoding="utf-8"))
    log("3dm 파일 여는 중")
    m = r.File3dm.Read(path)
    if m is None:
        raise ValueError("3dm 파일을 열 수 없어요. 라이노 8 이하에서 저장한 파일인지 확인해 주세요.")
    us = m.Settings.ModelUnitSystem
    scale = {r.UnitSystem.Millimeters: 0.001, r.UnitSystem.Centimeters: 0.01, r.UnitSystem.Meters: 1.0,
             r.UnitSystem.Inches: 0.0254, r.UnitSystem.Feet: 0.3048}.get(us, 1.0)
    layers = {L.Index: L for L in m.Layers}
    by_id = {str(L.Id): L for L in m.Layers}

    def eff_visible(L):
        while L is not None:
            if not L.Visible:
                return False
            L = by_id.get(str(L.ParentLayerId))
        return True
    mats = [mt.Name for mt in m.Materials]
    idefs = {d.Id: d for d in m.InstanceDefinitions}
    stats = {"no_mesh": 0, "subd": 0, "hidden_obj": 0, "curves": 0}
    parts = []

    def layer_of(attr):
        L = layers.get(attr.LayerIndex)
        name = L.FullPath if L else ""
        src = str(attr.MaterialSource)
        mi = attr.MaterialIndex if src.endswith("MaterialFromObject") else (L.RenderMaterialIndex if L else -1)
        return name, (mats[mi] if 0 <= mi < len(mats) else "")

    def add(g, attr, M, parent=None, depth=0):
        name, mat = layer_of(attr)
        if parent and name in ("", "Default", "기본값", "0"):
            name, mat = parent[0], (mat or parent[1])
        if any(name.startswith(s) for s in AUTO_SKIP):
            return
        t = type(g).__name__
        if t == "InstanceReference":
            if depth > 8:
                return
            d = idefs.get(g.ParentIdefId)
            if d is None:
                return
            M2 = M @ _xf(g.Xform)
            for oid in d.GetObjectIds():
                o = m.Objects.FindId(oid)
                if o is not None:
                    add(o.Geometry, o.Attributes, M2, (name, mat), depth + 1)
            return
        if "Curve" in t:
            stats["curves"] += 1
            return
        for v, f in _geom_meshes(g, stats):
            vh = np.c_[v, np.ones(len(v))] @ M.T
            parts.append((vh[:, :3] * scale, f, name, mat))

    n = len(m.Objects)
    for i, o in enumerate(m.Objects):
        a = o.Attributes
        if a.IsInstanceDefinitionObject:
            continue
        if not a.Visible:
            stats["hidden_obj"] += 1
            continue
        add(o.Geometry, a, np.eye(4))
        if i % 5000 == 0:
            log(f"형상 푸는 중 {i:,}/{n:,}", frac=i / max(n, 1))

    # 레이어 목록
    tri = {}
    cnt = {}
    zr = {}
    for v, f, ln, mt in parts:
        tri[ln] = tri.get(ln, 0) + len(f)
        cnt[ln] = cnt.get(ln, 0) + 1
        lo, hi = zr.get(ln, (1e18, -1e18))
        zr[ln] = (min(lo, float(v[:, 2].min())), max(hi, float(v[:, 2].max())))
    rows = []
    for L in m.Layers:
        fp = L.FullPath
        if fp not in tri:
            continue
        c = L.Color
        mat0 = next((p[3] for p in parts if p[2] == fp and p[3]), "")
        rows.append(dict(name=fp, visible=eff_visible(L), color="#%02x%02x%02x" % (c[0], c[1], c[2]),
                         parts=cnt[fp], tris=tri[fp], material_name=mat0,
                         zmin=round(zr[fp][0], 3), zmax=round(zr[fp][1], 3)))
    for ln in tri:  # 레이어 표에 없는 이름(블록 안 기본 레이어 등)
        if not any(rw["name"] == ln for rw in rows):
            rows.append(dict(name=ln or "(레이어 없음)", visible=True, color="#888888", parts=cnt[ln], tris=tri[ln],
                             material_name="", zmin=round(zr[ln][0], 3), zmax=round(zr[ln][1], 3)))
    V = np.vstack([p[0] for p in parts]) if parts else np.zeros((1, 3))
    warn = []
    if stats["no_mesh"]:
        warn.append(f"렌더 메시가 없는 객체 {stats['no_mesh']:,}개는 빠졌어요. 라이노에서 '작게 저장(Save Small)'이 아닌 일반 저장을 하거나, 음영 보기로 한 번 본 뒤 저장해 주세요.")
    if stats["subd"]:
        warn.append(f"SubD 객체 {stats['subd']:,}개는 아직 읽지 못해요. 라이노에서 ToNURBS 또는 Mesh로 바꿔 주세요.")
    if stats["hidden_obj"]:
        warn.append(f"숨긴(Hide) 객체 {stats['hidden_obj']:,}개는 뺐어요.")
    hidden_layers = [rw["name"] for rw in rows if not rw["visible"]]
    if hidden_layers:
        warn.append(f"꺼진 레이어 {len(hidden_layers)}개는 처음에 '빼기'로 두었어요: " + ", ".join(hidden_layers[:6]) + (" …" if len(hidden_layers) > 6 else ""))
    info = dict(file=os.path.basename(path), unit=str(us).split(".")[-1], layers=rows, warnings=warn,
                bbox=[V.min(0).round(3).tolist(), V.max(0).round(3).tolist()],
                tris=int(sum(len(p[1]) for p in parts)))
    if cache_dir:
        save_cache(parts, cp)
        json.dump(info, open(ip, "w", encoding="utf-8"), ensure_ascii=False)
    return parts, info


def save_cache(P, path):
    vo = np.cumsum([0] + [len(p[0]) for p in P]); fo = np.cumsum([0] + [len(p[1]) for p in P])
    np.savez(path, V=np.vstack([p[0] for p in P]).astype(np.float64), F=np.vstack([p[1] for p in P]).astype(np.int32),
             vo=vo, fo=fo, names=np.array([p[2] for p in P]), mats=np.array([p[3] for p in P]))


def load_cache(path):
    d = np.load(path)
    V, F, vo, fo = d["V"], d["F"], d["vo"], d["fo"]
    return [(V[vo[i]:vo[i + 1]], F[fo[i]:fo[i + 1]], str(n), str(mt)) for i, (n, mt) in enumerate(zip(d["names"], d["mats"]))]
