"""레이어 설정 저장·불러오기, 입면 만들기 작업."""
import datetime
import json
import os
import re
import shutil
import threading
import time

import numpy as np

from . import model3dm, analyze
from .elevation import Elevation, VIEWS, BEARINGS, Cancelled, classify
from . import draw

HOME = os.path.join(os.path.expanduser("~"), "입면도")
INPUT = os.path.join(HOME, "_입력")
CACHE = os.path.join(HOME, "_캐시")
SETTINGS = os.path.join(HOME, "_레이어설정")
MATLIB = os.path.join(HOME, "_재료")          # 재료 표현·텍스처 보관함(모든 모델 공통)
SAMPLES = os.path.join(os.path.dirname(__file__), "textures")

MATERIALS = ["유리", "금속", "콘크리트", "콘크리트블록", "벽돌", "목재", "폴리카보네이트", "기타", "수목"]
SURFACE_MATS = [m for m in MATERIALS if m != "수목"]   # 재료 표현 표(텍스처·해치)에 나오는 재료
CONTEXT_WORDS = ("주변", "인접", "기존건물", "기존_건물", "기존 건물", "context", "neighbo", "surround", "adjacent", "existing")
TERRAIN_WORDS = ("지형", "terrain", "topo", "콘타", "contour", "대지", "site", "ground", "지반", "landscape")
EXTRA_KEYS = ("north", "scale", "sun", "views", "formats", "quality", "levels", "gl", "clip", "glass", "tone",
              "dirmode", "paper", "project", "dims", "grids", "tags", "sheet", "style", "el_base", "collage")
CANCEL = threading.Event()


def _safe(s):
    return re.sub(r'[\\/:*?"<>|\s]+', "_", s).strip("_")[:60] or "model"


def default_cfg(info):
    cfg = {}
    for rw in info["layers"]:
        n = rw["name"]
        cfg[n] = dict(include=bool(rw["visible"]), material=classify(n, rw.get("material_name", "")),
                      terrain=any(w in n.lower() for w in TERRAIN_WORDS),
                      context=any(w in n.lower() for w in CONTEXT_WORDS))
    return cfg


def settings_path(info):
    return os.path.join(SETTINGS, _safe(os.path.splitext(info["file"])[0]) + ".json")


def load_cfg(info):
    """저장해 둔 설정이 있으면 그걸 쓰고, 새 레이어만 기본값으로 채운다."""
    cfg = default_cfg(info)
    p = settings_path(info)
    saved = False
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8")).get("layers", {})
        for k, v in old.items():
            if k in cfg:
                v = dict(v)
                if "context" not in v and cfg[k]["material"] == "수목" and v.get("material") == "목재":
                    v.pop("material")   # v0.7 이전 설정: '나무' 레이어를 목재로 저장해 둔 것은 수목으로 바로잡기
                cfg[k].update({kk: v[kk] for kk in ("include", "material", "terrain", "context") if kk in v})
        saved = True
    return cfg, saved


def model_sig(info):
    """같은 파일 이름이라도 모델이 바뀌었는지 알아보는 표시(범위 0.1 m + 삼각형 수)."""
    bb = info.get("bbox") or []
    return [round(float(x), 1) for x in np.ravel(bb)] + [int(info.get("tris", 0))]


def save_cfg(info, cfg, extra=None):
    os.makedirs(SETTINGS, exist_ok=True)
    data = dict(file=info["file"], saved=datetime.datetime.now().isoformat(timespec="seconds"), layers=cfg,
                model_sig=model_sig(info))
    if extra:
        data.update(extra)
    json.dump(data, open(settings_path(info), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def load_extra(info):
    p = settings_path(info)
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        ex = {k: d[k] for k in EXTRA_KEYS if k in d}
        if d.get("model_sig") and d["model_sig"] != model_sig(info):
            # 모델이 바뀜: 높이에 묶인 값(층·GL·자르기)은 버리고 새로 찾는다
            for k in ("levels", "gl", "clip"):
                ex.pop(k, None)
            ex["_changed"] = True
        return ex
    return {}


# ── 재료 보관함 ───────────────────────────────────────────────────────
DEFAULT_TEX = {"콘크리트": ("노출콘크리트_1.8x0.9m.png", 1.8, 0.9), "콘크리트블록": ("노출콘크리트_1.8x0.9m.png", 1.8, 0.9),
               "벽돌": ("붉은벽돌_0.4x0.134m.png", 0.4, 0.134), "목재": ("목재판재_0.6x1.2m.png", 0.6, 1.2),
               "금속": ("금속패널_1.0x0.5m.png", 1.0, 0.5), "유리": ("유리반사_3x3m.png", 3.0, 3.0)}


def _ensure_samples():
    os.makedirs(MATLIB, exist_ok=True)
    import shutil
    if os.path.isdir(SAMPLES):
        for f in os.listdir(SAMPLES):
            if os.path.isfile(os.path.join(SAMPLES, f)) and not os.path.exists(os.path.join(MATLIB, f)):
                shutil.copy(os.path.join(SAMPLES, f), MATLIB)


def load_matlib():
    _ensure_samples()
    p = os.path.join(MATLIB, "재료설정.json")
    lib = {"materials": {}, "fade": {"on": True, "start": 1.5, "end": 15.0, "min": 0.2, "ref": "local"}}
    for m in SURFACE_MATS:
        t = DEFAULT_TEX.get(m)
        lib["materials"][m] = dict(mode="hatch", texture=t[0] if t else "", w=t[1] if t else 1.0, h=t[2] if t else 1.0,
                                   scale=1.0, opacity=1.0)
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8"))
        for m, v in old.get("materials", {}).items():
            if m in lib["materials"]:
                lib["materials"][m].update(v)
        lib["fade"].update(old.get("fade", {}))
    lib["files"] = sorted(f for f in os.listdir(MATLIB) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")))
    return lib


def save_matlib(lib):
    _ensure_samples()
    json.dump({"materials": lib["materials"], "fade": lib["fade"]}, open(os.path.join(MATLIB, "재료설정.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def layer_textures(cfg, lib):
    """레이어 → 텍스처(재료가 '텍스처' 표현일 때), 레이어 → 표현 방식."""
    from PIL import Image
    imgs, tex, modes = {}, {}, {}
    for ln, c in cfg.items():
        if c.get("material") == "수목":   # 나무는 해치·텍스처 없이 선만
            modes[ln] = "none"
            continue
        mat = lib["materials"].get(c.get("material", "기타"), {})
        md = mat.get("mode", "hatch")
        modes[ln] = md
        if md != "texture":
            continue
        fn = os.path.join(MATLIB, os.path.basename(mat.get("texture", "")))
        if not os.path.isfile(fn):
            modes[ln] = "hatch"
            continue
        if fn not in imgs:
            imgs[fn] = Image.open(fn); imgs[fn].load()
        sc = float(mat.get("scale", 1.0) or 1.0)
        tex[ln] = dict(image=imgs[fn], w=float(mat.get("w", 1.0)) * sc, h=float(mat.get("h", 1.0)) * sc,
                       opacity=float(mat.get("opacity", 1.0)))
    return tex, modes


def collage_textures(lib):
    """콜라주: 재료마다 텍스처(재료 설정에 고른 것, 없으면 기본 견본)."""
    from PIL import Image
    out = {}
    for m in SURFACE_MATS:
        mat = lib["materials"].get(m, {})
        fn = os.path.join(MATLIB, os.path.basename(mat.get("texture") or ""))
        w, h = float(mat.get("w", 1.0)), float(mat.get("h", 1.0))
        if not os.path.isfile(fn) and m in DEFAULT_TEX:
            fn = os.path.join(MATLIB, DEFAULT_TEX[m][0]); w, h = DEFAULT_TEX[m][1], DEFAULT_TEX[m][2]
        if os.path.isfile(fn):
            im = Image.open(fn); im.load()
            sc = float(mat.get("scale", 1.0) or 1.0)
            out[m] = dict(image=im, w=w * sc, h=h * sc, opacity=1.0)
    return out


# 한 칸 크기 = 종이 0.1 mm × 품질 배수(축척에 맞춤). 최소 4 mm.
QUALITY = {"빠르게": 2.0, "보통": 1.0, "세밀": 0.5}


def pixel_size(scale, quality):
    return max(0.004, scale * 0.0001 * QUALITY.get(quality, 1.0))


def tidy_folders(current=None, keep_cache=3):
    """_입력에는 지금 파일만, _캐시는 최근 몇 개만 남긴다."""
    try:
        if os.path.isdir(INPUT):
            for f in os.listdir(INPUT):
                p = os.path.join(INPUT, f)
                if os.path.isfile(p) and (not current or os.path.abspath(p) != os.path.abspath(current)):
                    os.remove(p)
        if os.path.isdir(CACHE):
            keys = {}
            for f in os.listdir(CACHE):
                if f.endswith((".npz", ".json")):
                    k = os.path.splitext(f)[0]
                    keys[k] = max(keys.get(k, 0), os.path.getmtime(os.path.join(CACHE, f)))
            for k in sorted(keys, key=keys.get, reverse=True)[keep_cache:]:
                for ext in (".npz", ".json", ".png"):
                    p = os.path.join(CACHE, k + ext)
                    if os.path.exists(p):
                        os.remove(p)
    except OSError:
        pass


def history(n=12):
    """지난 결과 폴더(최근 순)."""
    out = []
    if not os.path.isdir(HOME):
        return out
    for d in os.listdir(HOME):
        p = os.path.join(HOME, d, "만든_설정.json")
        if os.path.isfile(p):
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            out.append(dict(folder=d, path=os.path.join(HOME, d), model=j.get("model", ""), time=j.get("time", ""),
                            scale=j.get("options", {}).get("scale"), views=[r["view"] for r in j.get("results", [])],
                            mtime=os.path.getmtime(p)))
    out.sort(key=lambda h: -h["mtime"])
    return out[:n]


def view_list(opts, el):
    """[(이름, 방위)] — 동서남북 또는 건물 벽에 맞춘 네 면."""
    names = [v for v in opts.get("views", list(VIEWS)) if v in VIEWS]
    if opts.get("dirmode") == "wall":
        a = analyze.wall_angle_arrays(el.N[~el.terrain_part[el.P]], el.A[~el.terrain_part[el.P]]) \
            if (~el.terrain_part).any() else 0.0
        return [(n, (BEARINGS[n] + a) % 360) for n in names], a
    return [(n, BEARINGS[n]) for n in names], 0.0


def grid_names(r):
    """축선 이름: 세계 X 방향으로 늘어서면 X1…, Y 방향이면 Y1… (서→동, 남→북 순).
    번호는 자른 범위 안 모든 기둥 줄 기준이라 마주 보는 면끼리 같은 이름이 된다. 보이는 줄만 그린다."""
    g = r.get("grids") or {}
    if isinstance(g, list):
        g = {"all": g, "vis": g}
    allu, vis = g.get("all") or [], g.get("vis") or []
    if not vis:
        return []
    rx, ry = r["right"][0], r["right"][1]
    axis, sgn = ("X", np.sign(rx) or 1) if abs(rx) >= abs(ry) else ("Y", np.sign(ry) or 1)
    order = sorted(allu, key=lambda u: u * sgn)
    out = []
    for u in vis:
        i = int(np.argmin([abs(u - a) for a in order]))
        out.append(dict(u=u, name=f"{axis}{i + 1}"))
    out.sort(key=lambda d: d["u"])
    return out


def legend_for(results):
    cats = []
    for r in results:
        for ln, it in r["layer_polys"].items():
            if r.get("modes", {}).get(ln, "hatch") != "none" and it["cat"] not in cats:
                cats.append(it["cat"])
    cats = [c for c in cats if c not in ("기타", "수목")]  # '기타'는 이름이 아니라서 번호를 붙이지 않는다
    cats.sort(key=lambda c: MATERIALS.index(c) if c in MATERIALS else 99)
    return {c: (i + 1, c) for i, c in enumerate(cats)}


ILLUSTRATOR_JSX = r"""// 일러스트레이터: 파일 > 스크립트 > 다른 스크립트… 로 이 파일을 실행하세요.
// 입면 SVG를 연 문서에서, 맨 위 그룹(선·재료_채움·그림자·레벨…)을 레이어로, 그 안 그룹을 하위 레이어로 바꿉니다.
(function () {
  if (app.documents.length === 0) { alert("먼저 입면 SVG를 여세요."); return; }
  var doc = app.activeDocument;
  var src = doc.layers[0];
  function nameOf(it, i) { return it.name && it.name.length ? it.name : ("그룹 " + (i + 1)); }
  function unwrap(g) { while (g.pageItems.length === 1 && g.pageItems[0].typename === "GroupItem") g = g.pageItems[0]; return g; }
  var top = unwrap(src.groupItems.length === 1 ? src.groupItems[0] : src);
  var items = [];
  for (var i = 0; i < top.pageItems.length; i++) items.push(top.pageItems[i]);
  for (var j = items.length - 1; j >= 0; j--) {
    var it = items[j];
    var L = doc.layers.add(); L.name = nameOf(it, j);
    if (it.typename === "GroupItem" && it.groupItems.length > 0) {
      var subs = [];
      for (var k = 0; k < it.pageItems.length; k++) subs.push(it.pageItems[k]);
      for (var m = subs.length - 1; m >= 0; m--) {
        var S = L.layers.add(); S.name = nameOf(subs[m], m);
        subs[m].move(S, ElementPlacement.PLACEATBEGINNING);
      }
    } else {
      it.move(L, ElementPlacement.PLACEATBEGINNING);
    }
  }
  try { if (src.pageItems.length === 0) src.remove(); } catch (e) {}
  alert("레이어로 바꿨어요.");
})();
"""


def make(parts, info, opts, log=print, preview=False, cache=None):
    """opts: cfg, views[], scale, north, sun, formats[], quality, levels[{z,name,show}], gl, clip{}, glass, tone,
    dirmode(nesw|wall), paper, project, dims, grids, tags, sheet → (결과 폴더, 결과 목록)."""
    t0 = time.time()
    CANCEL.clear()

    def chk(msg, frac=None):
        if CANCEL.is_set():
            raise Cancelled("취소했어요")
        log(msg, frac=frac)

    stem = _safe(os.path.splitext(info["file"])[0])
    collage = opts.get("style") == "collage"
    if collage:
        # 나무 레이어는 매스로 그리지 않고 나무 그림으로 바꿔 붙인다
        from . import collage as CL
        tc = opts["cfg"]
        tree_parts = [p for p in parts if CL.is_tree_layer(p[2], tc) and tc.get(p[2], {}).get("include", True)]
        opts = dict(opts, cfg={k: (dict(v, include=False) if CL.is_tree_layer(k, tc) else v) for k, v in opts["cfg"].items()})
    if preview:
        out = os.path.join(CACHE, "미리보기")
        shutil.rmtree(out, ignore_errors=True)
    else:
        out = os.path.join(HOME, f"{stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True)
    scale = int(opts.get("scale", 100))
    px = pixel_size(scale, "빠르게" if preview else opts.get("quality", "보통")) * (2.5 if preview else 1.0)
    # 메시 합치기·모서리 이웃 계산은 레이어 설정·북쪽이 같으면 다시 쓰기(미리보기 → 본 결과가 빨라짐)
    key = json.dumps([sorted((k, bool(v.get("include", True)), bool(v.get("terrain")), v.get("material"))
                             for k, v in opts["cfg"].items()), float(opts.get("north", 0) or 0)], ensure_ascii=False)
    if cache is not None and cache.get("key") == key:
        el = cache["el"]
        chk("입면 준비(앞에서 합친 메시 다시 씀)", frac=0.02)
    else:
        chk("입면 준비 중(메시 합치기)", frac=0.02)
        if cache is not None:
            cache.clear()
        el = Elevation(parts, px=px, layer_cfg=opts["cfg"], north_deg=float(opts.get("north", 0) or 0))
        if cache is not None:
            cache.update(key=key, el=el)
    fm = {"png"} if preview else set(opts.get("formats", ["pdf", "svg", "dxf", "png"]))
    sun = opts.get("sun") or {"mode": "conv"}
    vl, wall = view_list(opts, el)
    if preview:
        vl = vl[:1]
    if wall and opts.get("dirmode") == "wall":
        log(f"건물 벽 방향 {wall:+.1f}°에 맞춰 봐요")
    lib = opts.get("matlib") or load_matlib()
    if opts.get("matlib") and not preview:
        save_matlib(lib)
    tex, modes = layer_textures(opts["cfg"], lib)
    mat_tex = collage_textures(lib) if collage else None
    if collage:
        tex = {}
    clip = {k: (None if v in ("", None) else float(v)) for k, v in (opts.get("clip") or {}).items()}
    # 레벨·GL
    gl = opts.get("gl")
    gl = None if gl in ("", None) else float(gl)
    lv = opts.get("levels")
    if lv is None:
        sm = analyze.summary(parts, opts["cfg"])
        lv = sm["levels"]
        gl = sm["gl"] if gl is None else gl
    levels = [(l["name"], float(l["z"])) for l in lv if l.get("show", True) and l.get("name")]
    rs = []
    for i, (name, bearing) in enumerate(vl):
        base = i / len(vl)
        step = lambda msg, b=base: chk(msg, frac=0.05 + 0.85 * b)
        r = el.run(name, bearing=bearing, sun=sun if sun.get("mode") == "real" else (45.0, 45.0), progress=step,
                   textures=tex, fade=lib.get("fade"), clip=clip, glass_through=bool(opts.get("glass")),
                   depth_tone=opts.get("tone", True) is not False, keep_buffers=collage)
        r["modes"] = modes
        if collage:
            step(f"{name}: 콜라주 그리기")
            r["collage"] = CL.compose(r, scale, mat_tex, opts.get("collage"), parts=tree_parts,
                                      north=float(opts.get("north", 0) or 0), user_trees=os.path.join(HOME, "_점경"), log=log)
            r.pop("buf", None)
        if el.px_used > px * 1.01:
            log(f"{name}: 메모리 한도 때문에 한 칸을 {el.px_used * 1000:.0f} mm로 키웠어요")
        rs.append(r)
    tags = legend_for(rs) if opts.get("tags", True) else {}
    title = lambda r: r["name"]
    sub = f"벽에 맞춤(방위 {wall:+.1f}°)" if opts.get("dirmode") == "wall" and wall else ""
    anns = {}
    leg_all = [(n, c, draw.MAT_STYLE.get(c, draw.MAT_STYLE["기타"])[0]) for c, (n, _) in tags.items()]
    for r in rs:
        g = gl if gl is not None else r["gl"]
        tp = draw.tag_points(r, scale, tags) if tags else []
        shown = {p[2] for p in tp}  # 이 면에 번호가 실제로 붙은 재료만 범례에
        elb = opts.get("el_base")
        anns[r["name"]] = dict(style="collage" if collage else "standard",
                               el_base=None if elb in ("", None) else float(elb), levels=levels, gl=g, dims=opts.get("dims", True), grids=opts.get("grids", True),
                               grid_list=grid_names(r), title=title(r), subtitle=sub, tag_pts=tp,
                               legend=[l for l in leg_all if l[0] in shown], legend_all=leg_all)
    results = []
    for i, r in enumerate(rs):
        name = r["name"]; ann = anns[name]
        chk(f"{name}: 그리기·내보내기", frac=0.05 + 0.85 * (i + 0.9) / len(rs))
        fn = f"{name}_1-{scale}"
        files = []
        mp = [os.path.join(out, fn + ".png")]
        if "pdf" in fm:
            mp.append(os.path.join(out, fn + ".pdf"))
        draw.render_mpl(r, scale, mp, ann, dpi=110 if preview else 200)
        files += [os.path.basename(p) for p in mp if p.endswith(".pdf") or "png" in fm]
        if "svg" in fm:
            draw.export_svg(r, scale, os.path.join(out, fn + ".svg"), ann); files.append(fn + ".svg")
        std = dict(ann, style="standard")  # CAD 파일은 표준 표기(레벨선·해치)
        if "dxf" in fm:
            draw.export_dxf(r, scale, os.path.join(out, fn + ".dxf"), std); files.append(fn + ".dxf")
        if "3dm" in fm:
            draw.export_3dm(r, scale, os.path.join(out, fn + ".3dm"), std); files.append(fn + ".3dm")
        g = ann["gl"]
        results.append(dict(view=name, preview=fn + ".png", files=files, lines=int(len(r["lines"])),
                            px=round(r["px"] * 1000, 1),
                            levels=[[n, round((z - g) * 1000)] for n, z in sorted(levels, key=lambda t: t[1])],
                            grids=[gg["name"] for gg in ann["grid_list"]]))
    sheet = None
    if not preview and opts.get("sheet", True) and rs:
        chk("도면 시트 만들기", frac=0.95)
        sp = os.path.join(out, f"입면도_시트_1-{scale}.pdf")
        pname, suggest = draw.render_sheet(rs, scale, sp, lambda r: anns[r["name"]], paper=opts.get("paper", "A1"),
                                  project=opts.get("project") or stem, log=log)
        sheet = dict(file=os.path.basename(sp), paper=pname, asked=opts.get("paper", "A1"), suggest=suggest)
    if not preview:
        if "svg" in fm:
            open(os.path.join(out, "일러스트_레이어로.jsx"), "w", encoding="utf-8-sig").write(ILLUSTRATOR_JSX)
        json.dump(dict(model=info["file"], time=datetime.datetime.now().isoformat(timespec="seconds"),
                       options={k: v for k, v in opts.items() if k not in ("cfg", "matlib")}, results=results, sheet=sheet),
                  open(os.path.join(out, "만든_설정.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    bundle = None
    if not preview:
        # 결과 폴더 전체(텍스처 옆 폴더 포함)를 zip 하나로: DXF만 옮겨서 그림이 빠지는 일 막기
        import zipfile
        bundle = f"{os.path.basename(out)}_전체.zip"
        with zipfile.ZipFile(os.path.join(out, bundle), "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, fs in os.walk(out):
                for f in fs:
                    if f != bundle:
                        p = os.path.join(root, f)
                        zf.write(p, os.path.relpath(p, out))
    log(f"끝났어요 ({time.time() - t0:.0f}초)", frac=1.0)
    return out, dict(views=results, sheet=sheet, wall=wall, bundle=bundle)
