"""예제 입력 시험: 합성 모델로 입면도를 실제로 만들고 결과 파일·그림을 확인한다."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from conftest import ROOT

REF = ROOT / "tests" / "reference"


def _read(sample_model):
    from elevation_ai import jobs, model3dm
    parts, info = model3dm.read(str(sample_model), cache_dir=jobs.CACHE, log=lambda *a, **k: None)
    cfg, _ = jobs.load_cfg(info)
    return parts, info, cfg


def _check_png(path):
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    assert im.shape[0] > 400 and im.shape[1] > 400, f"그림이 너무 작음: {im.shape}"
    dark = (im < 200).mean()
    assert 0.02 < dark < 0.6, f"그림이 비었거나 새카맣다(어두운 픽셀 비율 {dark:.3f})"
    return im


def _compare_reference(png, name, artifacts):
    """기준 그림과 비교. 글꼴 안티앨리어싱 차이 정도는 넘어가고, 그림이 크게 바뀌면 문제로 돌려준다.
    다르면 빨간 점으로 차이를 표시한 그림(diff_*.png)을 산출물에 남긴다.
    그림을 일부러 바꾼 PR에서는 tests/reference 의 기준 그림도 같이 바꾼다."""
    ref = REF / name
    if not ref.exists():
        return f"기준 그림 없음: {name}"
    a = np.asarray(Image.open(png).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(ref).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return f"{name}: 그림 크기가 기준과 다름 {a.shape[:2]} vs {b.shape[:2]}"
    mask = np.abs(a - b).max(axis=2) > 48
    diff = mask.mean()
    if diff >= 0.02:
        vis = (b * 0.35 + 165).astype(np.uint8)
        vis[mask] = (220, 30, 30)
        Image.fromarray(vis).save(artifacts / f"diff_{name}")
        return f"{name}: 기준 그림과 {diff * 100:.2f}% 픽셀이 다름(허용 2%)"
    return None


def test_native_raster_loaded():
    """윈도우 DLL / 리눅스 so 가 실제로 읽히는지(못 읽으면 느린 numpy로 돌아감)."""
    from elevation_ai import raster
    raster._load()
    assert raster.BACKEND == "c", f"네이티브 래스터 라이브러리를 못 읽음 → {raster.BACKEND}"


def test_read_model(sample_model):
    parts, info, cfg = _read(sample_model)
    names = {rw["name"] for rw in info["layers"]}
    assert {"A_glass", "B_brick wall", "지형", "나무", "주변 건물"} <= names
    assert cfg["지형"]["terrain"] and cfg["주변 건물"]["context"]
    assert cfg["나무"]["material"] == "수목"
    assert cfg["B_brick wall"]["material"] == "벽돌"
    assert info["tris"] > 1000


@pytest.mark.parametrize("style", ["normal", "collage"])
def test_make_elevations(sample_model, artifacts, style):
    from elevation_ai import jobs
    parts, info, cfg = _read(sample_model)
    opts = dict(cfg=cfg, views=["남측입면", "동측입면"], scale=200, north=0.0, quality="보통",
                formats=["pdf", "svg", "dxf", "png"])
    if style == "collage":
        opts["style"] = "collage"
    out, res = jobs.make(parts, info, opts, log=lambda *a, **k: None)
    out = pathlib.Path(out)
    problems = []
    assert [v["view"] for v in res["views"]] == ["남측입면", "동측입면"]
    for v in res["views"]:
        for ext in ("png", "pdf", "svg", "dxf"):
            f = out / f"{v['view']}_1-200.{ext}"
            assert f.exists() and f.stat().st_size > 1000, f"{f.name} 없음"
        assert len(v["levels"]) >= 3, "층 레벨을 못 찾음"
        png = out / v["preview"]
        _check_png(png)
        dst = artifacts / f"{style}_{v['view']}_1-200.png"
        shutil.copy(png, dst)
        msg = _compare_reference(png, dst.name, artifacts)
        if msg:
            problems.append(msg)
    assert (out / "입면도_시트_1-200.pdf").exists()
    assert res["bundle"] and (out / res["bundle"]).exists()
    svg = (out / "남측입면_1-200.svg").read_text(encoding="utf-8")
    assert svg.lstrip().startswith("<") and "</svg>" in svg
    import ezdxf
    doc = ezdxf.readfile(str(out / "남측입면_1-200.dxf"))
    assert len(doc.modelspace()) > 50
    assert not problems, "\n".join(problems)


def test_cli(sample_model, tmp_path):
    """명령줄 실행(python -m elevation_ai 모델.3dm)이 끝까지 도는지."""
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "elevation_ai", str(sample_model), "--scale", "300",
                        "--views", "북측입면", "--quality", "빠르게"],
                       cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "결과:" in p.stdout


def test_web_flow(sample_model):
    """브라우저 화면이 쓰는 서버 흐름: 첫 화면 → 모델 올리기 → 레이어 확인 → 남측 입면 만들기 → 결과 받기."""
    import threading
    import time
    import urllib.request
    from urllib.parse import quote
    from elevation_ai import app

    srv = app.ThreadingHTTPServer(("127.0.0.1", 0), app.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path):
        return urllib.request.urlopen(base + path, timeout=30).read()

    def post(path, data, headers=None):
        rq = urllib.request.Request(base + path, data=data, headers=headers or {}, method="POST")
        return json.loads(urllib.request.urlopen(rq, timeout=30).read())

    def wait(stages, limit=600):
        t = time.time()
        while time.time() - t < limit:
            st = json.loads(get("/status"))
            if st["stage"] in stages:
                return st
            assert st["stage"] != "error", st["error"]
            time.sleep(0.5)
        pytest.fail(f"시간 초과: {st['stage']}")

    try:
        assert "입면" in get("/").decode("utf-8")
        post("/upload", sample_model.read_bytes(), {"X-Filename": quote(sample_model.name)})
        st = wait({"layers"})
        assert len(st["info"]["layers"]) >= 8
        post("/make", json.dumps(dict(cfg=st["cfg"], views=["남측입면"], scale=200, quality="빠르게",
                                      formats=["png", "svg"])).encode())
        st = wait({"done"})
        v = st["results"]["views"][0]
        png = get("/out/" + quote(v["preview"]))
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        srv.shutdown()
