"""입면도 자동 생성 화면: 브라우저에서 3dm을 넣고 → 레이어를 정리하고 → 만든다.

python -m elevation_ai   → http://127.0.0.1:8766
"""
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from . import analyze, jobs, model3dm
from .elevation import Cancelled

WEB = os.path.join(os.path.dirname(__file__), "web")
S = {"stage": "idle", "log": [], "frac": 0.0, "info": None, "cfg": None, "saved": False, "extra": {},
     "parts": None, "out": None, "results": None, "error": None, "path": None,
     "analysis": None, "plan": None, "preview": None, "notice": None}
LOCK = threading.Lock()
EL_CACHE = {}  # 합친 메시·모서리 이웃(모델이 바뀌면 비움)


def _errlog(where):
    r"""오류 내용을 사용자 폴더\입면도\_오류기록.txt 에 남긴다(보내 주기 쉽게)."""
    try:
        os.makedirs(jobs.HOME, exist_ok=True)
        p = os.path.join(jobs.HOME, "_오류기록.txt")
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {where} =====\n")
            f.write(f"파일: {S.get('path')}\n")
            f.write("\n".join(S["log"][-40:]) + "\n")
            f.write(traceback.format_exc())
        return p
    except Exception:
        return None


def _log(msg, frac=None):
    S["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
    S["log"] = S["log"][-200:]
    if frac is not None:
        S["frac"] = float(frac)


def _read(path):
    try:
        S.update(stage="reading", log=[], frac=0.0, error=None, info=None, results=None, out=None, path=path)
        EL_CACHE.clear()
        parts, info = model3dm.read(path, cache_dir=jobs.CACHE, log=_log)
        if not parts:
            raise ValueError("읽을 수 있는 3D 형상이 없어요(선만 있는 파일인지 확인해 주세요).")
        cfg, saved = jobs.load_cfg(info)
        _log("층 높이·벽 방향 살펴보는 중", 0.95)
        an = analyze.summary(parts, cfg)
        pp = os.path.join(jobs.CACHE, model3dm.file_key(path) + ".png")
        bbox, wh = analyze.plan_png(parts, pp, cfg)
        ex = jobs.load_extra(info)
        note = "같은 이름의 저장된 설정이 있지만 모델이 바뀌어서, 층 높이·GL·자르기는 새로 찾았어요." if ex.pop("_changed", False) else None
        S.update(parts=parts, info=info, cfg=cfg, saved=saved, extra=ex, analysis=an, notice=note,
                 plan=dict(file=pp, bbox=bbox, size=wh), preview=None, stage="layers", frac=1.0)
        jobs.tidy_folders(current=path)
        _log(f"레이어 {len(info['layers'])}개, 삼각형 {info['tris']:,}개를 읽었어요" + (" · 저장해 둔 레이어 설정을 불러왔어요" if saved else ""))
    except Exception as e:
        traceback.print_exc()
        p = _errlog("작업")
        S.update(stage="error", error=str(e) + (f"  (자세한 기록: {p})" if p else ""))


def _make(opts):
    pv = bool(opts.get("preview"))
    try:
        S.update(stage="making", frac=0.0, error=None, notice=None, log=[], **({} if pv else {"results": None}))
        extra = {k: opts[k] for k in jobs.EXTRA_KEYS if k in opts}
        jobs.save_cfg(S["info"], opts["cfg"], extra)
        S["cfg"] = opts["cfg"]; S["extra"] = extra
        out, res = jobs.make(S["parts"], S["info"], opts, log=_log, preview=pv, cache=EL_CACHE)
        if pv:
            v = res["views"][0]
            S.update(stage="layers", preview=dict(file=os.path.join(out, v["preview"]), view=v["view"], px=v["px"], t=time.time()))
        else:
            S.update(stage="done", out=out, results=res)
    except Cancelled:
        S.update(stage="layers", notice="취소했어요")
    except Exception as e:
        traceback.print_exc()
        p = _errlog("작업")
        S.update(stage="error", error=str(e) + (f"  (자세한 기록: {p})" if p else ""))


def open_folder(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if not os.path.isfile(path):
            return self._send(404, {"error": "없음"})
        ct = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            self._send(200, f.read(), ct)

    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        if p in ("/", "/index.html"):
            return self._file(os.path.join(WEB, "index.html"))
        if p.startswith("/web/"):
            return self._file(os.path.join(WEB, os.path.basename(p)))
        if p.startswith("/fonts/"):  # 화면 글꼴은 도면용 글꼴 파일을 같이 쓴다
            return self._file(os.path.join(os.path.dirname(WEB), "fonts", os.path.basename(p)))
        if p == "/status":
            return self._send(200, {k: S[k] for k in ("stage", "log", "frac", "info", "cfg", "saved", "extra", "out", "results", "error",
                                                       "analysis", "notice")}
                              | {"materials": jobs.MATERIALS, "views": list(jobs.VIEWS),
                                 "plan": {k: v for k, v in (S["plan"] or {}).items() if k != "file"} or None,
                                 "preview": {k: v for k, v in (S["preview"] or {}).items() if k != "file"} or None})
        if p == "/plan" and S["plan"]:
            return self._file(S["plan"]["file"])
        if p == "/preview" and S["preview"]:
            return self._file(S["preview"]["file"])
        if p == "/history":
            return self._send(200, jobs.history())
        if p.startswith("/hist/"):
            _, _, folder, fn = p.split("/", 3)
            return self._file(os.path.join(jobs.HOME, os.path.basename(folder), os.path.basename(fn)))
        if p == "/matlib":
            return self._send(200, jobs.load_matlib())
        if p.startswith("/tex/"):
            return self._file(os.path.join(jobs.MATLIB, os.path.basename(p)))
        if p.startswith("/out/") and S["out"]:
            return self._file(os.path.join(S["out"], os.path.basename(p)))
        return self._send(404, {"error": "없음"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0) or 0)
        if u.path == "/upload":
            name = os.path.basename(unquote(self.headers.get("X-Filename", "model.3dm")))
            if not name.lower().endswith(".3dm"):
                return self._send(400, {"error": "3dm 파일만 넣을 수 있어요"})
            os.makedirs(jobs.INPUT, exist_ok=True)
            dst = os.path.join(jobs.INPUT, name)
            left = n
            with open(dst, "wb") as f:
                while left > 0:
                    b = self.rfile.read(min(left, 4 << 20))
                    if not b:
                        break
                    f.write(b); left -= len(b)
            threading.Thread(target=_read, args=(dst,), daemon=True).start()
            return self._send(200, {"ok": True})
        if u.path == "/texture_upload":
            name = os.path.basename(unquote(self.headers.get("X-Filename", "texture.png")))
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
                return self._send(400, {"error": "그림 파일(png, jpg 등)만 넣을 수 있어요"})
            os.makedirs(jobs.MATLIB, exist_ok=True)
            with open(os.path.join(jobs.MATLIB, name), "wb") as f:
                f.write(self.rfile.read(n))
            return self._send(200, {"ok": True, "file": name, "lib": jobs.load_matlib()})
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/open_path":  # 파일 경로를 직접 붙여 넣은 경우(복사 없이 읽음)
            pth = body.get("path", "").strip().strip('"')
            if not os.path.isfile(pth) or not pth.lower().endswith(".3dm"):
                return self._send(400, {"error": "그 위치에 3dm 파일이 없어요"})
            threading.Thread(target=_read, args=(pth,), daemon=True).start()
            return self._send(200, {"ok": True})
        if u.path == "/save_layers":
            if not S["info"]:
                return self._send(400, {"error": "먼저 모델을 넣어 주세요"})
            S["cfg"] = body["cfg"]
            jobs.save_cfg(S["info"], body["cfg"], body.get("extra"))
            if body.get("matlib"):
                jobs.save_matlib(body["matlib"])
            S["saved"] = True
            return self._send(200, {"ok": True})
        if u.path == "/make":
            if S["stage"] in ("reading", "making"):
                return self._send(409, {"error": "다른 작업이 돌고 있어요"})
            if not S["parts"]:
                return self._send(400, {"error": "먼저 모델을 넣어 주세요"})
            threading.Thread(target=_make, args=(body,), daemon=True).start()
            return self._send(200, {"ok": True})
        if u.path == "/open_folder":
            f = os.path.join(jobs.HOME, os.path.basename(body["folder"])) if body.get("folder") else S["out"]
            if f and os.path.isdir(f):
                open_folder(f)
            return self._send(200, {"ok": True})
        if u.path == "/cancel":
            jobs.CANCEL.set()
            return self._send(200, {"ok": True})
        if u.path == "/analyze":  # 지형·넣기 설정이 바뀌면 층 후보를 다시
            if not S["parts"]:
                return self._send(400, {"error": "먼저 모델을 넣어 주세요"})
            S["analysis"] = analyze.summary(S["parts"], body.get("cfg") or S["cfg"])
            return self._send(200, S["analysis"])
        if u.path == "/back":
            if S["info"]:
                S.update(stage="layers", error=None)
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "없음"})


def main(port=8766, open_browser=True):
    os.makedirs(jobs.HOME, exist_ok=True)
    srv = None
    for p in range(port, port + 20):  # 이미 쓰는 포트면 다음 번호로
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), H)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit("빈 포트를 못 찾았어요")
    url = f"http://127.0.0.1:{port}"
    print(f"입면도 자동 생성 화면: {url}")
    print(f"결과 폴더: {jobs.HOME}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()
