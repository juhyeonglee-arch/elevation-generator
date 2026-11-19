"""윈도우 묶음(더블클릭 실행) 만들기. Actions의 windows 러너에서 돈다.

python packaging/build_windows.py --python <nuget python tools 폴더> --version v0.9 --out dist

결과: dist/elevation_windows_<버전>_<날짜>.zip
  입면도_윈도우/
    입면도 실행.bat, 먼저_읽어주세요.txt, 모델_체크리스트.md
    python/  (파이썬 3.12 + 필요한 라이브러리 + elevation_ai)

zip은 파이썬 zipfile로 묶는다: 한글 파일 이름에 UTF-8 표시(0x800)가 붙어야
윈도우 탐색기가 "압축 폴더가 올바르지 않습니다" 없이 연다.
"""
import argparse
import datetime
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOP = "입면도_윈도우"

# 실행에 필요 없는 것(크기 줄이기)
DROP_DIRS = {"__pycache__", "tests", "test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3",
             "include", "libs", "Scripts", "tcl", "pip", "setuptools", "_distutils_hack"}
DROP_FILES_SUFFIX = (".pdb", ".pyc", ".lib", ".a")


def slim(root):
    for d, subs, files in os.walk(root, topdown=True):
        for s in list(subs):
            if s in DROP_DIRS or s.startswith(("pip-", "setuptools-")):
                shutil.rmtree(os.path.join(d, s), ignore_errors=True)
                subs.remove(s)
        for f in files:
            if f.endswith(DROP_FILES_SUFFIX):
                os.remove(os.path.join(d, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True, help="nuget 'python' 패키지의 tools 폴더")
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()

    today = datetime.date.today()
    out = pathlib.Path(a.out)
    stage = out / "stage" / TOP
    shutil.rmtree(out / "stage", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    py = stage / "python"
    shutil.copytree(a.python, py)
    pyexe = py / "python.exe"
    subprocess.check_call([str(pyexe), "-m", "pip", "install", "--no-warn-script-location", "--no-compile",
                           "-r", str(ROOT / "requirements.txt")])
    site = py / "Lib" / "site-packages"
    dst = site / "elevation_ai"
    shutil.copytree(ROOT / "elevation_ai", dst, ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    if not (dst / "raster_win64.dll").exists():
        sys.exit("elevation_ai/raster_win64.dll 이 없음(native 빌드 산출물을 먼저 받아야 함)")

    for f in (ROOT / "packaging" / "windows").iterdir():
        data = f.read_bytes()
        if f.suffix == ".txt":
            data = data.replace("{VERSION}".encode(), a.version.encode()).replace("{DATE}".encode(), today.isoformat().encode())
        (stage / f.name).write_bytes(data)
    slim(py)

    zpath = out / f"elevation_windows_{a.version}_{today:%Y%m%d}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for d, _, files in os.walk(stage.parent):
            for f in sorted(files):
                p = pathlib.Path(d) / f
                zf.write(p, p.relative_to(stage.parent).as_posix())  # 폴더 항목은 넣지 않음
    with zipfile.ZipFile(zpath) as zf:
        bad = [i.filename for i in zf.infolist() if not i.filename.isascii() and not i.flag_bits & 0x800]
        if bad:
            sys.exit(f"UTF-8 표시가 없는 한글 이름: {bad[:5]}")
        n = len(zf.infolist())
    print(f"묶음: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB, 파일 {n}개)")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as fh:
            fh.write(f"zip={zpath.as_posix()}\nname={zpath.name}\nstage={stage.as_posix()}\n")


if __name__ == "__main__":
    main()
