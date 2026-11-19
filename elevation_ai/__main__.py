import argparse
import sys


def cli():
    ap = argparse.ArgumentParser(prog="elevation_ai", description="3dm → 동서남북 입면도")
    ap.add_argument("model", nargs="?", help="3dm 파일(없으면 화면이 열림)")
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--north", type=float, default=0.0, help="진북이 모델 +Y에서 시계방향 몇 도")
    ap.add_argument("--views", default="남측입면,북측입면,동측입면,서측입면")
    ap.add_argument("--quality", default="보통", choices=["빠르게", "보통", "세밀"])
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    if not a.model:
        from .app import main
        return main(a.port)
    from . import jobs, model3dm
    log = lambda m, frac=None: print(m, flush=True)
    parts, info = model3dm.read(a.model, cache_dir=jobs.CACHE, log=log)
    cfg, _ = jobs.load_cfg(info)
    for w in info["warnings"]:
        print("주의:", w)
    out, res = jobs.make(parts, info, dict(cfg=cfg, views=a.views.split(","), scale=a.scale, north=a.north,
                                           quality=a.quality, formats=["pdf", "svg", "dxf", "png"]), log=log)
    print("결과:", out)


if __name__ == "__main__":
    sys.exit(cli())
