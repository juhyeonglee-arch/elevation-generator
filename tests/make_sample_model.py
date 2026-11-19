"""시험용 합성 3dm 모델 만들기(개인 도면 대신 쓰는 가상의 작은 대지).

python tests/make_sample_model.py 출력.3dm

- 건물A(4층): 콘크리트 슬래브·벽, 유리 커튼월, 금속 멀리언
- 건물B(3층): 벽돌 벽, 창, 목재 루버
- 주변 건물(층 계산 제외), 경사진 지형, 나무 3그루
단위는 mm(라이노 기본과 같게).
"""
import math
import sys

import rhino3dm as r

MM = 1000.0


def box(x0, y0, z0, x1, y1, z1):
    m = r.Mesh()
    pts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
           (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    for p in pts:
        m.Vertices.Add(*(c * MM for c in p))
    for f in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
        m.Faces.AddFace(*f)
    m.Normals.ComputeNormals()
    return m


def cone_tree(cx, cy, z, h, rad, n=12):
    """줄기(가는 상자) + 수관(구에 가까운 다면체)."""
    trunk = box(cx - 0.15, cy - 0.15, z, cx + 0.15, cy + 0.15, z + h * 0.4)
    crown = r.Mesh()
    rings = 6
    zc = z + h * 0.65
    crown.Vertices.Add(cx * MM, cy * MM, (zc + rad) * MM)
    for i in range(1, rings):
        a = math.pi * i / rings
        for j in range(n):
            b = 2 * math.pi * j / n
            crown.Vertices.Add((cx + rad * math.sin(a) * math.cos(b)) * MM,
                               (cy + rad * math.sin(a) * math.sin(b)) * MM,
                               (zc + rad * math.cos(a)) * MM)
    crown.Vertices.Add(cx * MM, cy * MM, (zc - rad) * MM)
    last = 1 + (rings - 1) * n
    for j in range(n):
        crown.Faces.AddFace(0, 1 + j, 1 + (j + 1) % n)
        crown.Faces.AddFace(last, 1 + (rings - 2) * n + (j + 1) % n, 1 + (rings - 2) * n + j)
    for i in range(rings - 2):
        for j in range(n):
            a0, a1 = 1 + i * n + j, 1 + i * n + (j + 1) % n
            crown.Faces.AddFace(a0, a0 + n, a1 + n, a1)
    crown.Normals.ComputeNormals()
    return trunk, crown


def terrain(x0, y0, x1, y1, nx=24, ny=24):
    m = r.Mesh()
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * i / nx
            y = y0 + (y1 - y0) * j / ny
            z = -0.6 + 0.04 * (y - y0) + 0.25 * math.sin(x / 9.0)  # 북쪽으로 완만히 오르는 땅
            m.Vertices.Add(x * MM, y * MM, z * MM)
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            m.Faces.AddFace(a, a + 1, a + nx + 2, a + nx + 1)
    m.Normals.ComputeNormals()
    return m


def main(out):
    f = r.File3dm()
    f.Settings.ModelUnitSystem = r.UnitSystem.Millimeters

    def layer(name, rgb):
        L = r.Layer()
        L.Name = name
        L.Color = (*rgb, 255)
        return f.Layers.Add(L)

    lay = {
        "concrete": layer("A_concrete slab", (160, 160, 160)),
        "wall": layer("A_concrete wall", (140, 140, 140)),
        "glass": layer("A_glass", (120, 180, 220)),
        "frame": layer("A_metal frame", (60, 60, 60)),
        "brick": layer("B_brick wall", (170, 90, 70)),
        "bglass": layer("B_window glass", (120, 180, 220)),
        "louver": layer("B_wood louver", (180, 140, 90)),
        "context": layer("주변 건물", (200, 200, 200)),
        "terrain": layer("지형", (150, 170, 120)),
        "tree": layer("나무", (90, 140, 80)),
    }

    def add(mesh, key):
        a = r.ObjectAttributes()
        a.LayerIndex = lay[key]
        f.Objects.AddMesh(mesh, a)

    # 건물A: 14 x 10 m, 4층(층고 3.6 m, 1층 4.2 m)
    ax0, ay0, ax1, ay1 = 0.0, 0.0, 14.0, 10.0
    zs = [0.0, 4.2, 7.8, 11.4, 15.0]
    for k in range(len(zs)):
        add(box(ax0 - 0.3, ay0 - 0.3, zs[k] - 0.3, ax1 + 0.3, ay1 + 0.3, zs[k]), "concrete")
    for k in range(4):
        z0, z1 = zs[k], zs[k + 1] - 0.3
        add(box(ax0, ay0, z0, ax0 + 0.3, ay1, z1), "wall")          # 서쪽 벽
        add(box(ax1 - 0.3, ay0, z0, ax1, ay1, z1), "wall")          # 동쪽 벽
        add(box(ax0, ay1 - 0.3, z0, ax1, ay1, z1), "wall")          # 북쪽 벽
        add(box(ax0 + 0.3, ay0 + 0.05, z0, ax1 - 0.3, ay0 + 0.1, z1), "glass")  # 남쪽 커튼월
        x = ax0 + 0.3
        while x < ax1 - 0.3:                                          # 남쪽 멀리언 1.5 m 간격
            add(box(x, ay0 - 0.1, z0, x + 0.08, ay0 + 0.05, z1), "frame")
            x += 1.5
        for wx in (3.0, 7.0, 10.5):                                    # 동쪽 벽 창(벽 바깥에 붙인 유리)
            add(box(ax1, wx, z0 + 0.9, ax1 + 0.05, wx + 1.8, z1 - 0.4), "glass")
    add(box(ax0, ay0, 15.0, ax1, ay1, 16.2), "wall")                  # 옥상 파라펫 겸 기계실

    # 건물B: 동쪽으로 6 m 떨어진 10 x 8 m, 3층 벽돌
    bx0, by0, bx1, by1 = 20.0, 2.0, 30.0, 10.0
    for k in range(3):
        z0, z1 = k * 3.3, (k + 1) * 3.3
        add(box(bx0, by0, z0, bx1, by1, z1 - 0.25), "brick")
        add(box(bx0 - 0.2, by0 - 0.2, z1 - 0.25, bx1 + 0.2, by1 + 0.2, z1), "concrete")
        for wx in (21.5, 25.0, 28.0):
            add(box(wx, by0 - 0.05, z0 + 0.9, wx + 1.2, by0, z1 - 0.8), "bglass")
        for i in range(12):                                            # 동쪽 면 목재 루버
            y = by0 + 0.5 + i * 0.6
            add(box(bx1, y, z0 + 0.3, bx1 + 0.3, y + 0.08, z1 - 0.5), "louver")

    # 주변 건물(뒤쪽, 층 계산 제외 대상)
    add(box(-6.0, 18.0, 0.0, 8.0, 26.0, 21.0), "context")
    add(box(12.0, 20.0, 0.0, 34.0, 28.0, 9.0), "context")

    add(terrain(-10.0, -8.0, 40.0, 30.0), "terrain")

    for cx, cy, h in ((4.0, -4.0, 7.0), (17.0, -3.0, 6.0), (33.0, -2.0, 8.0)):
        t, c = cone_tree(cx, cy, -0.6, h, h * 0.28)
        add(t, "tree")
        add(c, "tree")

    if not f.Write(out, 7):
        raise SystemExit("3dm 저장 실패: " + out)
    print("시험 모델:", out, "객체", len(f.Objects))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample.3dm")
