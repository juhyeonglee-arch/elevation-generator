"""사람 점경 — 실제 키(mm) 비율로 뼈대를 세우고 팔다리 굵기를 입혀 외곽선을 만든다.

보기: side(옆모습, 단면·입면), front(정면), plan(평면)
동작: stand, walk, sit, phone(휴대폰 보며 서기), lean(기대 서기), run
"""
import math
from .geom import capsule, ellipse, polar, union, rect, mirror_x
from shapely import affinity

# 동작별 관절 각도 (도). 0 = 수직 아래, + = 앞으로.
#   torso: 상체 기울기(+앞), th/kn: 허벅지 각, 무릎 굽힘, ua/el: 위팔 각, 팔꿈치 굽힘
POSES = {
    "stand": dict(torso=0, near=dict(th=3, kn=0, ua=4, el=8), far=dict(th=-2, kn=0, ua=-3, el=6)),
    "walk": dict(torso=3, near=dict(th=24, kn=8, ua=-20, el=18), far=dict(th=-14, kn=22, ua=18, el=30)),
    "run": dict(torso=10, near=dict(th=48, kn=70, ua=-40, el=95), far=dict(th=-22, kn=60, ua=45, el=90)),
    "sit": dict(torso=-6, near=dict(th=88, kn=90, ua=12, el=55), far=dict(th=84, kn=80, ua=8, el=60)),
    "phone": dict(torso=2, near=dict(th=2, kn=0, ua=12, el=115), far=dict(th=-3, kn=0, ua=4, el=10)),
    "lean": dict(torso=-8, near=dict(th=10, kn=6, ua=-6, el=12), far=dict(th=-6, kn=0, ua=-10, el=20)),
}


class Body:
    def __init__(self, height=1700, sex="m"):
        H = self.H = float(height)
        f = sex == "f"
        self.sex = sex
        self.L = dict(
            thigh=0.245 * H, shank=0.245 * H, ankle=0.042 * H, foot=0.148 * H,
            torso=0.285 * H, neck=0.045 * H, head=0.125 * H,
            uarm=0.186 * H, farm=0.146 * H, hand=0.095 * H,
        )
        s = 0.93 if f else 1.0
        self.R = dict(
            thigh=0.047 * H, knee=0.029 * H, calf=0.033 * H, ankle=0.019 * H,
            uarm=0.027 * H * s, elbow=0.022 * H * s, wrist=0.016 * H * s,
            pelvis=(0.072 * H, 0.06 * H), chest=(0.072 * H * (1.03 if f else 1), 0.095 * H),
        )


def _leg(b, hip, th, kn):
    L, R = b.L, b.R
    knee = polar(hip, L["thigh"], th)
    ankle = polar(knee, L["shank"], th - kn)
    g = union(capsule(hip, knee, R["thigh"], R["knee"]),
              capsule(knee, polar(knee, L["shank"] * 0.35, th - kn), R["knee"], R["calf"]),
              capsule(polar(knee, L["shank"] * 0.35, th - kn), ankle, R["calf"], R["ankle"]))
    # 발: 정강이 방향(down)과 그에 직각인 앞쪽(fwd) 기준으로 발바닥을 만든다
    sa = th - kn
    dn = polar((0, 0), 1, sa)
    fw = polar((0, 0), 1, sa + 90)
    La, F = L["ankle"], L["foot"]
    def at(d, f):
        return (ankle[0] + dn[0] * d + fw[0] * f, ankle[1] + dn[1] * d + fw[1] * f)
    heel = at(La * 0.55, -F * 0.2)
    toe = at(La * 0.72, F * 0.74)
    foot = union(capsule(heel, toe, R["ankle"] * 0.95, R["ankle"] * 0.55),
                 capsule(ankle, at(La * 0.4, F * 0.45), R["ankle"] * 1.05, R["ankle"] * 0.8))
    return union(g, foot), ankle, toe, heel


def _arm(b, sh, ua, el):
    L, R = b.L, b.R
    elbow = polar(sh, L["uarm"], ua)
    wrist = polar(elbow, L["farm"], ua + el)
    hand_tip = polar(wrist, L["hand"] * 0.85, ua + el + 4)
    g = union(capsule(sh, elbow, R["uarm"], R["elbow"]),
              capsule(elbow, wrist, R["elbow"], R["wrist"]),
              capsule(wrist, hand_tip, R["wrist"] * 1.15, R["wrist"] * 0.8))
    return g, wrist, hand_tip


def side(height=1700, sex="m", pose="stand", bag=False, skirt=None, seat_h=None):
    """옆모습. 오른쪽(+x)을 향함. 반환: 뒤→앞 순서의 면 목록 + 메타."""
    b = Body(height, sex)
    P = POSES[pose]
    L, R, H = b.L, b.R, b.H
    tl = math.radians(P["torso"])
    up = (math.sin(tl), math.cos(tl))

    def T(d, fwd=0.0):  # 엉덩이 기준 몸통 위쪽으로 d, 앞쪽으로 fwd
        return (hip[0] + up[0] * d + up[1] * fwd, hip[1] + up[1] * d - up[0] * fwd)

    hip = (0.0, 0.0)
    legs = {k: _leg(b, hip, P[k]["th"], P[k]["kn"]) for k in ("near", "far")}
    # 바닥 맞추기
    lowest = min(legs[k][0].bounds[1] for k in legs)
    dy = -lowest
    if pose == "run":
        dy += 0.03 * H
    hip = (0.0, dy)
    legs = {k: _leg(b, hip, P[k]["th"], P[k]["kn"]) for k in ("near", "far")}

    pelvis = ellipse(T(0.02 * H, 0.004 * H), *R["pelvis"], rot=-P["torso"])
    chest_c = T(L["torso"] * 0.72, 0.008 * H)
    chest = ellipse(chest_c, R["chest"][0], R["chest"][1], rot=-P["torso"])
    waist = capsule(T(0.03 * H), T(L["torso"] * 0.65), 0.064 * H, 0.068 * H)
    shoulder = T(L["torso"], -0.004 * H)
    neck_b = T(L["torso"] + 0.01 * H, 0.004 * H)
    neck_t = T(L["torso"] + L["neck"] + 0.01 * H, 0.008 * H)
    neck = capsule(neck_b, neck_t, 0.026 * H, 0.022 * H)
    hc = T(L["torso"] + L["neck"] + L["head"] * 0.5 + 0.004 * H, 0.012 * H)
    head = ellipse(hc, 0.052 * H, L["head"] * 0.5, rot=-P["torso"] - 8)
    # 턱·코를 아주 살짝 (건축 점경이므로 얼굴 표현은 최소)
    jaw = ellipse(T(L["torso"] + L["neck"] + 0.03 * H, 0.03 * H), 0.03 * H, 0.025 * H, rot=-P["torso"] - 20)
    hair = None
    if sex == "f":
        hair = union(ellipse((hc[0] - 0.034 * H, hc[1] - 0.03 * H), 0.034 * H, 0.06 * H, rot=-12))
    core = union(pelvis, chest, waist, neck, head, jaw, hair)
    if skirt:
        from shapely.geometry import Polygon
        k_len = 0.55 if skirt == "short" else 0.92
        hem = -L["thigh"] * k_len
        sk = Polygon([T(0.08 * H, -0.066 * H), T(0.08 * H, 0.07 * H), T(hem, 0.1 * H), T(hem, -0.085 * H)])
        core = union(core, sk.buffer(0.01 * H).buffer(-0.008 * H))
    arms = {k: _arm(b, shoulder, P[k]["ua"], P[k]["el"]) for k in ("near", "far")}

    layers = []  # (geometry, kind)
    layers.append((arms["far"][0], "far"))
    layers.append((legs["far"][0], "far"))
    layers.append((union(core, legs["near"][0]), "body"))
    if bag:
        bc = T(0.01 * H, -0.065 * H)
        layers.append((rect(bc[0] - 0.05 * H, bc[1] - 0.06 * H, bc[0] + 0.05 * H, bc[1] + 0.035 * H, r=0.008 * H), "prop"))
    layers.append((arms["near"][0], "body"))
    if pose == "phone":
        w, tip = arms["near"][1], arms["near"][2]
        ph = affinity.rotate(rect(-0.008 * H, -0.04 * H, 0.008 * H, 0.04 * H, r=0.003 * H), -60)
        ph = affinity.translate(ph, tip[0] + 0.005 * H, tip[1])
        layers.append((ph, "prop"))
    return layers, dict(hip=hip, wrist=arms["near"][1], seat=hip[1] - R["pelvis"][1] * 0.9)


def front(height=1700, sex="m", bag=False, skirt=None):
    """정면 서 있는 모습(좌우 대칭)."""
    H = float(height)
    sw = 0.255 * H * (0.92 if sex == "f" else 1)
    hipw = 0.19 * H * (1.04 if sex == "f" else 1)
    parts_r = []
    hip_y = 0.53 * H
    # 다리 (오른쪽 하나 만들고 대칭)
    hipj = (hipw * 0.25, hip_y)
    knee = (0.052 * H, 0.285 * H)
    ank = (0.048 * H, 0.045 * H)
    leg = union(capsule(hipj, knee, 0.058 * H, 0.033 * H), capsule(knee, ank, 0.035 * H, 0.02 * H),
                capsule(ank, (0.062 * H, 0.012 * H), 0.022 * H, 0.016 * H))
    sh = (sw * 0.42, 0.815 * H)
    elbow = (sw * 0.5 + 0.012 * H, 0.63 * H)
    wrist = (sw * 0.5 + 0.02 * H, 0.475 * H)
    arm = union(capsule(sh, elbow, 0.03 * H, 0.023 * H), capsule(elbow, wrist, 0.023 * H, 0.016 * H),
                capsule(wrist, (wrist[0] + 0.004 * H, wrist[1] - 0.075 * H), 0.019 * H, 0.013 * H))
    torso = union(
        capsule((hipw * 0.22, hip_y + 0.01 * H), (sw * 0.36, 0.79 * H), 0.075 * H, 0.05 * H),
        rect(-0.001, hip_y - 0.03 * H, sw * 0.3, 0.8 * H, r=0.03 * H),
    )
    torso = union(torso, capsule((0, 0.79 * H), (sh[0], sh[1] - 0.005 * H), 0.04 * H, 0.032 * H))
    half = union(leg, torso)
    body = union(half, mirror_x(half))
    neck = capsule((0, 0.82 * H), (0, 0.875 * H), 0.036 * H, 0.032 * H)
    head = ellipse((0, 0.935 * H), 0.047 * H, 0.064 * H)
    hair = ellipse((0, 0.925 * H), 0.058 * H, 0.07 * H) if sex == "f" else None
    core = union(body, neck, head)
    if skirt:
        k = 0.36 * H if skirt != "short" else 0.42 * H
        core = union(core, union(rect(-hipw * 0.5, k, hipw * 0.5, hip_y, r=0.01 * H)).convex_hull)
    layers = []
    if hair is not None:
        layers.append((hair, "far"))
    layers.append((core, "body"))
    layers.append((union(arm, mirror_x(arm)), "body"))
    if bag:
        layers.append((rect(sw * 0.48, 0.44 * H, sw * 0.48 + 0.07 * H, 0.54 * H, r=0.008 * H), "prop"))
    return layers, {}


def plan(height=1700, pose="stand", heading=0):
    """평면(위에서 본) 사람. heading: 바라보는 방향(도, 0=오른쪽)."""
    H = float(height)
    sw = 0.255 * H
    layers = []
    if pose == "sit":
        legs = union(capsule((0.02 * H, 0.05 * H), (0.27 * H, 0.055 * H), 0.05 * H, 0.035 * H),
                     capsule((0.02 * H, -0.05 * H), (0.27 * H, -0.055 * H), 0.05 * H, 0.035 * H),
                     capsule((0.27 * H, 0.055 * H), (0.31 * H, 0.06 * H), 0.03 * H, 0.025 * H),
                     capsule((0.27 * H, -0.055 * H), (0.31 * H, -0.06 * H), 0.03 * H, 0.025 * H))
        layers.append((legs, "far"))
    elif pose in ("walk", "run"):
        s = 0.13 * H if pose == "walk" else 0.22 * H
        feet = union(capsule((s * 0.5, 0.045 * H), (s * 0.5 + 0.1 * H, 0.045 * H), 0.024 * H, 0.02 * H),
                     capsule((-s * 0.5 - 0.02 * H, -0.045 * H), (-s * 0.5 + 0.08 * H, -0.045 * H), 0.024 * H, 0.02 * H))
        layers.append((feet, "far"))
    else:
        feet = union(capsule((0.0, 0.045 * H), (0.1 * H, 0.055 * H), 0.024 * H, 0.02 * H),
                     capsule((0.0, -0.045 * H), (0.1 * H, -0.055 * H), 0.024 * H, 0.02 * H))
        layers.append((feet, "far"))
    shoulders = ellipse((0, 0), 0.065 * H, sw * 0.5)
    arms = union(ellipse((0.01 * H, sw * 0.47), 0.04 * H, 0.03 * H), ellipse((0.01 * H, -sw * 0.47), 0.04 * H, 0.03 * H))
    if pose == "walk":
        arms = union(ellipse((0.05 * H, -sw * 0.48), 0.05 * H, 0.028 * H), ellipse((-0.04 * H, sw * 0.48), 0.05 * H, 0.028 * H))
    layers.append((union(shoulders, arms), "body"))
    layers.append((ellipse((0.012 * H, 0), 0.058 * H, 0.05 * H), "body"))
    if heading:
        layers = [(affinity.rotate(g, heading, origin=(0, 0)), k) for g, k in layers]
    return layers, {}
