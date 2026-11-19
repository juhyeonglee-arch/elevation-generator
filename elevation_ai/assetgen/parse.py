"""자연어(한국어) → 에셋 설명(JSON 같은 dict) → 면 목록.

시험판은 낱말 규칙으로 읽는다(인터넷 없이 동작). 정식판에서는 이 자리에
Claude API 를 붙여 더 자유로운 문장을 같은 dict 로 바꾸게 할 예정.
"""
import re
from shapely import affinity
from . import person, objects
from .geom import union


def _num_mm(text, keys):
    """'키 170', '높이 8m', '170cm' 같은 표현에서 mm 값."""
    for k in keys:
        m = re.search(k + r"\s*(?:는|은|가|이)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|미터|센티)?", text)
        if m:
            return _to_mm(float(m.group(1)), m.group(2))
    return None


def _to_mm(v, unit):
    if unit in ("m", "미터"):
        return v * 1000
    if unit in ("cm", "센티"):
        return v * 10
    if unit == "mm":
        return v
    # 단위 없음: 크기로 짐작
    return v * 1000 if v < 30 else (v * 10 if v < 300 else v)


def parse(text):
    t = text.replace(" ", " ")
    spec = {"text": text}
    m = re.search(r"1\s*[:/]\s*(\d+)", t)
    spec["scale"] = int(m.group(1)) if m else 50
    spec["view"] = "plan" if re.search(r"평면|위에서", t) else ("front" if re.search(r"정면|앞모습", t) else "side")
    spec["style"] = "solid" if re.search(r"실루엣|검은|채운", t) else "line"
    dims = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)?\s*[x×*]\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)?", t)
    if dims:
        u = dims.group(2) or dims.group(4)
        spec["w"] = _to_mm(float(dims.group(1)), u)
        spec["d"] = _to_mm(float(dims.group(3)), u)

    if re.search(r"나무|수목|가로수|느티|소나무|은행|메타세|벚", t):
        spec["type"] = "tree"
        spec["kind"] = ("conifer" if re.search(r"소나무|침엽|전나무|구상|주목", t) else
                        "columnar" if re.search(r"메타세|은행|좁고|기둥형|포플러", t) else
                        "spread" if re.search(r"넓게|퍼진|느티|정자|벚", t) else "broad")
        spec["height"] = _num_mm(t, ["높이", "키", "수고", r"(?=\d)"]) or 8000
        spec["crown"] = _num_mm(t, ["수관폭", "수관", "폭", "너비"])
        m = re.search(r"(\d+)\s*(그루|개)", t)
        spec["count"] = int(m.group(1)) if m else 1
        return spec
    for key, typ in [("식탁|테이블|책상", "table"), ("소파|쇼파", "sofa"), ("침대", "bed")]:
        if re.search(key, t) and not re.search(r"앉", t):
            spec["type"] = typ
            m = re.search(r"(\d+)\s*인", t)
            spec["chairs"] = int(m.group(1)) if m else (4 if typ == "table" and re.search(r"의자", t) else 0)
            return spec
    if re.search(r"의자", t) and not re.search(r"사람|남자|여자|앉은|앉아", t):
        spec["type"] = "chair"
        return spec

    spec["type"] = "person"
    spec["sex"] = "f" if re.search(r"여자|여성|여학생|엄마|할머니|소녀", t) else "m"
    child = re.search(r"아이|어린이|아기|소녀|소년|꼬마", t)
    h = _num_mm(t, ["키", "신장", r"(?=\d+\s*(?:cm|센티))"])
    spec["height"] = h or (1150 if child else (1620 if spec["sex"] == "f" else 1740))
    spec["pose"] = ("run" if re.search(r"뛰|달리|조깅", t) else
                    "sit" if re.search(r"앉", t) else
                    "walk" if re.search(r"걷|걸어|산책|지나가", t) else
                    "phone" if re.search(r"휴대폰|핸드폰|폰|전화", t) else
                    "lean" if re.search(r"기대", t) else "stand")
    spec["bag"] = bool(re.search(r"가방|백팩|숄더", t))
    spec["skirt"] = "short" if re.search(r"짧은\s*치마|미니", t) else ("long" if re.search(r"치마|원피스", t) else None)
    spec["with_chair"] = spec["pose"] == "sit" and not re.search(r"벤치|계단|바닥", t)
    spec["bench"] = spec["pose"] == "sit" and bool(re.search(r"벤치", t))
    spec["facing"] = -1 if re.search(r"왼쪽", t) else 1
    return spec


def build(spec):
    typ = spec["type"]
    v = spec["view"]
    if typ == "person":
        if v == "plan":
            layers, _ = person.plan(spec["height"], spec["pose"])
            if spec.get("with_chair"):
                ch = [(affinity.rotate(g, 180, origin=(0, 0)), "far") for g, _ in objects.chair_plan()]
                ch = [(affinity.translate(g, spec["height"] * 0.06, 0), k) for g, k in ch]
                layers = ch + layers
        elif v == "front" and spec["pose"] == "stand":
            layers, _ = person.front(spec["height"], spec["sex"], spec["bag"], spec["skirt"])
        else:
            layers, meta = person.side(spec["height"], spec["sex"], spec["pose"], spec["bag"], spec["skirt"])
            if spec.get("with_chair") or spec.get("bench"):
                hip = meta["hip"]
                seat = meta["seat"]
                if spec.get("bench"):
                    furn = [(affinity.translate(objects.rect(-250, seat - 40, 250, seat, r=5), hip[0], 0), "far"),
                            (affinity.translate(objects.rect(-200, 0, -160, seat - 40), hip[0], 0), "far"),
                            (affinity.translate(objects.rect(160, 0, 200, seat - 40), hip[0], 0), "far")]
                else:
                    furn = [(affinity.translate(g, hip[0] + 40, 0), "far")
                            for g, _ in objects.chair_side(seat_h=seat, facing=1)]
                    # 등받이는 사람 뒤에 있으므로 사람보다 먼저 그림
                layers = furn + layers
        if spec.get("facing", 1) < 0:
            layers = [(affinity.scale(g, -1, 1, origin=(0, 0)), k) for g, k in layers]
        return layers
    if typ == "tree":
        out = []
        n = spec.get("count", 1)
        for i in range(n):
            if v == "plan":
                d = spec.get("crown") or spec["height"] * 0.65
                L = objects.tree_plan(d, seed=i + 3, kind=spec["kind"])
                dx = i * d * 1.1
            else:
                L = objects.tree_side(spec["height"] * (1 + 0.06 * ((i * 7) % 3 - 1)), spec.get("crown"),
                                      spec["kind"], seed=i + 3)
                dx = i * (spec.get("crown") or spec["height"] * 0.65) * 1.05
            out += [(affinity.translate(g, dx, 0), k) for g, k in L]
        return out
    if typ == "table":
        w, d = spec.get("w", 1200 if spec["chairs"] <= 4 else 1800), spec.get("d", 800)
        if v == "plan":
            return objects.table_plan(w, d, spec["chairs"])
        return objects.table_side(w)
    if typ == "sofa":
        return objects.sofa_plan(spec.get("w", 2000), spec.get("d", 900)) if v == "plan" else objects.sofa_side()
    if typ == "bed":
        return objects.bed_plan(spec.get("w", 1500), spec.get("d", 2000))
    if typ == "chair":
        return objects.chair_plan() if v == "plan" else objects.chair_side()
    raise ValueError(typ)
