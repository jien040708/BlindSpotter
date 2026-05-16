"""
Step 4: Blind Zone 계산 + Emergence 라벨링 → emergence_samples.pkl
virtual_scenes.pkl의 각 장면에서 blind zone emergence 이벤트를 탐지합니다.
"""
import pickle, math
from pathlib import Path
from collections import Counter

# ── 설정 ──────────────────────────────────────────────────────
IN_PATH      = Path("data/generated/virtual_scenes.pkl")
OUT_PATH     = Path("data/generated/emergence_samples.pkl")
OCC_RADIUS   = 2.5    # 차량 반경 (m)
BZ_DEPTH     = 25.0   # 사각지대 깊이 (m)
PRED_STEPS   = 5      # 출현 판단 미래 스텝 수
MIN_HIDDEN   = 2      # 최소 은폐 스텝 수 (노이즈 제거)

# ── Blind Zone 기하 ───────────────────────────────────────────
def compute_blind_zone(ego_pos, occ_pos, occ_r=OCC_RADIUS, depth=BZ_DEPTH):
    ex, ey = ego_pos
    ox, oy = occ_pos
    dist = math.hypot(ox - ex, oy - ey)
    if dist <= occ_r:
        return None
    ac = math.atan2(oy - ey, ox - ex)
    ha = math.asin(min(occ_r / dist, 1.0))
    al, ar = ac + ha, ac - ha
    return [
        (ox - occ_r * math.sin(ac), oy + occ_r * math.cos(ac)),
        (ox + depth * math.cos(al), oy + depth * math.sin(al)),
        (ox + depth * math.cos(ac), oy + depth * math.sin(ac)),
        (ox + depth * math.cos(ar), oy + depth * math.sin(ar)),
        (ox + occ_r * math.sin(ac), oy - occ_r * math.cos(ac)),
    ]


def point_in_polygon(point, polygon):
    if not polygon or len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def in_any_bz(ego_pos, vehicles, pm_pos):
    """PM이 어느 차량의 blind zone 안에 있는지 확인"""
    for veh in vehicles:
        if not veh.get("active", True):
            continue
        occ_pos = (veh["x"], veh["y"])
        poly = compute_blind_zone(ego_pos, occ_pos)
        if poly and point_in_polygon(pm_pos, poly):
            return True, occ_pos
    return False, None


def find_ego(vehicles):
    """가장 빠른 차량을 ego로 선택"""
    active = [v for v in vehicles if v.get("active", True)]
    if not active:
        return None
    return max(active, key=lambda v: v["speed"])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ── 라벨링 ───────────────────────────────────────────────────
print("virtual_scenes.pkl 로딩 중...")
with open(IN_PATH, "rb") as f:
    scenes = pickle.load(f)
print(f"  장면 수: {len(scenes)}")

samples = []
pos_count = 0
neg_count = 0

for scene in scenes:
    frames = scene["frames"]
    n      = len(frames)

    for fi, frame in enumerate(frames):
        if fi + PRED_STEPS >= n:
            break

        pm_pos   = (frame["pm"]["x"], frame["pm"]["y"])
        vehicles = frame["vehicles"]

        ego = find_ego(vehicles)
        if ego is None:
            continue
        ego_pos = (ego["x"], ego["y"])

        occ_vehs = [v for v in vehicles if v.get("active", True) and
                    (v["x"], v["y"]) != ego_pos]
        if not occ_vehs:
            continue

        # 현재 프레임에서 PM이 BZ 안에 있어야 샘플 생성
        in_bz, occ_pos = in_any_bz(ego_pos, occ_vehs, pm_pos)
        if not in_bz:
            continue

        # 미래 PRED_STEPS 안에 BZ 밖으로 나오면 label=1, 계속 안에 있으면 label=0
        label = 0
        for ff in frames[fi + 1: fi + PRED_STEPS + 1]:
            fp_pm  = (ff["pm"]["x"], ff["pm"]["y"])
            fp_ego = find_ego(ff["vehicles"])
            if fp_ego is None:
                continue
            fp_occ_vehs = [v for v in ff["vehicles"] if v.get("active", True) and
                           (v["x"], v["y"]) != (fp_ego["x"], fp_ego["y"])]
            still_in, _ = in_any_bz((fp_ego["x"], fp_ego["y"]), fp_occ_vehs, fp_pm)
            if not still_in:
                label = 1
                break

        dist_to_ego = euclidean(ego_pos, pm_pos)

        samples.append({
            "scene_id":    scene["scene_id"],
            "split":       scene["split"],
            "pm_class":    scene["pm_class"],
            "pm_id":       scene["pm_id"],
            "frame_idx":   fi,
            "label":       label,
            "ego_pos":     ego_pos,
            "occ_pos":     occ_pos,
            "pm_pos":      pm_pos,
            "dist_to_ego": round(dist_to_ego, 3),
            "hidden_sec":  round(fi * 0.04, 3),
            "ego_speed":   round(ego["speed"], 3),
            "pm_speed":    round(frame["pm"]["speed"], 3),
            "frame_snap":  {
                "pm":       frame["pm"],
                "vehicles": frame["vehicles"],
                "occ_pos":  occ_pos,
                "ego_pos":  ego_pos,
            },
        })

        if label == 1:
            pos_count += 1
        else:
            neg_count += 1

# ── 결과 출력 ─────────────────────────────────────────────────
total = len(samples)
print(f"\n{'='*55}")
print(f"총 샘플: {total:,}")
print(f"  Positive (출현): {pos_count:,}  ({100*pos_count/max(total,1):.1f}%)")
print(f"  Negative (은폐): {neg_count:,}  ({100*neg_count/max(total,1):.1f}%)")
print(f"{'='*55}")

if samples:
    print(f"\nPM 클래스별: {dict(Counter(s['pm_class'] for s in samples))}")
    print(f"Split별:     {dict(Counter(s['split']    for s in samples))}")
    dists = [s["dist_to_ego"] for s in samples]
    times = [s["hidden_sec"]  for s in samples]
    print(f"\nego까지 거리  평균={sum(dists)/len(dists):.1f}m  "
          f"최소={min(dists):.1f}m  최대={max(dists):.1f}m")
    print(f"은폐 시간     평균={sum(times)/len(times):.1f}s  "
          f"최소={min(times):.1f}s  최대={max(times):.1f}s")

# ── 저장 ─────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(samples, f)
print(f"\n저장 완료 → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
