"""
Blind Zone Emergence 분석 스크립트 (imptc_samples 기반)
킥보드/자전거가 차량 뒤 사각지대에서 갑자기 출현하는 케이스를 탐색합니다.
"""
import json, math, csv
from pathlib import Path
from collections import Counter

# ── 설정 ──────────────────────────────────────────────────────
DATA_PATH  = Path("data/imptc_samples")
PM_CLASSES = {"scooter", "bicycle", "motorcycle"}
OCC_RADIUS = 2.5   # 차량 반경 (m)
BZ_DEPTH   = 25.0  # 사각지대 깊이 (m)
GRID_US    = 40_000

# ── 유틸 ──────────────────────────────────────────────────────
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_blind_zone(ego_pos, occ_pos, occ_r=OCC_RADIUS, depth=BZ_DEPTH):
    ex, ey = ego_pos; ox, oy = occ_pos
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
    x, y = point; inside = False; j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi:
            inside = not inside
        j = i
    return inside

def point_in_any_bz(ego_pos, occ_positions, point):
    for occ in occ_positions:
        poly = compute_blind_zone(ego_pos, occ)
        if poly and point_in_polygon(point, poly):
            return True, occ
    return False, None

# ── 트랙 로드 ─────────────────────────────────────────────────
def load_track(fp):
    try:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
    except Exception:
        return None
    ov         = data.get("overview", {})
    class_name = ov.get("class_name", "unknown").lower()
    track_data = data.get("track_data", {})
    if not track_data:
        return None

    frames = []
    for _, f in track_data.items():
        coords = f.get("coordinates", [0, 0, 0])
        frames.append({
            "ts_us": int(f["ts"]),
            "x":     float(coords[0]),
            "y":     float(coords[1]),
            "speed": float(f.get("velocity", 0)),
        })
    frames.sort(key=lambda r: r["ts_us"])

    pos_at = {}
    for fr in frames:
        tg = round(fr["ts_us"] / GRID_US) * GRID_US
        if tg not in pos_at:
            pos_at[tg] = (fr["x"], fr["y"], fr["speed"])

    return {
        "id":          Path(fp).parent.name,
        "class_name":  class_name,
        "ts_min":      frames[0]["ts_us"] / 1e6,
        "ts_max":      frames[-1]["ts_us"] / 1e6,
        "pos_at":      pos_at,
        "ts_grid_set": set(pos_at.keys()),
    }

# ── 시퀀스별 분석 ─────────────────────────────────────────────
print("imptc_samples 분석 중...\n")

all_events = []
seq_dirs   = sorted(DATA_PATH.iterdir()) if DATA_PATH.exists() else []

for seq_dir in seq_dirs:
    if not seq_dir.is_dir() or seq_dir.name == "samples_overview.csv":
        continue

    vru_dir = seq_dir / "vrus"
    veh_dir = seq_dir / "vehicles"
    if not vru_dir.exists() or not veh_dir.exists():
        continue

    # 트랙 로드
    vru_tracks = [t for d in sorted(vru_dir.iterdir()) if (fp := d / "track.json").exists()
                  for t in [load_track(fp)] if t]
    veh_tracks = [t for d in sorted(veh_dir.iterdir()) if (fp := d / "track.json").exists()
                  for t in [load_track(fp)] if t]

    pm_tracks = [t for t in vru_tracks if t["class_name"] in PM_CLASSES]
    if not pm_tracks or not veh_tracks:
        continue

    # 공통 타임스탬프
    all_ts = sorted(set(ts for t in vru_tracks + veh_tracks for ts in t["ts_grid_set"]))

    for pm in pm_tracks:
        was_hidden = False
        hidden_occ = None
        hidden_ego = None
        hidden_pos = None
        hidden_ts  = None

        for ts_g in all_ts:
            if ts_g not in pm["pos_at"]:
                was_hidden = False
                continue

            pm_pos = pm["pos_at"][ts_g][:2]

            # 가장 빠른 차량 → ego (접근 중인 차량)
            active_veh = [(t, t["pos_at"][ts_g]) for t in veh_tracks if ts_g in t["pos_at"]]
            if not active_veh:
                continue

            ego_tr, ego_data = max(active_veh, key=lambda x: x[1][2])
            ego_pos = ego_data[:2]

            # 나머지 차량 = occluder 후보
            occ_positions = [d[:2] for t, d in active_veh if t["id"] != ego_tr["id"]]
            if not occ_positions:
                continue

            in_bz, occ_pos = point_in_any_bz(ego_pos, occ_positions, pm_pos)

            if in_bz and not was_hidden:
                was_hidden = True
                hidden_occ = occ_pos
                hidden_ego = ego_pos
                hidden_pos = pm_pos
                hidden_ts  = ts_g

            elif not in_bz and was_hidden:
                # Emergence!
                dist_to_ego  = euclidean(ego_pos, pm_pos)
                travel_dist  = euclidean(hidden_pos, pm_pos)
                hidden_time  = (ts_g - hidden_ts) / 1e6  # 초

                all_events.append({
                    "sequence":    seq_dir.name,
                    "pm_id":       pm["id"],
                    "pm_class":    pm["class_name"],
                    "ts_emerge":   ts_g,
                    "hidden_sec":  round(hidden_time, 2),
                    "ego_pos":     ego_pos,
                    "occ_pos":     hidden_occ,
                    "hidden_pos":  hidden_pos,
                    "emerge_pos":  pm_pos,
                    "dist_to_ego": round(dist_to_ego, 2),
                    "travel_dist": round(travel_dist, 2),
                })
                was_hidden = False

    print(f"  [{seq_dir.name}]  VRU {len(vru_tracks)}개 / 차량 {len(veh_tracks)}개 / "
          f"PM {len(pm_tracks)}개  →  이벤트 {sum(1 for e in all_events if e['sequence']==seq_dir.name)}건")

# ── 결과 출력 ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Blind Zone Emergence 이벤트 총 {len(all_events)}건")
print(f"{'='*60}")

if all_events:
    print(f"\n클래스별: {dict(Counter(e['pm_class'] for e in all_events))}")
    dists = [e['dist_to_ego'] for e in all_events]
    times = [e['hidden_sec']  for e in all_events]
    print(f"ego까지 거리  평균: {sum(dists)/len(dists):.1f}m  최소: {min(dists):.1f}m  최대: {max(dists):.1f}m")
    print(f"사각지대 은폐 시간  평균: {sum(times)/len(times):.1f}s  최소: {min(times):.1f}s  최대: {max(times):.1f}s")

    print(f"\n--- 위험 TOP 10 (ego 근접 순) ---")
    for e in sorted(all_events, key=lambda x: x["dist_to_ego"])[:10]:
        print(f"  [{e['sequence']}] {e['pm_class']} #{e['pm_id']}  "
              f"출현위치=({e['emerge_pos'][0]:.1f}, {e['emerge_pos'][1]:.1f})  "
              f"ego까지 {e['dist_to_ego']:.1f}m  "
              f"은폐시간 {e['hidden_sec']:.1f}s  이동 {e['travel_dist']:.1f}m")

    # CSV 저장
    out = Path("outputs/blindzone_emergence_events.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sequence","pm_id","pm_class","ts_emerge","hidden_sec",
            "ego_x","ego_y","occ_x","occ_y",
            "hidden_x","hidden_y","emerge_x","emerge_y",
            "dist_to_ego","travel_dist"
        ])
        w.writeheader()
        for e in all_events:
            w.writerow({
                "sequence":  e["sequence"], "pm_id": e["pm_id"], "pm_class": e["pm_class"],
                "ts_emerge": e["ts_emerge"], "hidden_sec": e["hidden_sec"],
                "ego_x": round(e["ego_pos"][0],2),  "ego_y": round(e["ego_pos"][1],2),
                "occ_x": round(e["occ_pos"][0],2),  "occ_y": round(e["occ_pos"][1],2),
                "hidden_x": round(e["hidden_pos"][0],2), "hidden_y": round(e["hidden_pos"][1],2),
                "emerge_x": round(e["emerge_pos"][0],2), "emerge_y": round(e["emerge_pos"][1],2),
                "dist_to_ego": e["dist_to_ego"], "travel_dist": e["travel_dist"],
            })
    print(f"\n결과 저장 → {out}")
else:
    print("\n이벤트 없음 — OCC_RADIUS/BZ_DEPTH 파라미터 조정 필요")
