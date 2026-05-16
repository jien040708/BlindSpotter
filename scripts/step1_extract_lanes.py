"""
Step 1: 차선 경로 추출
imptc_samples 차량 트랙에서 교차로 차선 waypoint를 추출합니다.
"""
import json, math, random
from pathlib import Path
from collections import defaultdict
import statistics

# ── 설정 ──────────────────────────────────────────────────────
DATA_PATH  = Path("data/imptc_samples")
OUT_PATH   = Path("data/generated/lanes.json")
N_CLUSTERS = 6      # 차선 수 (교차로 진입/진출 방향)
MIN_TRACK_LEN = 5   # 최소 프레임 수 (너무 짧은 트랙 제외)
WAYPOINT_STEP = 10  # waypoint 간격 (프레임 수)

# ── 유틸 ──────────────────────────────────────────────────────
def load_vehicle_track(fp):
    try:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
    except Exception:
        return None
    td = data.get("track_data", {})
    if not td:
        return None
    frames = []
    for _, f in td.items():
        c = f.get("coordinates", [])
        if len(c) < 2:
            continue
        frames.append({
            "ts_us": int(f["ts"]),
            "x":     float(c[0]),
            "y":     float(c[1]),
            "speed": float(f.get("velocity", 0)),
        })
    frames.sort(key=lambda r: r["ts_us"])
    return frames if len(frames) >= MIN_TRACK_LEN else None


def track_heading(frames):
    """트랙 전체 이동 방향 (시작 → 끝 벡터)"""
    dx = frames[-1]["x"] - frames[0]["x"]
    dy = frames[-1]["y"] - frames[0]["y"]
    return math.atan2(dy, dx)


def heading_distance(a, b):
    """두 heading 각도 차이 (0~pi)"""
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


# ── 차량 트랙 전체 수집 ───────────────────────────────────────
print("차량 트랙 수집 중...")
all_tracks = []
speed_samples = []

for seq_dir in sorted(DATA_PATH.iterdir()):
    if not seq_dir.is_dir():
        continue
    veh_dir = seq_dir / "vehicles"
    if not veh_dir.exists():
        continue
    for td in sorted(veh_dir.iterdir()):
        fp = td / "track.json"
        if not fp.exists():
            continue
        frames = load_vehicle_track(str(fp))
        if frames:
            all_tracks.append(frames)
            speed_samples.extend(f["speed"] for f in frames if f["speed"] > 1.0)

print(f"  트랙 수: {len(all_tracks)}")
print(f"  속도 샘플: {len(speed_samples)}개  "
      f"평균={statistics.mean(speed_samples):.1f}  "
      f"std={statistics.stdev(speed_samples):.1f}  "
      f"min={min(speed_samples):.1f}  "
      f"max={max(speed_samples):.1f} m/s")

# ── heading 기반 클러스터링 ───────────────────────────────────
print(f"\n{N_CLUSTERS}개 차선으로 클러스터링 중...")

headings = [track_heading(t) for t in all_tracks]

# K-Means (각도 공간)
def circular_kmeans(headings, k, n_iter=100):
    indices = random.sample(range(len(headings)), k)
    centers = [headings[i] for i in indices]
    labels  = [0] * len(headings)
    for _ in range(n_iter):
        labels = [
            min(range(k), key=lambda i: heading_distance(h, centers[i]))
            for h in headings
        ]
        new_centers = []
        for i in range(k):
            members = [headings[j] for j, l in enumerate(labels) if l == i]
            if not members:
                new_centers.append(centers[i])
            else:
                sin_mean = sum(math.sin(m) for m in members) / len(members)
                cos_mean = sum(math.cos(m) for m in members) / len(members)
                new_centers.append(math.atan2(sin_mean, cos_mean))
        if all(abs(new_centers[i] - centers[i]) < 1e-4 for i in range(k)):
            break
        centers = new_centers
    return labels, centers

random.seed(42)
labels, centers = circular_kmeans(headings, N_CLUSTERS)

for i, c in enumerate(centers):
    n = sum(1 for l in labels if l == i)
    deg = math.degrees(c)
    print(f"  차선 {i}: heading={deg:+.1f}°  트랙 수={n}")

# ── 차선별 waypoint 추출 ──────────────────────────────────────
print("\nWaypoint 추출 중...")
lanes = []

for lane_id in range(N_CLUSTERS):
    member_tracks = [t for t, lbl in zip(all_tracks, labels) if lbl == lane_id]
    if not member_tracks:
        continue

    # 대표 궤적: 가장 긴 트랙 기준
    ref_track = max(member_tracks, key=len)

    # waypoint: 일정 간격으로 샘플링
    waypoints = []
    for i in range(0, len(ref_track), WAYPOINT_STEP):
        f = ref_track[i]
        waypoints.append({"x": round(f["x"], 2), "y": round(f["y"], 2)})
    if len(ref_track) % WAYPOINT_STEP != 0:
        f = ref_track[-1]
        waypoints.append({"x": round(f["x"], 2), "y": round(f["y"], 2)})

    # 차선 길이 (m)
    total_len = sum(
        math.hypot(waypoints[i+1]["x"] - waypoints[i]["x"],
                   waypoints[i+1]["y"] - waypoints[i]["y"])
        for i in range(len(waypoints) - 1)
    )

    lanes.append({
        "lane_id":   lane_id,
        "heading":   round(math.degrees(centers[lane_id]), 1),
        "n_tracks":  len(member_tracks),
        "length_m":  round(total_len, 1),
        "waypoints": waypoints,
    })
    print(f"  차선 {lane_id}: {len(waypoints)} waypoints  길이={total_len:.1f}m")

# ── 속도 분포 통계 저장 ───────────────────────────────────────
def percentile(data, p):
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)

speed_stats = {
    "mean": round(statistics.mean(speed_samples), 2),
    "std":  round(statistics.stdev(speed_samples), 2),
    "p10":  round(percentile(speed_samples, 10), 2),
    "p50":  round(percentile(speed_samples, 50), 2),
    "p90":  round(percentile(speed_samples, 90), 2),
}

# ── 저장 ─────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result = {"lanes": lanes, "speed_stats": speed_stats}
OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n저장 완료 → {OUT_PATH}")
print(f"속도 통계: {speed_stats}")
print(f"총 차선 수: {len(lanes)}")
