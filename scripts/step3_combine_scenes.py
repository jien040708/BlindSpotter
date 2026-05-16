"""
Step 3: VRU 트랙 + 가상 차량 조합 → virtual_scenes.pkl
imptc_trajectory의 PM 트랙 하나당 N회 다른 가상 차량 배치를 반복합니다.
"""
import json, pickle, math
from pathlib import Path
from step2_place_vehicles import simulate_vehicles

# ── 설정 ──────────────────────────────────────────────────────
TRAJ_PATH    = Path("data/imptc_trajectory")
LANES_PATH   = Path("data/generated/lanes.json")
OUT_PATH     = Path("data/generated/virtual_scenes.pkl")
SPLITS       = ["train", "eval", "test"]
PM_CLASSES   = {"scooter", "bicycle", "motorcycle"}
N_REPEAT     = 4      # 트랙당 가상 차량 배치 반복 횟수
N_PER_LANE   = 1      # 차선당 차량 수
MAX_TRACKS   = 500    # 사용할 PM 트랙 최대 수
DT_S         = 0.04   # 시뮬레이션 타임스텝 (40ms = 25Hz)
GRID_US      = 40_000 # 타임스탬프 그리드 (us)

# ── 유틸 ──────────────────────────────────────────────────────
def load_pm_track(fp, split):
    try:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
    except Exception:
        return None
    ov = data.get("overview", {})
    cls = ov.get("class_name", "").lower()
    if cls not in PM_CLASSES:
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
    if len(frames) < 10:
        return None

    # ts_grid 인덱스 생성
    pos_at = {}
    for fr in frames:
        tg = round(fr["ts_us"] / GRID_US) * GRID_US
        if tg not in pos_at:
            pos_at[tg] = (fr["x"], fr["y"], fr["speed"])

    duration_s = (frames[-1]["ts_us"] - frames[0]["ts_us"]) / 1e6

    return {
        "id":         Path(fp).parent.name,
        "split":      split,
        "class_name": cls,
        "duration_s": round(duration_s, 2),
        "pos_at":     pos_at,
        "ts_grid_set": sorted(pos_at.keys()),
    }


# ── PM 트랙 로드 ──────────────────────────────────────────────
print("PM 트랙 로딩 중...")
pm_tracks = []
for split in SPLITS:
    sp = TRAJ_PATH / split
    if not sp.exists():
        continue
    for td in sorted(sp.iterdir()):
        fp = td / "track.json"
        if not fp.exists():
            continue
        tr = load_pm_track(str(fp), split)
        if tr:
            pm_tracks.append(tr)

from collections import Counter
cls_cnt = Counter(t["class_name"] for t in pm_tracks)
print(f"  PM 트랙 총 {len(pm_tracks)}개: {dict(cls_cnt)}")

# 트랙 수 제한 (클래스 비율 유지)
if len(pm_tracks) > MAX_TRACKS:
    import random as _rnd
    _rnd.seed(42)
    pm_tracks = _rnd.sample(pm_tracks, MAX_TRACKS)
    print(f"  → {MAX_TRACKS}개로 샘플링")

# ── 차선/속도 통계 로드 ───────────────────────────────────────
lanes_data  = json.loads(LANES_PATH.read_text(encoding="utf-8"))
lanes       = lanes_data["lanes"]
speed_stats = lanes_data["speed_stats"]
print(f"  차선 수: {len(lanes)}")

# ── 장면 조합 ─────────────────────────────────────────────────
print(f"\n장면 조합 중... (트랙 {len(pm_tracks)}개 × {N_REPEAT}회 반복)")

scenes = []
for track_idx, pm in enumerate(pm_tracks):
    duration_s = pm["duration_s"]
    ts_list    = pm["ts_grid_set"]

    for rep in range(N_REPEAT):
        seed = track_idx * N_REPEAT + rep

        # 가상 차량 시뮬레이션 (PM 트랙 길이만큼)
        veh_timeline = simulate_vehicles(
            lanes, speed_stats,
            duration_s=duration_s,
            dt_s=DT_S,
            n_per_lane=N_PER_LANE,
            seed=seed,
        )

        # 타임스탬프 매핑: PM ts_grid → veh_timeline 스텝
        n_steps   = len(veh_timeline)
        ts_min    = ts_list[0]
        ts_max    = ts_list[-1]
        ts_span   = max(ts_max - ts_min, 1)

        def ts_to_step(ts_g):
            ratio = (ts_g - ts_min) / ts_span
            return int(max(0, min(n_steps - 1, ratio * n_steps)))

        # 장면 프레임 구성
        frames = []
        for ts_g in ts_list:
            if ts_g not in pm["pos_at"]:
                continue
            pm_x, pm_y, pm_spd = pm["pos_at"][ts_g]
            step_idx = ts_to_step(ts_g)
            veh_states = veh_timeline[step_idx]

            frames.append({
                "ts_g":      ts_g,
                "pm": {
                    "x":     pm_x,
                    "y":     pm_y,
                    "speed": pm_spd,
                    "class": pm["class_name"],
                },
                "vehicles":  veh_states,   # 가상 차량 상태 리스트
            })

        if len(frames) < 5:
            continue

        scenes.append({
            "scene_id":    f"{pm['split']}_{pm['id']}_rep{rep}",
            "split":       pm["split"],
            "pm_class":    pm["class_name"],
            "pm_id":       pm["id"],
            "rep":         rep,
            "frames":      frames,
        })

    if (track_idx + 1) % 100 == 0:
        print(f"  {track_idx + 1}/{len(pm_tracks)} 트랙 처리 완료  (장면 {len(scenes)}개)")

print(f"\n총 장면: {len(scenes)}개")
split_cnt = Counter(s["split"] for s in scenes)
print(f"Split별: {dict(split_cnt)}")

# ── 저장 ─────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(scenes, f)
print(f"\n저장 완료 → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
