"""
Step 5: Emergence 샘플 → GNN 그래프 데이터셋 → graph_dataset.pkl
각 샘플을 노드/엣지 피처 그래프로 변환합니다.

노드 피처 (8차원):
  [x, y, vx, vy, heading, type_id, speed, is_occluder]

엣지 피처 (5차원):
  [distance, rel_vx, rel_vy, rel_heading, visibility_blocked]

노드 타입:
  0=unknown, 1=ego_vehicle, 2=scooter, 3=bicycle,
  4=motorcycle, 5=vehicle(가상), 6=blind_zone
"""
import pickle, math
from pathlib import Path
from collections import Counter

# ── 설정 ──────────────────────────────────────────────────────
IN_PATH      = Path("data/generated/emergence_samples.pkl")
OUT_PATH     = Path("data/generated/graph_dataset.pkl")
EDGE_RADIUS  = 20.0   # 엣지 연결 최대 거리 (m)
OCC_RADIUS   = 2.5
BZ_DEPTH     = 25.0
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO  = 0.15  (나머지)

NODE_TYPE = {
    "ego_vehicle": 1,
    "scooter":     2,
    "bicycle":     3,
    "motorcycle":  4,
    "vehicle":     5,
    "blind_zone":  6,
}

NODE_FEATURE_NAMES = ["x", "y", "vx", "vy", "heading", "type_id", "speed", "is_occluder"]
EDGE_FEATURE_NAMES = ["distance", "rel_vx", "rel_vy", "rel_heading", "visibility_blocked"]

# ── 유틸 ──────────────────────────────────────────────────────
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def normalize_angle(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def compute_bz_centroid(ego_pos, occ_pos, occ_r=OCC_RADIUS, depth=BZ_DEPTH):
    ex, ey = ego_pos; ox, oy = occ_pos
    dist = math.hypot(ox - ex, oy - ey)
    if dist <= occ_r:
        return None
    ac = math.atan2(oy - ey, ox - ex)
    cx = ox + (depth * 0.6) * math.cos(ac)
    cy = oy + (depth * 0.6) * math.sin(ac)
    return (cx, cy)


# ── 그래프 변환 ───────────────────────────────────────────────
def sample_to_graph(sample: dict) -> dict | None:
    snap      = sample["frame_snap"]
    pm        = snap["pm"]
    vehicles  = snap["vehicles"]
    ego_pos   = snap["ego_pos"]
    occ_pos   = snap["occ_pos"]

    if ego_pos is None or occ_pos is None:
        return None

    nodes = []  # {"x","y","vx","vy","heading","type_id","speed","is_occluder"}

    # ── 노드 1: ego 차량 ──────────────────────────────────────
    ego_veh = next(
        (v for v in vehicles if v.get("active", True) and
         abs(v["x"] - ego_pos[0]) < 0.1 and abs(v["y"] - ego_pos[1]) < 0.1),
        None
    )
    ego_spd     = ego_veh["speed"]   if ego_veh else 10.0
    ego_heading = ego_veh["heading"] if ego_veh else 0.0
    ego_vx      = ego_spd * math.cos(ego_heading)
    ego_vy      = ego_spd * math.sin(ego_heading)

    nodes.append({
        "x": ego_pos[0], "y": ego_pos[1],
        "vx": ego_vx,    "vy": ego_vy,
        "heading": ego_heading, "type_id": NODE_TYPE["ego_vehicle"],
        "speed": ego_spd, "is_occluder": 0,
    })

    # ── 노드 2: PM (킥보드/자전거) ────────────────────────────
    pm_heading = math.atan2(pm.get("vy", 0), pm.get("vx", 0)) if pm.get("vx") or pm.get("vy") else 0.0
    nodes.append({
        "x": pm["x"], "y": pm["y"],
        "vx": pm.get("vx", 0), "vy": pm.get("vy", 0),
        "heading": pm_heading,
        "type_id": NODE_TYPE.get(pm["class"], NODE_TYPE["bicycle"]),
        "speed": pm["speed"], "is_occluder": 0,
    })

    # ── 노드 3~N: 주변 차량 (occluder 포함) ──────────────────
    occ_node_idx = None
    for vi, veh in enumerate(vehicles):
        if not veh.get("active", True):
            continue
        vpos = (veh["x"], veh["y"])
        if abs(vpos[0] - ego_pos[0]) < 0.1 and abs(vpos[1] - ego_pos[1]) < 0.1:
            continue  # ego 제외
        is_occ = int(
            abs(vpos[0] - occ_pos[0]) < 0.5 and abs(vpos[1] - occ_pos[1]) < 0.5
        )
        nodes.append({
            "x": veh["x"], "y": veh["y"],
            "vx": veh["vx"], "vy": veh["vy"],
            "heading": veh["heading"], "type_id": NODE_TYPE["vehicle"],
            "speed": veh["speed"], "is_occluder": is_occ,
        })
        if is_occ:
            occ_node_idx = len(nodes) - 1

    # ── 노드 N+1: Blind Zone 가상 노드 ───────────────────────
    bz_center = compute_bz_centroid(ego_pos, occ_pos)
    if bz_center:
        nodes.append({
            "x": bz_center[0], "y": bz_center[1],
            "vx": 0, "vy": 0, "heading": 0,
            "type_id": NODE_TYPE["blind_zone"],
            "speed": 0, "is_occluder": 0,
        })
        bz_node_idx = len(nodes) - 1
    else:
        bz_node_idx = None

    if len(nodes) < 3:
        return None

    # ── 노드 피처 행렬 ────────────────────────────────────────
    x = [[n["x"], n["y"], n["vx"], n["vy"],
          n["heading"], float(n["type_id"]),
          n["speed"], float(n["is_occluder"])]
         for n in nodes]

    # ── 엣지 구성 (거리 기반) ─────────────────────────────────
    src_list, dst_list, edge_attr, edge_type = [], [], [], []

    for si in range(len(nodes)):
        for di in range(len(nodes)):
            if si == di:
                continue
            sn, dn = nodes[si], nodes[di]
            dist = euclidean((sn["x"], sn["y"]), (dn["x"], dn["y"]))
            if dist > EDGE_RADIUS:
                continue

            rel_vx      = dn["vx"] - sn["vx"]
            rel_vy      = dn["vy"] - sn["vy"]
            rel_heading = normalize_angle(dn["heading"] - sn["heading"])
            vis_blocked = float(
                sn["is_occluder"] == 1 and
                dn["type_id"] == NODE_TYPE.get(pm["class"], 3) and
                dist < 12.0
            )

            src_list.append(si)
            dst_list.append(di)
            edge_attr.append([dist, rel_vx, rel_vy, rel_heading, vis_blocked])

            # 엣지 타입
            if sn["type_id"] == NODE_TYPE["blind_zone"] or dn["type_id"] == NODE_TYPE["blind_zone"]:
                etype = "blind_zone_relation"
            elif sn["is_occluder"] and dn["type_id"] in (2, 3, 4) and dist < 10.0:
                etype = "occludes"
            elif dist < 5.0:
                etype = "potential_conflict"
            else:
                etype = "spatial_near"
            edge_type.append(etype)

    return {
        "scene_id":    sample["scene_id"],
        "split":       sample["split"],
        "pm_class":    sample["pm_class"],
        "label":       sample["label"],
        "x":           x,
        "edge_index":  [src_list, dst_list],
        "edge_attr":   edge_attr,
        "edge_type":   edge_type,
        "node_types":  [n["type_id"] for n in nodes],
        "bz_node_idx": bz_node_idx,
        "occ_node_idx": occ_node_idx,
        "meta": {
            "dist_to_ego": sample["dist_to_ego"],
            "hidden_sec":  sample["hidden_sec"],
            "ego_speed":   sample["ego_speed"],
            "pm_speed":    sample["pm_speed"],
        },
    }


# ── 메인 ─────────────────────────────────────────────────────
print("emergence_samples.pkl 로딩 중...")
with open(IN_PATH, "rb") as f:
    samples = pickle.load(f)
print(f"  샘플 수: {len(samples):,}")

print("그래프 변환 중...")
graphs = []
skip   = 0
for s in samples:
    g = sample_to_graph(s)
    if g is None:
        skip += 1
    else:
        graphs.append(g)

print(f"  변환 완료: {len(graphs):,}  스킵: {skip}")

# ── Train / Val / Test 분할 ───────────────────────────────────
# split 필드 기준으로 유지 (imptc_trajectory 원본 split 존중)
# 추가로 scene 단위 랜덤 분할
import random
random.seed(42)

by_split = {"train": [], "eval": [], "test": []}
for g in graphs:
    sp = g["split"]
    if sp not in by_split:
        by_split["train"].append(g)
    else:
        by_split[sp].append(g)

# eval → val 로 rename
dataset = {
    "train": by_split["train"],
    "val":   by_split["eval"],
    "test":  by_split["test"],
}

# ── 최종 통계 출력 ────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"GNN 그래프 데이터셋 완성")
print(f"{'='*55}")
for sp, gs in dataset.items():
    if not gs:
        continue
    n_pos = sum(g["label"] for g in gs)
    print(f"  {sp:5s}: {len(gs):,}개  positive={n_pos:,} ({100*n_pos/max(len(gs),1):.1f}%)")

all_graphs = [g for gs in dataset.values() for g in gs]
if all_graphs:
    n_nodes = [len(g["x"])          for g in all_graphs]
    n_edges = [len(g["edge_attr"])  for g in all_graphs]
    print(f"\n노드 수  평균={sum(n_nodes)/len(n_nodes):.1f}  최대={max(n_nodes)}")
    print(f"엣지 수  평균={sum(n_edges)/len(n_edges):.1f}  최대={max(n_edges)}")
    print(f"노드 피처 dim: {len(NODE_FEATURE_NAMES)}  → {NODE_FEATURE_NAMES}")
    print(f"엣지 피처 dim: {len(EDGE_FEATURE_NAMES)}  → {EDGE_FEATURE_NAMES}")
    etypes = Counter(et for g in all_graphs for et in g["edge_type"])
    print(f"엣지 타입: {dict(etypes.most_common())}")

# ── 저장 ─────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump({"dataset": dataset,
                 "node_feature_names": NODE_FEATURE_NAMES,
                 "edge_feature_names": EDGE_FEATURE_NAMES}, f)
print(f"\n저장 완료 → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
