# Canonical Graph Dataset Contract

이 문서는 모델별 실험이 공유해야 하는 graph JSON 형식을 정의합니다.

## 공통 입력 파일

```text
outputs/graphs/*.json
```

이 파일들은 `scripts/preprocess_sample.py`로 생성합니다.

```bash
python scripts/preprocess_sample.py \
  --root data/sample \
  --output outputs/graphs \
  --max-sequences 4 \
  --max-frames 200 \
  --frame-stride 10
```

검증은 다음으로 합니다.

```bash
python scripts/validate_preprocessing.py --graphs outputs/graphs --write-summary
```

## Scene-level fields

각 graph JSON 최상위에는 다음이 있습니다.

```text
scene_id
source_path
reference_track
node_feature_names
edge_feature_names
y
frames
temporal_edges
```

## Frame-level fields

각 frame에는 다음이 있습니다.

```text
frame_id
timestamp
node_ids
node_types
x
edge_index
edge_attr
edge_type
y
blind_node_indices
blind_y
```

## Node

현재 node type:

```text
ego_vehicle
car
truck
pedestrian
cyclist
e_scooter
occlusion_zone
```

`occlusion_zone`이 blind-zone 후보 node입니다.

## Node features

현재 node feature 순서:

```text
x
y
vx
vy
heading
object_type_id
distance_to_reference
relative_angle_to_reference
visibility
is_occluder
is_vulnerable_road_user
speed
acceleration
blind_zone_area
```

모델별 adapter는 이 feature 중 일부만 사용하거나, normalize해서 사용해도 됩니다. 단, 원본 graph JSON 자체를 모델별로 다르게 저장하지 않는 것을 권장합니다.

## Edge

현재 edge type:

```text
spatial_near
potential_conflict
occludes
blind_zone_relation
```

`temporal_next`는 scene-level `temporal_edges`에 저장됩니다.

## Edge features

현재 edge feature 순서:

```text
distance
relative_velocity_x
relative_velocity_y
relative_heading
time_to_collision
visibility_blocked
```

EIGCN 계열 모델은 이 `edge_attr`를 expert-informed attention bias로 사용하면 됩니다.

## Label

예측 대상은 frame 안의 blind-zone node입니다.

```text
blind_node_indices = 예측 대상 node index
blind_y = 각 blind-zone node의 label
```

```text
blind_y = 1
```

이면 해당 blind-zone 근처에 미래 window 안에서 scooter-like VRU가 등장했다는 뜻입니다.

```text
blind_y = 0
```

이면 그렇지 않은 경우입니다.

## 모델별 adapter 방향

공통 graph JSON을 아래처럼 변환해서 사용합니다.

```text
MR-GCN:
  edge_type → relation_id
  relation별 adjacency/message passing

EIGCN:
  edge_attr → expert feature / attention bias
  distance, TTC, relative heading 등을 사용

STGCN:
  frame sequence window 생성
  current frame의 blind_y 예측
```

