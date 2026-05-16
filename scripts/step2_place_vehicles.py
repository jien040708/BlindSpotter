"""
Step 2: 가상 차량 생성 모듈
step3에서 import해서 사용합니다. 직접 실행하지 않습니다.
"""
import math, random


# ── 가상 차량 클래스 ──────────────────────────────────────────
class VirtualVehicle:
    """
    차선 waypoint를 따라 이동하는 가상 차량.
    """
    OCC_RADIUS = 2.5   # 차량 반경 (m), blind zone 계산에 사용

    def __init__(self, lane: dict, speed_stats: dict,
                 start_offset: float = 0.0, speed_override: float = None):
        """
        lane         : step1의 lanes.json 차선 딕셔너리
        speed_stats  : {"mean", "std", "p10", "p50", "p90"}
        start_offset : 차선 시작점에서의 초기 거리 오프셋 (m)
        """
        self.lane       = lane
        self.waypoints  = [(wp["x"], wp["y"]) for wp in lane["waypoints"]]
        self.speed      = speed_override if speed_override else self._sample_speed(speed_stats)
        self.wp_idx     = 0
        self.x, self.y  = self.waypoints[0]
        self.heading    = self._heading_to_next()
        self.active     = True

        # 초기 오프셋 적용
        if start_offset > 0:
            self._advance(start_offset)

    def _sample_speed(self, stats):
        """실제 속도 분포에서 샘플링 (p10~p90 범위 clamp)"""
        spd = random.gauss(stats["mean"], stats["std"] * 0.5)
        return max(stats["p10"], min(stats["p90"], spd))

    def _heading_to_next(self):
        if self.wp_idx + 1 >= len(self.waypoints):
            return self.heading if hasattr(self, "heading") else 0.0
        nx, ny = self.waypoints[self.wp_idx + 1]
        return math.atan2(ny - self.y, nx - self.x)

    def _advance(self, dist_m):
        """차량을 dist_m 만큼 앞으로 이동"""
        remaining = dist_m
        while remaining > 0 and self.active:
            if self.wp_idx + 1 >= len(self.waypoints):
                self.active = False
                break
            nx, ny = self.waypoints[self.wp_idx + 1]
            seg_len = math.hypot(nx - self.x, ny - self.y)
            if seg_len <= 0:
                self.wp_idx += 1
                continue
            if remaining < seg_len:
                ratio = remaining / seg_len
                self.x += ratio * (nx - self.x)
                self.y += ratio * (ny - self.y)
                remaining = 0
            else:
                self.x, self.y = nx, ny
                self.wp_idx += 1
                remaining -= seg_len
        self.heading = self._heading_to_next()

    def step(self, dt: float):
        """dt초 만큼 이동"""
        if not self.active:
            return
        self._advance(self.speed * dt)

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def vel(self):
        return (self.speed * math.cos(self.heading),
                self.speed * math.sin(self.heading))

    def state_dict(self):
        return {
            "x":       round(self.x, 3),
            "y":       round(self.y, 3),
            "vx":      round(self.vel[0], 3),
            "vy":      round(self.vel[1], 3),
            "speed":   round(self.speed, 3),
            "heading": round(self.heading, 4),
            "active":  self.active,
        }


# ── 가상 차량 세트 생성 ───────────────────────────────────────
def create_virtual_vehicles(lanes: list, speed_stats: dict,
                             n_per_lane: int = 2,
                             rng: random.Random = None) -> list[VirtualVehicle]:
    """
    각 차선에 n_per_lane 대의 가상 차량을 생성합니다.
    차량 간격은 차선 길이 / n_per_lane 으로 균등 배치.
    """
    if rng is None:
        rng = random.Random()

    vehicles = []
    for lane in lanes:
        lane_len = lane.get("length_m", 50.0)
        interval = lane_len / max(n_per_lane, 1)
        for i in range(n_per_lane):
            offset = interval * i + rng.uniform(0, interval * 0.3)
            spd    = None  # speed_stats에서 샘플링
            veh    = VirtualVehicle(lane, speed_stats, start_offset=offset, speed_override=spd)
            if veh.active:
                vehicles.append(veh)
    return vehicles


# ── 시뮬레이션: 타임스탬프별 차량 상태 시퀀스 생성 ───────────
def simulate_vehicles(lanes: list, speed_stats: dict,
                      duration_s: float, dt_s: float = 0.04,
                      n_per_lane: int = 2,
                      seed: int = None) -> list[list[dict]]:
    """
    duration_s 동안 가상 차량을 시뮬레이션하고
    타임스탬프별 차량 상태 리스트를 반환합니다.

    반환값: [ [step0의 차량 상태들], [step1의 차량 상태들], ... ]
    """
    rng = random.Random(seed)
    vehicles = create_virtual_vehicles(lanes, speed_stats, n_per_lane, rng)

    n_steps = int(duration_s / dt_s)
    timeline = []

    for _ in range(n_steps):
        frame_states = [v.state_dict() for v in vehicles if v.active]
        timeline.append(frame_states)
        for v in vehicles:
            v.step(dt_s)

    return timeline


if __name__ == "__main__":
    import json
    from pathlib import Path

    lanes_data  = json.loads(Path("data/generated/lanes.json").read_text(encoding="utf-8"))
    lanes       = lanes_data["lanes"]
    speed_stats = lanes_data["speed_stats"]

    timeline = simulate_vehicles(lanes, speed_stats, duration_s=5.0, n_per_lane=2, seed=0)
    print(f"시뮬레이션 {len(timeline)} 스텝")
    print(f"첫 스텝 차량 수: {len(timeline[0])}")
    for i, v in enumerate(timeline[0]):
        print(f"  차량 {i}: pos=({v['x']:.1f}, {v['y']:.1f})  speed={v['speed']:.1f}m/s  active={v['active']}")
