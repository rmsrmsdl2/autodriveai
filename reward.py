# reward.py — 보상 함수를 독립 모듈로 분리
#
# 보상 설계 철학
# ─────────────────────────────────────────────────────────────────────
# 1. 생존 보상 (Survive)       : 매 step 소액. 오래 살아남는 것 자체가 가치.
# 2. 중앙 정렬 보상 (Center)   : 차선 중심에 가까울수록 커지는 연속 보상.
#    → 단순히 '차선 안에 있음'이 아니라 '얼마나 정중앙이냐'를 미분 가능하게.
# 3. 차선 변경 패널티 (Lane)   : 쓸데없는 좌우 이동을 억제.
#    → 매 step 패널티 대신 '이전 차선과 달라졌을 때'만 부과.
# 4. 벽 근접 패널티 (Wall)     : 도로 경계 근처에서 강한 음의 신호.
# 5. 장애물 근접 패널티 (Near) : 전방 위험 구간에 장애물이 있으면 경고.
# 6. 추월 보상 (Overtake)      : 장애물을 성공적으로 지나쳤을 때 +보상.
# 7. 충돌 패널티 (Collision)   : 에피소드 종료 + 큰 음수.

import math
import pygame
from config import (
    ROAD_LEFT, ROAD_RIGHT, CAR_WIDTH, CAR_HEIGHT,
    R_SURVIVE, R_CENTER_MAX, R_LANE_CHANGE,
    R_COLLISION, R_NEAR_WALL, R_NEAR_OBS,
    R_SAFE_OVERTAKE, R_WALL_DIST_THR,
    LANE_WIDTH, NUM_LANES,
)


def _lane_center(lane_idx: int) -> float:
    """lane_idx(0~2)의 차선 중앙 x 좌표 반환."""
    return ROAD_LEFT + LANE_WIDTH * lane_idx + LANE_WIDTH / 2


def compute_reward(
    car_rect: pygame.Rect,
    obstacles: list,
    prev_lane: int,
    curr_lane: int,
    collision: bool,
    overtaken_count: int,   # 이번 step에서 새로 추월한 장애물 수
) -> tuple[float, dict]:
    """
    Returns
    -------
    total_reward : float
    breakdown    : dict  (디버깅 / 시각화용 상세 내역)
    """
    bd = {}   # breakdown

    # 1. 생존 보상
    bd["survive"] = R_SURVIVE

    # 2. 차선 중앙 정렬 보상 (가우시안 형태)
    #    차선 중앙과의 거리가 0일 때 최대, 멀어질수록 급격히 감소
    cx = car_rect.centerx
    lane_cx = _lane_center(curr_lane)
    dist_to_center = abs(cx - lane_cx)
    sigma = LANE_WIDTH * 0.35          # 표준편차 → 감쇠 폭 조절
    center_reward = R_CENTER_MAX * math.exp(-0.5 * (dist_to_center / sigma) ** 2)
    bd["center"] = center_reward

    # 3. 차선 변경 패널티 (변경했을 때만)
    lane_penalty = R_LANE_CHANGE if prev_lane != curr_lane else 0.0
    bd["lane_change"] = lane_penalty

    # 4. 벽 근접 패널티
    dist_left  = car_rect.left  - ROAD_LEFT
    dist_right = ROAD_RIGHT     - car_rect.right
    wall_penalty = 0.0
    if dist_left < R_WALL_DIST_THR:
        wall_penalty += R_NEAR_WALL * (1 - dist_left / R_WALL_DIST_THR)
    if dist_right < R_WALL_DIST_THR:
        wall_penalty += R_NEAR_WALL * (1 - dist_right / R_WALL_DIST_THR)
    bd["wall"] = wall_penalty

    # 5. 장애물 근접 패널티
    #    전방 위험 구간 = 차량 y - 100px 이상 위에 있는 장애물 중 같은 차선
    near_penalty = 0.0
    DANGER_ZONE = 120   # px
    for obs in obstacles:
        dy = car_rect.top - obs.bottom   # 양수 = 내 차 위에 장애물
        if 0 < dy < DANGER_ZONE:
            dx = abs(car_rect.centerx - obs.centerx)
            if dx < LANE_WIDTH * 0.8:
                # 거리에 반비례한 패널티 (가까울수록 강함)
                near_penalty += R_NEAR_OBS * (1 - dy / DANGER_ZONE)
    bd["near_obs"] = near_penalty

    # 6. 추월 보상
    overtake_reward = R_SAFE_OVERTAKE * overtaken_count
    bd["overtake"] = overtake_reward

    # 7. 충돌 패널티
    collision_penalty = R_COLLISION if collision else 0.0
    bd["collision"] = collision_penalty

    total = sum(bd.values())
    bd["total"] = total
    return total, bd
