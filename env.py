# env.py — 자율주행 차선 유지 환경 (OpenAI Gym 스타일 인터페이스)

import random
import numpy as np
import pygame
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    ROAD_LEFT, ROAD_RIGHT, ROAD_CENTER, LANE_WIDTH, NUM_LANES,
    CAR_WIDTH, CAR_HEIGHT, CAR_SPEED, CAR_Y,
    OBS_SPEED_MIN, OBS_SPEED_MAX, OBS_SPAWN_PROB, MAX_OBSTACLES,
    STATE_DIM, ACTION_DIM,
)
from reward import compute_reward


def _lane_center(lane_idx: int) -> int:
    return int(ROAD_LEFT + LANE_WIDTH * lane_idx + LANE_WIDTH / 2 - CAR_WIDTH / 2)


class DrivingEnv:
    """
    State  (12차원)
    ──────────────────────────────────────────────
    [0]  내 차 x 정규화        (0~1)
    [1]  내 차 현재 차선        (0~1 per lane)
    [2]  차선 중앙까지 거리     (-1~1)
    [3]  왼쪽 벽까지 거리      (0~1)
    [4]  오른쪽 벽까지 거리    (0~1)
    [5~11] 각 차선 × 전방 구간별 장애물 존재 여부 (7개 grid cell)

    Action (3)
    ──────────────────────────────────────────────
    0: 왼쪽 이동  1: 직진  2: 오른쪽 이동
    """

    def __init__(self):
        self.car_rect  = None
        self.obstacles = []
        self.step_count   = 0
        self.score        = 0
        self.total_reward = 0.0
        self._prev_lane   = 1
        self._passed_obs  = set()   # 이미 추월 보상 준 장애물 id
        self.reward_breakdown = {}
        self.reset()

    # ── 공개 인터페이스 ────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        start_lane = 1   # 항상 중앙 차선 시작
        x = _lane_center(start_lane)
        self.car_rect = pygame.Rect(x, CAR_Y, CAR_WIDTH, CAR_HEIGHT)
        self.obstacles    = []
        self.step_count   = 0
        self.score        = 0
        self.total_reward = 0.0
        self._prev_lane   = start_lane
        self._passed_obs  = set()
        self.reward_breakdown = {}
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Returns: (next_state, reward, done, info)
        """
        self.step_count += 1

        # ── 차량 이동 ────────────────────────────────────────────────
        if action == 0:
            self.car_rect.x = max(ROAD_LEFT, self.car_rect.x - CAR_SPEED)
        elif action == 2:
            self.car_rect.x = min(ROAD_RIGHT - CAR_WIDTH, self.car_rect.x + CAR_SPEED)
        # action == 1: 직진, x 변화 없음

        # ── 장애물 생성 & 이동 ─────────────────────────────────────
        self._spawn_obstacle()
        self._move_obstacles()

        # ── 현재 차선 판별 ─────────────────────────────────────────
        curr_lane = self._get_lane()

        # ── 추월 감지 ─────────────────────────────────────────────
        overtaken = self._count_overtaken()

        # ── 충돌 감지 ─────────────────────────────────────────────
        collision = any(self.car_rect.colliderect(o) for o in self.obstacles)

        # ── 보상 계산 ─────────────────────────────────────────────
        reward, breakdown = compute_reward(
            car_rect       = self.car_rect,
            obstacles      = self.obstacles,
            prev_lane      = self._prev_lane,
            curr_lane      = curr_lane,
            collision      = collision,
            overtaken_count= overtaken,
        )

        self.reward_breakdown = breakdown
        self.total_reward    += reward
        if overtaken:
            self.score += overtaken * 10

        done = collision or self.step_count >= 3000

        self._prev_lane = curr_lane
        info = {
            "step":     self.step_count,
            "score":    self.score,
            "lane":     curr_lane,
            "breakdown": breakdown,
        }
        return self._get_state(), reward, done, info

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _get_lane(self) -> int:
        cx = self.car_rect.centerx
        lane = int((cx - ROAD_LEFT) / LANE_WIDTH)
        return max(0, min(NUM_LANES - 1, lane))

    def _spawn_obstacle(self):
        if len(self.obstacles) < MAX_OBSTACLES and random.random() < OBS_SPAWN_PROB:
            lane = random.randint(0, NUM_LANES - 1)
            x    = _lane_center(lane) + (LANE_WIDTH - CAR_WIDTH) // 2
            speed= random.uniform(OBS_SPEED_MIN, OBS_SPEED_MAX)
            rect = pygame.Rect(x, -CAR_HEIGHT, CAR_WIDTH, CAR_HEIGHT)
            # speed를 rect에 직접 붙일 수 없으므로 별도 리스트 관리
            self.obstacles.append(rect)
            self._obs_speeds[id(rect)] = speed

    def _move_obstacles(self):
        remaining = []
        for obs in self.obstacles:
            spd = self._obs_speeds.get(id(obs), OBS_SPEED_MIN)
            obs.y += int(spd)
            if obs.top < SCREEN_HEIGHT + 20:
                remaining.append(obs)
            else:
                del self._obs_speeds[id(obs)]
        self.obstacles = remaining

    def _count_overtaken(self) -> int:
        count = 0
        for obs in self.obstacles:
            oid = id(obs)
            if oid not in self._passed_obs and obs.top > self.car_rect.bottom:
                self._passed_obs.add(oid)
                count += 1
        return count

    def _get_state(self) -> np.ndarray:
        s = np.zeros(STATE_DIM, dtype=np.float32)
        cx = self.car_rect.centerx

        s[0] = (cx - ROAD_LEFT) / (ROAD_RIGHT - ROAD_LEFT)   # 정규화 x
        s[1] = self._get_lane() / (NUM_LANES - 1)             # 차선

        lane_cx = ROAD_LEFT + self._get_lane() * LANE_WIDTH + LANE_WIDTH / 2
        s[2] = (cx - lane_cx) / (LANE_WIDTH / 2)             # 중앙 편차

        s[3] = (self.car_rect.left - ROAD_LEFT)  / (ROAD_RIGHT - ROAD_LEFT)
        s[4] = (ROAD_RIGHT - self.car_rect.right) / (ROAD_RIGHT - ROAD_LEFT)

        # 전방 7개 구간 × 3차선 그리드 — 장애물 있으면 1.0
        grid_zones = [50, 100, 150, 200, 300, 400, 600]
        for zi, zone in enumerate(grid_zones[:7]):
            for lane in range(NUM_LANES):
                lx = ROAD_LEFT + lane * LANE_WIDTH
                for obs in self.obstacles:
                    dy = self.car_rect.top - obs.bottom
                    if 0 < dy <= zone and lx <= obs.centerx <= lx + LANE_WIDTH:
                        s[5 + zi] = max(s[5 + zi], 1 - dy / zone)
                        break
        return s

    # ── 초기화 보조 ───────────────────────────────────────────────────
    def __init__(self):
        self._obs_speeds: dict[int, float] = {}
        self.car_rect     = None
        self.obstacles    = []
        self.step_count   = 0
        self.score        = 0
        self.total_reward = 0.0
        self._prev_lane   = 1
        self._passed_obs  = set()
        self.reward_breakdown = {}
        self.reset()
