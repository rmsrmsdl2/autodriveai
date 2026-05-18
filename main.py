# main.py — 진입점 & 메인 루프
#
# 파일 구조
# ─────────────────────────────────────────
#  config.py   — 모든 상수 & 하이퍼파라미터
#  reward.py   — 보상 함수 (독립 모듈)
#  env.py      — 자율주행 환경 (gym 스타일)
#  model.py    — Dueling DQN 신경망
#  agent.py    — DQNAgent (ε-greedy, Replay, Target Net, Double DQN)
#  renderer.py — pygame 렌더링 전담
#  main.py     — 진입점, 이벤트 루프

import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, C
from env import DrivingEnv
from agent import DQNAgent
from renderer import (
    init_fonts, draw_road, draw_car, draw_obstacles,
    draw_side_panel, draw_menu, draw_paused,
)


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Lane Keeper RL — Dueling DQN")
        self.clock = pygame.time.Clock()
        init_fonts()

        self.env   = DrivingEnv()
        self.agent = DQNAgent()
        self.agent.load()

        self.state   = "MENU"       # MENU | PLAYING
        self.mode    = 1            # 1:Human  2:AI  3:Train
        self.paused  = False
        self.speed   = 1            # 시뮬레이션 배속
        self.scroll  = 0            # 차선 스크롤 오프셋

    # ── 메인 루프 ─────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            if self.state == "MENU":
                running = self._menu_loop()
            else:
                running = self._game_loop()
            pygame.display.flip()
        pygame.quit()

    # ── 메뉴 ──────────────────────────────────────────────────────────

    def _menu_loop(self) -> bool:
        draw_menu(self.screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                key_map = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3}
                if event.key in key_map:
                    self.mode  = key_map[event.key]
                    self.state = "PLAYING"
                    self.env.reset()
                    self.scroll = 0
                    self.speed  = 1
        self.clock.tick(FPS)
        return True

    # ── 게임 루프 ────────────────────────────────────────────────────

    def _game_loop(self) -> bool:
        # 이벤트
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.state = "MENU"
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_s and self.mode == 3:
                    self.agent.save()
                    print("[Main] 모델 저장 완료")
                elif event.key == pygame.K_UP:
                    self.speed = min(self.speed * 10, 1000)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(self.speed // 10, 1)

        if not self.paused:
            for _ in range(self.speed):
                done = self._step()
                if done:
                    break

        # 렌더링 — 고속 시뮬레이션 중엔 5 episode마다 한 번만
        if self.speed < 100 or self.agent.episode % 5 == 0:
            self._render()

        if self.speed == 1:
            self.clock.tick(FPS)
        return True

    def _step(self) -> bool:
        obs = self.env._get_state()

        # 행동 선택
        if self.mode == 1:
            keys   = pygame.key.get_pressed()
            action = 0 if keys[pygame.K_LEFT] else 2 if keys[pygame.K_RIGHT] else 1
        else:
            train  = (self.mode == 3)
            action = self.agent.act(obs, train=train)

        next_obs, reward, done, info = self.env.step(action)

        # 학습 (Training 모드만)
        if self.mode == 3:
            self.agent.push(obs, action, reward, next_obs, done)
            self.agent.train_step()

        # 스크롤 애니메이션
        self.scroll = (self.scroll + 4) % 140

        if done:
            if self.mode == 3:
                self.agent.on_episode_end(self.env.total_reward, self.env.score)
            self.env.reset()
            return True
        return False

    def _render(self):
        mode_str = {1: "HUMAN", 2: "AI TEST", 3: "TRAINING"}[self.mode]

        draw_road(self.screen, self.scroll)
        draw_obstacles(self.screen, self.env.obstacles)
        draw_car(self.screen, self.env.car_rect)
        draw_side_panel(self.screen, self.agent, self.env, mode_str, self.speed)

        if self.paused:
            draw_paused(self.screen)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().run()
