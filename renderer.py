# renderer.py — 렌더링 전담 모듈 (게임 로직 없음)

import math
import pygame
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    ROAD_LEFT, ROAD_RIGHT, LANE_WIDTH, NUM_LANES, CAR_Y,
    C,
)

# 폰트는 모듈 수준에서 초기화 — pygame.init() 이후 호출해야 함
_fonts: dict = {}


def init_fonts():
    candidates = ["malgungothic", "malgun gothic", "applegothic", "nanum gothic", "gulim", None]
    for name in candidates:
        try:
            _fonts["lg"] = pygame.font.SysFont(name, 28, bold=True)
            _fonts["md"] = pygame.font.SysFont(name, 20)
            _fonts["sm"] = pygame.font.SysFont(name, 15)
            _fonts["xl"] = pygame.font.SysFont(name, 48, bold=True)
            break
        except Exception:
            continue


def _txt(screen, text, x, y, color, font_key="md", center=False):
    f = _fonts.get(font_key, _fonts.get("md"))
    surf = f.render(str(text), True, color)
    rect = surf.get_rect(center=(x, y)) if center else surf.get_rect(topleft=(x, y))
    screen.blit(surf, rect)


# ── 도로 렌더링 ──────────────────────────────────────────────────────

def draw_road(screen, scroll_offset: int):
    """배경 + 도로 + 차선 마킹."""
    screen.fill(C["bg"])

    # 도로 면
    pygame.draw.rect(screen, C["road"], (ROAD_LEFT, 0, ROAD_RIGHT - ROAD_LEFT, SCREEN_HEIGHT))

    # 양쪽 실선
    pygame.draw.line(screen, C["lane_solid"], (ROAD_LEFT, 0),  (ROAD_LEFT, SCREEN_HEIGHT),  3)
    pygame.draw.line(screen, C["lane_solid"], (ROAD_RIGHT, 0), (ROAD_RIGHT, SCREEN_HEIGHT), 3)

    # 차선 점선 (스크롤 애니메이션)
    dash_h, gap_h = 40, 30
    period = dash_h + gap_h
    for lane in range(1, NUM_LANES):
        x = ROAD_LEFT + lane * LANE_WIDTH
        y = -(period - scroll_offset % period)
        while y < SCREEN_HEIGHT:
            pygame.draw.rect(screen, C["lane_dash"], (x - 2, y, 4, dash_h))
            y += period


# ── 차량 렌더링 ──────────────────────────────────────────────────────

def draw_car(screen, car_rect: pygame.Rect, glow: bool = True):
    """내 차 — 글로우 효과 포함."""
    if glow:
        glow_surf = pygame.Surface((car_rect.w + 20, car_rect.h + 20), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*C["car_glow"], 40),
                         glow_surf.get_rect(), border_radius=12)
        screen.blit(glow_surf, (car_rect.x - 10, car_rect.y - 10))

    pygame.draw.rect(screen, C["car"], car_rect, border_radius=6)
    # 헤드라이트
    hw = 6
    pygame.draw.rect(screen, (255, 255, 200),
                     (car_rect.left + 4, car_rect.top + 4, hw, 5), border_radius=2)
    pygame.draw.rect(screen, (255, 255, 200),
                     (car_rect.right - 4 - hw, car_rect.top + 4, hw, 5), border_radius=2)


def draw_obstacles(screen, obstacles: list):
    for obs in obstacles:
        pygame.draw.rect(screen, C["obs"], obs, border_radius=5)
        # 테일라이트
        pygame.draw.rect(screen, (255, 30, 30),
                         (obs.left + 3, obs.bottom - 8, 8, 5), border_radius=2)
        pygame.draw.rect(screen, (255, 30, 30),
                         (obs.right - 11, obs.bottom - 8, 8, 5), border_radius=2)


# ── 사이드 패널 ──────────────────────────────────────────────────────

def draw_side_panel(screen, agent, env, mode_str: str, speed: int):
    """왼쪽 + 오른쪽 사이드 패널 — RL 정보 표시."""
    # ── 왼쪽 패널 ─────────────────────────────────────
    lw = ROAD_LEFT - 5
    pygame.draw.rect(screen, C["panel"], (0, 0, lw, SCREEN_HEIGHT))

    _txt(screen, "LANE KEEPER", lw // 2, 30, C["accent"], "lg", center=True)
    _txt(screen, "RL", lw // 2, 58, C["car"], "xl", center=True)

    y = 110
    items_left = [
        ("MODE",    mode_str),
        ("SPEED",   f"{speed}x"),
        ("EP",      str(agent.episode)),
        ("SCORE",   str(env.score)),
        ("STEPS",   str(env.step_count)),
    ]
    for label, val in items_left:
        _txt(screen, label, 10, y, C["text"], "sm")
        _txt(screen, val, lw - 10, y, C["accent"], "md")
        y += 30

    # ε bar
    y += 10
    _txt(screen, f"ε  {agent.epsilon:.3f}", 10, y, C["text"], "sm")
    y += 18
    bar_w = lw - 20
    pygame.draw.rect(screen, C["bar_bg"], (10, y, bar_w, 10), border_radius=5)
    pygame.draw.rect(screen, C["bar_fg"],
                     (10, y, int(bar_w * agent.epsilon), 10), border_radius=5)

    # Loss bar
    y += 22
    avg_loss = agent.avg_loss()
    _txt(screen, f"Loss {avg_loss:.4f}", 10, y, C["text"], "sm")
    y += 18
    loss_norm = min(avg_loss / 2.0, 1.0)
    pygame.draw.rect(screen, C["bar_bg"], (10, y, bar_w, 10), border_radius=5)
    pygame.draw.rect(screen, C["bar_loss"],
                     (10, y, int(bar_w * loss_norm), 10), border_radius=5)

    # ── 오른쪽 패널 ───────────────────────────────────
    rx = ROAD_RIGHT + 5
    rw = SCREEN_WIDTH - rx
    pygame.draw.rect(screen, C["panel"], (rx, 0, rw, SCREEN_HEIGHT))

    _txt(screen, "REWARD", rx + rw // 2, 30, C["accent"], "lg", center=True)

    bd = env.reward_breakdown
    if bd:
        ry = 70
        for key in ["survive", "center", "lane_change", "wall", "near_obs", "overtake", "collision", "total"]:
            val = bd.get(key, 0.0)
            color = C["reward_pos"] if val >= 0 else C["reward_neg"]
            if key == "total":
                pygame.draw.line(screen, C["lane_solid"], (rx + 5, ry - 3), (rx + rw - 5, ry - 3), 1)
                ry += 4
            _txt(screen, key[:8], rx + 5, ry, C["text"], "sm")
            _txt(screen, f"{val:+.3f}", rx + rw - 8, ry, color, "sm")
            # 작은 막대
            bar_len = int(min(abs(val) / 0.5, 1.0) * (rw - 12))
            bx = rx + 5
            pygame.draw.rect(screen, C["bar_bg"], (bx, ry + 14, rw - 12, 4), border_radius=2)
            pygame.draw.rect(screen, color, (bx, ry + 14, bar_len, 4), border_radius=2)
            ry += 30

    # 평균 보상 추이 (미니 그래프)
    _draw_mini_graph(screen, rx + 5, 370, rw - 10, 100,
                     list(agent.reward_history)[-60:], "Avg Reward")


def _draw_mini_graph(screen, x, y, w, h, data, label):
    _txt(screen, label, x, y, C["text"], "sm")
    y += 16
    pygame.draw.rect(screen, C["bar_bg"], (x, y, w, h - 16))
    if len(data) < 2:
        return
    mn, mx = min(data), max(data)
    rng = max(mx - mn, 1)
    pts = []
    for i, v in enumerate(data):
        px = x + int(i / (len(data) - 1) * (w - 2))
        py = y + (h - 16) - int((v - mn) / rng * (h - 20))
        pts.append((px, py))
    if len(pts) >= 2:
        pygame.draw.lines(screen, C["car"], False, pts, 2)


# ── 오버레이 ─────────────────────────────────────────────────────────

def draw_menu(screen):
    screen.fill(C["bg"])
    cx = SCREEN_WIDTH // 2

    # 타이틀
    _txt(screen, "LANE KEEPER RL", cx, 180, C["accent"], "xl", center=True)
    _txt(screen, "Reinforcement Learning Demo", cx, 235, C["text"], "md", center=True)

    items = [
        ("[1]  Human Play",                 C["text"]),
        ("[2]  AI  — 저장 모델 테스트",     C["text"]),
        ("[3]  Training — 강화학습 실행",   C["car"]),
    ]
    for i, (txt, col) in enumerate(items):
        _txt(screen, txt, cx, 350 + i * 60, col, "lg", center=True)

    tips = [
        "ESC : 메뉴로   P : 일시정지",
        "S : 모델 저장  ↑↓ : 속도 조절",
        "Human Play → ← 방향키",
    ]
    for i, t in enumerate(tips):
        _txt(screen, t, cx, 620 + i * 28, C["lane_solid"], "sm", center=True)


def draw_paused(screen):
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    screen.blit(overlay, (0, 0))
    _txt(screen, "PAUSED", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, C["accent"], "xl", center=True)
