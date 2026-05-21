import os
import random
from dataclasses import dataclass, field

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

WIDTH = 700
HEIGHT = 900

ROAD_LEFT = 150
ROAD_RIGHT = 550
LANE_WIDTH = (ROAD_RIGHT - ROAD_LEFT) // 3
NUM_LANES = 3

CAR_W = 40
CAR_H = 70
CAR_Y = HEIGHT - 150

OBS_W = 40
OBS_H = 70
OBS_SPAWN_PROB = 0.08
OBS_MIN_SPEED = 8
OBS_MAX_SPEED = 16
MAX_OBS = 7


def lane_center_x(lane):
    return int(ROAD_LEFT + LANE_WIDTH * lane + LANE_WIDTH / 2)


@dataclass
class Obstacle:
    lane: int
    y: float
    speed: float

    @property
    def x(self):
        return lane_center_x(self.lane) - OBS_W // 2


@dataclass
class SimState:
    car_lane: int = 1
    car_x: float = field(default_factory=lambda: lane_center_x(1) - CAR_W // 2)
    obstacles: list = field(default_factory=list)
    step_count: int = 0
    score: int = 0
    crash_count: int = 0
    total_reward: float = 0.0
    scroll: int = 0
    last_reward: float = 0.0
    mode: str = "AI"


state = SimState()


def reset_state():
    global state
    state = SimState()
    return render(), status_text()


def rects_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def choose_ai_action():
    danger = {0: 9999, 1: 9999, 2: 9999}
    for obs in state.obstacles:
        dy = CAR_Y - (obs.y + OBS_H)
        if 0 < dy < danger[obs.lane]:
            danger[obs.lane] = dy

    current = state.car_lane
    if danger[current] < 180:
        candidates = [0, 1, 2]
        candidates.sort(key=lambda l: danger[l], reverse=True)
        target = candidates[0]
        if target < current:
            return "LEFT"
        if target > current:
            return "RIGHT"
    return "STRAIGHT"


def step(mode, human_action="STRAIGHT"):
    state.mode = mode
    state.step_count += 1
    state.scroll = (state.scroll + 8) % 70

    action = choose_ai_action() if mode == "AI" else human_action

    if action == "LEFT":
        state.car_lane = max(0, state.car_lane - 1)
    elif action == "RIGHT":
        state.car_lane = min(NUM_LANES - 1, state.car_lane + 1)

    target_x = lane_center_x(state.car_lane) - CAR_W // 2
    state.car_x += 0.35 * (target_x - state.car_x)

    if len(state.obstacles) < MAX_OBS and random.random() < OBS_SPAWN_PROB:
        lane = random.randint(0, NUM_LANES - 1)
        state.obstacles.append(Obstacle(lane, -OBS_H, random.uniform(OBS_MIN_SPEED, OBS_MAX_SPEED)))

    reward = 0.05
    new_obs = []
    for obs in state.obstacles:
        obs.y += obs.speed
        if obs.y > HEIGHT + 30:
            state.score += 10
            reward += 1.0
        else:
            new_obs.append(obs)
    state.obstacles = new_obs

    car_rect = (int(state.car_x), CAR_Y, CAR_W, CAR_H)
    crashed = False
    for obs in state.obstacles:
        obs_rect = (obs.x, int(obs.y), OBS_W, OBS_H)
        if rects_overlap(car_rect, obs_rect):
            crashed = True
            break

    if crashed:
        reward -= 15.0
        state.crash_count += 1
        state.obstacles.clear()
        state.car_lane = 1
        state.car_x = lane_center_x(1) - CAR_W // 2

    center_x = lane_center_x(state.car_lane) - CAR_W // 2
    reward += max(0.0, 0.3 * (1 - abs(state.car_x - center_x) / (LANE_WIDTH / 2)))

    state.last_reward = reward
    state.total_reward += reward


def run_once(mode, human_action):
    step(mode, human_action)
    return render(), status_text()


def run_many(mode, human_action, n_steps):
    for _ in range(int(n_steps)):
        step(mode, human_action)
    return render(), status_text()


def render():
    img = Image.new("RGB", (WIDTH, HEIGHT), (8, 10, 20))
    d = ImageDraw.Draw(img)

    d.rectangle([ROAD_LEFT, 0, ROAD_RIGHT, HEIGHT], fill=(28, 30, 45))
    d.line([ROAD_LEFT, 0, ROAD_LEFT, HEIGHT], fill=(80, 85, 110), width=3)
    d.line([ROAD_RIGHT, 0, ROAD_RIGHT, HEIGHT], fill=(80, 85, 110), width=3)

    for lane in range(1, NUM_LANES):
        x = ROAD_LEFT + LANE_WIDTH * lane
        y = -70 + state.scroll
        while y < HEIGHT:
            d.rectangle([x - 2, y, x + 2, y + 40], fill=(220, 220, 90))
            y += 70

    d.rectangle([0, 0, ROAD_LEFT - 5, HEIGHT], fill=(15, 18, 35))
    d.rectangle([ROAD_RIGHT + 5, 0, WIDTH, HEIGHT], fill=(15, 18, 35))

    for obs in state.obstacles:
        x, y = obs.x, int(obs.y)
        d.rounded_rectangle([x, y, x + OBS_W, y + OBS_H], radius=6, fill=(220, 50, 50))

    cx = int(state.car_x)
    d.rounded_rectangle([cx, CAR_Y, cx + CAR_W, CAR_Y + CAR_H], radius=6, fill=(0, 220, 180))
    d.rectangle([cx + 5, CAR_Y + 5, cx + 12, CAR_Y + 12], fill=(255, 255, 200))
    d.rectangle([cx + CAR_W - 12, CAR_Y + 5, cx + CAR_W - 5, CAR_Y + 12], fill=(255, 255, 200))

    d.text((18, 25), "AUTODRIVE", fill=(255, 50, 140))
    d.text((18, 55), "NO PYGAME", fill=(0, 220, 180))
    d.text((18, 110), f"MODE: {state.mode}", fill=(230, 235, 255))
    d.text((18, 140), f"STEP: {state.step_count}", fill=(230, 235, 255))
    d.text((18, 170), f"SCORE: {state.score}", fill=(230, 235, 255))
    d.text((18, 200), f"CRASH: {state.crash_count}", fill=(230, 235, 255))

    d.text((ROAD_RIGHT + 22, 25), "REWARD", fill=(255, 50, 140))
    d.text((ROAD_RIGHT + 22, 65), f"last {state.last_reward:+.3f}", fill=(230, 235, 255))
    d.text((ROAD_RIGHT + 22, 100), f"total {state.total_reward:+.1f}", fill=(230, 235, 255))

    return np.array(img)


def status_text():
    return (
        f"mode={state.mode} | step={state.step_count} | score={state.score} | "
        f"crash={state.crash_count} | last_reward={state.last_reward:+.3f} | "
        f"total_reward={state.total_reward:+.3f}"
    )


with gr.Blocks(title="Autodrive AI") as demo:
    gr.Markdown("# Autodrive AI — Render Safe Version")
    gr.Markdown("이 버전은 Render 배포 안정성을 위해 pygame 없이 Gradio + Pillow만 사용합니다.")

    image = gr.Image(value=render(), label="Simulation", type="numpy")
    status = gr.Textbox(value=status_text(), label="Status")

    with gr.Row():
        mode = gr.Radio(["AI", "HUMAN"], value="AI", label="Mode")
        human_action = gr.Radio(["LEFT", "STRAIGHT", "RIGHT"], value="STRAIGHT", label="Human Action")
        n_steps = gr.Slider(1, 200, value=20, step=1, label="Steps")

    with gr.Row():
        step_btn = gr.Button("Step")
        run_btn = gr.Button("Run Steps")
        reset_btn = gr.Button("Reset")

    step_btn.click(run_once, inputs=[mode, human_action], outputs=[image, status])
    run_btn.click(run_many, inputs=[mode, human_action, n_steps], outputs=[image, status])
    reset_btn.click(reset_state, outputs=[image, status])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
