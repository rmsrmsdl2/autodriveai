import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys
import importlib.util
from pathlib import Path

# 업로드 과정에서 env.py가 env(1).py로 바뀐 경우도 대응
if not Path("env.py").exists() and Path("env(1).py").exists():
    spec = importlib.util.spec_from_file_location("env", "env(1).py")
    env_mod = importlib.util.module_from_spec(spec)
    sys.modules["env"] = env_mod
    spec.loader.exec_module(env_mod)

import gradio as gr
from PIL import Image, ImageDraw

from config import SCREEN_WIDTH, SCREEN_HEIGHT, ROAD_LEFT, ROAD_RIGHT, LANE_WIDTH, NUM_LANES, CAR_WIDTH, CAR_HEIGHT, CAR_Y
from env import DrivingEnv
from agent import DQNAgent

# Render에서 한 프로세스 안에서 유지되는 전역 객체
ENV = DrivingEnv()
AGENT = DQNAgent()
try:
    AGENT.load()
except Exception as e:
    print(f"[App] saved model load skipped: {e}")

MODE = {"value": "Human"}
LAST_STATE = {"obs": ENV._get_state()}


def draw_frame():
    img = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (8, 10, 20))
    d = ImageDraw.Draw(img)

    # road
    d.rectangle([ROAD_LEFT, 0, ROAD_RIGHT, SCREEN_HEIGHT], fill=(28, 30, 45))
    d.line([ROAD_LEFT, 0, ROAD_LEFT, SCREEN_HEIGHT], fill=(60, 65, 90), width=3)
    d.line([ROAD_RIGHT, 0, ROAD_RIGHT, SCREEN_HEIGHT], fill=(60, 65, 90), width=3)
    for lane in range(1, NUM_LANES):
        x = ROAD_LEFT + lane * LANE_WIDTH
        for y in range(0, SCREEN_HEIGHT, 70):
            d.rectangle([x - 2, y, x + 2, y + 40], fill=(200, 200, 80))

    # obstacles
    for obs in ENV.obstacles:
        d.rounded_rectangle([obs.left, obs.top, obs.right, obs.bottom], radius=5, fill=(220, 50, 50))

    # car
    car = ENV.car_rect
    d.rounded_rectangle([car.left, car.top, car.right, car.bottom], radius=6, fill=(0, 220, 180))

    # panels text
    d.text((15, 20), "LANE KEEPER RL", fill=(255, 50, 140))
    d.text((15, 55), f"Mode: {MODE['value']}", fill=(230, 235, 255))
    d.text((15, 80), f"Step: {ENV.step_count}", fill=(230, 235, 255))
    d.text((15, 105), f"Score: {ENV.score}", fill=(230, 235, 255))
    d.text((15, 130), f"Reward: {ENV.total_reward:.2f}", fill=(230, 235, 255))
    d.text((565, 20), f"Episode: {AGENT.episode}", fill=(230, 235, 255))
    d.text((565, 45), f"epsilon: {AGENT.epsilon:.3f}", fill=(230, 235, 255))
    d.text((565, 70), f"loss: {AGENT.avg_loss():.4f}", fill=(230, 235, 255))

    return img


def status_text():
    bd = ENV.reward_breakdown or {}
    return (
        f"mode={MODE['value']} | step={ENV.step_count} | score={ENV.score} | "
        f"total_reward={ENV.total_reward:.2f} | epsilon={AGENT.epsilon:.3f} | "
        f"avg_loss={AGENT.avg_loss():.4f}\n"
        f"reward_breakdown={bd}"
    )


def reset(mode):
    MODE["value"] = mode
    LAST_STATE["obs"] = ENV.reset()
    return draw_frame(), status_text()


def step(action_name, mode):
    MODE["value"] = mode
    action_map = {"Left": 0, "Straight": 1, "Right": 2}
    obs = ENV._get_state()

    if mode == "AI Test":
        action = AGENT.act(obs, train=False)
    else:
        action = action_map[action_name]

    next_obs, reward, done, info = ENV.step(action)
    LAST_STATE["obs"] = next_obs
    if done:
        ENV.reset()
    return draw_frame(), status_text()


def train(n_steps, mode):
    MODE["value"] = "Training"
    obs = ENV._get_state()
    episodes_finished = 0
    for _ in range(int(n_steps)):
        action = AGENT.act(obs, train=True)
        next_obs, reward, done, info = ENV.step(action)
        AGENT.push(obs, action, reward, next_obs, done)
        AGENT.train_step()
        obs = next_obs
        if done:
            AGENT.on_episode_end(ENV.total_reward, ENV.score)
            obs = ENV.reset()
            episodes_finished += 1
    LAST_STATE["obs"] = obs
    return draw_frame(), status_text() + f"\ntrained_steps={int(n_steps)}, episodes_finished={episodes_finished}"


def save_model():
    AGENT.save()
    return "모델 저장 완료: driving_model.pth"


with gr.Blocks(title="Autodrive AI") as demo:
    gr.Markdown("# Autodrive AI — Lane Keeper RL")
    with gr.Row():
        frame = gr.Image(type="pil", label="Simulation")
        with gr.Column():
            mode = gr.Radio(["Human", "AI Test", "Training"], value="Human", label="Mode")
            action = gr.Radio(["Left", "Straight", "Right"], value="Straight", label="Human Action")
            status = gr.Textbox(label="Status", lines=8)
            with gr.Row():
                step_btn = gr.Button("Step")
                reset_btn = gr.Button("Reset")
            n_steps = gr.Slider(1, 500, value=50, step=1, label="Training steps")
            train_btn = gr.Button("Train")
            save_btn = gr.Button("Save model")
            save_out = gr.Textbox(label="Save result")

    demo.load(fn=lambda: (draw_frame(), status_text()), outputs=[frame, status])
    step_btn.click(fn=step, inputs=[action, mode], outputs=[frame, status])
    reset_btn.click(fn=reset, inputs=[mode], outputs=[frame, status])
    train_btn.click(fn=train, inputs=[n_steps, mode], outputs=[frame, status])
    save_btn.click(fn=save_model, outputs=save_out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
