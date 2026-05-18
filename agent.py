# agent.py — DQN 에이전트 (Experience Replay + Target Network + Dueling DQN)

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from config import (
    STATE_DIM, ACTION_DIM,
    LR, GAMMA, BATCH_SIZE, MEMORY_SIZE,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    TARGET_UPDATE,
)

MODEL_PATH = "driving_model.pth"


class DQNAgent:
    """
    알고리즘 구성
    ──────────────────────────────────────────────────────────────────
    - Dueling DQN          : model.py 참조
    - Experience Replay    : deque(maxlen=MEMORY_SIZE)
    - Target Network       : TARGET_UPDATE step마다 hard copy
    - ε-greedy 탐색        : 에피소드 종료마다 decay
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online  = DQN(STATE_DIM, ACTION_DIM).to(self.device)
        self.target  = DQN(STATE_DIM, ACTION_DIM).to(self.device)
        self._sync_target()
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LR)
        self.memory    = collections.deque(maxlen=MEMORY_SIZE)

        self.epsilon   = EPSILON_START
        self.step_cnt  = 0
        self.episode   = 0

        # 로그용
        self.loss_history    = collections.deque(maxlen=200)
        self.reward_history  = collections.deque(maxlen=200)
        self.score_history   = collections.deque(maxlen=200)
        self.current_loss    = 0.0

    # ── 행동 선택 ─────────────────────────────────────────────────────

    def act(self, state: np.ndarray, train: bool = True) -> int:
        """ε-greedy 정책."""
        if train and random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online(t).argmax().item()

    # ── 학습 ─────────────────────────────────────────────────────────

    def push(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, float(done)))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)

        s   = torch.FloatTensor(np.array(s)).to(self.device)
        a   = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r   = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns  = torch.FloatTensor(np.array(ns)).to(self.device)
        d   = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # 현재 Q값
        curr_q = self.online(s).gather(1, a)

        # Double DQN: online으로 action 선택, target으로 평가
        with torch.no_grad():
            best_a   = self.online(ns).argmax(1, keepdim=True)
            next_q   = self.target(ns).gather(1, best_a)
            target_q = r + GAMMA * next_q * (1 - d)

        loss = nn.SmoothL1Loss()(curr_q, target_q)   # Huber loss — 이상값에 강건
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10)
        self.optimizer.step()

        self.current_loss = loss.item()
        self.loss_history.append(self.current_loss)

        self.step_cnt += 1
        if self.step_cnt % TARGET_UPDATE == 0:
            self._sync_target()

    def on_episode_end(self, total_reward: float, score: int):
        """에피소드 종료 시 호출 — epsilon decay + 로그."""
        self.episode  += 1
        self.epsilon   = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.reward_history.append(total_reward)
        self.score_history.append(score)

    # ── 저장 / 불러오기 ──────────────────────────────────────────────

    def save(self):
        torch.save({
            "online":  self.online.state_dict(),
            "target":  self.target.state_dict(),
            "epsilon": self.epsilon,
            "episode": self.episode,
        }, MODEL_PATH)

    def load(self):
        try:
            ckpt = torch.load(MODEL_PATH, map_location=self.device)
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
            self.epsilon = ckpt.get("epsilon", EPSILON_MIN)
            self.episode = ckpt.get("episode", 0)
            print(f"[Agent] 모델 로드 완료 — episode {self.episode}, ε={self.epsilon:.3f}")
        except FileNotFoundError:
            print("[Agent] 저장된 모델 없음 — 새로 시작합니다.")

    # ── 통계 ─────────────────────────────────────────────────────────

    def avg_reward(self, n: int = 50) -> float:
        hist = list(self.reward_history)[-n:]
        return sum(hist) / len(hist) if hist else 0.0

    def avg_loss(self, n: int = 50) -> float:
        hist = list(self.loss_history)[-n:]
        return sum(hist) / len(hist) if hist else 0.0

    # ── 내부 ─────────────────────────────────────────────────────────

    def _sync_target(self):
        self.target.load_state_dict(self.online.state_dict())
