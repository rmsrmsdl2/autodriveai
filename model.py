# model.py — 신경망 아키텍처

import torch
import torch.nn as nn
from config import HIDDEN_DIM


class DQN(nn.Module):
    """
    Dueling DQN 아키텍처
    ──────────────────────────────────────────────────────────────────
    - Value stream    : V(s)       — 상태 자체의 가치
    - Advantage stream: A(s, a)    — 각 행동의 상대적 이점
    - Q(s, a) = V(s) + (A(s,a) - mean(A))
    
    일반 DQN보다 학습 안정성이 높음.
    특히 action이 Q값에 큰 차이를 안 만드는 상황(직진)이 많은
    자율주행 환경에 적합.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )
        # Value head
        self.value = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Advantage head
        self.advantage = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        v    = self.value(feat)          # (B, 1)
        a    = self.advantage(feat)      # (B, A)
        # Dueling 결합: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
