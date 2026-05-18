# config.py — 모든 설정값을 한 곳에서 관리

# ── 화면 ──────────────────────────────────────────────────────────────
SCREEN_WIDTH  = 700
SCREEN_HEIGHT = 900
FPS           = 60

# ── 도로 / 차량 ───────────────────────────────────────────────────────
ROAD_LEFT      = 150          # 도로 왼쪽 경계 x
ROAD_RIGHT     = 550          # 도로 오른쪽 경계 x
ROAD_CENTER    = (ROAD_LEFT + ROAD_RIGHT) // 2   # 350
LANE_WIDTH     = (ROAD_RIGHT - ROAD_LEFT) // 3   # 133 px, 3차선
NUM_LANES      = 3

CAR_WIDTH  = 40
CAR_HEIGHT = 70
CAR_SPEED  = 5               # px/step — 내 차 이동 속도
CAR_Y      = SCREEN_HEIGHT - 150   # 화면 고정 위치

# 장애물 차량
OBS_SPEED_MIN  = 2
OBS_SPEED_MAX  = 6
OBS_SPAWN_PROB = 0.015        # 매 step 장애물 생성 확률
MAX_OBSTACLES  = 8

# ── 강화학습 하이퍼파라미터 ────────────────────────────────────────────
STATE_DIM  = 12   # 상태 벡터 크기 (env.py 참조)
ACTION_DIM = 3    # 0:좌, 1:직진, 2:우

LR           = 3e-4
GAMMA        = 0.99
BATCH_SIZE   = 128
MEMORY_SIZE  = 30_000
EPSILON_START = 1.0
EPSILON_MIN   = 0.02
EPSILON_DECAY = 0.997          # episode 종료마다 곱함
TARGET_UPDATE  = 200           # step마다 target network 동기화

HIDDEN_DIM = 256

# ── 보상 설계 (reward.py 에서 사용) ────────────────────────────────────
R_SURVIVE        =  0.05   # 살아있을 때 매 step 소액 지급
R_CENTER_MAX     =  0.30   # 차선 중앙에 완벽히 있을 때
R_LANE_CHANGE    = -0.10   # 불필요한 차선 변경 패널티
R_COLLISION      = -15.0   # 충돌
R_NEAR_WALL      = -0.50   # 도로 경계에 너무 가까울 때
R_NEAR_OBS       = -0.20   # 장애물 바로 앞 위험 거리
R_SAFE_OVERTAKE  =  1.50   # 장애물을 안전하게 추월했을 때
R_WALL_DIST_THR  = 20      # 이 픽셀 이하면 near_wall 패널티

# ── 색상 팔레트 ────────────────────────────────────────────────────────
C = {
    "bg":          (8,  10,  20),
    "road":        (28,  30,  45),
    "lane_solid":  (60,  65,  90),
    "lane_dash":   (200, 200,  80),
    "car":         (0,  220, 180),
    "car_glow":    (0,  255, 200),
    "obs":         (220,  50,  50),
    "obs_glow":    (255,  80,  80),
    "text":        (230, 235, 255),
    "accent":      (255,  50, 140),
    "panel":       (15,  18,  35),
    "bar_bg":      (40,  40,  65),
    "bar_fg":      (0,  200, 150),
    "bar_loss":    (255, 140,  0),
    "reward_pos":  (0,  255, 150),
    "reward_neg":  (255,  60,  60),
}
