import numpy as np
import random
from typing import Tuple, Optional


P_RATED_W = 25e6
V_GRID_V = 35e3
SECONDS_PER_DAY = 24 * 3600


def _seasonal_baseline(day_type: str, step_seconds: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回功率基线(按分钟, W)与负载权重，用于模拟不同季节/日型下的一充一放日循环。
    - 放电: 正功率 (向电网送电)
    - 充电: 负功率 (从电网取电)
    """
    num_steps = SECONDS_PER_DAY // step_seconds
    t_sec = np.arange(num_steps) * step_seconds
    t_hr = t_sec / 3600.0

    P = np.zeros_like(t_hr, dtype=float)

    # 默认日循环：夜间充电、白天放电
    # 充电窗口（低电价/夜间）：2-6, 22-24
    P[(t_hr >= 2) & (t_hr < 6)] = -0.8 * P_RATED_W
    P[(t_hr >= 22) & (t_hr < 24)] = -0.8 * P_RATED_W

    # 放电窗口（高电价/白天）：8-12, 14-18
    P[(t_hr >= 8) & (t_hr < 12)] = 0.9 * P_RATED_W
    P[(t_hr >= 14) & (t_hr < 18)] = 0.9 * P_RATED_W

    # 基于季节/日型的微调（峰谷比例/时段微偏移）
    dt = day_type.lower() if isinstance(day_type, str) else ""
    if "summer" in dt or "夏" in dt:
        # 夏季白天用电更高：抬高下午放电脉冲
        P[(t_hr >= 14) & (t_hr < 19)] = 1.0 * P_RATED_W
    elif "winter" in dt or "冬" in dt:
        # 冬季早晚用电更高：上午/傍晚略增
        P[(t_hr >= 8) & (t_hr < 12)] = 1.0 * P_RATED_W
        P[(t_hr >= 17) & (t_hr < 20)] = 0.8 * P_RATED_W
    elif "weekend" in dt or "周末" in dt:
        # 周末中午更高，早晚稍低
        P[(t_hr >= 11) & (t_hr < 15)] = 0.95 * P_RATED_W
        P[(t_hr >= 8) & (t_hr < 10)] = 0.6 * P_RATED_W
        P[(t_hr >= 18) & (t_hr < 20)] = 0.6 * P_RATED_W

    # 添加缓慢的随机波动（真实度）
    rng = np.random.default_rng()
    noise = rng.normal(0.0, 0.05, size=P.shape)  # 5% 标准差
    P = P * (1.0 + noise)

    # 平滑边缘，防止尖锐跳变（核宽约等于5个采样点）
    kernel = np.ones(5) / 5.0
    P = np.convolve(P, kernel, mode='same')

    return P, t_hr


def _inject_overload_shocks(P: np.ndarray, step_seconds: int, seed: Optional[int] = None) -> None:
    """
    向功率序列中随机插入瞬时过载冲击（3×额定电流，持续~10秒）。
    由于序列按分钟采样，将冲击平均化到所在分钟：等效附加功率 ≈ 3×P_rated × (10/60)。
    可随机插入1~3个，极性随机（正/负）。原地修改。
    """
    rnd = random.Random(seed)
    n_shocks = rnd.randint(1, 3)
    # 10s 冲击，跨越的采样点数
    N = max(1, int(round(10.0 / float(step_seconds))))
    # 每个覆盖点叠加的功率，使总能量等效：sum(P_add*dt) = 3*P_rated*10s
    P_add = 3.0 * P_RATED_W * 10.0 / (N * float(step_seconds))

    for _ in range(n_shocks):
        idx = rnd.randrange(0, len(P))
        sign = rnd.choice([-1.0, 1.0])
        for k in range(N):
            j = idx + k
            if j < len(P):
                P[j] += sign * P_add

    # 限幅，避免非物理值
    np.clip(P, -3.0 * P_RATED_W, 3.0 * P_RATED_W, out=P)


def generate_environment_temperature(day_type: str, size: int, seed: Optional[int] = None) -> np.ndarray:
    """生成与功率曲线对齐的环境温度序列（°C），在 20~40 范围内随机波动，含昼夜正弦分量。"""
    rnd = np.random.default_rng(seed)
    t = np.arange(size) / 60.0

    # 季节基线温度
    dt = day_type.lower() if isinstance(day_type, str) else ""
    if "summer" in dt or "夏" in dt:
        base = 32.0
        amp = 5.5
    elif "winter" in dt or "冬" in dt:
        base = 22.0
        amp = 4.0
    else:  # 春秋或默认
        base = 27.0
        amp = 4.5

    temp = base + amp * np.sin(2 * np.pi * (t - 6.0) / 24.0)  # 下午略高
    temp += rnd.normal(0.0, 0.8, size=size)  # 小扰动
    return np.clip(temp, 20.0, 40.0)


def generate_load_profile(day_type: str, step_seconds: int = 60) -> np.ndarray:
    """
    生成 24 小时逐分钟的功率序列（单位 W，长度 1440）。
    满足：
    - 日循环：至少一天一充一放；按 25 MW 顶峰设定。
    - 过载冲击：随机插入 3×额定电流的10秒冲击，折算到分钟平均。
    - 可与 `generate_environment_temperature(day_type)` 同步使用，得到同时间段的温度序列（20–40 ℃）。
    """
    P, t_hr = _seasonal_baseline(day_type, step_seconds)
    _inject_overload_shocks(P, step_seconds)

    # 日内能量平衡处理：使正负能量尽量相等（∫P dt ≈ 0）
    dt_h = 1.0 / 60.0
    pos_energy = float(np.sum(P[P > 0]) * dt_h)
    neg_energy = float(np.sum(P[P < 0]) * dt_h)  # 负值
    if pos_energy > 0 and neg_energy < 0:
        scale = pos_energy / abs(neg_energy) if abs(neg_energy) > 1e-9 else 1.0
        # 仅缩放充电段，使|充电能量|≈放电能量
        charge_mask = P < 0
        P[charge_mask] *= scale

    return P.astype(float)


def generate_profiles(day_type: str, step_seconds: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """便捷函数：同时返回功率与温度两个对齐序列。step_seconds 支持 60, 30, 10, 1 等。"""
    P = generate_load_profile(day_type, step_seconds=step_seconds)
    T = generate_environment_temperature(day_type, size=len(P))
    return P, T


