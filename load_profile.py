import numpy as np
import random
from typing import Tuple, Optional


P_RATED_W = 25e6
V_GRID_V = 35e3
SECONDS_PER_DAY = 24 * 3600


def _seasonal_baseline(day_type: str, step_seconds: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回功率基线(按分钟, W)与负载权重，用于模拟0-25MW动态波动的充放电循环。
    - 放电: 正功率 (向电网送电) 0-25MW
    - 充电: 负功率 (从电网取电) 0到-25MW
    修改：实现连续波动而非分段常数，确保24小时内完成至少一个充放电循环
    """
    num_steps = SECONDS_PER_DAY // step_seconds
    t_sec = np.arange(num_steps) * step_seconds
    t_hr = t_sec / 3600.0

    # 基于25MW额定功率设计动态功率曲线
    base_power_mw = 25.0  # 基准功率25MW，符合系统额定功率
    
    # 设计多频率叠加的动态波动曲线
    # 主频率：24小时一个完整充放电循环
    main_cycle = np.sin(2 * np.pi * t_hr / 24.0 - np.pi/2)  # 主周期：从最低点开始
    
    # 副频率：增加更细致的波动
    sub_cycle1 = 0.3 * np.sin(2 * np.pi * t_hr / 12.0)  # 12小时副周期
    sub_cycle2 = 0.2 * np.sin(2 * np.pi * t_hr / 6.0)   # 6小时副周期
    sub_cycle3 = 0.1 * np.sin(2 * np.pi * t_hr / 3.0)   # 3小时副周期
    
    # 叠加所有频率分量
    combined_signal = main_cycle + sub_cycle1 + sub_cycle2 + sub_cycle3
    
    # 添加随机噪声增加真实性
    rng = np.random.default_rng(42)  # 固定种子保证可重复性
    noise = rng.normal(0.0, 0.08, size=combined_signal.shape)  # 8%标准差噪声
    combined_signal += noise
    
    # 将信号映射到0-25MW范围（双向）
    # 正值表示放电（0-25MW），负值表示充电（0到-25MW）
    P_normalized = combined_signal / np.max(np.abs(combined_signal))  # 归一化到[-1, 1]
    P_mw = P_normalized * base_power_mw  # 缩放到[-25, 25]MW
    
    # 转换为瓦特
    P = P_mw * 1e6  # MW转换为W
    
    # 基于季节/日型的微调
    dt = day_type.lower() if isinstance(day_type, str) else ""
    if "summer" in dt or "夏" in dt:
        # 夏季：增强白天放电，减少夜间充电
        summer_weight = 1.0 + 0.3 * np.sin(2 * np.pi * (t_hr - 6) / 24.0)
        P *= summer_weight
    elif "winter" in dt or "冬" in dt:
        # 冬季：增强早晚功率变化
        winter_weight = 1.0 + 0.2 * np.sin(2 * np.pi * (t_hr - 8) / 24.0)
        P *= winter_weight
    elif "weekend" in dt or "周末" in dt:
        # 周末：功率变化更平缓
        weekend_weight = 0.8 + 0.2 * np.sin(2 * np.pi * t_hr / 24.0)
        P *= weekend_weight
    
    # 确保功率在合理范围内
    P = np.clip(P, -25e6, 25e6)  # 限制在±25MW，符合系统额定功率
    
    # 轻微平滑以避免过于尖锐的变化
    kernel_size = max(1, step_seconds // 30)  # 根据时间步长调整平滑核
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
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

    # 修改：允许充放电能量不平衡，从而产生更大的SOC变化
    # 原来的能量平衡处理被注释掉，允许SOC在24小时内产生累积变化
    # dt_h = 1.0 / 60.0
    # pos_energy = float(np.sum(P[P > 0]) * dt_h)
    # neg_energy = float(np.sum(P[P < 0]) * dt_h)  # 负值
    # if pos_energy > 0 and neg_energy < 0:
    #     scale = pos_energy / abs(neg_energy) if abs(neg_energy) > 1e-9 else 1.0
    #     # 仅缩放充电段，使|充电能量|≈放电能量
    #     charge_mask = P < 0
    #     P[charge_mask] *= scale

    return P.astype(float)


def generate_load_profile_new(
    day_type: str,
    step_seconds: int = 60,
    *,
    peak_w: float = 25e6,           # 25 MW 峰值
    num_packs: int = 120,           # 电池包数量
    capacity_ah: float = 314.0,     # 每包容量 [Ah]
    series_cells: int = 312,        # 每包串数（估算名义电压用）
    v_cell_nominal: float = 3.35,   # 单体名义电压 [V]（LFP 平台约 3.3~3.4）
    soc_min: float = 0.0,           # 修改：允许SOC降到0%
    soc_max: float = 1.0,           # 修改：允许SOC升到100%
    soc0: float = 0.5,      # 起始 SOC；默认窗口中点
    safety_margin: float = 0.80,    # 进一步减少安全裕度，允许更大的SOC变化
) -> np.ndarray:
    """
    生成 24h 的功率序列 P[t] (W)，按 25 MW 顶峰缩放，并保证 SOC 不越过 [soc_min, soc_max]。
    要求你已有：
      - _seasonal_baseline(day_type, step_seconds) -> (P_base, t_hr)
      - _inject_overload_shocks(P, step_seconds)   # 可选：对 P 就地注入过载片段
    """
    # 1) 基线 + 过载片段（你已有的函数）
    P, t_hr = _seasonal_baseline(day_type, step_seconds)   # 形状 (T,)
    _inject_overload_shocks(P, step_seconds)               # 可选：如果不需要可注释

    # 修改：不强制去均值，允许能量不平衡，从而产生更大的SOC变化
    # P = P.astype(float)
    # P -= np.mean(P)

    # 3) 峰值约束缩放系数（25 MW）
    peak_raw = float(np.max(np.abs(P))) if np.any(P) else 1.0
    scale_peak = (peak_w / peak_raw) if peak_raw > 1e-12 else 1.0

    # 4) 能量窗约束（不越 SOC 窗口）
    #    可用能量窗口（Wh）：总能量 * 起始点可上下摆幅的最小边
    if soc0 is None:
        soc0 = 0.5 * (soc_min + soc_max) + 0.5 * (soc_max - soc_min)  # = 0.5; 写长是为了看清逻辑
        soc0 = 0.5  # 上面那句等价于 0.5，这里直给
    swing = min(soc0 - soc_min, soc_max - soc0)  # 能上下摆动的SOC幅度
    v_nom = series_cells * v_cell_nominal        # 模块名义电压
    E_total_Wh = num_packs * capacity_ah * v_nom # 总名义能量
    E_window_Wh = E_total_Wh * swing             # 可摆动能量窗口

    # 基于未缩放曲线计算"最大累积能量偏移"
    dt_h = step_seconds / 3600.0
    cum_Wh = np.cumsum(P) * dt_h                 # 单位 Wh
    max_abs_cum_Wh = float(np.max(np.abs(cum_Wh))) if cum_Wh.size else 0.0
    scale_energy = (E_window_Wh / max_abs_cum_Wh) if max_abs_cum_Wh > 1e-12 else 1.0

    # 5) 取两者最小，并加安全裕度
    scale = safety_margin * min(scale_peak, scale_energy)
    P_scaled = P * scale

    # 修改：不强制再次去均值，允许能量不平衡
    # P_scaled -= np.mean(P_scaled)

    return P_scaled


def generate_profiles(day_type: str, step_seconds: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """便捷函数：同时返回功率与温度两个对齐序列。step_seconds 支持 60, 30, 10, 1 等。"""
    P = generate_load_profile(day_type, step_seconds=step_seconds)
    T = generate_environment_temperature(day_type, size=len(P))
    return P, T


