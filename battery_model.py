"""
BatteryModel: 35 kV/25 MW 级联储能 PCS 的电池模块模型（312S, 314 Ah）。

该模型面向单个模块（312 串电芯，标称容量 314 Ah，直流母线约 35 kV/√3 ÷ 40 ≈ 500 V/模块），
可在更高层的 PCS 级联中以多个模块串并联方式组合。

功能要点：
- 内置 OCV(SOC, T) 与内阻 R(SOC, T) 模型（可调参数）。
- 库仑计量更新 SOC；温度采用一阶热网络（I^2R 损耗发热 + 对流散热）。
- 寿命模型同时考虑日历老化与循环老化，并对高倍率/过载（≥ 3× 额定电流）施加加速因子。
- 端电压 = OCV - I·R；支持 get_voltage() 查询。

电流方向约定：current_a > 0 表示放电（从电池流出），会降低 SOC。

注意：参数为工程近似，已调至在 20~40 ℃、额定电流 ~420 A、
以及 3×额定电流过载场景下具有合理数值范围，便于系统级仿真。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


KELVIN_OFFSET: float = 273.15


@dataclass
class BatteryModelConfig:
    """电池模型参数配置。

    该配置用于集中管理 OCV、内阻、热参数以及寿命模型的灵敏度。
    可根据器件实测或供应商数据进行二次标定。
    """

    # 额定/几何参数
    series_cells: int = 312
    rated_capacity_ah: float = 314.0
    rated_current_a: float = 420.0  # 模块级额定电流（与 25 MW/级联规模相容的量级）

    # OCV 与内阻基准参数
    nominal_voltage_per_cell_v: float = 3.6
    base_string_resistance_ohm_25c: float = 0.22  # 312 串等效内阻在 25 ℃、中等 SOC

    # 温度/容量效应参数
    # 低温可用容量衰减/高温轻微提升（工程近似，夹紧 0.8~1.1）
    low_temp_capacity_loss_per_k_c: float = 0.003
    high_temp_capacity_gain_per_k_c: float = 0.001

    # 热模型（一阶 RC）：
    # Rth 取较小值以反映液冷/风冷系统的热阻（单位 K/W，越小散热越强）
    thermal_resistance_k_per_w: float = 3.0e-4
    thermal_capacity_j_per_k: float = 2.0e6

    # 日历老化（25 ℃ 年损失占比）
    calendar_fade_per_year_at_25c: float = 0.02

    # 循环老化基准（100%DOD、25 ℃、1C 每等效循环的容量损失占比）
    cycle_fade_per_equivalent_full_cycle_base: float = 5.0e-4
    cycle_fade_temp_sensitivity_per_k_c: float = math.log(2) / 10.0  # ~Q10 模型
    cycle_fade_c_rate_exponent: float = 1.3

    # 过载倍率对循环老化的加速（I/Irated > 1 时生效；≥3×时额外放大）
    overload_extra_multiplier_at_3x: float = 3.0


class BatteryModel:
    """储能 PCS 模块级电池模型。

    - 初始化参数可覆盖 `BatteryModelConfig` 中的默认值。
    - 使用 `update_state(current, dt, ambient_temp)` 连续更新状态（SOC、温度、寿命）。
    - 使用 `get_voltage()` 查询端电压（基于最近一次电流与状态）。

    电流方向：current > 0 放电（SOC 下降）。
    """

    def __init__(
        self,
        config: Optional[BatteryModelConfig] = None,
        initial_soc: float = 0.5,
        initial_temperature_c: float = 25.0,
    ) -> None:
        self.config: BatteryModelConfig = config or BatteryModelConfig()

        # 状态量
        self.state_of_charge: float = float(max(0.0, min(1.0, initial_soc)))
        self.cell_temperature_c: float = float(initial_temperature_c)
        self.ambient_temperature_c: float = float(initial_temperature_c)
        self.last_current_a: float = 0.0

        # 寿命/退化状态
        self.capacity_fade_fraction: float = 0.0  # 容量损失占比（0~1）
        self.resistance_growth_fraction: float = 0.0  # 内阻增长占比（0~1）
        self.cumulative_throughput_ah: float = 0.0  # 绝对安时吞吐累计
        self.equivalent_full_cycles: float = 0.0  # 等效满循环计数（EFC）

    # ----------------------------- 核心物理子模型 -----------------------------
    def _ocv_per_cell_v(self, soc: float, temperature_c: float) -> float:
        """OCV-SOC-T 近似：单体电芯开路电压（单位 V）。

        采用平滑的经验函数，能覆盖 0~1 SOC 的单调关系，并在 20~40 ℃ 具备微弱温度依赖。
        """
        soc_clamped = max(0.0, min(1.0, soc))

        # 基础 OCV 曲线（类 NMC/LFP 的平坦区 + 双指数端区）；数值为工程近似
        ocv_base = (
            3.0
            + 0.7 * soc_clamped
            + 0.20 * math.exp(-10.0 * (1.0 - soc_clamped))
            - 0.10 * math.exp(-10.0 * soc_clamped)
        )

        # 温度微调：每升高 1 ℃，OCV 轻微提升 ~0.8 mV
        d_ocv_temp = 0.0008 * (temperature_c - 25.0)
        return ocv_base + d_ocv_temp

    def _string_ocv_v(self, soc: float, temperature_c: float) -> float:
        return self.config.series_cells * self._ocv_per_cell_v(soc, temperature_c)

    def _soc_factor_for_resistance(self, soc: float) -> float:
        """SOC 对内阻的影响：两端升高，中间较低。

        取 0.5 SOC 处为 1 倍，0/1 SOC 端提升至 ~1.4 倍（可调）。
        """
        x = abs(max(0.0, min(1.0, soc)) - 0.5) / 0.5
        return 1.0 + 0.4 * (x ** 1.5)

    def _temp_factor_for_resistance(self, temperature_c: float) -> float:
        """温度对内阻的影响：低温升高、高温降低（Q10 近似）。"""
        return math.exp(0.025 * (25.0 - temperature_c))

    def _string_resistance_ohm(self, soc: float, temperature_c: float) -> float:
        base_r = self.config.base_string_resistance_ohm_25c
        r = base_r * self._soc_factor_for_resistance(soc) * self._temp_factor_for_resistance(temperature_c)
        r *= (1.0 + self.resistance_growth_fraction)
        return max(1e-6, r)

    def _capacity_temp_factor(self, temperature_c: float) -> float:
        """温度对可用容量的影响，限制在 [0.8, 1.1]。"""
        if temperature_c < 25.0:
            factor = 1.0 - self.config.low_temp_capacity_loss_per_k_c * (25.0 - temperature_c)
        else:
            factor = 1.0 + self.config.high_temp_capacity_gain_per_k_c * (temperature_c - 25.0)
        return float(max(0.8, min(1.1, factor)))

    # ----------------------------- 寿命模型 -----------------------------
    def _calendar_fade_increment(self, temperature_c: float, dt_s: float) -> float:
        """日历老化：按 25 ℃ 年化 2% 基准，并用 Q10≈2 在 10 ℃ 升温时加倍。"""
        per_sec_25c = self.config.calendar_fade_per_year_at_25c / (365.0 * 24.0 * 3600.0)
        q10 = math.exp(self.config.cycle_fade_temp_sensitivity_per_k_c * (temperature_c - 25.0))
        return per_sec_25c * q10 * dt_s

    def _cycle_fade_increment(self, current_a: float, temperature_c: float, dt_s: float) -> float:
        cfg = self.config
        abs_current = abs(current_a)
        # 名义 C 率（相对退化中的额定容量使用名义额定值定义倍率）
        c_rate = abs_current / max(1e-6, cfg.rated_capacity_ah)
        # 基于安时吞吐的 EFC 近似：dEFC = dAh / (2*Capacity)
        d_ah = abs_current * dt_s / 3600.0
        d_efc = d_ah / (2.0 * max(1e-6, cfg.rated_capacity_ah))

        temp_multiplier = math.exp(cfg.cycle_fade_temp_sensitivity_per_k_c * (temperature_c - 25.0))
        c_rate_multiplier = max(0.0, c_rate) ** cfg.cycle_fade_c_rate_exponent

        base = cfg.cycle_fade_per_equivalent_full_cycle_base
        fade = base * d_efc * temp_multiplier * max(1e-6, c_rate_multiplier)

        # 过载加速：当倍率超过 1× 逐步增加；≥3× 时再乘以一个额外放大系数
        overload_ratio = abs_current / max(1e-6, cfg.rated_current_a)
        if overload_ratio > 1.0:
            fade *= overload_ratio ** 2  # 倍率的平方加速
            if overload_ratio >= 3.0:
                fade *= cfg.overload_extra_multiplier_at_3x
        return fade

    # ----------------------------- 对外接口 -----------------------------
    def update_state(self, current_a: float, dt_s: float, ambient_temp_c: float) -> None:
        """状态更新（适用于 24 h 一充一放等任意时序）：

        - 输入电流 current_a（>0 放电）、步长 dt_s（秒）、环境温度 ambient_temp_c（20~40 ℃）。
        - 更新 SOC、温度、容量衰减、等效循环等状态量。
        - 同时考虑高倍率/过载对热与寿命的影响。
        """
        cfg = self.config
        self.last_current_a = float(current_a)
        self.ambient_temperature_c = float(ambient_temp_c)

        # 1) 电学：基于当前 SOC/T 估算 OCV 与内阻
        ocv_v = self._string_ocv_v(self.state_of_charge, self.cell_temperature_c)
        r_ohm = self._string_resistance_ohm(self.state_of_charge, self.cell_temperature_c)

        # 2) 热学：I^2 R 发热与对流散热（单步欧拉）
        joule_heat_w = (current_a ** 2) * r_ohm
        delta_t_source = joule_heat_w * cfg.thermal_resistance_k_per_w
        # 一阶 RC：dT/dt = (P - (T - Ta)/Rth) / Cth
        dT_dt = (joule_heat_w - (self.cell_temperature_c - ambient_temp_c) / cfg.thermal_resistance_k_per_w) / cfg.thermal_capacity_j_per_k
        self.cell_temperature_c += dT_dt * dt_s
        # 数值安全：限制温度变化与范围（工程仿真，防止极端步长导致爆炸）
        self.cell_temperature_c = float(max(-20.0, min(80.0, self.cell_temperature_c)))

        # 3) 容量与 SOC：考虑温度影响与容量衰减
        temp_capacity_factor = self._capacity_temp_factor(self.cell_temperature_c)
        effective_capacity_ah = max(1e-6, cfg.rated_capacity_ah * (1.0 - self.capacity_fade_fraction) * temp_capacity_factor)
        d_soc = -(current_a * dt_s / 3600.0) / effective_capacity_ah
        self.state_of_charge = float(max(0.0, min(1.0, self.state_of_charge + d_soc)))

        # 4) 吞吐/循环计量
        d_ah_abs = abs(current_a) * dt_s / 3600.0
        self.cumulative_throughput_ah += d_ah_abs
        self.equivalent_full_cycles = self.cumulative_throughput_ah / (2.0 * max(1e-6, cfg.rated_capacity_ah))

        # 5) 寿命衰减（容量）与内阻增长（简化：用容量衰减的 1/2 映射到内阻增长）
        d_fade_calendar = self._calendar_fade_increment(self.cell_temperature_c, dt_s)
        d_fade_cycle = self._cycle_fade_increment(current_a, self.cell_temperature_c, dt_s)
        d_fade = d_fade_calendar + d_fade_cycle
        if d_fade > 0.0:
            self.capacity_fade_fraction = float(max(0.0, min(0.4, self.capacity_fade_fraction + d_fade)))
            self.resistance_growth_fraction = float(max(0.0, min(1.0, self.resistance_growth_fraction + 0.5 * d_fade)))

        # 6) 端电压可在外部通过 get_voltage() 查询
        _ = ocv_v  # 仅为清晰，端电压在 get_voltage 中即时计算

    def get_voltage(self) -> float:
        """返回当前端电压（V）= OCV(SOC,T) - I·R(SOC,T)。"""
        ocv_v = self._string_ocv_v(self.state_of_charge, self.cell_temperature_c)
        r_ohm = self._string_resistance_ohm(self.state_of_charge, self.cell_temperature_c)
        v = ocv_v - self.last_current_a * r_ohm
        # 避免出现非物理负电压
        return float(max(0.0, v))

    # ----------------------------- 便捷属性/查询 -----------------------------
    @property
    def effective_capacity_ah(self) -> float:
        cfg = self.config
        temp_capacity_factor = self._capacity_temp_factor(self.cell_temperature_c)
        return max(1e-6, cfg.rated_capacity_ah * (1.0 - self.capacity_fade_fraction) * temp_capacity_factor)

    @property
    def ocv_v(self) -> float:
        return self._string_ocv_v(self.state_of_charge, self.cell_temperature_c)

    @property
    def resistance_ohm(self) -> float:
        return self._string_resistance_ohm(self.state_of_charge, self.cell_temperature_c)

    # ----------------------------- 示例 -----------------------------
    def example_daily_profile(self) -> None:
        """示例：以 24 小时一充一放为例（仅说明，非单元测试）。

        放电 12 h（额定电流），充电 12 h（同倍率，反向电流），环境温度 20~40 ℃ 正弦波动。
        """
        total_seconds = 24 * 3600
        dt = 1.0
        for k in range(int(total_seconds)):
            t = k * dt
            # 简单温度波动（20~40 ℃）
            ambient = 30.0 + 10.0 * math.sin(2.0 * math.pi * t / total_seconds)
            # 12 h 放电 + 12 h 充电
            current = self.config.rated_current_a if t < 12 * 3600 else -self.config.rated_current_a
            self.update_state(current, dt, ambient)


