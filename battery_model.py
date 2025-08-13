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

import os
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from load_profile import generate_profiles

KELVIN_OFFSET: float = 273.15

ocv_data = loadmat('ocv_data.mat')
soc_lut = np.linspace(0, 1, 201)
soc_lut = soc_lut.reshape(-1, 1)
OCV0 = ocv_data['OCV0'][0]
OCV_rel = ocv_data['OCVrel'][0]
temp_lut = np.array([-25, -15, -5, 5, 15, 25, 35, 45]) + 273.15

em_lut = OCV0.reshape(-1, 1) @ np.ones_like(temp_lut.reshape(1, -1)) + OCV_rel.reshape(-1, 1) @ temp_lut.reshape(1, -1)


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
    base_string_resistance_ohm_25c: float = 0.00022 * 312  # 312 串等效内阻在 25 ℃、中等 SOC

    # 温度/容量效应参数
    # 低温可用容量衰减/高温轻微提升（工程近似，夹紧 0.8~1.1）
    low_temp_capacity_loss_per_k_c: float = 0.003
    high_temp_capacity_gain_per_k_c: float = 0.001

    # 热模型（一阶 RC）：
    # Rth 取较小值以反映液冷/风冷系统的热阻（单位 K/W，越小散热越强）
    thermal_resistance_k_per_w: float = 3.0e-4
    thermal_capacity_j_per_k: float = 2.0e6


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

    # ----------------------------- 核心物理子模型 -----------------------------
    def _ocv_per_cell_v(self, soc: float, temp: float) -> float:
        if np.isscalar(soc):  # replicate a scalar temperature for all socs
            soc = np.array([[soc]])
        soccol = soc  # 将 soc 转置为列向量
        SOC = soc_lut  # 将 SOC 转置为列向量
        OCV0_ = OCV0  # 将 OCV0_ 转置为列向量
        OCVrel0 = OCV_rel  # 将 OCVrel转置为列向量
        temp = temp + 273.15
        if np.isscalar(temp):  # replicate a scalar temperature for all socs
            tempcol = temp * np.ones_like(soccol)
        else:
            tempcol = temp  # force matrix temperature to be column vector
            if tempcol.size != soccol.size:
                raise ValueError(
                    'Function inputs "soc" and "temp" must either have the same number of elements, or "temp" must be a scalar')

        diffSOC = SOC[1] - SOC[0]  # spacing between SOC points - assume uniform
        ocv = np.zeros_like(soccol)  # initialize output to zero

        I1 = np.where(soccol <= SOC[0])[0]  # indices of socs below model-stored data
        I2 = np.where(soccol >= SOC[-1])[0]  # and of socs above model-stored data
        I3 = np.where((soccol > SOC[0]) & (soccol < SOC[-1]))[0]  # the rest of them
        I6 = np.isnan(soccol)  # if input is "not a number" for any locations

        # for voltages less than lowest stored soc datapoint, extrapolate off low end of table
        if len(I1) > 0:
            dv = (OCV0_[1] + tempcol[I1] * OCVrel0[1]) - (OCV0_[0] + tempcol[I1] * OCVrel0[0])
            ocv[I1] = (soccol[I1] - SOC[0]) * dv / diffSOC + OCV0_[0] + tempcol[I1] * OCVrel0[0]

        # for voltages greater than highest stored soc datapoint, extrapolate off high end of table
        if len(I2) > 0:
            dv = (OCV0_[-1] + tempcol[I2] * OCVrel0[-1]) - (OCV0_[-2] + tempcol[I2] * OCVrel0[-2])
            ocv[I2] = (soccol[I2] - SOC[-1]) * dv / diffSOC + OCV0_[-1] + tempcol[I2] * OCVrel0[-1]

        # for normal soc range, manually interpolate (10x faster than "interp1")
        I4 = (soccol[I3] - SOC[0]) / diffSOC  # using linear interpolation
        I5 = np.floor(I4).astype(int)
        I45 = I4 - I5
        omI45 = 1 - I45
        ocv[I3] = OCV0_[I5] * omI45 + OCV0_[I5 + 1] * I45
        ocv[I3] = ocv[I3] + tempcol[I3] * (OCVrel0[I5] * omI45 + OCVrel0[I5 + 1] * I45)
        ocv[I6] = 0  # replace NaN SOCs with zero voltage
        return ocv[0, 0]

    def _string_ocv_v(self, soc: float, temperature_c: float) -> float:
        return self.config.series_cells * self._ocv_per_cell_v(soc, temperature_c)

    def _string_resistance_ohm(self, soc: float, temperature_c: float) -> float:
        base_r = self.config.base_string_resistance_ohm_25c
        r = base_r
        return max(1e-6, r)

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

        # 一阶 RC：dT/dt = (P - (T - Ta)/Rth) / Cth
        dT_dt = (joule_heat_w - (
                    self.cell_temperature_c - ambient_temp_c) / cfg.thermal_resistance_k_per_w) / cfg.thermal_capacity_j_per_k
        self.cell_temperature_c += dT_dt * dt_s
        # 数值安全：限制温度变化与范围（工程仿真，防止极端步长导致爆炸）
        self.cell_temperature_c = float(max(-20.0, min(80.0, self.cell_temperature_c)))

        # 3) 容量与 SOC：考虑温度影响与容量衰减

        effective_capacity_ah = cfg.rated_capacity_ah
        d_soc = -(current_a * dt_s / 3600.0) / effective_capacity_ah
        self.state_of_charge = float(max(0.0, min(1.0, self.state_of_charge + d_soc)))

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
        temp_capacity_factor = 1
        return max(1e-6, cfg.rated_capacity_ah * temp_capacity_factor)

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

    # ----------------------------- 新增实用方法 -----------------------------

    def get_battery_status(self) -> dict:
        """获取电池完整状态信息，用于监控和诊断。"""
        return {
            'soc': self.state_of_charge,
            'voltage_v': self.get_voltage(),
            'ocv_v': self.ocv_v,
            'current_a': self.last_current_a,
            'cell_temperature_c': self.cell_temperature_c,
            'ambient_temperature_c': self.ambient_temperature_c,
            'effective_capacity_ah': self.effective_capacity_ah,
            'resistance_ohm': self.resistance_ohm,
            'c_rate': abs(self.last_current_a) / max(1e-6, self.config.rated_capacity_ah)
        }

    def check_safety_limits(self) -> dict:
        """检查电池是否在安全运行范围内。"""
        warnings = []
        critical_alerts = []

        # SOC 检查
        if self.state_of_charge < 0.05:
            warnings.append("SOC过低 (<5%)")
        elif self.state_of_charge > 0.95:
            warnings.append("SOC过高 (>95%)")

        # 温度检查
        if self.cell_temperature_c > 60.0:
            critical_alerts.append(f"电池温度过高: {self.cell_temperature_c:.1f}°C")
        elif self.cell_temperature_c < 0.0:
            warnings.append(f"电池温度过低: {self.cell_temperature_c:.1f}°C")

        # 过载检查
        overload_ratio = abs(self.last_current_a) / max(1e-6, self.config.rated_current_a)
        if overload_ratio > 3.0:
            critical_alerts.append(f"严重过载: {overload_ratio:.1f}x 额定电流")
        elif overload_ratio > 2.0:
            warnings.append(f"中等过载: {overload_ratio:.1f}x 额定电流")

        return {
            'warnings': warnings,
            'critical_alerts': critical_alerts,
            'is_safe': len(critical_alerts) == 0,
            'overload_ratio': overload_ratio
        }

    def simulate_constant_power_discharge(self, power_w: float, ambient_temp_c: float,
                                          max_duration_hours: float = 2.0) -> dict:
        """模拟恒功率放电过程，返回放电曲线和关键指标。"""
        dt = 1.0  # 1秒步长
        max_steps = int(max_duration_hours * 3600)

        time_points = []
        voltage_points = []
        soc_points = []
        current_points = []
        temp_points = []

        initial_soc = self.state_of_charge
        initial_temp = self.cell_temperature_c

        def get_curr(E, P, R):
            """E: OCV[V], R: internal resistance[Ω], P: port power[W] (discharge>0, charge<0)"""
            disc = E * E - 4 * R * P
            if disc < 0:
                raise ValueError(f"P 超过可达范围：P ≤ E^2/(4R) = {E * E / (4 * R):.4f} W")
            I_minus = (E - math.sqrt(disc)) / (2 * R)
            I_plus = (E + math.sqrt(disc)) / (2 * R)
            return I_minus

        E = self._string_ocv_v(self.state_of_charge, self.cell_temperature_c)
        P = power_w
        R = self._string_resistance_ohm(self.state_of_charge, self.cell_temperature_c)
        current = get_curr(E, P, R)

        for step in range(max_steps):
            # 计算当前电流（基于功率和电压
            if current > self.config.rated_current_a * 3.0:  # 限制最大电流
                current = self.config.rated_current_a * 3.0

            # 更新状态
            self.update_state(current, dt, ambient_temp_c)

            # 记录数据
            time_points.append(step * dt / 3600.0)  # 转换为小时
            voltage_points.append(self.get_voltage())
            soc_points.append(self.state_of_charge)
            current_points.append(current)
            temp_points.append(self.cell_temperature_c)

            current = power_w / self.get_voltage()

            # 检查终止条件
            if self.state_of_charge <= 0.05 or self.get_voltage() <= 0:
                break

        # 恢复初始状态
        self.state_of_charge = initial_soc
        self.cell_temperature_c = initial_temp

        return {
            'time_hours': time_points,
            'voltage_v': voltage_points,
            'soc': soc_points,
            'current_a': current_points,
            'temperature_c': temp_points,
            'discharge_capacity_ah': (initial_soc - self.state_of_charge) * self.config.rated_capacity_ah,
            'discharge_energy_wh': power_w * (len(time_points) * dt / 3600.0),
            'max_current_a': max(current_points) if current_points else 0.0,
            'min_voltage_v': min(voltage_points) if voltage_points else 0.0
        }

    def step(self, power_w: float, ambient_temp_c: float):
        """模拟恒功率放电过程，返回放电曲线和关键指标。"""
        dt = 1.0  # 1秒步长

        def get_curr(E, P, R):
            """E: OCV[V], R: internal resistance[Ω], P: port power[W] (discharge>0, charge<0)"""
            disc = E * E - 4 * R * P
            if disc < 0:
                raise ValueError(f"P 超过可达范围：P ≤ E^2/(4R) = {E * E / (4 * R):.4f} W")
            I_minus = (E - math.sqrt(disc)) / (2 * R)
            I_plus = (E + math.sqrt(disc)) / (2 * R)
            return I_minus

        E = self._string_ocv_v(self.state_of_charge, self.cell_temperature_c)
        P = power_w
        R = self._string_resistance_ohm(self.state_of_charge, self.cell_temperature_c)
        current = get_curr(E, P, R)

        # 计算当前电流（基于功率和电压
        if current > self.config.rated_current_a * 3.0:  # 限制最大电流
            current = self.config.rated_current_a * 3.0

        # 更新状态
        self.update_state(current, dt, ambient_temp_c)
        return {
            'voltage_v': self.get_voltage(),
            'soc': self.state_of_charge,
            'current_a': current,
            'temperature_c': self.cell_temperature_c,
        }


if __name__ == "__main__":
    # 创建电池模型实例
    t_env = 25
    battery = BatteryModel(initial_soc=0.5, initial_temperature_c=t_env)

    print("=== 35 kV/25 MW 级联储能 PCS 电池模型测试 ===")
    print(f"初始状态: SOC={battery.state_of_charge:.1%}, 温度={battery.cell_temperature_c:.1f}°C")
    print(f"额定容量: {battery.config.rated_capacity_ah:.1f} Ah")
    print(f"额定电流: {battery.config.rated_current_a:.1f} A")
    print(f"串联电芯数: {battery.config.series_cells}")

    from device_parameters import get_optimized_parameters

    sys_params = get_optimized_parameters()['system']
    step_seconds = int(getattr(sys_params, 'time_step_seconds', 60))
    P_profile_raw, T_amb = generate_profiles(day_type="summer-weekday", step_seconds=step_seconds)
    p_profile = np.repeat(P_profile_raw, step_seconds) / 10000

    time_s = []
    voltage_v = []
    soc = []
    current_a = []
    temp_c = []

    t = 0
    for p in p_profile:
        rec = battery.step(float(p), t_env)  # 或用固定 t_env：battery.step(float(p), t_env)
        time_s.append(t)
        voltage_v.append(rec['voltage_v'])
        soc.append(rec['soc'])
        current_a.append(rec['current_a'])
        temp_c.append(rec['temperature_c'])
        t += 1  # 每步1秒

    time_s = np.asarray(time_s, dtype=float)
    voltage_v = np.asarray(voltage_v, dtype=float)
    soc = np.asarray(soc, dtype=float)
    current_a = np.asarray(current_a, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)

    # 时间轴（小时）
    th = np.asarray(time_s, dtype=float) / 3600.0

    # 创建 4 行 1 列子图（也可把 nrows=4, figsize 调大/调小）
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), sharex=True)

    # 1) 电压
    axs[0].plot(th, voltage_v)
    axs[0].set_ylabel("Voltage [V]")
    axs[0].set_title("Terminal Voltage")
    axs[0].grid(True)

    # 2) SOC（转百分比更直观）
    axs[1].plot(th, soc * 100.0)
    axs[1].set_ylabel("SOC [%]")
    axs[1].set_title("State of Charge")
    axs[1].grid(True)

    # 3) 电流
    axs[2].plot(th, current_a)
    axs[2].axhline(0, linestyle=":")  # 零参考线
    axs[2].set_ylabel("Current [A]")
    axs[2].set_title("Current")
    axs[2].grid(True)

    # 4) 功率（kW），显示实际与指令
    axs[3].plot(th, p_profile / 1000.0, label="Actual")
    axs[3].axhline(0, linestyle=":")  # 零参考线（充/放为正负）
    axs[3].set_ylabel("Power [kW]")
    axs[3].set_title("Power")
    axs[3].grid(True)

    axs[-1].set_xlabel("Time [h]")
    fig.suptitle("Battery Run — Voltage / SOC / Current / Power vs Time", y=0.995)
    fig.tight_layout()
    plt.show()


