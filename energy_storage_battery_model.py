"""
储能电池模型：基于PyBaMM的高精度储能系统建模

该模型专门为储能PCS系统设计，满足以下规格：
- 电池容量：314 Ah
- 串联数：312 串
- 模块电压范围：30 kV 至 40.5 kV
- 电池类型：磷酸铁锂（LFP）或镍钴钛（NCM）
- 额定功率：25 MW级联系统

功能特点：
1. 基于PyBaMM的电化学建模
2. 支持LFP和NCM两种化学体系
3. 精确的温度-电化学耦合
4. 详细的老化和寿命预测
5. 安全性分析和故障模式
6. 与PCS系统集成接口
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, Literal, List
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta

try:
    import pybamm
    PYBAMM_AVAILABLE = True
except ImportError:
    PYBAMM_AVAILABLE = False
    warnings.warn("PyBaMM未安装，将使用简化模型。请运行: pip install pybamm")

from battery_model import BatteryModelConfig


@dataclass
class EnergyStorageConfig:
    """储能电池系统配置"""
    
    # 基本规格参数
    battery_capacity_ah: float = 314.0  # 电池容量 (Ah)
    series_cells: int = 312  # 串联数
    module_voltage_min_kv: float = 30.0  # 最小模块电压 (kV)
    module_voltage_max_kv: float = 40.5  # 最大模块电压 (kV)
    rated_power_mw: float = 25.0  # 额定功率 (MW)
    
    # 电池化学体系
    battery_type: Literal["LFP", "NCM"] = "LFP"  # 电池类型
    
    # PyBaMM模型配置
    pybamm_model_type: Literal["SPM", "DFN", "SPMe"] = "SPM"
    thermal_model: str = "lumped"  # 热模型类型
    ageing_model: bool = True  # 是否启用老化模型
    
    # 安全参数
    max_overload_ratio: float = 3.0  # 最大过载倍数
    max_temperature_c: float = 60.0  # 最大工作温度
    min_temperature_c: float = -20.0  # 最小工作温度
    
    # 寿命参数
    target_life_years: float = 15.0  # 目标寿命（年）
    calendar_fade_per_year: float = 0.02  # 年日历衰减率
    cycle_fade_per_cycle: float = 5e-4  # 每循环衰减率


class EnergyStorageBatteryModel:
    """储能电池模型：基于PyBaMM的高精度建模"""
    
    def __init__(
        self,
        config: Optional[EnergyStorageConfig] = None,
        initial_soc: float = 0.5,
        initial_temperature_c: float = 25.0,
        use_pybamm: bool = True
    ):
        self.config = config or EnergyStorageConfig()
        self.use_pybamm = use_pybamm and PYBAMM_AVAILABLE
        
        # 初始化状态变量
        self.state_of_charge = float(max(0.0, min(1.0, initial_soc)))
        self.cell_temperature_c = float(initial_temperature_c)
        self.ambient_temperature_c = float(initial_temperature_c)
        self.current_a = 0.0
        self.voltage_v = 0.0
        self.power_w = 0.0
        
        # 寿命和健康状态
        self.capacity_fade_fraction = 0.0
        self.resistance_growth_fraction = 0.0
        self.health_state = 1.0  # 健康状态 (0-1)
        self.equivalent_full_cycles = 0.0
        self.calendar_age_years = 0.0
        
        # 安全状态
        self.safety_status = {
            'overload': False,
            'over_temperature': False,
            'over_voltage': False,
            'under_voltage': False,
            'over_current': False
        }
        
        # PyBaMM模型
        if self.use_pybamm:
            self._setup_pybamm_model()
        else:
            self.pybamm_model = None
            self.pybamm_sim = None
        
        # 仿真历史
        self.simulation_history = []
        self.operation_mode = "standby"  # standby, charging, discharging, fault
        
    def _setup_pybamm_model(self):
        """设置PyBaMM模型"""
        try:
            # 根据电池类型选择参数集
            if self.config.battery_type == "LFP":
                parameter_set = "Prada2013"
            else:  # NCM
                parameter_set = "Chen2020"
            
            # 创建模型
            if self.config.pybamm_model_type == "SPM":
                self.pybamm_model = pybamm.lithium_ion.SPM()
            elif self.config.pybamm_model_type == "DFN":
                self.pybamm_model = pybamm.lithium_ion.DFN()
            elif self.config.pybamm_model_type == "SPMe":
                self.pybamm_model = pybamm.lithium_ion.SPMe()
            
            # 添加热模型
            if self.config.thermal_model != "isothermal":
                self.pybamm_model = pybamm.thermal.lumped(self.pybamm_model)
            
            # 添加老化模型
            if self.config.ageing_model:
                self.pybamm_model = pybamm.ageing.SEI(
                    self.pybamm_model, 
                    "solvent-diffusion-limited"
                )
                self.pybamm_model = pybamm.ageing.lithium_plating(self.pybamm_model)
            
            # 设置参数
            self.param = pybamm.ParameterValues(parameter_set)
            
            # 调整参数以匹配实际规格
            self._adjust_parameters_for_storage()
            
            # 设置求解器
            self.solver = pybamm.CasadiSolver(mode="fast")
            
            # 初始化仿真
            self.pybamm_sim = None
            
        except Exception as e:
            warnings.warn(f"PyBaMM模型设置失败: {e}")
            self.use_pybamm = False
            self.pybamm_model = None
    
    def _adjust_parameters_for_storage(self):
        """调整参数以匹配储能系统规格"""
        try:
            # 调整容量参数
            if self.config.battery_type == "LFP":
                # LFP电池参数调整
                self.param["Negative electrode capacity [A.h/m2]"] *= 1.2
                self.param["Positive electrode capacity [A.h/m2]"] *= 1.2
            else:
                # NCM电池参数调整
                self.param["Negative electrode capacity [A.h/m2]"] *= 1.0
                self.param["Positive electrode capacity [A.h/m2]"] *= 1.0
            
            # 调整热参数
            self.param["Cell thermal mass [J/K]"] *= 2.0  # 储能电池热容更大
            self.param["Cell thermal conductivity [W/m/K]"] *= 1.5
            
        except Exception as e:
            warnings.warn(f"参数调整失败: {e}")
    
    def update_state(
        self, 
        current_a: float, 
        dt_s: float, 
        ambient_temp_c: float,
        operation_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新电池状态
        
        Args:
            current_a: 电流 (A)，正值表示放电
            dt_s: 时间步长 (s)
            ambient_temp_c: 环境温度 (°C)
            operation_mode: 操作模式
            
        Returns:
            状态更新结果
        """
        if operation_mode:
            self.operation_mode = operation_mode
        
        # 更新基本状态
        self.current_a = current_a
        self.ambient_temperature_c = ambient_temp_c
        
        # 计算功率
        self.power_w = abs(current_a * self.voltage_v)
        
        # 更新SOC
        if self.use_pybamm:
            self._update_pybamm_state(current_a, dt_s, ambient_temp_c)
        else:
            self._update_simplified_state(current_a, dt_s, ambient_temp_c)
        
        # 更新温度
        self._update_temperature(current_a, dt_s, ambient_temp_c)
        
        # 更新寿命和健康状态
        self._update_life_parameters(current_a, dt_s, ambient_temp_c)
        
        # 检查安全状态
        self._check_safety_limits()
        
        # 记录历史
        self._record_history(dt_s)
        
        return self.get_status()
    
    def _update_pybamm_state(self, current_a: float, dt_s: float, ambient_temp_c: float):
        """更新PyBaMM模型状态"""
        try:
            if self.pybamm_sim is None:
                # 首次运行
                self.pybamm_sim = pybamm.Simulation(
                    self.pybamm_model,
                    parameter_values=self.param,
                    solver=self.solver
                )
                
                t_eval = np.array([0, dt_s])
                solution = self.pybamm_sim.solve(t_eval)
                self.last_pybamm_solution = solution
            else:
                # 继续仿真
                t_eval = np.array([
                    self.last_pybamm_solution.t[-1],
                    self.last_pybamm_solution.t[-1] + dt_s
                ])
                solution = self.pybamm_sim.solve(t_eval, initial_conditions=self.last_pybamm_solution)
                self.last_pybamm_solution = solution
            
            # 提取结果
            self._extract_pybamm_results(solution)
            
        except Exception as e:
            warnings.warn(f"PyBaMM仿真失败: {e}")
            self._update_simplified_state(current_a, dt_s, ambient_temp_c)
    
    def _extract_pybamm_results(self, solution):
        """从PyBaMM结果中提取关键参数"""
        try:
            # 提取电压
            voltage = solution["Terminal voltage [V]"].entries[-1]
            self.voltage_v = voltage * self.config.series_cells
            
            # 提取SOC
            discharge_capacity = solution["Discharge capacity [A.h]"].entries[-1]
            initial_capacity = solution["Discharge capacity [A.h]"].entries[0]
            self.state_of_charge = 1.0 - (discharge_capacity / initial_capacity)
            
            # 提取温度
            if "Cell temperature [K]" in solution:
                temp_k = solution["Cell temperature [K]"].entries[-1]
                self.cell_temperature_c = temp_k - 273.15
            
            # 提取老化信息
            if "Loss of lithium inventory [%]" in solution:
                lli = solution["Loss of lithium inventory [%]"].entries[-1]
                self.capacity_fade_fraction = lli / 100.0
            
        except Exception as e:
            warnings.warn(f"PyBaMM结果提取失败: {e}")
    
    def _update_simplified_state(self, current_a: float, dt_s: float, ambient_temp_c: float):
        """更新简化模型状态（PyBaMM不可用时）"""
        # 简化的SOC更新
        capacity_ah = self.config.battery_capacity_ah * (1.0 - self.capacity_fade_fraction)
        soc_change = current_a * dt_s / 3600.0 / capacity_ah
        self.state_of_charge = max(0.0, min(1.0, self.state_of_charge - soc_change))
        
        # 简化的电压计算
        nominal_voltage = (self.config.module_voltage_min_kv + self.config.module_voltage_max_kv) / 2
        self.voltage_v = nominal_voltage * 1000 * self.state_of_charge
    
    def _update_temperature(self, current_a: float, dt_s: float, ambient_temp_c: float):
        """更新温度模型"""
        # 热模型参数
        thermal_resistance = 0.1  # K/W
        thermal_capacity = 1000.0  # J/K
        
        # 热源（焦耳热）
        resistance = 0.1  # 欧姆
        heat_generation = current_a**2 * resistance
        
        # 热传导
        heat_conduction = (self.cell_temperature_c - ambient_temp_c) / thermal_resistance
        
        # 温度变化
        net_heat = heat_generation - heat_conduction
        temperature_change = net_heat * dt_s / thermal_capacity
        
        self.cell_temperature_c += temperature_change
        
        # 限制温度范围
        self.cell_temperature_c = max(
            self.config.min_temperature_c,
            min(self.config.max_temperature_c, self.cell_temperature_c)
        )
    
    def _update_life_parameters(self, current_a: float, dt_s: float, ambient_temp_c: float):
        """更新寿命和健康状态参数"""
        # 日历老化
        calendar_time = dt_s / (365.25 * 24 * 3600)  # 转换为年
        self.calendar_age_years += calendar_time
        
        calendar_fade = self.config.calendar_fade_per_year * calendar_time
        self.capacity_fade_fraction += calendar_fade
        
        # 循环老化
        if abs(current_a) > 0:
            # 计算等效循环
            capacity_ah = self.config.battery_capacity_ah
            cycle_equivalent = abs(current_a) * dt_s / 3600.0 / capacity_ah
            self.equivalent_full_cycles += cycle_equivalent
            
            # 循环衰减
            cycle_fade = self.config.cycle_fade_per_cycle * cycle_equivalent
            self.capacity_fade_fraction += cycle_fade
        
        # 温度加速老化
        if ambient_temp_c > 25.0:
            temp_acceleration = 2.0**((ambient_temp_c - 25.0) / 10.0)
            self.capacity_fade_fraction *= temp_acceleration
        
        # 限制衰减范围
        self.capacity_fade_fraction = min(0.8, self.capacity_fade_fraction)
        
        # 更新健康状态
        self.health_state = 1.0 - self.capacity_fade_fraction
        
        # 内阻增长（简化模型）
        self.resistance_growth_fraction = self.capacity_fade_fraction * 0.5
    
    def _check_safety_limits(self):
        """检查安全限制"""
        # 过载检查
        rated_current = self.config.rated_power_mw * 1e6 / (self.config.module_voltage_max_kv * 1e3)
        if abs(self.current_a) > rated_current * self.config.max_overload_ratio:
            self.safety_status['overload'] = True
        else:
            self.safety_status['overload'] = False
        
        # 温度检查
        if self.cell_temperature_c > self.config.max_temperature_c:
            self.safety_status['over_temperature'] = True
        elif self.cell_temperature_c < self.config.min_temperature_c:
            self.safety_status['over_temperature'] = True
        else:
            self.safety_status['over_temperature'] = False
        
        # 电压检查
        voltage_kv = self.voltage_v / 1000.0
        if voltage_kv > self.config.module_voltage_max_kv:
            self.safety_status['over_voltage'] = True
        elif voltage_kv < self.config.module_voltage_min_kv:
            self.safety_status['under_voltage'] = True
        else:
            self.safety_status['over_voltage'] = False
            self.safety_status['under_voltage'] = False
        
        # 电流检查
        if abs(self.current_a) > rated_current * 2.0:
            self.safety_status['over_current'] = True
        else:
            self.safety_status['over_current'] = False
    
    def _record_history(self, dt_s: float):
        """记录仿真历史"""
        timestamp = len(self.simulation_history) * dt_s
        
        self.simulation_history.append({
            'timestamp': timestamp,
            'soc': self.state_of_charge,
            'voltage_v': self.voltage_v,
            'current_a': self.current_a,
            'power_w': self.power_w,
            'temperature_c': self.cell_temperature_c,
            'health_state': self.health_state,
            'operation_mode': self.operation_mode,
            'safety_status': self.safety_status.copy()
        })
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'soc': self.state_of_charge,
            'voltage_v': self.voltage_v,
            'voltage_kv': self.voltage_v / 1000.0,
            'current_a': self.current_a,
            'power_w': self.power_w,
            'power_mw': self.power_w / 1e6,
            'temperature_c': self.cell_temperature_c,
            'health_state': self.health_state,
            'capacity_fade_fraction': self.capacity_fade_fraction,
            'resistance_growth_fraction': self.resistance_growth_fraction,
            'equivalent_full_cycles': self.equivalent_full_cycles,
            'calendar_age_years': self.calendar_age_years,
            'operation_mode': self.operation_mode,
            'safety_status': self.safety_status.copy(),
            'pybamm_available': self.use_pybamm
        }
    
    def get_battery_specs(self) -> Dict[str, Any]:
        """获取电池规格信息"""
        return {
            'battery_type': self.config.battery_type,
            'battery_capacity_ah': self.config.battery_capacity_ah,
            'series_cells': self.config.series_cells,
            'module_voltage_range_kv': [
                self.config.module_voltage_min_kv,
                self.config.module_voltage_max_kv
            ],
            'rated_power_mw': self.config.rated_power_mw,
            'max_overload_ratio': self.config.max_overload_ratio,
            'temperature_range_c': [
                self.config.min_temperature_c,
                self.config.config.max_temperature_c
            ],
            'target_life_years': self.config.target_life_years
        }
    
    def estimate_remaining_life(self) -> Dict[str, Any]:
        """估计剩余寿命"""
        # 基于容量衰减的寿命估计
        if self.capacity_fade_fraction > 0:
            remaining_capacity = 1.0 - self.capacity_fade_fraction
            life_consumed = 1.0 - remaining_capacity
            
            # 假设线性衰减
            if life_consumed > 0:
                remaining_life_ratio = remaining_capacity / life_consumed
                remaining_life_years = self.calendar_age_years * remaining_life_ratio
            else:
                remaining_life_years = self.config.target_life_years
        else:
            remaining_life_years = self.config.target_life_years
        
        # 基于健康状态的寿命估计
        health_based_life = self.health_state * self.config.target_life_years
        
        return {
            'remaining_life_years': min(remaining_life_years, health_based_life),
            'life_consumed_percent': (1.0 - self.health_state) * 100.0,
            'capacity_health_percent': (1.0 - self.capacity_fade_fraction) * 100.0,
            'resistance_health_percent': (1.0 - self.resistance_growth_fraction) * 100.0,
            'overall_health_percent': self.health_state * 100.0
        }
    
    def simulate_operation_scenario(
        self, 
        scenario: str,
        duration_hours: float = 24.0
    ) -> Dict[str, Any]:
        """模拟特定操作场景"""
        scenarios = {
            'daily_cycle': self._simulate_daily_cycle,
            'peak_shaving': self._simulate_peak_shaving,
            'frequency_regulation': self._simulate_frequency_regulation,
            'emergency_backup': self._simulate_emergency_backup
        }
        
        if scenario in scenarios:
            return scenarios[scenario](duration_hours)
        else:
            raise ValueError(f"未知场景: {scenario}")
    
    def _simulate_daily_cycle(self, duration_hours: float) -> Dict[str, Any]:
        """模拟日循环场景"""
        dt = 300  # 5分钟时间步
        total_steps = int(duration_hours * 3600 / dt)
        
        # 日负荷曲线（简化）
        load_profile = []
        for i in range(total_steps):
            hour = i * dt / 3600.0
            # 模拟日负荷变化
            if 6 <= hour <= 9:  # 早高峰
                load = 0.8
            elif 18 <= hour <= 21:  # 晚高峰
                load = 0.9
            elif 23 <= hour or hour <= 6:  # 夜间低谷
                load = 0.2
            else:  # 平时
                load = 0.5
            
            load_profile.append(load)
        
        return self._run_scenario_simulation(load_profile, dt, "daily_cycle")
    
    def _simulate_peak_shaving(self, duration_hours: float) -> Dict[str, Any]:
        """模拟削峰填谷场景"""
        dt = 600  # 10分钟时间步
        total_steps = int(duration_hours * 3600 / dt)
        
        # 削峰填谷负荷曲线
        load_profile = []
        for i in range(total_steps):
            hour = i * dt / 3600.0
            # 模拟电网负荷变化
            if 10 <= hour <= 16:  # 午间高峰
                load = 0.9
            elif 19 <= hour <= 22:  # 晚间高峰
                load = 0.95
            elif 2 <= hour <= 6:  # 凌晨低谷
                load = 0.1
            else:
                load = 0.6
            
            load_profile.append(load)
        
        return self._run_scenario_simulation(load_profile, dt, "peak_shaving")
    
    def _simulate_frequency_regulation(self, duration_hours: float) -> Dict[str, Any]:
        """模拟频率调节场景"""
        dt = 60  # 1分钟时间步
        total_steps = int(duration_hours * 3600 / dt)
        
        # 频率调节负荷曲线（快速变化）
        load_profile = []
        for i in range(total_steps):
            # 模拟频率调节的快速变化
            load = 0.5 + 0.3 * np.sin(i * 0.1) + 0.1 * np.random.random()
            load = max(0.1, min(0.9, load))
            load_profile.append(load)
        
        return self._run_scenario_simulation(load_profile, dt, "frequency_regulation")
    
    def _simulate_emergency_backup(self, duration_hours: float) -> Dict[str, Any]:
        """模拟应急备用场景"""
        dt = 300  # 5分钟时间步
        total_steps = int(duration_hours * 3600 / dt)
        
        # 应急备用负荷曲线
        load_profile = []
        for i in range(total_steps):
            hour = i * dt / 3600.0
            # 模拟应急情况
            if hour < 2:  # 前2小时高负荷
                load = 0.95
            elif hour < 4:  # 2-4小时中等负荷
                load = 0.7
            else:  # 4小时后低负荷
                load = 0.3
            
            load_profile.append(load)
        
        return self._run_scenario_simulation(load_profile, dt, "emergency_backup")
    
    def _run_scenario_simulation(
        self, 
        load_profile: List[float], 
        dt: float, 
        scenario_name: str
    ) -> Dict[str, Any]:
        """运行场景仿真"""
        # 保存当前状态
        original_state = self.get_status()
        
        # 重置仿真历史
        self.simulation_history = []
        
        # 运行仿真
        start_time = datetime.now()
        
        for i, load_ratio in enumerate(load_profile):
            # 计算电流（基于功率和电压）
            target_power = load_ratio * self.config.rated_power_mw * 1e6
            if self.voltage_v > 0:
                target_current = target_power / self.voltage_v
            else:
                target_current = 0
            
            # 更新状态
            self.update_state(target_current, dt, self.ambient_temperature_c)
            
            # 检查安全状态
            if any(self.safety_status.values()):
                break
        
        simulation_time = (datetime.now() - start_time).total_seconds()
        
        # 分析结果
        results = self._analyze_simulation_results(scenario_name)
        results['simulation_time_seconds'] = simulation_time
        results['total_steps'] = len(self.simulation_history)
        
        # 恢复原始状态
        self.state_of_charge = original_state['soc']
        self.voltage_v = original_state['voltage_v']
        self.cell_temperature_c = original_state['temperature_c']
        
        return results
    
    def _analyze_simulation_results(self, scenario_name: str) -> Dict[str, Any]:
        """分析仿真结果"""
        if not self.simulation_history:
            return {}
        
        # 提取数据
        socs = [h['soc'] for h in self.simulation_history]
        voltages = [h['voltage_v'] for h in self.simulation_history]
        currents = [h['current_a'] for h in self.simulation_history]
        temperatures = [h['temperature_c'] for h in self.simulation_history]
        health_states = [h['health_state'] for h in self.simulation_history]
        
        # 计算统计信息
        results = {
            'scenario_name': scenario_name,
            'soc_range': [min(socs), max(socs)],
            'voltage_range_kv': [min(voltages)/1000, max(voltages)/1000],
            'current_range_a': [min(currents), max(currents)],
            'temperature_range_c': [min(temperatures), max(temperatures)],
            'health_degradation': 1.0 - min(health_states),
            'safety_violations': sum(1 for h in self.simulation_history if any(h['safety_status'].values())),
            'energy_throughput_mwh': sum(abs(h['power_w'] * 0.001) for h in self.simulation_history) / 3600,
            'peak_power_mw': max(h['power_w'] for h in self.simulation_history) / 1e6
        }
        
        return results
    
    def plot_operation_results(self, save_path: Optional[str] = None):
        """绘制操作结果"""
        if not self.simulation_history:
            print("没有仿真历史数据")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 提取数据
        timestamps = [h['timestamp'] for h in self.simulation_history]
        socs = [h['soc'] for h in self.simulation_history]
        voltages = [h['voltage_v']/1000 for h in self.simulation_history]
        currents = [h['current_a'] for h in self.simulation_history]
        temperatures = [h['temperature_c'] for h in self.simulation_history]
        
        # SOC曲线
        ax1.plot(timestamps, socs, 'b-', linewidth=2)
        ax1.set_ylabel('SOC')
        ax1.set_title('荷电状态变化')
        ax1.grid(True)
        
        # 电压曲线
        ax2.plot(timestamps, voltages, 'r-', linewidth=2)
        ax2.set_ylabel('电压 (kV)')
        ax2.set_title('模块电压变化')
        ax2.grid(True)
        
        # 电流曲线
        ax3.plot(timestamps, currents, 'g-', linewidth=2)
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('电流 (A)')
        ax3.set_title('电流变化')
        ax3.grid(True)
        
        # 温度曲线
        ax4.plot(timestamps, temperatures, 'm-', linewidth=2)
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('温度 (°C)')
        ax4.set_title('电池温度变化')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_to_pcs_format(self) -> Dict[str, Any]:
        """导出为PCS系统可用的格式"""
        return {
            'battery_specs': self.get_battery_specs(),
            'current_status': self.get_status(),
            'life_estimation': self.estimate_remaining_life(),
            'safety_alarms': self.safety_status,
            'operation_limits': {
                'max_current_a': self.config.rated_power_mw * 1e6 / (self.config.module_voltage_min_kv * 1e3),
                'min_voltage_kv': self.config.module_voltage_min_kv,
                'max_voltage_kv': self.config.module_voltage_max_kv,
                'min_temperature_c': self.config.min_temperature_c,
                'max_temperature_c': self.config.max_temperature_c,
                'max_soc': 0.95,
                'min_soc': 0.05
            },
            'control_parameters': {
                'charge_current_limit_a': self.config.rated_power_mw * 1e6 / (self.config.module_voltage_max_kv * 1e3),
                'discharge_current_limit_a': self.config.rated_power_mw * 1e6 / (self.config.module_voltage_min_kv * 1e3),
                'temperature_control_c': 25.0,
                'soc_control_range': [0.1, 0.9]
            }
        }


def create_storage_battery_example():
    """创建储能电池模型示例"""
    # 配置储能电池
    config = EnergyStorageConfig(
        battery_type="LFP",
        pybamm_model_type="SPM",
        thermal_model="lumped",
        ageing_model=True
    )
    
    # 创建模型
    battery = EnergyStorageBatteryModel(
        config=config,
        initial_soc=0.8,
        initial_temperature_c=25.0,
        use_pybamm=True
    )
    
    print("储能电池模型创建成功！")
    print(f"电池类型: {config.battery_type}")
    print(f"电池容量: {config.battery_capacity_ah} Ah")
    print(f"串联数: {config.series_cells}")
    print(f"电压范围: {config.module_voltage_min_kv}-{config.module_voltage_max_kv} kV")
    
    return battery


if __name__ == "__main__":
    # 创建示例模型
    battery = create_storage_battery_example()
    
    # 运行日循环仿真
    print("\n开始日循环仿真...")
    results = battery.simulate_operation_scenario('daily_cycle', 24.0)
    
    print(f"\n仿真结果:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # 绘制结果
    battery.plot_operation_results()
    
    # 导出PCS格式
    pcs_data = battery.export_to_pcs_format()
    print(f"\nPCS集成数据已准备就绪")
