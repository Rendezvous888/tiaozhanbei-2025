"""
构网型级联储能PCS核心模型
实现35kV/25MW级联储能系统的核心建模
包含H桥级联、IGBT、母线电容等关键器件
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

@dataclass
class SystemParameters:
    """系统级参数配置"""
    # 系统规格
    rated_power: float = 25e6  # 25 MW
    rated_voltage: float = 35e3  # 35 kV
    rated_current: float = 420.0  # A
    frequency: float = 50.0  # Hz
    
    # 级联配置
    h_bridge_per_phase: int = 40  # 每相H桥数量
    total_phases: int = 3  # 三相系统
    
    # 开关频率
    switching_frequency: float = 1000.0  # Hz
    
    # 温度范围
    min_temp: float = 20.0  # ℃
    max_temp: float = 40.0  # ℃
    nominal_temp: float = 25.0  # ℃
    
    # 过载能力
    overload_capability: float = 3.0  # 3倍过载
    overload_duration: float = 10.0  # 10秒

@dataclass
class IGBTParameters:
    """IGBT器件参数"""
    # 基本参数
    rated_voltage: float = 1700.0  # V
    rated_current: float = 1500.0  # A
    switching_frequency: float = 1000.0  # Hz
    
    # 热参数
    thermal_resistance_jc: float = 0.1  # K/W, 结到壳热阻
    thermal_resistance_cs: float = 0.05  # K/W, 壳到散热器热阻
    thermal_resistance_sa: float = 0.02  # K/W, 散热器到环境热阻
    
    # 损耗参数
    conduction_loss_coeff: float = 2.0  # V, 导通压降
    switching_loss_coeff: float = 0.001  # J/A, 开关损耗系数

@dataclass
class CapacitorParameters:
    """母线电容参数"""
    # 基本参数
    capacitance: float = 15e-3  # 15 mF
    rated_voltage: float = 1200.0  # V
    max_current: float = 80.0  # A
    
    # 寿命参数
    rated_lifetime: float = 100000.0  # 小时
    max_temperature: float = 70.0  # ℃
    thermal_coefficient: float = 2.0  # 温度系数

class HBridgeModule:
    """H桥模块类"""
    
    def __init__(self, igbt_params: IGBTParameters, cap_params: CapacitorParameters):
        self.igbt_params = igbt_params
        self.cap_params = cap_params
        
        # 状态变量
        self.igbt_temperature = 25.0  # ℃
        self.cap_temperature = 25.0  # ℃
        self.cap_voltage = 0.0  # V
        self.output_voltage = 0.0  # V
        self.output_current = 0.0  # A
        
        # 损耗计算
        self.igbt_conduction_loss = 0.0  # W
        self.igbt_switching_loss = 0.0  # W
        self.cap_loss = 0.0  # W
        
        # 寿命消耗
        self.igbt_life_consumption = 0.0  # 归一化寿命消耗
        self.cap_life_consumption = 0.0  # 归一化寿命消耗
    
    def update_temperature(self, ambient_temp: float, time_step: float):
        """更新器件温度"""
        # IGBT温度更新（简化热模型）
        total_igbt_loss = self.igbt_conduction_loss + self.igbt_switching_loss
        thermal_resistance_total = (self.igbt_params.thermal_resistance_jc + 
                                  self.igbt_params.thermal_resistance_cs + 
                                  self.igbt_params.thermal_resistance_sa)
        
        # 指数响应热模型
        temp_diff = total_igbt_loss * thermal_resistance_total
        self.igbt_temperature = ambient_temp + temp_diff * (1 - np.exp(-time_step / 10.0))
        
        # 电容温度更新
        self.cap_temperature = ambient_temp + self.cap_loss * 0.01  # 简化模型
    
    def calculate_losses(self, voltage: float, current: float, duty_cycle: float):
        """计算器件损耗"""
        # IGBT导通损耗
        self.igbt_conduction_loss = (self.igbt_params.conduction_loss_coeff * 
                                   abs(current) * duty_cycle)
        
        # IGBT开关损耗
        self.igbt_switching_loss = (self.igbt_params.switching_loss_coeff * 
                                  abs(current) * self.igbt_params.switching_frequency)
        
        # 电容损耗（ESR损耗）
        self.cap_loss = 0.001 * current**2  # 简化模型
    
    def calculate_life_consumption(self, time_step: float):
        """计算寿命消耗"""
        # IGBT寿命消耗（基于温度循环）
        if self.igbt_temperature > 125:  # 超过125℃加速老化
            temp_factor = (self.igbt_temperature - 125) / 25
            self.igbt_life_consumption += time_step * (1 + temp_factor) / 87600  # 年化
        
        # 电容寿命消耗（基于温度）
        if self.cap_temperature > 70:
            temp_factor = (self.cap_temperature - 70) / 10
            self.cap_life_consumption += time_step * (1 + temp_factor) / 87600

class CascadedPCS:
    """级联PCS系统类"""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        
        # 创建H桥模块
        igbt_params = IGBTParameters()
        cap_params = CapacitorParameters()
        
        self.h_bridges = []
        for phase in range(system_params.total_phases):
            phase_bridges = []
            for i in range(system_params.h_bridge_per_phase):
                bridge = HBridgeModule(igbt_params, cap_params)
                phase_bridges.append(bridge)
            self.h_bridges.append(phase_bridges)
        
        # 系统状态
        self.dc_voltage = 0.0  # V
        self.ac_voltage = np.zeros(3)  # V
        self.ac_current = np.zeros(3)  # A
        self.active_power = 0.0  # W
        self.reactive_power = 0.0  # VAR
        
        # 时间相关
        self.simulation_time = 0.0  # s
        self.time_step = 0.001  # s (1ms)
    
    def set_power_reference(self, active_power: float, reactive_power: float = 0.0):
        """设置功率参考值"""
        self.active_power = active_power
        self.reactive_power = reactive_power
        
        # 计算电流参考值
        if abs(self.ac_voltage[0]) > 1e-6:
            self.ac_current[0] = (self.active_power + 1j * self.reactive_power) / (3 * self.ac_voltage[0])
            self.ac_current[1] = self.ac_current[0] * np.exp(-1j * 2 * np.pi / 3)
            self.ac_current[2] = self.ac_current[0] * np.exp(1j * 2 * np.pi / 3)
    
    def update_control(self, time: float):
        """更新控制逻辑"""
        # 简化的PWM控制
        for phase in range(self.system_params.total_phases):
            for i, bridge in enumerate(self.h_bridges[phase]):
                # 计算调制比
                modulation_index = 0.8  # 固定调制比
                
                # 计算占空比
                phase_angle = 2 * np.pi * self.system_params.frequency * time + phase * 2 * np.pi / 3
                duty_cycle = 0.5 + 0.5 * modulation_index * np.sin(phase_angle)
                
                # 更新H桥状态
                bridge.output_voltage = self.dc_voltage * (2 * duty_cycle - 1)
                bridge.output_current = self.ac_current[phase].real
                
                # 计算损耗
                bridge.calculate_losses(bridge.output_voltage, bridge.output_current, duty_cycle)
    
    def update_temperature(self, ambient_temp: float):
        """更新所有器件的温度"""
        for phase_bridges in self.h_bridges:
            for bridge in phase_bridges:
                bridge.update_temperature(ambient_temp, self.time_step)
                bridge.calculate_life_consumption(self.time_step)
    
    def step_simulation(self, time: float, ambient_temp: float):
        """仿真步进"""
        self.simulation_time = time
        
        # 更新控制
        self.update_control(time)
        
        # 更新温度
        self.update_temperature(ambient_temp)
        
        # 更新系统状态
        self._update_system_state()
    
    def _update_system_state(self):
        """更新系统状态"""
        # 计算总输出电压
        total_voltage = np.zeros(3)
        for phase in range(self.system_params.total_phases):
            for bridge in self.h_bridges[phase]:
                total_voltage[phase] += bridge.output_voltage
        
        self.ac_voltage = total_voltage
    
    def get_health_status(self) -> Dict:
        """获取系统健康状态"""
        total_igbt_life = 0.0
        total_cap_life = 0.0
        total_modules = 0
        
        for phase_bridges in self.h_bridges:
            for bridge in phase_bridges:
                total_igbt_life += bridge.igbt_life_consumption
                total_cap_life += bridge.cap_life_consumption
                total_modules += 1
        
        avg_igbt_life = total_igbt_life / total_modules
        avg_cap_life = total_cap_life / total_modules
        
        # 计算健康度（0-100分）
        igbt_health = max(0, 100 * (1 - avg_igbt_life))
        cap_health = max(0, 100 * (1 - avg_cap_life))
        
        # 综合健康度（取最小值）
        overall_health = min(igbt_health, cap_health)
        
        return {
            'overall_health': overall_health,
            'igbt_health': igbt_health,
            'capacitor_health': cap_health,
            'igbt_life_consumption': avg_igbt_life,
            'capacitor_life_consumption': avg_cap_life,
            'total_modules': total_modules
        }
    
    def get_temperature_distribution(self) -> Dict:
        """获取温度分布"""
        igbt_temps = []
        cap_temps = []
        
        for phase_bridges in self.h_bridges:
            for bridge in phase_bridges:
                igbt_temps.append(bridge.igbt_temperature)
                cap_temps.append(bridge.cap_temperature)
        
        return {
            'igbt_temperatures': igbt_temps,
            'capacitor_temperatures': cap_temps,
            'max_igbt_temp': max(igbt_temps),
            'min_igbt_temp': min(igbt_temps),
            'max_cap_temp': max(cap_temps),
            'min_cap_temp': min(cap_temps)
        }
