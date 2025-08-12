"""
关键器件寿命预测模型
实现IGBT和母线电容的寿命预测算法
基于温度循环、应力分析等关键因素
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

@dataclass
class LifeModelParameters:
    """寿命模型参数"""
    # IGBT寿命模型参数
    igbt_base_lifetime: float = 100000.0  # 小时
    igbt_temperature_coefficient: float = 2.0  # 温度系数
    igbt_thermal_cycling_coefficient: float = 1.5  # 热循环系数
    igbt_current_stress_coefficient: float = 1.2  # 电流应力系数
    
    # 电容寿命模型参数
    cap_base_lifetime: float = 100000.0  # 小时
    cap_temperature_coefficient: float = 2.0  # 温度系数
    cap_voltage_coefficient: float = 1.5  # 电压应力系数
    cap_ripple_coefficient: float = 1.3  # 纹波电流系数

class IGBTLifeModel:
    """IGBT寿命预测模型"""
    
    def __init__(self, params: LifeModelParameters):
        self.params = params
        self.temperature_history = []
        self.current_history = []
        self.time_history = []
        
        # 寿命消耗累计
        self.total_life_consumption = 0.0
        self.remaining_life = 1.0  # 归一化剩余寿命
        
        # 热循环统计
        self.thermal_cycles = []
        self.cycle_amplitudes = []
    
    def update_operating_conditions(self, temperature: float, current: float, time: float):
        """更新运行条件"""
        self.temperature_history.append(temperature)
        self.current_history.append(current)
        self.time_history.append(time)
        
        # 保持历史记录长度
        if len(self.temperature_history) > 1000:
            self.temperature_history.pop(0)
            self.current_history.pop(0)
            self.time_history.pop(0)
    
    def calculate_thermal_cycles(self) -> List[Tuple[float, float]]:
        """计算热循环"""
        if len(self.temperature_history) < 2:
            return []
        
        cycles = []
        amplitudes = []
        
        # 简化的峰值检测算法
        for i in range(1, len(self.temperature_history) - 1):
            if (self.temperature_history[i] > self.temperature_history[i-1] and 
                self.temperature_history[i] > self.temperature_history[i+1]):
                # 峰值
                if len(cycles) > 0:
                    # 计算与前一个峰值的循环
                    cycle_amplitude = abs(self.temperature_history[i] - cycles[-1][1])
                    amplitudes.append(cycle_amplitude)
                
                cycles.append((self.time_history[i], self.temperature_history[i]))
        
        self.thermal_cycles = cycles
        self.cycle_amplitudes = amplitudes
        
        return cycles
    
    def calculate_life_consumption(self, time_step: float) -> float:
        """计算寿命消耗"""
        if len(self.temperature_history) < 2:
            return 0.0
        
        # 计算热循环
        self.calculate_thermal_cycles()
        
        # 基于温度的寿命消耗
        current_temp = self.temperature_history[-1]
        temp_factor = 1.0
        
        if current_temp > 125:  # 超过125℃加速老化
            temp_factor = 2.0 ** ((current_temp - 125) / 10)
        elif current_temp > 100:  # 100-125℃中等加速
            temp_factor = 1.5 ** ((current_temp - 100) / 25)
        
        # 基于电流应力的寿命消耗
        current_factor = 1.0
        current_stress = abs(self.current_history[-1]) / 1500.0  # 归一化到额定电流
        
        if current_stress > 1.0:  # 过载情况
            current_factor = 1.0 + self.params.igbt_current_stress_coefficient * (current_stress - 1.0)
        
        # 基于热循环的寿命消耗
        cycle_factor = 1.0
        if self.cycle_amplitudes:
            avg_amplitude = np.mean(self.cycle_amplitudes)
            if avg_amplitude > 20:  # 超过20℃的热循环
                cycle_factor = 1.0 + self.params.igbt_thermal_cycling_coefficient * (avg_amplitude - 20) / 50
        
        # 综合寿命消耗
        life_consumption = (time_step / self.params.igbt_base_lifetime) * temp_factor * current_factor * cycle_factor
        
        self.total_life_consumption += life_consumption
        self.remaining_life = max(0.0, 1.0 - self.total_life_consumption)
        
        return life_consumption
    
    def predict_remaining_lifetime(self, current_conditions: Dict) -> float:
        """预测剩余寿命"""
        # 基于当前条件预测
        temp = current_conditions.get('temperature', 25.0)
        current = current_conditions.get('current', 0.0)
        
        # 计算当前条件下的寿命消耗率
        temp_factor = 1.0
        if temp > 125:
            temp_factor = 2.0 ** ((temp - 125) / 10)
        elif temp > 100:
            temp_factor = 1.5 ** ((temp - 100) / 25)
        
        current_factor = 1.0
        current_stress = abs(current) / 1500.0
        if current_stress > 1.0:
            current_factor = 1.0 + self.params.igbt_current_stress_coefficient * (current_stress - 1.0)
        
        # 预测剩余寿命
        consumption_rate = temp_factor * current_factor / self.params.igbt_base_lifetime
        if consumption_rate > 0:
            remaining_hours = self.remaining_life / consumption_rate
            return remaining_hours
        
        return float('inf')
    
    def get_life_status(self) -> Dict:
        """获取寿命状态"""
        return {
            'remaining_life': self.remaining_life,
            'total_consumption': self.total_life_consumption,
            'thermal_cycles': len(self.thermal_cycles),
            'max_cycle_amplitude': max(self.cycle_amplitudes) if self.cycle_amplitudes else 0.0,
            'avg_cycle_amplitude': np.mean(self.cycle_amplitudes) if self.cycle_amplitudes else 0.0
        }

class CapacitorLifeModel:
    """母线电容寿命预测模型"""
    
    def __init__(self, params: LifeModelParameters):
        self.params = params
        self.temperature_history = []
        self.voltage_history = []
        self.ripple_current_history = []
        self.time_history = []
        
        # 寿命消耗累计
        self.total_life_consumption = 0.0
        self.remaining_life = 1.0
        
        # 应力统计
        self.voltage_stress_history = []
        self.ripple_stress_history = []
    
    def update_operating_conditions(self, temperature: float, voltage: float, 
                                  ripple_current: float, time: float):
        """更新运行条件"""
        self.temperature_history.append(temperature)
        self.voltage_history.append(voltage)
        self.ripple_current_history.append(ripple_current)
        self.time_history.append(time)
        
        # 保持历史记录长度
        if len(self.temperature_history) > 1000:
            self.temperature_history.pop(0)
            self.voltage_history.pop(0)
            self.ripple_current_history.pop(0)
            self.time_history.pop(0)
    
    def calculate_life_consumption(self, time_step: float) -> float:
        """计算寿命消耗"""
        if len(self.temperature_history) < 1:
            return 0.0
        
        current_temp = self.temperature_history[-1]
        current_voltage = self.voltage_history[-1]
        current_ripple = self.ripple_current_history[-1]
        
        # 基于温度的寿命消耗
        temp_factor = 1.0
        if current_temp > 70:  # 超过70℃加速老化
            temp_factor = 2.0 ** ((current_temp - 70) / 10)
        elif current_temp > 50:  # 50-70℃中等加速
            temp_factor = 1.5 ** ((current_temp - 50) / 20)
        
        # 基于电压应力的寿命消耗
        voltage_factor = 1.0
        voltage_stress = current_voltage / 1200.0  # 归一化到额定电压
        
        if voltage_stress > 0.8:  # 超过80%额定电压
            voltage_factor = 1.0 + self.params.cap_voltage_coefficient * (voltage_stress - 0.8) / 0.2
        
        # 基于纹波电流的寿命消耗
        ripple_factor = 1.0
        ripple_stress = current_ripple / 80.0  # 归一化到额定纹波电流
        
        if ripple_stress > 0.8:  # 超过80%额定纹波电流
            ripple_factor = 1.0 + self.params.cap_ripple_coefficient * (ripple_stress - 0.8) / 0.2
        
        # 综合寿命消耗
        life_consumption = (time_step / self.params.cap_base_lifetime) * temp_factor * voltage_factor * ripple_factor
        
        self.total_life_consumption += life_consumption
        self.remaining_life = max(0.0, 1.0 - self.total_life_consumption)
        
        return life_consumption
    
    def predict_remaining_lifetime(self, current_conditions: Dict) -> float:
        """预测剩余寿命"""
        temp = current_conditions.get('temperature', 25.0)
        voltage = current_conditions.get('voltage', 0.0)
        ripple_current = current_conditions.get('ripple_current', 0.0)
        
        # 计算当前条件下的寿命消耗率
        temp_factor = 1.0
        if temp > 70:
            temp_factor = 2.0 ** ((temp - 70) / 10)
        elif temp > 50:
            temp_factor = 1.5 ** ((temp - 50) / 20)
        
        voltage_factor = 1.0
        voltage_stress = voltage / 1200.0
        if voltage_stress > 0.8:
            voltage_factor = 1.0 + self.params.cap_voltage_coefficient * (voltage_stress - 0.8) / 0.2
        
        ripple_factor = 1.0
        ripple_stress = ripple_current / 80.0
        if ripple_stress > 0.8:
            ripple_factor = 1.0 + self.params.cap_ripple_coefficient * (ripple_stress - 0.8) / 0.2
        
        # 预测剩余寿命
        consumption_rate = temp_factor * voltage_factor * ripple_factor / self.params.cap_base_lifetime
        if consumption_rate > 0:
            remaining_hours = self.remaining_life / consumption_rate
            return remaining_hours
        
        return float('inf')
    
    def get_life_status(self) -> Dict:
        """获取寿命状态"""
        return {
            'remaining_life': self.remaining_life,
            'total_consumption': self.total_life_consumption,
            'max_temperature': max(self.temperature_history) if self.temperature_history else 0.0,
            'max_voltage': max(self.voltage_history) if self.voltage_history else 0.0,
            'max_ripple_current': max(self.ripple_current_history) if self.ripple_current_history else 0.0
        }

class IntegratedLifeModel:
    """综合寿命预测模型"""
    
    def __init__(self):
        self.params = LifeModelParameters()
        self.igbt_models = {}  # 按模块ID存储
        self.cap_models = {}   # 按模块ID存储
        
        # 系统级寿命统计
        self.system_life_consumption = 0.0
        self.system_remaining_life = 1.0
    
    def add_module(self, module_id: str):
        """添加模块的寿命模型"""
        self.igbt_models[module_id] = IGBTLifeModel(self.params)
        self.cap_models[module_id] = CapacitorLifeModel(self.params)
    
    def update_module_conditions(self, module_id: str, igbt_temp: float, igbt_current: float,
                               cap_temp: float, cap_voltage: float, cap_ripple: float, time: float):
        """更新模块运行条件"""
        if module_id in self.igbt_models:
            self.igbt_models[module_id].update_operating_conditions(igbt_temp, igbt_current, time)
            self.cap_models[module_id].update_operating_conditions(cap_temp, cap_voltage, cap_ripple, time)
    
    def calculate_module_life_consumption(self, module_id: str, time_step: float) -> Dict:
        """计算模块寿命消耗"""
        if module_id not in self.igbt_models:
            return {}
        
        igbt_consumption = self.igbt_models[module_id].calculate_life_consumption(time_step)
        cap_consumption = self.cap_models[module_id].calculate_life_consumption(time_step)
        
        return {
            'igbt_consumption': igbt_consumption,
            'capacitor_consumption': cap_consumption,
            'total_consumption': igbt_consumption + cap_consumption
        }
    
    def get_module_health(self, module_id: str) -> float:
        """获取单个模块健康度（0-100）
        健康度定义为 IGBT 与电容剩余寿命的最小值乘以100。
        若模块不存在，返回100.0 作为默认健康度。
        """
        if module_id not in self.igbt_models or module_id not in self.cap_models:
            return 100.0
        
        igbt_status = self.igbt_models[module_id].get_life_status()
        cap_status = self.cap_models[module_id].get_life_status()
        igbt_life = igbt_status.get('remaining_life', 1.0)
        cap_life = cap_status.get('remaining_life', 1.0)
        module_health = min(igbt_life, cap_life) * 100.0
        
        # 限幅到 [0, 100]
        return float(max(0.0, min(100.0, module_health)))
    
    def get_system_health_status(self) -> Dict:
        """获取系统整体健康状态"""
        if not self.igbt_models:
            return {}
        
        # 计算所有模块的平均寿命状态
        total_igbt_life = 0.0
        total_cap_life = 0.0
        module_count = len(self.igbt_models)
        
        for module_id in self.igbt_models:
            igbt_status = self.igbt_models[module_id].get_life_status()
            cap_status = self.cap_models[module_id].get_life_status()
            
            total_igbt_life += igbt_status['remaining_life']
            total_cap_life += cap_status['remaining_life']
        
        avg_igbt_life = total_igbt_life / module_count
        avg_cap_life = total_cap_life / module_count
        
        # 系统健康度（取最小值）
        system_health = min(avg_igbt_life, avg_cap_life) * 100
        
        return {
            'system_health': system_health,
            'igbt_health': avg_igbt_life * 100,
            'capacitor_health': avg_cap_life * 100,
            'module_count': module_count,
            'critical_modules': self._identify_critical_modules()
        }
    
    def _identify_critical_modules(self) -> List[str]:
        """识别关键模块（寿命消耗最严重的）"""
        critical_modules = []
        
        for module_id in self.igbt_models:
            igbt_status = self.igbt_models[module_id].get_life_status()
            cap_status = self.cap_models[module_id].get_life_status()
            
            # 如果任一器件剩余寿命低于20%，标记为关键模块
            if igbt_status['remaining_life'] < 0.2 or cap_status['remaining_life'] < 0.2:
                critical_modules.append(module_id)
        
        return critical_modules
    
    def predict_system_lifetime(self, operating_profile: Dict) -> float:
        """预测系统整体寿命"""
        # 基于运行工况预测
        avg_temp = operating_profile.get('average_temperature', 25.0)
        avg_power = operating_profile.get('average_power', 0.0)
        duty_cycle = operating_profile.get('duty_cycle', 0.5)
        
        # 简化的系统寿命预测
        temp_factor = 1.0
        if avg_temp > 30:
            temp_factor = 1.5 ** ((avg_temp - 30) / 10)
        
        power_factor = 1.0
        if avg_power > 0.8:  # 超过80%额定功率
            power_factor = 1.0 + 0.5 * (avg_power - 0.8) / 0.2
        
        # 预测剩余寿命
        base_lifetime = 100000.0  # 小时
        consumption_rate = temp_factor * power_factor / base_lifetime
        
        if consumption_rate > 0:
            remaining_hours = self.system_remaining_life / consumption_rate
            return remaining_hours
        
        return float('inf')
