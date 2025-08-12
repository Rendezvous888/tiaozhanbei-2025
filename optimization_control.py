"""
优化控制策略模块
实现充放电功率优化、器件应力抑制、健康度优化等算法
包含遗传算法、MPC控制等优化方法
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
import math

@dataclass
class OptimizationParameters:
    """优化算法参数"""
    # 遗传算法参数
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # MPC参数
    prediction_horizon: int = 10
    control_horizon: int = 5
    
    # 约束参数
    max_power_ramp_rate: float = 0.1  # 功率变化率限制
    min_power: float = 0.0  # 最小功率
    max_power: float = 25e6  # 最大功率 (25 MW)
    
    # 健康度权重
    health_weight: float = 0.4
    efficiency_weight: float = 0.3
    stress_weight: float = 0.3

class PowerProfileOptimizer:
    """功率曲线优化器"""
    
    def __init__(self, params: OptimizationParameters):
        self.params = params
        self.daily_cycle_hours = 24
        self.time_resolution = 1.0  # 小时
        
    def optimize_daily_cycle(self, grid_demand: List[float], 
                           battery_soc: float, 
                           health_status: Dict) -> List[float]:
        """优化24小时充放电功率曲线"""
        
        # 基于健康状态调整优化策略
        health_factor = health_status.get('overall_health', 100.0) / 100.0
        
        if health_factor < 0.5:  # 健康度低，采用保守策略
            return self._conservative_power_profile(grid_demand, battery_soc)
        elif health_factor < 0.8:  # 健康度中等，平衡策略
            return self._balanced_power_profile(grid_demand, battery_soc)
        else:  # 健康度高，激进策略
            return self._aggressive_power_profile(grid_demand, battery_soc)
    
    def _conservative_power_profile(self, grid_demand: List[float], 
                                  battery_soc: float) -> List[float]:
        """保守功率策略：优先保护设备"""
        power_profile = []
        
        for hour in range(self.daily_cycle_hours):
            if hour < 6:  # 夜间充电
                power = min(5e6, 10e6 * (1 - battery_soc))  # 限制充电功率
            elif 6 <= hour < 18:  # 日间放电
                power = -min(8e6, abs(grid_demand[hour]))  # 限制放电功率
            else:  # 晚间充电
                power = min(6e6, 10e6 * (1 - battery_soc))
            
            power_profile.append(power)
        
        return power_profile
    
    def _balanced_power_profile(self, grid_demand: List[float], 
                              battery_soc: float) -> List[float]:
        """平衡功率策略：兼顾效率和健康"""
        power_profile = []
        
        for hour in range(self.daily_cycle_hours):
            if hour < 6:  # 夜间充电
                power = min(8e6, 15e6 * (1 - battery_soc))
            elif 6 <= hour < 18:  # 日间放电
                power = -min(12e6, abs(grid_demand[hour]))
            else:  # 晚间充电
                power = min(10e6, 15e6 * (1 - battery_soc))
            
            power_profile.append(power)
        
        return power_profile
    
    def _aggressive_power_profile(self, grid_demand: List[float], 
                                battery_soc: float) -> List[float]:
        """激进功率策略：最大化经济效益"""
        power_profile = []
        
        for hour in range(self.daily_cycle_hours):
            if hour < 6:  # 夜间充电
                power = min(12e6, 20e6 * (1 - battery_soc))
            elif 6 <= hour < 18:  # 日间放电
                power = -min(18e6, abs(grid_demand[hour]))
            else:  # 晚间充电
                power = min(15e6, 20e6 * (1 - battery_soc))
            
            power_profile.append(power)
        
        return power_profile

class StressOptimization:
    """应力优化器"""
    
    def __init__(self, params: OptimizationParameters):
        self.params = params
    
    def optimize_switching_frequency(self, current_health: Dict, 
                                   temperature_distribution: Dict) -> float:
        """优化开关频率以减少应力"""
        
        # 基于健康状态和温度分布调整开关频率
        health_factor = current_health.get('overall_health', 100.0) / 100.0
        max_temp = temperature_distribution.get('max_igbt_temp', 25.0)
        
        # 基础开关频率
        base_frequency = 1000.0  # Hz
        
        # 健康度低或温度高时降低开关频率
        if health_factor < 0.6 or max_temp > 80:
            optimized_frequency = base_frequency * 0.7  # 降低30%
        elif health_factor < 0.8 or max_temp > 60:
            optimized_frequency = base_frequency * 0.85  # 降低15%
        else:
            optimized_frequency = base_frequency
        
        # 确保在允许范围内
        optimized_frequency = max(500.0, min(1000.0, optimized_frequency))
        
        return optimized_frequency
    
    def optimize_modulation_strategy(self, power_level: float, 
                                   health_status: Dict) -> Dict:
        """优化调制策略以减少应力"""
        
        health_factor = health_status.get('overall_health', 100.0) / 100.0
        
        # 基于健康状态调整调制策略
        if health_factor < 0.6:
            # 健康度低：采用多电平调制减少开关应力
            strategy = {
                'modulation_type': 'multilevel',
                'modulation_index': 0.6,
                'switching_reduction': 0.3
            }
        elif health_factor < 0.8:
            # 健康度中等：平衡调制
            strategy = {
                'modulation_type': 'balanced',
                'modulation_index': 0.75,
                'switching_reduction': 0.15
            }
        else:
            # 健康度高：标准调制
            strategy = {
                'modulation_type': 'standard',
                'modulation_index': 0.9,
                'switching_reduction': 0.0
            }
        
        return strategy

class HealthOptimizationController:
    """健康度优化控制器"""
    
    def __init__(self, params: OptimizationParameters):
        self.params = params
        self.power_optimizer = PowerProfileOptimizer(params)
        self.stress_optimizer = StressOptimization(params)
    
    def optimize_system_operation(self, current_conditions: Dict, 
                                grid_demand: List[float],
                                health_status: Dict) -> Dict:
        """优化系统运行"""
        
        # 1. 功率曲线优化
        optimized_power = self.power_optimizer.optimize_daily_cycle(
            grid_demand, 
            current_conditions.get('battery_soc', 0.5),
            health_status
        )
        
        # 2. 应力优化
        optimized_frequency = self.stress_optimizer.optimize_switching_frequency(
            health_status,
            current_conditions.get('temperature_distribution', {})
        )
        
        optimized_modulation = self.stress_optimizer.optimize_modulation_strategy(
            current_conditions.get('power', 0.0),
            health_status
        )
        
        return {
            'optimized_power_profile': optimized_power,
            'optimized_switching_frequency': optimized_frequency,
            'optimized_modulation': optimized_modulation,
            'health_optimization_score': self._calculate_optimization_score(
                optimized_power, health_status
            )
        }
    
    def _calculate_optimization_score(self, power_profile: List[float], 
                                    health_status: Dict) -> float:
        """计算优化得分"""
        
        # 功率平滑度得分
        power_variations = [abs(power_profile[i] - power_profile[i-1]) 
                           for i in range(1, len(power_profile))]
        smoothness_score = 100.0 / (1.0 + np.mean(power_variations) / 1e6)
        
        # 健康度得分
        health_score = health_status.get('overall_health', 100.0)
        
        # 综合得分
        total_score = (self.params.health_weight * health_score + 
                      self.params.efficiency_weight * smoothness_score + 
                      self.params.stress_weight * 100.0)
        
        return total_score
