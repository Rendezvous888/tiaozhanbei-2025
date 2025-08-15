#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的IGBT热模型
解决温度直线和heavy工况寿命问题
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class ThermalParams:
    """修正的热参数"""
    # 热阻参数 (K/W) - 基于真实IGBT数据
    Rth_jc: float = 0.05      # 结到壳热阻
    Rth_ch: float = 0.02      # 壳到散热器热阻  
    Rth_ha: float = 0.4       # 散热器到环境热阻
    
    # 热容参数 (J/K) - 增大以获得合理时间常数
    Cth_j: float = 1000       # 结热容
    Cth_c: float = 5000       # 壳热容
    Cth_h: float = 20000      # 散热器热容

class FixedThermalModel:
    """修复的三阶热模型"""
    
    def __init__(self, params: ThermalParams = None):
        self.params = params or ThermalParams()
        
        # 温度状态 [结温, 壳温, 散热器温度]
        self.Tj = 25.0
        self.Tc = 25.0  
        self.Th = 25.0
        
        # 历史记录
        self.temperature_history = []
        
        # 计算时间常数
        self.tau_jc = self.params.Rth_jc * self.params.Cth_j
        self.tau_ch = self.params.Rth_ch * self.params.Cth_c
        self.tau_ha = self.params.Rth_ha * self.params.Cth_h
        
        print(f"热时间常数:")
        print(f"  τ_jc = {self.tau_jc:.0f}s = {self.tau_jc/60:.1f}min")
        print(f"  τ_ch = {self.tau_ch:.0f}s = {self.tau_ch/60:.1f}min") 
        print(f"  τ_ha = {self.tau_ha:.0f}s = {self.tau_ha/3600:.1f}h")
    
    def update_temperature(self, power_loss_W: float, ambient_temp_C: float, 
                         dt_s: float = 60) -> Tuple[float, float, float]:
        """
        更新温度状态 - 使用数值稳定的算法
        
        Args:
            power_loss_W: 功率损耗 (W)
            ambient_temp_C: 环境温度 (°C)
            dt_s: 时间步长 (s)
            
        Returns:
            (结温, 壳温, 散热器温度) (°C)
        """
        # 使用较小的内部时间步长来保证数值稳定性
        internal_dt = min(dt_s, self.tau_jc / 10)  # 内部时间步长为最小时间常数的1/10
        num_steps = int(dt_s / internal_dt)
        
        for _ in range(num_steps):
            # 热流计算
            q_jc = (self.Tj - self.Tc) / self.params.Rth_jc
            q_ch = (self.Tc - self.Th) / self.params.Rth_ch  
            q_ha = (self.Th - ambient_temp_C) / self.params.Rth_ha
            
            # 温度变化率
            dTj_dt = (power_loss_W - q_jc) / self.params.Cth_j
            dTc_dt = (q_jc - q_ch) / self.params.Cth_c
            dTh_dt = (q_ch - q_ha) / self.params.Cth_h
            
            # 欧拉积分更新
            self.Tj += dTj_dt * internal_dt
            self.Tc += dTc_dt * internal_dt
            self.Th += dTh_dt * internal_dt
            
            # 物理合理性检查
            self.Tj = max(ambient_temp_C - 5, min(self.Tj, 300))
            self.Tc = max(ambient_temp_C - 2, min(self.Tc, 280))
            self.Th = max(ambient_temp_C - 1, min(self.Th, 250))
        
        # 记录历史
        self.temperature_history.append(self.Tj)
        
        return self.Tj, self.Tc, self.Th

def test_fixed_thermal_model():
    """测试修复后的热模型"""
    print("=" * 60)
    print("测试修复后的热模型")
    print("=" * 60)
    
    thermal = FixedThermalModel()
    
    # 测试1: 阶跃响应
    print(f"\n测试1: 阶跃功率响应")
    
    # 重置状态
    thermal.Tj = thermal.Tc = thermal.Th = 25.0
    thermal.temperature_history = []
    
    time_minutes = np.arange(0, 120, 1)  # 2小时，每分钟一点
    power_profile = np.where(time_minutes < 60, 1000, 3000)  # 1小时后功率阶跃
    
    temperatures = []
    for power in power_profile:
        Tj, Tc, Th = thermal.update_temperature(power, 25, 60)  # 1分钟步长
        temperatures.append([Tj, Tc, Th])
    
    temperatures = np.array(temperatures)
    
    # 分析阶跃响应
    step_idx = 60
    initial_temp = temperatures[step_idx-1, 0]
    final_temp = temperatures[-1, 0]
    temp_63 = initial_temp + 0.63 * (final_temp - initial_temp)
    
    # 找到63%响应时间
    step_temps = temperatures[step_idx:, 0]
    try:
        tau_idx = np.where(step_temps >= temp_63)[0][0]
        tau_minutes = tau_idx
        print(f"阶跃响应分析:")
        print(f"  初始结温: {initial_temp:.1f}°C")
        print(f"  最终结温: {final_temp:.1f}°C")
        print(f"  温度变化: {final_temp - initial_temp:.1f}K")
        print(f"  63%响应时间: {tau_minutes}分钟")
    except:
        print(f"无法确定响应时间")
    
    # 绘制阶跃响应
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_minutes, power_profile / 1000, 'k-', linewidth=2)
    plt.ylabel('功率 (kW)')
    plt.title('功率阶跃输入')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=60, color='r', linestyle='--', alpha=0.5)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_minutes, temperatures[:, 0], 'r-', linewidth=2, label='结温')
    plt.plot(time_minutes, temperatures[:, 1], 'b-', linewidth=2, label='壳温')
    plt.plot(time_minutes, temperatures[:, 2], 'g-', linewidth=2, label='散热器温度')
    plt.axvline(x=60, color='r', linestyle='--', alpha=0.5, label='阶跃时刻')
    plt.xlabel('时间 (分钟)')
    plt.ylabel('温度 (°C)')
    plt.title('温度阶跃响应')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/修复热模型_阶跃响应.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 测试2: 动态负载响应
    print(f"\n测试2: 动态负载响应")
    
    # 重置状态
    thermal.Tj = thermal.Tc = thermal.Th = 25.0
    thermal.temperature_history = []
    
    # 48小时仿真
    time_hours = np.linspace(0, 48, 48*4)  # 每15分钟一个点
    
    # 复杂的功率和环境变化
    power_base = 2000
    power_daily = power_base * (0.7 + 0.3 * np.sin(2 * np.pi * time_hours / 24))
    power_noise = power_base * 0.1 * np.random.normal(0, 1, len(time_hours))
    power_profile = power_daily + power_noise
    
    ambient_base = 35
    ambient_daily = ambient_base + 8 * np.sin(2 * np.pi * (time_hours - 12) / 24)
    ambient_noise = 2 * np.random.normal(0, 1, len(time_hours))
    ambient_profile = ambient_daily + ambient_noise
    
    dynamic_temps = []
    for i, (power, ambient) in enumerate(zip(power_profile, ambient_profile)):
        Tj, Tc, Th = thermal.update_temperature(power, ambient, 900)  # 15分钟步长
        dynamic_temps.append([Tj, Tc, Th])
    
    dynamic_temps = np.array(dynamic_temps)
    
    # 分析温度变化
    temp_range = np.max(dynamic_temps[:, 0]) - np.min(dynamic_temps[:, 0])
    temp_std = np.std(dynamic_temps[:, 0])
    
    print(f"动态响应分析:")
    print(f"  结温范围: {temp_range:.1f}K")
    print(f"  结温标准差: {temp_std:.1f}K")
    print(f"  平均结温: {np.mean(dynamic_temps[:, 0]):.1f}°C")
    
    if temp_range > 10:
        print(f"  ✓ 温度有良好的动态变化")
    else:
        print(f"  ⚠ 温度变化仍然较小")
    
    # 绘制动态响应
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_hours, power_profile / 1000, 'purple', linewidth=1.5)
    plt.ylabel('功率 (kW)')
    plt.title('功率变化曲线')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_hours, ambient_profile, 'orange', linewidth=1.5)
    plt.ylabel('环境温度 (°C)')
    plt.title('环境温度变化')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_hours, dynamic_temps[:, 0], 'r-', linewidth=2, label='结温')
    plt.plot(time_hours, dynamic_temps[:, 1], 'b-', linewidth=1.5, label='壳温')
    plt.plot(time_hours, dynamic_temps[:, 2], 'g-', linewidth=1.5, label='散热器温度')
    plt.plot(time_hours, ambient_profile, 'k--', linewidth=1, alpha=0.7, label='环境温度')
    plt.xlabel('时间 (小时)')
    plt.ylabel('温度 (°C)')
    plt.title(f'温度动态响应 (范围: {temp_range:.1f}K)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/修复热模型_动态响应.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return thermal

def compare_load_scenarios():
    """对比不同负载场景的寿命"""
    print(f"\n" + "=" * 60)
    print("对比不同负载场景的寿命")
    print("=" * 60)
    
    scenarios = {
        'light': {'power_base': 1000, 'description': '轻载工况'},
        'medium': {'power_base': 2000, 'description': '中载工况'},
        'heavy': {'power_base': 3500, 'description': '重载工况'}
    }
    
    results = {}
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n分析{scenario['description']}...")
        
        thermal = FixedThermalModel()
        thermal.Tj = thermal.Tc = thermal.Th = 25.0
        thermal.temperature_history = []
        
        # 仿真一年
        hours_per_year = 8760
        time_steps = np.linspace(0, hours_per_year, 365)  # 每天一个点
        
        for t in time_steps:
            # 功率变化
            daily_var = 0.8 + 0.4 * np.sin(2 * np.pi * t / 24)
            seasonal_var = 0.9 + 0.1 * np.sin(2 * np.pi * t / (365 * 24))
            power = scenario['power_base'] * daily_var * seasonal_var
            
            # 环境温度变化
            ambient = 35 + 10 * np.sin(2 * np.pi * t / (365 * 24)) + 5 * np.sin(2 * np.pi * t / 24)
            
            # 更新温度 (24小时步长)
            thermal.update_temperature(power, ambient, 24 * 3600)
        
        # 分析结果
        temps = np.array(thermal.temperature_history)
        avg_temp = np.mean(temps)
        max_temp = np.max(temps)
        temp_range = np.max(temps) - np.min(temps)
        
        # 简化寿命计算
        if avg_temp < 100:
            life_consumption = 0.02  # 2%/年
        elif avg_temp < 125:
            life_consumption = 0.05  # 5%/年
        elif avg_temp < 150:
            life_consumption = 0.15  # 15%/年
        else:
            life_consumption = 0.4   # 40%/年
        
        remaining_life = max(0, 100 - life_consumption * 100)
        
        results[scenario_name] = {
            'avg_temp': avg_temp,
            'max_temp': max_temp,
            'temp_range': temp_range,
            'life_consumption': life_consumption,
            'remaining_life': remaining_life
        }
        
        print(f"  平均结温: {avg_temp:.1f}°C")
        print(f"  最高结温: {max_temp:.1f}°C")
        print(f"  温度变化范围: {temp_range:.1f}K")
        print(f"  年寿命消耗: {life_consumption*100:.1f}%")
        print(f"  1年后剩余寿命: {remaining_life:.1f}%")
    
    # 验证heavy工况寿命最短
    light_life = results['light']['remaining_life']
    medium_life = results['medium']['remaining_life']
    heavy_life = results['heavy']['remaining_life']
    
    print(f"\n寿命对比验证:")
    print(f"  Light工况剩余寿命: {light_life:.1f}%")
    print(f"  Medium工况剩余寿命: {medium_life:.1f}%")  
    print(f"  Heavy工况剩余寿命: {heavy_life:.1f}%")
    
    if heavy_life < medium_life < light_life:
        print(f"  ✓ 寿命排序正确：Light > Medium > Heavy")
    else:
        print(f"  ⚠ 寿命排序异常")
    
    return results

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 测试修复后的热模型
    test_fixed_thermal_model()
    
    # 对比负载场景
    compare_load_scenarios()
    
    print(f"\n" + "=" * 60)
    print("修复后的热模型测试完成！")
    print("主要改进:")
    print("1. 合理的热时间常数 (分钟到小时级别)")
    print("2. 数值稳定的积分算法")
    print("3. 物理合理的温度限制")
    print("4. 正确的寿命排序逻辑")
    print("=" * 60)
