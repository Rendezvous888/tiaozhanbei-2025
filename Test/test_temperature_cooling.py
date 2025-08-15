#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PCS温度冷却效果
验证当功率为0时，结温和壳温是否能正确冷却到环境温度
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcs_simulation_model import PCSSimulation

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_temperature_cooling():
    """测试温度冷却效果"""
    print("=" * 60)
    print("测试PCS温度冷却效果")
    print("=" * 60)
    
    # 创建PCS仿真实例
    pcs_sim = PCSSimulation()
    print(f"环境温度设定: {pcs_sim.params.T_amb}°C")
    
    # 测试场景：先加热再冷却
    print("\n1. 模拟加热阶段...")
    
    # 第一阶段：高功率运行30分钟，使温度上升
    dt = 60  # 时间步长1分钟
    heating_time = 30  # 加热30分钟
    cooling_time = 60   # 冷却60分钟
    
    heating_power = 20e6  # 20MW高功率
    cooling_power = 0     # 0MW无功率
    
    # 存储结果
    time_history = []
    Tj_history = []
    Tc_history = []
    power_history = []
    
    # 初始化温度
    thermal_model = pcs_sim.thermal
    thermal_model.Tj = pcs_sim.params.T_amb  # 初始结温等于环境温度
    thermal_model.Tc = pcs_sim.params.T_amb  # 初始壳温等于环境温度
    
    print(f"初始状态:")
    print(f"  - 结温: {thermal_model.Tj:.1f}°C")
    print(f"  - 壳温: {thermal_model.Tc:.1f}°C")
    
    # 加热阶段
    current_time = 0
    for step in range(heating_time):
        # 计算功率损耗
        P_loss = heating_power * 0.05  # 假设5%的损耗
        
        # 更新温度
        Tj, Tc = thermal_model.update_temperature(P_loss, dt)
        
        # 记录数据
        time_history.append(current_time / 60.0)  # 转换为小时
        Tj_history.append(Tj)
        Tc_history.append(Tc)
        power_history.append(heating_power / 1e6)  # 转换为MW
        
        current_time += dt
    
    print(f"\n加热后状态 (30分钟后):")
    print(f"  - 结温: {thermal_model.Tj:.1f}°C")
    print(f"  - 壳温: {thermal_model.Tc:.1f}°C")
    print(f"  - 温升: {thermal_model.Tj - pcs_sim.params.T_amb:.1f}°C")
    
    # 冷却阶段
    print(f"\n2. 模拟冷却阶段...")
    for step in range(cooling_time):
        # 无功率损耗
        P_loss = cooling_power
        
        # 更新温度
        Tj, Tc = thermal_model.update_temperature(P_loss, dt)
        
        # 记录数据
        time_history.append(current_time / 60.0)  # 转换为小时
        Tj_history.append(Tj)
        Tc_history.append(Tc)
        power_history.append(cooling_power / 1e6)  # 转换为MW
        
        current_time += dt
    
    print(f"\n冷却后状态 (60分钟后):")
    print(f"  - 结温: {thermal_model.Tj:.1f}°C")
    print(f"  - 壳温: {thermal_model.Tc:.1f}°C")
    print(f"  - 与环境温度差: {thermal_model.Tj - pcs_sim.params.T_amb:.1f}°C")
    
    # 计算冷却效果
    max_temp = max(Tj_history)
    final_temp = Tj_history[-1]
    cooling_effect = max_temp - final_temp
    
    print(f"\n3. 冷却效果分析:")
    print(f"  - 最高结温: {max_temp:.1f}°C")
    print(f"  - 最终结温: {final_temp:.1f}°C") 
    print(f"  - 总冷却量: {cooling_effect:.1f}°C")
    print(f"  - 冷却效率: {cooling_effect/(max_temp-pcs_sim.params.T_amb)*100:.1f}%")
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上子图：温度变化
    ax1.plot(time_history, Tj_history, 'r-', linewidth=2, label='结温 (Tj)')
    ax1.plot(time_history, Tc_history, 'g-', linewidth=2, label='壳温 (Tc)')
    ax1.axhline(y=pcs_sim.params.T_amb, color='b', linestyle='--', alpha=0.7, label=f'环境温度 ({pcs_sim.params.T_amb}°C)')
    ax1.axvline(x=heating_time/60, color='orange', linestyle=':', alpha=0.7, label='开始冷却')
    
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('PCS温度冷却测试 - 温度响应')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下子图：功率变化
    ax2.plot(time_history, power_history, 'purple', linewidth=2, label='功率')
    ax2.axvline(x=heating_time/60, color='orange', linestyle=':', alpha=0.7, label='开始冷却')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('功率 (MW)')
    ax2.set_title('功率变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    os.makedirs('Test', exist_ok=True)
    plt.savefig('Test/temperature_cooling_test.png', dpi=300, bbox_inches='tight')
    print(f"\n图形已保存: Test/temperature_cooling_test.png")
    
    plt.show()
    
    # 验证结果
    print(f"\n4. 验证结果:")
    if final_temp <= pcs_sim.params.T_amb + 2:  # 允许2°C误差
        print("✅ 冷却效果正常 - 结温基本降至环境温度")
    else:
        print("❌ 冷却效果异常 - 结温未能充分冷却")
        
    if cooling_effect > 10:  # 至少冷却10°C
        print("✅ 冷却幅度充分")
    else:
        print("❌ 冷却幅度不足")
    
    return time_history, Tj_history, Tc_history, power_history

if __name__ == "__main__":
    test_temperature_cooling()
