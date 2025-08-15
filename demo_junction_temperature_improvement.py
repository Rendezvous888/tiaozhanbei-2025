#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示IGBT结温动态响应改进效果
对比改进前后的温度变化特性
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimized_igbt_model import OptimizedIGBTModel

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def simulate_old_thermal_model(power_profile, ambient_profile, time_hours):
    """模拟简化的热模型（结温近似直线）"""
    # 简化热模型参数
    Rth_total = 0.15  # 总热阻 K/W
    
    # 计算稳态温度（几乎是直线）
    temperatures = []
    for power, ambient in zip(power_profile, ambient_profile):
        temp_rise = power * Rth_total
        junction_temp = ambient + temp_rise
        temperatures.append(junction_temp)
    
    return np.array(temperatures)

def simulate_improved_thermal_model(power_profile, ambient_profile, time_hours):
    """使用改进的IGBT模型"""
    igbt = OptimizedIGBTModel()
    
    # 重置模型状态
    igbt.junction_temperature_C = ambient_profile[0]
    igbt.case_temperature_C = ambient_profile[0]
    igbt.temperature_history = []
    
    temperatures = []
    dt = 3600  # 1小时时间步长
    
    for i, (power, ambient) in enumerate(zip(power_profile, ambient_profile)):
        # 更新温度状态
        Tj, Tc = igbt.update_thermal_state(power, ambient, dt)
        temperatures.append(Tj)
    
    return np.array(temperatures), igbt.temperature_history

def generate_realistic_profiles(duration_hours=48):
    """生成真实的功率和环境温度曲线"""
    time_hours = np.linspace(0, duration_hours, int(duration_hours))
    
    # 功率曲线：日负载变化 + 随机波动
    base_power = 2000  # W
    daily_power = base_power * (0.6 + 0.4 * np.sin(2 * np.pi * (time_hours - 6) / 24))
    
    # 添加随机负载波动
    np.random.seed(42)  # 确保可重复
    power_noise = base_power * 0.1 * np.random.normal(0, 1, len(time_hours))
    power_profile = daily_power + power_noise
    power_profile = np.clip(power_profile, base_power * 0.3, base_power * 1.2)
    
    # 环境温度曲线：日变化 + 天气变化
    base_ambient = 35  # °C
    daily_temp = base_ambient + 8 * np.sin(2 * np.pi * (time_hours - 12) / 24)
    
    # 添加天气变化
    weather_noise = 3 * np.random.normal(0, 1, len(time_hours))
    ambient_profile = daily_temp + weather_noise
    ambient_profile = np.clip(ambient_profile, 20, 50)
    
    return time_hours, power_profile, ambient_profile

def compare_thermal_models():
    """对比不同热模型的结温响应"""
    print("=" * 60)
    print("IGBT结温动态响应改进效果演示")
    print("=" * 60)
    
    # 生成仿真数据
    duration = 48  # 48小时
    time_hours, power_profile, ambient_profile = generate_realistic_profiles(duration)
    
    print(f"仿真设置:")
    print(f"  时长: {duration} 小时")
    print(f"  平均功率: {np.mean(power_profile):.0f} W")
    print(f"  功率范围: {np.min(power_profile):.0f} - {np.max(power_profile):.0f} W")
    print(f"  平均环境温度: {np.mean(ambient_profile):.1f} °C")
    print(f"  环境温度范围: {np.min(ambient_profile):.1f} - {np.max(ambient_profile):.1f} °C")
    
    # 运行不同模型
    print(f"\n运行热模型仿真...")
    
    # 简化模型（直线结温）
    old_temps = simulate_old_thermal_model(power_profile, ambient_profile, time_hours)
    
    # 改进模型（动态结温）
    improved_temps, temp_history = simulate_improved_thermal_model(power_profile, ambient_profile, time_hours)
    
    # 分析结果
    print(f"\n仿真结果对比:")
    print(f"简化模型:")
    print(f"  平均结温: {np.mean(old_temps):.1f} °C")
    print(f"  结温范围: {np.max(old_temps) - np.min(old_temps):.1f} K")
    print(f"  温度标准差: {np.std(old_temps):.2f} K")
    
    print(f"改进模型:")
    print(f"  平均结温: {np.mean(improved_temps):.1f} °C")
    print(f"  结温范围: {np.max(improved_temps) - np.min(improved_temps):.1f} K")
    print(f"  温度标准差: {np.std(improved_temps):.2f} K")
    
    # 绘制对比图
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('IGBT结温动态响应改进效果对比', fontsize=16, fontweight='bold')
    
    # 1. 输入条件
    ax1 = axes[0, 0]
    ax1.plot(time_hours, power_profile / 1000, 'b-', linewidth=2, label='功率损耗')
    ax1.set_ylabel('功率 (kW)')
    ax1.set_title('功率损耗输入')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(time_hours, ambient_profile, 'g-', linewidth=2, label='环境温度')
    ax2.set_ylabel('温度 (°C)')
    ax2.set_title('环境温度输入')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2. 温度响应对比
    ax3 = axes[1, 0]
    ax3.plot(time_hours, old_temps, 'r--', linewidth=2, label='简化模型（近似直线）')
    ax3.plot(time_hours, ambient_profile, 'g:', linewidth=1, label='环境温度', alpha=0.7)
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('温度 (°C)')
    ax3.set_title('简化热模型结温响应')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(time_hours, improved_temps, 'r-', linewidth=2, label='改进模型（动态响应）')
    ax4.plot(time_hours, ambient_profile, 'g:', linewidth=1, label='环境温度', alpha=0.7)
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('温度 (°C)')
    ax4.set_title('改进热模型结温响应')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 3. 直接对比
    ax5 = axes[2, 0]
    ax5.plot(time_hours, old_temps, 'r--', linewidth=2, label='简化模型', alpha=0.8)
    ax5.plot(time_hours, improved_temps, 'r-', linewidth=2, label='改进模型')
    ax5.set_xlabel('时间 (小时)')
    ax5.set_ylabel('结温 (°C)')
    ax5.set_title('结温响应直接对比')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 4. 温度变化率分析
    ax6 = axes[2, 1]
    old_gradient = np.gradient(old_temps)
    improved_gradient = np.gradient(improved_temps)
    
    ax6.plot(time_hours[:-1], old_gradient[:-1], 'b--', linewidth=2, label='简化模型变化率', alpha=0.8)
    ax6.plot(time_hours[:-1], improved_gradient[:-1], 'b-', linewidth=2, label='改进模型变化率')
    ax6.set_xlabel('时间 (小时)')
    ax6.set_ylabel('温度变化率 (K/h)')
    ax6.set_title('温度变化率对比')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/IGBT结温动态响应改进对比.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 统计分析
    print(f"\n详细统计分析:")
    print(f"温度变化特性:")
    print(f"  简化模型变化率标准差: {np.std(old_gradient):.3f} K/h")
    print(f"  改进模型变化率标准差: {np.std(improved_gradient):.3f} K/h")
    print(f"  改进效果: {np.std(improved_gradient)/np.std(old_gradient):.1f}倍变化")
    
    # 频谱分析
    print(f"\n频谱特性分析:")
    old_fft = np.fft.fft(old_temps - np.mean(old_temps))
    improved_fft = np.fft.fft(improved_temps - np.mean(improved_temps))
    
    print(f"  简化模型主要频率分量: {np.sum(np.abs(old_fft)**2):.1f}")
    print(f"  改进模型主要频率分量: {np.sum(np.abs(improved_fft)**2):.1f}")
    
    return time_hours, old_temps, improved_temps

def demonstrate_step_response():
    """演示阶跃响应特性"""
    print(f"\n" + "=" * 60)
    print("阶跃响应特性演示")
    print("=" * 60)
    
    igbt = OptimizedIGBTModel()
    
    # 阶跃输入：功率从1kW突然跳到3kW
    time_minutes = np.arange(0, 120, 1)  # 2小时，每分钟一个点
    power_step = np.where(time_minutes < 60, 1000, 3000)  # 1小时后阶跃
    ambient_temp = 25  # 恒定环境温度
    
    # 仿真阶跃响应
    temperatures = []
    igbt.junction_temperature_C = ambient_temp
    igbt.case_temperature_C = ambient_temp
    
    for i, power in enumerate(power_step):
        Tj, Tc = igbt.update_thermal_state(power, ambient_temp, 60)  # 1分钟步长
        temperatures.append(Tj)
    
    # 分析时间常数
    step_start_idx = 60  # 阶跃开始点
    step_temps = np.array(temperatures[step_start_idx:])
    
    # 63%上升时间
    initial_temp = temperatures[step_start_idx - 1]
    final_temp = temperatures[-1]
    temp_63 = initial_temp + 0.63 * (final_temp - initial_temp)
    
    try:
        tau_idx = np.where(step_temps >= temp_63)[0][0]
        tau_minutes = tau_idx
        print(f"时间常数分析:")
        print(f"  初始温度: {initial_temp:.1f} °C")
        print(f"  最终温度: {final_temp:.1f} °C")
        print(f"  温度变化: {final_temp - initial_temp:.1f} K")
        print(f"  63%上升时间: {tau_minutes:.1f} 分钟")
    except:
        print(f"时间常数计算失败")
    
    # 绘制阶跃响应
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_minutes, power_step / 1000, 'k-', linewidth=2)
    plt.ylabel('功率 (kW)')
    plt.title('功率阶跃输入')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=60, color='r', linestyle='--', alpha=0.5, label='阶跃时刻')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_minutes, temperatures, 'r-', linewidth=2, label='结温响应')
    plt.axhline(y=temp_63, color='b', linestyle='--', alpha=0.5, label=f'63%响应 ({temp_63:.1f}°C)')
    plt.axvline(x=60, color='r', linestyle='--', alpha=0.5, label='阶跃时刻')
    if 'tau_minutes' in locals():
        plt.axvline(x=60 + tau_minutes, color='g', linestyle='--', alpha=0.5, label=f'时间常数 ({tau_minutes:.1f}min)')
    plt.xlabel('时间 (分钟)')
    plt.ylabel('结温 (°C)')
    plt.title('结温阶跃响应')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/IGBT结温阶跃响应分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return time_minutes, temperatures

def analyze_thermal_improvements():
    """分析热模型改进的具体效果"""
    print(f"\n" + "=" * 60)
    print("热模型改进效果分析")
    print("=" * 60)
    
    # 运行主要对比
    time_hours, old_temps, improved_temps = compare_thermal_models()
    
    # 运行阶跃响应
    time_minutes, step_temps = demonstrate_step_response()
    
    # 生成改进总结
    print(f"\n" + "=" * 60)
    print("改进效果总结")
    print("=" * 60)
    
    print(f"问题分析:")
    print(f"  ✗ 原始模型结温近似直线，缺乏动态特性")
    print(f"  ✗ 忽略热网络的时间常数效应")
    print(f"  ✗ 缺少温度对功率损耗的反馈影响")
    print(f"  ✗ 环境条件过于简化")
    
    print(f"\n改进措施:")
    print(f"  ✓ 双RC热网络模型 - 结到壳 + 壳到环境")
    print(f"  ✓ 动态时间常数响应 - 指数衰减特性")
    print(f"  ✓ 温度-功率反馈耦合 - 温度影响损耗")
    print(f"  ✓ 复杂环境和负载变化 - 日/周/季节/随机变化")
    print(f"  ✓ 合理的温度限制 - 避免过度削峰")
    
    print(f"\n改进效果:")
    temp_range_old = np.max(old_temps) - np.min(old_temps)
    temp_range_new = np.max(improved_temps) - np.min(improved_temps)
    
    print(f"  结温变化范围: {temp_range_old:.1f}K → {temp_range_new:.1f}K")
    print(f"  动态特性提升: {temp_range_new/max(temp_range_old, 0.1):.1f}倍")
    print(f"  温度标准差: {np.std(old_temps):.2f}K → {np.std(improved_temps):.2f}K")
    print(f"  更真实的热响应特性和寿命分析基础")
    
    print(f"\n技术特点:")
    print(f"  • 基于物理的多阶RC热网络")
    print(f"  • 考虑热时间常数的动态响应")
    print(f"  • 温度与功率损耗的双向耦合")
    print(f"  • 真实的环境条件变化建模")
    print(f"  • 适用于寿命预测和热设计")

if __name__ == "__main__":
    # 设置随机种子确保可重复性
    np.random.seed(42)
    
    # 运行完整分析
    analyze_thermal_improvements()
    
    print(f"\n" + "=" * 60)
    print("IGBT结温动态响应改进完成！")
    print("现在结温展现真实的动态特性，不再是单调的直线")
    print("=" * 60)
