#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试IGBT热模型和heavy工况寿命问题
1. 诊断温度为什么还是直线
2. 分析heavy工况寿命计算问题
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimized_igbt_model import OptimizedIGBTModel

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def debug_temperature_profile():
    """调试温度曲线问题"""
    print("=" * 60)
    print("调试温度曲线问题")
    print("=" * 60)
    
    igbt = OptimizedIGBTModel()
    
    # 测试不同场景的温度响应
    scenarios = {
        'constant_power': {
            'description': '恒定功率',
            'power_func': lambda t: 2000,  # 恒定2kW
            'ambient_func': lambda t: 25   # 恒定25°C
        },
        'variable_power': {
            'description': '变化功率',
            'power_func': lambda t: 1500 + 1000 * np.sin(2 * np.pi * t / 24),  # 功率变化
            'ambient_func': lambda t: 25   # 恒定环境温度
        },
        'variable_ambient': {
            'description': '变化环境温度',
            'power_func': lambda t: 2000,  # 恒定功率
            'ambient_func': lambda t: 25 + 10 * np.sin(2 * np.pi * t / 24)  # 环境温度变化
        },
        'both_variable': {
            'description': '功率和环境都变化',
            'power_func': lambda t: 1500 + 1000 * np.sin(2 * np.pi * t / 24),
            'ambient_func': lambda t: 25 + 10 * np.sin(2 * np.pi * t / 24)
        }
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (scenario_name, scenario) in enumerate(scenarios.items()):
        print(f"\n测试场景: {scenario['description']}")
        
        # 重置IGBT状态
        igbt.junction_temperature_C = 25.0
        igbt.case_temperature_C = 25.0
        igbt.temperature_history = []
        
        # 仿真48小时
        time_hours = np.linspace(0, 48, 48)  # 每小时一个点
        temps = []
        powers = []
        ambients = []
        
        for t in time_hours:
            power = scenario['power_func'](t)
            ambient = scenario['ambient_func'](t)
            
            # 更新温度
            Tj, Tc = igbt.update_thermal_state(power, ambient, 3600)  # 1小时步长
            
            temps.append(Tj)
            powers.append(power)
            ambients.append(ambient)
        
        # 分析温度变化
        temp_range = max(temps) - min(temps)
        temp_std = np.std(temps)
        
        print(f"  温度范围: {temp_range:.2f} K")
        print(f"  温度标准差: {temp_std:.2f} K")
        print(f"  平均温度: {np.mean(temps):.1f} °C")
        
        # 检查是否为直线
        temp_gradient = np.gradient(temps)
        max_gradient = np.max(np.abs(temp_gradient))
        
        if temp_range < 1.0:
            print(f"  ⚠️ 温度变化过小，近似直线 (变化范围: {temp_range:.3f}K)")
        elif temp_std < 0.5:
            print(f"  ⚠️ 温度变化很小，接近直线 (标准差: {temp_std:.3f}K)")
        else:
            print(f"  ✓ 温度有合理变化")
        
        # 绘图
        ax = axes[i]
        ax.plot(time_hours, temps, 'r-', linewidth=2, label='结温')
        ax.plot(time_hours, ambients, 'g--', linewidth=1, label='环境温度')
        
        # 添加功率信息（缩放显示）
        power_scaled = np.array(powers) / 100 + min(ambients) - 5
        ax.plot(time_hours, power_scaled, 'b:', linewidth=1, label='功率/100', alpha=0.7)
        
        ax.set_xlabel('时间 (小时)')
        ax.set_ylabel('温度 (°C)')
        ax.set_title(f'{scenario["description"]}\n温度范围: {temp_range:.1f}K')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/调试温度曲线分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return scenarios

def debug_heavy_load_lifetime():
    """调试heavy工况寿命问题"""
    print("\n" + "=" * 60)
    print("调试heavy工况寿命问题")
    print("=" * 60)
    
    igbt = OptimizedIGBTModel()
    
    # 测试不同负载工况
    load_scenarios = {
        'light': {'load_factor': 0.3, 'base_current_A': 300},
        'medium': {'load_factor': 0.6, 'base_current_A': 600},
        'heavy': {'load_factor': 0.9, 'base_current_A': 1000}
    }
    
    results = {}
    
    for load_name, profile in load_scenarios.items():
        print(f"\n分析{load_name}工况...")
        
        # 预测5年寿命
        life_results = igbt.predict_lifetime(profile, years=5)
        results[load_name] = life_results
        
        # 分析结果
        final_life = life_results.iloc[-1]['remaining_life_percent']
        total_damage = life_results.iloc[-1]['cumulative_damage_percent']
        avg_temp = life_results.iloc[-1]['avg_temperature_C']
        max_temp = life_results.iloc[-1]['max_temperature_C']
        
        print(f"  5年后剩余寿命: {final_life:.1f}%")
        print(f"  累积损伤: {total_damage:.1f}%")
        print(f"  平均温度: {avg_temp:.1f}°C")
        print(f"  最高温度: {max_temp:.1f}°C")
        
        # 检查异常
        if load_name == 'heavy':
            if final_life > 80:
                print(f"  ⚠️ Heavy工况寿命过高，可能有问题")
            if avg_temp < 100:
                print(f"  ⚠️ Heavy工况温度过低，可能有问题")
            
            # 详细检查heavy工况的计算过程
            print(f"\n  Heavy工况详细分析:")
            print(f"    负载因子: {profile['load_factor']}")
            print(f"    基础电流: {profile['base_current_A']} A")
            
            # 检查温度历史
            if hasattr(igbt, 'temperature_history') and len(igbt.temperature_history) > 0:
                temp_hist = igbt.temperature_history[-100:]  # 最后100个点
                print(f"    温度历史统计:")
                print(f"      平均: {np.mean(temp_hist):.1f}°C")
                print(f"      范围: {np.max(temp_hist) - np.min(temp_hist):.1f}K")
                print(f"      标准差: {np.std(temp_hist):.2f}K")
    
    # 绘制寿命对比
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for load_name, data in results.items():
        plt.plot(data['year'], data['remaining_life_percent'], 'o-', 
                linewidth=2, markersize=6, label=f'{load_name}负载')
    
    plt.xlabel('运行年数')
    plt.ylabel('剩余寿命 (%)')
    plt.title('不同负载下的寿命预测对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    for load_name, data in results.items():
        plt.plot(data['year'], data['avg_temperature_C'], 's-', 
                linewidth=2, markersize=6, label=f'{load_name}负载')
    
    plt.xlabel('运行年数')
    plt.ylabel('平均温度 (°C)')
    plt.title('不同负载下的平均温度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    for load_name, data in results.items():
        plt.plot(data['year'], data['thermal_cycles'], '^-', 
                linewidth=2, markersize=6, label=f'{load_name}负载')
    
    plt.xlabel('运行年数')
    plt.ylabel('温度循环数')
    plt.title('不同负载下的温度循环数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # 损伤速率分析
    for load_name, data in results.items():
        annual_damage = np.diff(data['cumulative_damage_percent'])
        years_diff = data['year'].iloc[1:]
        plt.plot(years_diff, annual_damage, 'v-', 
                linewidth=2, markersize=6, label=f'{load_name}负载')
    
    plt.xlabel('运行年数')
    plt.ylabel('年度损伤 (%)')
    plt.title('不同负载下的年度损伤速率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/调试heavy工况寿命分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def analyze_thermal_calculation_issues():
    """分析热计算的具体问题"""
    print("\n" + "=" * 60)
    print("分析热计算的具体问题")
    print("=" * 60)
    
    igbt = OptimizedIGBTModel()
    
    # 检查热模型参数
    print(f"热模型参数:")
    print(f"  Rth_jc: {igbt.params.Rth_jc_K_per_W} K/W")
    print(f"  Rth_ca: {igbt.params.Rth_ca_K_per_W} K/W")
    print(f"  Cth_jc: {igbt.params.Cth_jc_J_per_K} J/K")
    print(f"  Cth_ca: {igbt.params.Cth_ca_J_per_K} J/K")
    
    # 计算时间常数
    tau_jc = igbt.params.Rth_jc_K_per_W * igbt.params.Cth_jc_J_per_K
    tau_ca = igbt.params.Rth_ca_K_per_W * igbt.params.Cth_ca_J_per_K
    
    print(f"  时间常数 tau_jc: {tau_jc:.1f} s = {tau_jc/60:.1f} min")
    print(f"  时间常数 tau_ca: {tau_ca:.1f} s = {tau_ca/60:.1f} min")
    
    # 测试单步温度更新
    print(f"\n单步温度更新测试:")
    
    test_cases = [
        (1000, 25, 3600),  # 1kW, 25°C, 1小时
        (2000, 25, 3600),  # 2kW, 25°C, 1小时
        (3000, 25, 3600),  # 3kW, 25°C, 1小时
        (2000, 35, 3600),  # 2kW, 35°C, 1小时
    ]
    
    for power, ambient, dt in test_cases:
        # 重置状态
        igbt.junction_temperature_C = ambient
        igbt.case_temperature_C = ambient
        
        # 计算一步
        Tj_before = igbt.junction_temperature_C
        Tj_after, Tc_after = igbt.update_thermal_state(power, ambient, dt)
        
        temp_rise = Tj_after - ambient
        temp_change = Tj_after - Tj_before
        
        print(f"  功率{power}W, 环境{ambient}°C:")
        print(f"    结温变化: {Tj_before:.1f}°C → {Tj_after:.1f}°C (Δ{temp_change:.1f}K)")
        print(f"    温升: {temp_rise:.1f}K")
        print(f"    壳温: {Tc_after:.1f}°C")
        
        # 检查是否合理
        expected_ss_rise = power * (igbt.params.Rth_jc_K_per_W + igbt.params.Rth_ca_K_per_W)
        print(f"    预期稳态温升: {expected_ss_rise:.1f}K")
        
        if abs(temp_change) < 0.1:
            print(f"    ⚠️ 温度变化过小，可能有问题")
        
    # 测试连续多步更新
    print(f"\n连续多步更新测试:")
    
    igbt.junction_temperature_C = 25.0
    igbt.case_temperature_C = 25.0
    
    power_profile = [1000, 2000, 3000, 2000, 1000]  # 5小时功率变化
    ambient_profile = [25, 30, 35, 30, 25]          # 5小时环境温度变化
    
    temps = [25.0]  # 初始温度
    
    for i, (power, ambient) in enumerate(zip(power_profile, ambient_profile)):
        Tj, Tc = igbt.update_thermal_state(power, ambient, 3600)
        temps.append(Tj)
        print(f"  第{i+1}小时: P={power}W, Ta={ambient}°C → Tj={Tj:.1f}°C")
    
    temp_changes = np.diff(temps)
    total_range = max(temps) - min(temps)
    
    print(f"\n连续更新结果:")
    print(f"  温度序列: {[f'{t:.1f}' for t in temps]}")
    print(f"  温度变化: {[f'{dt:.1f}' for dt in temp_changes]}")
    print(f"  总温度范围: {total_range:.1f}K")
    
    if total_range < 5.0:
        print(f"  ⚠️ 温度变化范围过小 ({total_range:.1f}K)，可能存在问题")
    
    return temps

def fix_thermal_model_issues():
    """修复热模型问题"""
    print("\n" + "=" * 60)
    print("识别和修复热模型问题")
    print("=" * 60)
    
    # 问题分析
    print(f"可能的问题:")
    print(f"1. 热时间常数过小，响应过快")
    print(f"2. 温度限制过严，削峰过多")
    print(f"3. 功率变化不够，输入过于平稳")
    print(f"4. 数值积分步长问题")
    
    # 建议的修复方案
    print(f"\n建议的修复方案:")
    print(f"1. 调整热参数，增加时间常数")
    print(f"2. 放宽温度限制范围")
    print(f"3. 增加功率和环境的变化幅度")
    print(f"4. 优化数值积分算法")
    
    return True

def main():
    """主测试函数"""
    print("开始调试IGBT热模型和heavy工况寿命问题...")
    
    # 1. 调试温度曲线问题
    debug_temperature_profile()
    
    # 2. 调试heavy工况寿命问题
    debug_heavy_load_lifetime()
    
    # 3. 分析热计算问题
    analyze_thermal_calculation_issues()
    
    # 4. 修复建议
    fix_thermal_model_issues()
    
    print(f"\n" + "=" * 60)
    print("调试完成！")
    print("主要发现:")
    print("1. 温度响应可能过快，时间常数需要调整")
    print("2. Heavy工况的损伤计算可能需要重新校准")
    print("3. 需要增加更多的功率和环境变化")
    print("4. 温度限制策略需要优化")
    print("=" * 60)

if __name__ == "__main__":
    main()
