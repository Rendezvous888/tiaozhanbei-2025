#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同调制策略是否真正产生不同的结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import CascadedHBridgeSystem

def test_modulation_strategies():
    """测试不同调制策略的输出差异"""
    print("=== 测试不同调制策略的输出差异 ===")
    
    # 系统参数
    N_modules = 8  # 减少模块数以便于观察差异
    Vdc_per_module = 1000
    fsw = 1000
    f_grid = 50
    
    # 创建不同调制策略的系统
    strategies = ["PS-PWM", "NLM", "SHE"]
    systems = {}
    
    for strategy in strategies:
        print(f"\n创建 {strategy} 系统...")
        systems[strategy] = CascadedHBridgeSystem(
            N_modules=N_modules,
            Vdc_per_module=Vdc_per_module,
            fsw=fsw,
            f_grid=f_grid,
            modulation_strategy=strategy
        )
        print(f"  调制策略: {systems[strategy].modulation_strategy}")
    
    # 仿真参数
    t = np.linspace(0, 0.02, 2000)  # 一个工频周期
    modulation_index = 0.8
    
    # 存储结果
    results = {}
    
    for strategy in strategies:
        print(f"\n分析 {strategy} 策略...")
        results[strategy] = {}
        
        # 生成输出电压
        V_total, V_modules = systems[strategy].generate_phase_shifted_pwm(t, modulation_index)
        
        # 计算谐波频谱
        freqs, magnitude = systems[strategy].calculate_harmonic_spectrum(V_total, t)
        
        # 计算THD
        thd = systems[strategy].calculate_thd_time_domain(V_total, t) * 100.0
        
        # 计算RMS值
        v_rms = np.sqrt(np.mean(V_total**2))
        v_peak = np.max(np.abs(V_total))
        
        # 存储结果
        results[strategy] = {
            'V_total': V_total,
            'V_modules': V_modules,
            'freqs': freqs,
            'magnitude': magnitude,
            'THD': thd,
            'V_rms': v_rms,
            'V_peak': v_peak
        }
        
        print(f"  THD: {thd:.2f}%")
        print(f"  V_rms: {v_rms:.1f} V")
        print(f"  V_peak: {v_peak:.1f} V")
        
        # 检查输出是否真的不同
        if strategy == "PS-PWM":
            ps_pwm_output = V_total.copy()
        elif strategy == "NLM":
            nlm_output = V_total.copy()
        elif strategy == "SHE":
            she_output = V_total.copy()
    
    # 比较输出差异
    print("\n=== 输出差异分析 ===")
    
    # 计算PS-PWM和NLM的差异
    ps_nlm_diff = np.mean(np.abs(ps_pwm_output - nlm_output))
    print(f"PS-PWM vs NLM 平均差异: {ps_nlm_diff:.2f} V")
    
    # 计算PS-PWM和SHE的差异
    ps_she_diff = np.mean(np.abs(ps_pwm_output - she_output))
    print(f"PS-PWM vs SHE 平均差异: {ps_she_diff:.2f} V")
    
    # 计算NLM和SHE的差异
    nlm_she_diff = np.mean(np.abs(nlm_output - she_output))
    print(f"NLM vs SHE 平均差异: {nlm_she_diff:.2f} V")
    
    # 检查是否有显著差异
    if ps_nlm_diff > 1.0 and ps_she_diff > 1.0 and nlm_she_diff > 1.0:
        print("✓ 不同调制策略产生了显著不同的输出！")
    else:
        print("✗ 不同调制策略的输出差异较小，可能存在实现问题")
    
    # 绘制比较图
    plot_strategy_comparison(t, results, strategies)
    
    return results

def plot_strategy_comparison(t, results, strategies):
    """绘制不同策略的比较图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('不同调制策略输出比较', fontsize=16)
    
    # 第一行：时域波形比较
    colors = ['blue', 'green', 'red']
    for i, strategy in enumerate(strategies):
        V_total = results[strategy]['V_total']
        axes[0, 0].plot(t * 1000, V_total / 1000, color=colors[i], 
                        linewidth=2, label=f'{strategy}', alpha=0.8)
    
    axes[0, 0].set_xlabel('时间 (ms)')
    axes[0, 0].set_ylabel('电压 (kV)')
    axes[0, 0].set_title('输出电压波形比较')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 第二行：THD比较
    thd_values = [results[strategy]['THD'] for strategy in strategies]
    bars = axes[0, 1].bar(strategies, thd_values, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel('THD (%)')
    axes[0, 1].set_title('THD性能比较')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, thd in zip(bars, thd_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{thd:.2f}%', ha='center', va='bottom')
    
    # 第三行：谐波频谱比较
    for i, strategy in enumerate(strategies):
        freqs = results[strategy]['freqs']
        magnitude = results[strategy]['magnitude']
        axes[1, 0].plot(freqs, magnitude, color=colors[i], 
                        linewidth=2, label=f'{strategy}', alpha=0.8)
    
    axes[1, 0].set_xlabel('频率 (Hz)')
    axes[1, 0].set_ylabel('幅值 (V)')
    axes[1, 0].set_title('谐波频谱比较')
    axes[1, 0].set_xlim(0, 2000)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 第四行：模块输出比较（显示前4个模块）
    for i, strategy in enumerate(strategies):
        V_modules = results[strategy]['V_modules']
        if len(V_modules) >= 4:
            for j in range(4):
                axes[1, 1].plot(t * 1000, V_modules[j] / 1000, 
                               color=colors[i], linewidth=1, 
                               label=f'{strategy}-模块{j+1}', alpha=0.6)
    
    axes[1, 1].set_xlabel('时间 (ms)')
    axes[1, 1].set_ylabel('电压 (kV)')
    axes[1, 1].set_title('前4个模块输出比较')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/Modulation_Strategies_Comparison.png', dpi=300, bbox_inches='tight')
    print("\n比较图已保存到: pic/Modulation_Strategies_Comparison.png")
    plt.show()

if __name__ == "__main__":
    # 运行测试
    results = test_modulation_strategies()
    print("\n测试完成！")

