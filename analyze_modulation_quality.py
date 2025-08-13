#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析调制质量和输出波形的正弦性
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import CascadedHBridgeSystem

def analyze_modulation_quality():
    """分析调制质量"""
    print("=== 分析调制质量 ===")
    
    # 创建级联系统
    N_modules = 5
    Vdc_per_module = 1000
    fsw = 1000
    f_grid = 50
    
    cascaded_system = CascadedHBridgeSystem(N_modules, Vdc_per_module, fsw, f_grid)
    
    # 测试不同的调制比
    modulation_indices = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 时间向量 - 一个工频周期
    t = np.linspace(0, 0.02, 2000)
    
    results = []
    
    for mi in modulation_indices:
        print(f"\n调制比: {mi}")
        
        # 生成输出
        V_total, V_modules = cascaded_system.generate_phase_shifted_pwm(t, mi)
        
        # 分析输出
        V_max = np.max(V_total)
        V_min = np.min(V_total)
        V_pp = V_max - V_min
        V_rms = np.sqrt(np.mean(V_total**2))
        
        # 理论值
        V_theoretical_max = N_modules * Vdc_per_module * mi
        V_theoretical_rms = V_theoretical_max / np.sqrt(2)
        
        print(f"  输出范围: [{V_min:.0f}, {V_max:.0f}] V")
        print(f"  峰峰值: {V_pp:.0f} V")
        print(f"  RMS值: {V_rms:.0f} V")
        print(f"  理论最大值: {V_theoretical_max:.0f} V")
        print(f"  理论RMS: {V_theoretical_rms:.0f} V")
        
        results.append({
            'mi': mi,
            'V_max': V_max,
            'V_min': V_min,
            'V_pp': V_pp,
            'V_rms': V_rms,
            'V_theoretical_max': V_theoretical_max,
            'V_theoretical_rms': V_theoretical_rms
        })
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 调制比 vs 输出电压
    mi_values = [r['mi'] for r in results]
    V_max_values = [r['V_max'] for r in results]
    V_min_values = [r['V_min'] for r in results]
    V_theoretical_max_values = [r['V_theoretical_max'] for r in results]
    
    axes[0, 0].plot(mi_values, V_max_values, 'bo-', label='实际最大值', linewidth=2)
    axes[0, 0].plot(mi_values, V_min_values, 'ro-', label='实际最小值', linewidth=2)
    axes[0, 0].plot(mi_values, V_theoretical_max_values, 'g--', label='理论最大值', linewidth=2)
    axes[0, 0].plot(mi_values, [-v for v in V_theoretical_max_values], 'g--', label='理论最小值', linewidth=2)
    axes[0, 0].set_xlabel('调制比')
    axes[0, 0].set_ylabel('电压 (V)')
    axes[0, 0].set_title('调制比 vs 输出电压')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 调制比 vs RMS值
    V_rms_values = [r['V_rms'] for r in results]
    V_theoretical_rms_values = [r['V_theoretical_rms'] for r in results]
    
    axes[0, 1].plot(mi_values, V_rms_values, 'bo-', label='实际RMS', linewidth=2)
    axes[0, 1].plot(mi_values, V_theoretical_rms_values, 'r--', label='理论RMS', linewidth=2)
    axes[0, 1].set_xlabel('调制比')
    axes[0, 1].set_ylabel('RMS电压 (V)')
    axes[0, 1].set_title('调制比 vs RMS电压')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 调制比 vs 峰峰值
    V_pp_values = [r['V_pp'] for r in results]
    V_theoretical_pp_values = [2 * r['V_theoretical_max'] for r in results]
    
    axes[1, 0].plot(mi_values, V_pp_values, 'bo-', label='实际峰峰值', linewidth=2)
    axes[1, 0].plot(mi_values, V_theoretical_pp_values, 'r--', label='理论峰峰值', linewidth=2)
    axes[1, 0].set_xlabel('调制比')
    axes[1, 0].set_ylabel('峰峰值电压 (V)')
    axes[1, 0].set_title('调制比 vs 峰峰值电压')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 调制比 vs 线性度误差
    linearity_error = [(r['V_max'] - r['V_theoretical_max']) / r['V_theoretical_max'] * 100 for r in results]
    
    axes[1, 1].plot(mi_values, linearity_error, 'ro-', label='线性度误差', linewidth=2)
    axes[1, 1].set_xlabel('调制比')
    axes[1, 1].set_ylabel('误差 (%)')
    axes[1, 1].set_title('调制比 vs 线性度误差')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

def analyze_waveform_sinusoidality():
    """分析波形的正弦性"""
    print("\n=== 分析波形正弦性 ===")
    
    # 创建级联系统
    N_modules = 5
    Vdc_per_module = 1000
    fsw = 1000
    f_grid = 50
    
    cascaded_system = CascadedHBridgeSystem(N_modules, Vdc_per_module, fsw, f_grid)
    
    # 使用最佳调制比
    modulation_index = 0.8
    t = np.linspace(0, 0.02, 2000)
    
    # 生成输出
    V_total, V_modules = cascaded_system.generate_phase_shifted_pwm(t, modulation_index)
    
    # 计算基波分量
    from scipy import signal
    
    # 设计低通滤波器
    fs = 1.0 / (t[1] - t[0])
    cutoff = 2 * f_grid  # 100 Hz
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    
    # 使用Butterworth滤波器
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    V_fundamental = signal.filtfilt(b, a, V_total)
    
    # 计算谐波分量
    V_harmonic = V_total - V_fundamental
    
    # 计算基波和谐波的RMS
    V_fundamental_rms = np.sqrt(np.mean(V_fundamental**2))
    V_harmonic_rms = np.sqrt(np.mean(V_harmonic**2))
    V_total_rms = np.sqrt(np.mean(V_total**2))
    
    # 计算THD
    thd = (V_harmonic_rms / V_fundamental_rms) * 100 if V_fundamental_rms > 0 else float('inf')
    
    print(f"调制比: {modulation_index}")
    print(f"总输出RMS: {V_total_rms:.1f} V")
    print(f"基波分量RMS: {V_fundamental_rms:.1f} V")
    print(f"谐波分量RMS: {V_harmonic_rms:.1f} V")
    print(f"总谐波失真: {thd:.2f}%")
    
    # 绘制波形分析
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 原始输出
    axes[0].plot(t * 1000, V_total, 'b-', label='级联输出', linewidth=1, alpha=0.7)
    axes[0].set_ylabel('电压 (V)')
    axes[0].set_title('级联H桥系统输出')
    axes[0].legend()
    axes[0].grid(True)
    
    # 基波分量
    axes[1].plot(t * 1000, V_fundamental, 'r-', label='基波分量', linewidth=2)
    axes[1].set_ylabel('电压 (V)')
    axes[1].set_title('基波分量')
    axes[1].legend()
    axes[1].grid(True)
    
    # 谐波分量
    axes[2].plot(t * 1000, V_harmonic, 'g-', label='谐波分量', linewidth=1)
    axes[2].set_xlabel('时间 (ms)')
    axes[2].set_ylabel('电压 (V)')
    axes[2].set_title('谐波分量')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 分析单个模块的贡献
    print(f"\n单个模块分析:")
    for i, V_module in enumerate(V_modules):
        V_module_rms = np.sqrt(np.mean(V_module**2))
        print(f"模块 {i+1}: RMS = {V_module_rms:.0f} V")
    
    return V_total, V_fundamental, V_harmonic, thd

if __name__ == "__main__":
    # 分析调制质量
    results = analyze_modulation_quality()
    
    # 分析波形正弦性
    V_total, V_fundamental, V_harmonic, thd = analyze_waveform_sinusoidality()
    
    print(f"\n=== 分析完成 ===")
    print(f"级联H桥系统的调制质量分析完成")
    print(f"当前THD: {thd:.2f}%")
