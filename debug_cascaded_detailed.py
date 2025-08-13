#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细调试级联H桥系统
显示每个模块的载波、参考波和输出
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import HBridgeUnit, CascadedHBridgeSystem

def debug_cascaded_detailed():
    """详细调试级联系统"""
    print("=== 详细调试级联H桥系统 ===")
    
    # 创建级联系统
    N_modules = 5
    Vdc_per_module = 1000
    fsw = 1000
    f_grid = 50
    
    cascaded_system = CascadedHBridgeSystem(N_modules, Vdc_per_module, fsw, f_grid)
    
    # 时间向量 - 几个载波周期
    t = np.linspace(0, 0.005, 1000)  # 5ms，5个载波周期
    modulation_index = 0.8
    
    print(f"系统参数:")
    print(f"- 模块数: {N_modules}")
    print(f"- 每模块直流电压: {Vdc_per_module} V")
    print(f"- 开关频率: {fsw} Hz")
    print(f"- 电网频率: {f_grid} Hz")
    print(f"- 调制比: {modulation_index}")
    
    # 计算相移
    phase_shifts = np.linspace(0, (N_modules - 1) / N_modules, N_modules, endpoint=True)
    print(f"\n载波相移:")
    for i, shift in enumerate(phase_shifts):
        print(f"- 模块 {i+1}: {shift:.3f} (载波周期比例)")
    
    # 创建大图显示所有模块的详细信息
    fig, axes = plt.subplots(N_modules, 3, figsize=(18, 4*N_modules))
    
    # 为每个模块生成载波、参考波和输出
    for i in range(N_modules):
        hbridge = cascaded_system.hbridge_units[i]
        carrier_shift = phase_shifts[i]
        
        # 生成载波（带相移）
        carrier = hbridge.generate_carrier_wave_with_shift(t, carrier_shift)
        # 生成参考波（无相移）
        reference = hbridge.generate_reference_wave(t, modulation_index, 0)
        # 生成输出
        V_out = hbridge.calculate_output_voltage_with_carrier_shift(t, modulation_index, carrier_shift)
        
        # 绘制载波
        axes[i, 0].plot(t * 1000, carrier, 'b-', linewidth=1)
        axes[i, 0].set_ylabel('载波幅值')
        axes[i, 0].set_title(f'模块 {i+1} 载波 (相移={carrier_shift:.3f})')
        axes[i, 0].grid(True)
        
        # 绘制参考波
        axes[i, 1].plot(t * 1000, reference, 'r-', linewidth=2)
        axes[i, 1].set_ylabel('参考波幅值')
        axes[i, 1].set_title(f'模块 {i+1} 参考波')
        axes[i, 1].grid(True)
        
        # 绘制输出
        axes[i, 2].plot(t * 1000, V_out, 'g-', linewidth=1)
        axes[i, 2].set_ylabel('输出电压 (V)')
        axes[i, 2].set_title(f'模块 {i+1} 输出')
        axes[i, 2].set_xlabel('时间 (ms)')
        axes[i, 2].grid(True)
        
        # 计算输出统计
        V_rms = np.sqrt(np.mean(V_out**2))
        print(f"模块 {i+1}: RMS = {V_rms:.0f} V, 范围 = [{np.min(V_out):.0f}, {np.max(V_out):.0f}] V")
    
    plt.tight_layout()
    plt.show()
    
    # 现在测试级联叠加
    print(f"\n=== 级联叠加测试 ===")
    
    # 使用更长的仿真时间
    t_long = np.linspace(0, 0.02, 2000)  # 一个工频周期
    
    # 生成级联输出
    V_total, V_modules = cascaded_system.generate_phase_shifted_pwm(t_long, modulation_index)
    
    # 分析级联输出
    print(f"级联输出分析:")
    print(f"- 总输出范围: [{np.min(V_total):.0f}, {np.max(V_total):.0f}] V")
    print(f"- 理论范围: [{-N_modules * Vdc_per_module}, {N_modules * Vdc_per_module}] V")
    
    # 计算每个模块的贡献
    V_total_calculated = np.sum(V_modules, axis=0)
    print(f"- 计算总输出范围: [{np.min(V_total_calculated):.0f}, {np.max(V_total_calculated):.0f}] V")
    
    # 检查是否一致
    if np.allclose(V_total, V_total_calculated):
        print("- 级联叠加计算正确")
    else:
        print("- 级联叠加计算有误")
    
    # 绘制级联输出
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 总输出
    axes[0].plot(t_long * 1000, V_total, 'b-', linewidth=1)
    axes[0].set_ylabel('电压 (V)')
    axes[0].set_title('级联H桥系统总输出电压')
    axes[0].grid(True)
    
    # 单个模块输出
    for i in range(min(3, len(V_modules))):
        axes[1].plot(t_long * 1000, V_modules[i], alpha=0.7, label=f'模块 {i+1}')
    axes[1].set_ylabel('电压 (V)')
    axes[1].set_title('单个模块输出电压')
    axes[1].legend()
    axes[1].grid(True)
    
    # 局部放大
    zoom_start = int(len(t_long) * 0.1)
    zoom_end = int(len(t_long) * 0.15)
    axes[2].plot(t_long[zoom_start:zoom_end] * 1000, V_total[zoom_start:zoom_end], 'b-', linewidth=1)
    axes[2].set_xlabel('时间 (ms)')
    axes[2].set_ylabel('电压 (V)')
    axes[2].set_title('总输出局部放大')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return cascaded_system, V_total, V_modules

if __name__ == "__main__":
    debug_cascaded_detailed()
