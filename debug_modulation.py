#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试调制过程和输出波形质量
检查为什么输出不像正弦波
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import HBridgeUnit, CascadedHBridgeSystem

def debug_modulation_process():
    """调试调制过程"""
    print("=== 调试调制过程 ===")
    
    # 创建单个H桥单元进行测试
    Vdc = 1000  # V
    fsw = 1000  # Hz
    f_grid = 50  # Hz
    
    hbridge = HBridgeUnit(Vdc, fsw, f_grid)
    
    # 时间向量 - 一个工频周期
    t = np.linspace(0, 0.02, 2000)  # 增加采样点数
    
    # 调制比
    modulation_index = 0.8
    
    print(f"测试参数:")
    print(f"- 直流电压: {Vdc} V")
    print(f"- 开关频率: {fsw} Hz")
    print(f"- 电网频率: {f_grid} Hz")
    print(f"- 调制比: {modulation_index}")
    print(f"- 时间点数: {len(t)}")
    
    # 生成载波和参考波
    carrier = hbridge.generate_carrier_wave(t)
    reference = hbridge.generate_reference_wave(t, modulation_index)
    
    # 生成PWM信号
    pwm_pos, pwm_neg = hbridge.pwm_comparison(t, modulation_index)
    
    # 计算输出电压
    V_out = hbridge.calculate_output_voltage(t, modulation_index)
    
    # 分析波形特征
    print(f"\n波形分析:")
    print(f"- 载波幅值范围: [{np.min(carrier):.3f}, {np.max(carrier):.3f}]")
    print(f"- 参考波幅值范围: [{np.min(reference):.3f}, {np.max(reference):.3f}]")
    print(f"- PWM正脉冲数量: {np.sum(pwm_pos)}")
    print(f"- PWM负脉冲数量: {np.sum(pwm_neg)}")
    print(f"- 输出电压范围: [{np.min(V_out):.0f}, {np.max(V_out):.0f}]")
    
    # 检查载波频率
    carrier_periods = np.sum(np.diff(carrier > 0) != 0) / 2
    actual_fsw = carrier_periods / (t[-1] - t[0])
    print(f"- 实际载波频率: {actual_fsw:.1f} Hz")
    
    # 检查参考波频率
    reference_zero_crossings = np.sum(np.diff(reference > 0) != 0)
    actual_f_grid = reference_zero_crossings / (2 * (t[-1] - t[0]))
    print(f"- 实际参考波频率: {actual_f_grid:.1f} Hz")
    
    # 绘制详细分析图
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # 载波和参考波
    axes[0].plot(t * 1000, carrier, 'b-', label='载波', linewidth=1)
    axes[0].plot(t * 1000, reference, 'r-', label='参考波', linewidth=2)
    axes[0].set_ylabel('幅值')
    axes[0].set_title('载波和参考波')
    axes[0].legend()
    axes[0].grid(True)
    
    # PWM信号
    axes[1].plot(t * 1000, pwm_pos, 'g-', label='PWM正脉冲', linewidth=2)
    axes[1].plot(t * 1000, pwm_neg, 'r-', label='PWM负脉冲', linewidth=2)
    axes[1].set_ylabel('开关状态')
    axes[1].set_title('PWM开关信号')
    axes[1].legend()
    axes[1].grid(True)
    
    # 输出电压
    axes[2].plot(t * 1000, V_out, 'b-', label='输出电压', linewidth=1)
    axes[2].set_ylabel('电压 (V)')
    axes[2].set_title('H桥输出电压')
    axes[2].legend()
    axes[2].grid(True)
    
    # 输出电压的局部放大（显示几个开关周期）
    zoom_start = int(len(t) * 0.1)  # 从10%开始
    zoom_end = int(len(t) * 0.15)   # 到15%结束
    axes[3].plot(t[zoom_start:zoom_end] * 1000, V_out[zoom_start:zoom_end], 'b-', linewidth=1)
    axes[3].set_xlabel('时间 (ms)')
    axes[3].set_ylabel('电压 (V)')
    axes[3].set_title('输出电压局部放大（显示开关细节）')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return hbridge, t, V_out

def debug_cascaded_system():
    """调试级联系统"""
    print("\n=== 调试级联系统 ===")
    
    # 创建级联系统
    N_modules = 5  # 减少模块数便于分析
    Vdc_per_module = 1000
    fsw = 1000
    f_grid = 50
    
    cascaded_system = CascadedHBridgeSystem(N_modules, Vdc_per_module, fsw, f_grid)
    
    # 时间向量
    t = np.linspace(0, 0.02, 2000)
    modulation_index = 0.8
    
    print(f"级联系统参数:")
    print(f"- 模块数: {N_modules}")
    print(f"- 每模块直流电压: {Vdc_per_module} V")
    print(f"- 总输出电压: {cascaded_system.V_total} V")
    
    # 生成输出电压
    V_total, V_modules = cascaded_system.generate_phase_shifted_pwm(t, modulation_index)
    
    print(f"\n级联系统输出分析:")
    print(f"- 总输出电压范围: [{np.min(V_total):.0f}, {np.max(V_total):.0f}]")
    print(f"- 理论最大电压: {N_modules * Vdc_per_module}")
    
    # 分析每个模块的输出
    for i, V_module in enumerate(V_modules):
        print(f"- 模块 {i+1} 输出范围: [{np.min(V_module):.0f}, {np.max(V_module):.0f}]")
    
    # 绘制级联系统输出
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 总输出电压
    axes[0].plot(t * 1000, V_total, 'b-', label='总输出电压', linewidth=1)
    axes[0].set_ylabel('电压 (V)')
    axes[0].set_title('级联H桥系统总输出电压')
    axes[0].legend()
    axes[0].grid(True)
    
    # 单个模块输出（显示前3个）
    for i in range(min(3, len(V_modules))):
        axes[1].plot(t * 1000, V_modules[i], alpha=0.7, label=f'模块 {i+1}')
    axes[1].set_ylabel('电压 (V)')
    axes[1].set_title('单个H桥模块输出电压')
    axes[1].legend()
    axes[1].grid(True)
    
    # 总输出的局部放大
    zoom_start = int(len(t) * 0.1)
    zoom_end = int(len(t) * 0.15)
    axes[2].plot(t[zoom_start:zoom_end] * 1000, V_total[zoom_start:zoom_end], 'b-', linewidth=1)
    axes[2].set_xlabel('时间 (ms)')
    axes[2].set_ylabel('电压 (V)')
    axes[2].set_title('总输出电压局部放大')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return cascaded_system, t, V_total, V_modules

def analyze_waveform_quality(V_output, t):
    """分析波形质量"""
    print("\n=== 波形质量分析 ===")
    
    # 计算RMS值
    V_rms = np.sqrt(np.mean(V_output**2))
    print(f"- 输出电压RMS值: {V_rms:.1f} V")
    
    # 计算基波分量（通过低通滤波）
    from scipy import signal
    
    # 设计低通滤波器，截止频率为电网频率的2倍
    fs = 1.0 / (t[1] - t[0])
    cutoff = 2 * 50  # 100 Hz
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    
    # 使用Butterworth滤波器
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    V_filtered = signal.filtfilt(b, a, V_output)
    
    # 计算基波RMS
    V_fundamental_rms = np.sqrt(np.mean(V_filtered**2))
    print(f"- 基波分量RMS值: {V_fundamental_rms:.1f} V")
    
    # 计算谐波含量
    V_harmonic_rms = np.sqrt(V_rms**2 - V_fundamental_rms**2)
    thd = (V_harmonic_rms / V_fundamental_rms) * 100
    print(f"- 谐波含量RMS值: {V_harmonic_rms:.1f} V")
    print(f"- 总谐波失真: {thd:.2f}%")
    
    # 绘制滤波结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t * 1000, V_output, 'b-', label='原始输出', linewidth=1, alpha=0.7)
    plt.plot(t * 1000, V_filtered, 'r-', label='基波分量', linewidth=2)
    plt.ylabel('电压 (V)')
    plt.title('原始输出与基波分量对比')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t * 1000, V_output - V_filtered, 'g-', label='谐波分量', linewidth=1)
    plt.xlabel('时间 (ms)')
    plt.ylabel('电压 (V)')
    plt.title('谐波分量')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return V_filtered, thd

if __name__ == "__main__":
    # 调试单个H桥调制过程
    hbridge, t, V_out = debug_modulation_process()
    
    # 调试级联系统
    cascaded_system, t, V_total, V_modules = debug_cascaded_system()
    
    # 分析波形质量
    V_filtered, thd = analyze_waveform_quality(V_total, t)
    
    print("\n=== 调制过程调试完成 ===")
    print("请检查上述图表，分析调制过程中的问题")

