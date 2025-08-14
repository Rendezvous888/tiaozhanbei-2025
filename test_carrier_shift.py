#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试载波相移功能
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import HBridgeUnit

def test_carrier_shift():
    """测试载波相移"""
    print("=== 测试载波相移 ===")
    
    # 创建H桥单元
    Vdc = 1000
    fsw = 1000
    f_grid = 50
    
    hbridge = HBridgeUnit(Vdc, fsw, f_grid)
    
    # 时间向量 - 几个载波周期
    t = np.linspace(0, 0.005, 1000)  # 5ms，5个载波周期
    
    # 测试不同的相移
    shifts = [0, 0.25, 0.5, 0.75]  # 0°, 90°, 180°, 270°
    
    plt.figure(figsize=(15, 10))
    
    for i, shift in enumerate(shifts):
        # 生成带相移的载波
        carrier_shifted = hbridge.generate_carrier_wave_with_shift(t, shift)
        carrier_normal = hbridge.generate_carrier_wave(t)
        
        # 绘制载波对比
        plt.subplot(2, 2, i+1)
        plt.plot(t * 1000, carrier_normal, 'b-', label='原始载波', linewidth=2)
        plt.plot(t * 1000, carrier_shifted, 'r--', label=f'相移 {shift:.2f}', linewidth=2)
        plt.xlabel('时间 (ms)')
        plt.ylabel('载波幅值')
        plt.title(f'载波相移测试 (相移 = {shift:.2f})')
        plt.legend()
        plt.grid(True)
        
        # 计算实际相移
        if shift > 0:
            # 找到载波的过零点
            zero_crossings_normal = np.where(np.diff(carrier_normal > 0) != 0)[0]
            zero_crossings_shifted = np.where(np.diff(carrier_shifted > 0) != 0)[0]
            
            if len(zero_crossings_normal) > 0 and len(zero_crossings_shifted) > 0:
                # 计算第一个过零点的偏移
                time_diff = t[zero_crossings_shifted[0]] - t[zero_crossings_normal[0]]
                phase_diff = time_diff * fsw
                print(f"相移 {shift:.2f}: 理论 = {shift:.2f}, 实际 = {phase_diff:.3f}")
    
    plt.tight_layout()
    plt.show()
    
    return hbridge

def test_pwm_with_shift():
    """测试带相移的PWM"""
    print("\n=== 测试带相移的PWM ===")
    
    hbridge = test_carrier_shift()
    
    # 时间向量
    t = np.linspace(0, 0.02, 2000)  # 一个工频周期
    modulation_index = 0.8
    
    # 测试不同的载波相移
    shifts = [0, 0.2, 0.4, 0.6, 0.8]
    
    plt.figure(figsize=(15, 10))
    
    for i, shift in enumerate(shifts):
        # 生成输出电压
        V_out = hbridge.calculate_output_voltage_with_carrier_shift(t, modulation_index, shift)
        
        # 绘制输出
        plt.subplot(2, 3, i+1)
        plt.plot(t * 1000, V_out, 'b-', linewidth=1)
        plt.xlabel('时间 (ms)')
        plt.ylabel('电压 (V)')
        plt.title(f'载波相移 {shift:.1f}')
        plt.grid(True)
        
        # 计算RMS值
        V_rms = np.sqrt(np.mean(V_out**2))
        print(f"相移 {shift:.1f}: RMS = {V_rms:.0f} V")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_pwm_with_shift()

