#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHE-PWM功能测试脚本
验证选择性谐波消除PWM的实现
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import HBridgeUnit, CascadedHBridgeSystem
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

def test_she_angles_calculation():
    """测试SHE开关角计算"""
    print("=== 测试SHE开关角计算 ===")
    
    # 创建H桥单元
    hbridge = HBridgeUnit(Vdc=1000, fsw=1000, f_grid=50)
    
    # 测试不同调制比
    modulation_indices = [0.6, 0.8, 0.9]
    harmonic_orders_list = [[5, 7], [5, 7, 11], [5, 7, 11, 13]]
    
    for mi in modulation_indices:
        print(f"\n调制比: {mi}")
        print("-" * 30)
        
        for harmonic_orders in harmonic_orders_list:
            print(f"消除谐波: {harmonic_orders}")
            
            # 计算开关角
            angles = hbridge.calculate_she_angles(mi, harmonic_orders)
            
            if angles is not None:
                print(f"  开关角: {np.degrees(angles)}°")
                print(f"  开关角(弧度): {angles}")
                
                # 验证开关角是否满足SHE方程
                verify_she_equations(angles, mi, harmonic_orders)
            else:
                print("  计算失败")
    
    return hbridge

def verify_she_equations(angles, modulation_index, harmonic_orders):
    """验证SHE方程是否满足"""
    print("  验证SHE方程:")
    
    # 基波幅值方程
    fundamental = 0
    for alpha in angles:
        fundamental += 4 * np.cos(alpha) / np.pi
    
    fundamental_error = abs(fundamental - modulation_index)
    print(f"    基波幅值: {fundamental:.4f} (目标: {modulation_index:.4f}, 误差: {fundamental_error:.4f})")
    
    # 谐波消除方程
    for h in harmonic_orders:
        harmonic = 0
        for alpha in angles:
            harmonic += 4 * np.cos(h * alpha) / (h * np.pi)
        
        harmonic_error = abs(harmonic)
        print(f"    {h}次谐波: {harmonic:.6f} (目标: 0, 误差: {harmonic_error:.6f})")

def test_she_waveform_generation():
    """测试SHE波形生成"""
    print("\n=== 测试SHE波形生成 ===")
    
    # 创建H桥单元
    hbridge = HBridgeUnit(Vdc=1000, fsw=1000, f_grid=50)
    
    # 仿真参数
    t = np.linspace(0, 0.02, 10000)  # 一个工频周期
    modulation_index = 0.8
    
    # 计算SHE开关角
    angles = hbridge.calculate_she_angles(modulation_index, [5, 7])
    
    if angles is not None:
        print(f"使用开关角: {np.degrees(angles)}°")
        
        # 生成SHE波形
        V_she = hbridge.generate_she_waveform(t, modulation_index)
        
        # 生成SPWM波形作为对比
        V_spwm = hbridge.calculate_output_voltage(t, modulation_index)
        
        # 绘制波形对比
        plot_she_vs_spwm(t, V_she, V_spwm, modulation_index, angles)
        
        return V_she, V_spwm
    else:
        print("SHE角度计算失败")
        return None, None

def plot_she_vs_spwm(t, V_she, V_spwm, modulation_index, angles):
    """绘制SHE vs SPWM波形对比"""
    # 创建自适应图形
    fig, axes = create_adaptive_figure(2, 2, title='SHE-PWM vs SPWM Comparison')
    
    # 第一行：时域波形对比
    axes[0, 0].plot(t * 1000, V_she, 'b-', linewidth=2, label='SHE-PWM')
    axes[0, 0].plot(t * 1000, V_spwm, 'r--', linewidth=1, alpha=0.7, label='SPWM')
    format_axis_labels(axes[0, 0], 'Time (ms)', 'Voltage (V)', 'Output Voltage Comparison')
    axes[0, 0].legend()
    add_grid(axes[0, 0])
    set_adaptive_ylim(axes[0, 0], np.concatenate([V_she, V_spwm]))
    
    # 第二行：一个周期的详细对比
    # 找到第一个完整周期
    period_samples = int(0.02 / (t[1] - t[0]))
    t_period = t[:period_samples] * 1000
    V_she_period = V_she[:period_samples]
    V_spwm_period = V_spwm[:period_samples]
    
    axes[0, 1].plot(t_period, V_she_period, 'b-', linewidth=2, label='SHE-PWM')
    axes[0, 1].plot(t_period, V_spwm_period, 'r--', linewidth=1, alpha=0.7, label='SPWM')
    format_axis_labels(axes[0, 1], 'Time (ms)', 'Voltage (V)', 'One Period Detail')
    axes[0, 1].legend()
    add_grid(axes[0, 1])
    set_adaptive_ylim(axes[0, 1], np.concatenate([V_she_period, V_spwm_period]))
    
    # 第三行：开关角标记
    axes[1, 0].plot(t_period, V_she_period, 'b-', linewidth=2, label='SHE-PWM')
    
    # 标记开关角
    omega = 2 * np.pi * 50  # 50Hz
    for i, angle in enumerate(angles):
        time_point = angle / omega * 1000  # 转换为毫秒
        if time_point <= 10:  # 只显示前10ms内的开关角
            axes[1, 0].axvline(x=time_point, color='red', linestyle='--', alpha=0.7, 
                              label=f'Switch {i+1}: {np.degrees(angle):.1f}°')
    
    format_axis_labels(axes[1, 0], 'Time (ms)', 'Voltage (V)', 'SHE-PWM with Switching Angles')
    axes[1, 0].legend()
    add_grid(axes[1, 0])
    set_adaptive_ylim(axes[1, 0], V_she_period)
    
    # 第四行：开关角分布
    angles_deg = np.degrees(angles)
    axes[1, 1].bar(range(1, len(angles)+1), angles_deg, color='green', alpha=0.7)
    format_axis_labels(axes[1, 1], 'Switch Number', 'Angle (degrees)', 'Switching Angles Distribution')
    add_grid(axes[1, 1])
    set_adaptive_ylim(axes[1, 1], angles_deg)
    
    # 优化布局
    optimize_layout(fig)
    finalize_plot(fig, 'SHE-PWM vs SPWM Analysis')
    
    # 保存图片
    plt.savefig('result/she_pwm_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSHE-PWM分析结果已保存到: result/she_pwm_analysis.png")

def test_cascaded_she_system():
    """测试级联SHE系统"""
    print("\n=== 测试级联SHE系统 ===")
    
    # 创建级联SHE系统
    N_modules = 10  # 使用较少的模块进行测试
    cascaded_she = CascadedHBridgeSystem(
        N_modules=N_modules,
        Vdc_per_module=1000,
        fsw=1000,
        f_grid=50,
        modulation_strategy="SHE"
    )
    
    # 仿真参数
    t = np.linspace(0, 0.02, 10000)
    modulation_index = 0.8
    
    # 生成输出电压
    V_total, V_modules = cascaded_she.generate_phase_shifted_pwm(t, modulation_index)
    
    # 计算THD
    thd = cascaded_she.calculate_thd_time_domain(V_total, t) * 100.0
    
    print(f"级联SHE系统结果:")
    print(f"- 模块数: {N_modules}")
    print(f"- 调制比: {modulation_index}")
    print(f"- THD: {thd:.2f}%")
    print(f"- 输出电压范围: [{np.min(V_total):.1f}, {np.max(V_total):.1f}] V")
    
    # 绘制级联SHE结果
    plot_cascaded_she_results(t, V_total, V_modules, freqs, magnitude, cascaded_she)
    
    return cascaded_she, V_total, thd

def calculate_thd(freqs, magnitude, fundamental_freq):
    """计算THD"""
    try:
        # 找到基频分量
        fundamental_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fundamental_magnitude = magnitude[fundamental_idx]
        
        if fundamental_magnitude <= 0:
            return float('inf')
        
        # 计算谐波功率（排除基频）
        harmonic_power = 0
        for i, freq in enumerate(freqs):
            if freq > fundamental_freq and magnitude[i] > 0:
                harmonic_power += magnitude[i]**2
        
        # 计算THD
        thd = np.sqrt(harmonic_power) / fundamental_magnitude * 100
        return thd
        
    except Exception as e:
        print(f"THD计算错误: {e}")
        return float('inf')

def plot_cascaded_she_results(t, V_total, V_modules, freqs, magnitude, system):
    """绘制级联SHE结果"""
    # 创建自适应图形
    fig, axes = create_adaptive_figure(2, 2, title='Cascaded SHE-PWM System Results')
    
    # 第一行：总输出电压和单个模块
    axes[0, 0].plot(t * 1000, V_total / 1000, 'b-', linewidth=2, label='Total Output')
    format_axis_labels(axes[0, 0], 'Time (ms)', 'Voltage (kV)', 'Cascaded SHE-PWM Output')
    axes[0, 0].legend()
    add_grid(axes[0, 0])
    set_adaptive_ylim(axes[0, 0], V_total / 1000)
    
    # 显示前5个模块的输出
    for i in range(min(5, len(V_modules))):
        axes[0, 1].plot(t * 1000, V_modules[i], alpha=0.7, label=f'Module {i+1}')
    format_axis_labels(axes[0, 1], 'Time (ms)', 'Voltage (V)', 'Individual Module Outputs')
    axes[0, 1].legend()
    add_grid(axes[0, 1])
    set_adaptive_ylim(axes[0, 1], np.array(V_modules[:5]).flatten())
    
    # 第二行：谐波频谱
    axes[1, 0].plot(freqs, magnitude, 'r-', linewidth=2)
    format_axis_labels(axes[1, 0], 'Frequency (Hz)', 'Magnitude (V)', 'Harmonic Spectrum')
    axes[1, 0].set_xlim(0, 5000)
    add_grid(axes[1, 0])
    set_adaptive_ylim(axes[1, 0], magnitude)
    
    # 谐波含量分析
    harmonic_orders = [1, 3, 5, 7, 9, 11, 13, 15]
    harmonic_magnitudes = []
    for order in harmonic_orders:
        freq_target = order * 50
        idx = np.argmin(np.abs(freqs - freq_target))
        harmonic_magnitudes.append(magnitude[idx])
    
    axes[1, 1].bar(harmonic_orders, harmonic_magnitudes, color='orange', alpha=0.7)
    format_axis_labels(axes[1, 1], 'Harmonic Order', 'Magnitude (V)', 'Harmonic Content Analysis')
    add_grid(axes[1, 1])
    set_adaptive_ylim(axes[1, 1], harmonic_magnitudes)
    
    # 优化布局
    optimize_layout(fig)
    finalize_plot(fig, 'Cascaded SHE-PWM System Analysis')
    
    # 保存图片
    plt.savefig('result/cascaded_she_analysis.png', dpi=300, bbox_inches='tight')
    print("\n级联SHE-PWM分析结果已保存到: result/cascaded_she_analysis.png")

def main():
    """主函数"""
    print("开始SHE-PWM功能测试...")
    
    # 测试1：开关角计算
    hbridge = test_she_angles_calculation()
    
    # 测试2：SHE波形生成
    V_she, V_spwm = test_she_waveform_generation()
    
    # 测试3：级联SHE系统
    cascaded_she, V_total, thd = test_cascaded_she_system()
    
    print("\n" + "="*60)
    print("SHE-PWM测试完成！")
    print("="*60)
    print(f"级联SHE系统THD: {thd:.2f}%")
    
    if V_she is not None and V_spwm is not None:
        # 计算波形质量改进
        v_she_rms = np.sqrt(np.mean(V_she**2))
        v_spwm_rms = np.sqrt(np.mean(V_spwm**2))
        print(f"SHE-PWM RMS: {v_she_rms:.1f} V")
        print(f"SPWM RMS: {v_spwm_rms:.1f} V")
        print(f"RMS改进: {((v_she_rms - v_spwm_rms) / v_spwm_rms * 100):.2f}%")

if __name__ == "__main__":
    main()
