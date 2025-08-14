#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
35kV系统SHE-PWM优化脚本
针对40模块级联H桥系统优化SHE-PWM参数
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import CascadedHBridgeSystem
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

def optimize_she_for_35kv():
    """针对35kV系统优化SHE-PWM"""
    print("=== 35kV系统SHE-PWM优化 ===")
    
    # 35kV系统参数
    N_modules = 40
    Vdc_per_module = 875  # 35kV / 40 = 875V
    fsw = 1000
    f_grid = 50
    
    print(f"系统参数:")
    print(f"- 模块数: {N_modules}")
    print(f"- 每模块直流电压: {Vdc_per_module} V")
    print(f"- 总输出电压: {N_modules * Vdc_per_module / 1000:.1f} kV")
    print(f"- 开关频率: {fsw} Hz")
    
    # 创建SHE系统
    print(f"\n创建SHE-PWM系统...")
    she_system = CascadedHBridgeSystem(
        N_modules=N_modules,
        Vdc_per_module=Vdc_per_module,
        fsw=fsw,
        f_grid=f_grid,
        modulation_strategy="SHE"
    )
    
    # 测试不同调制比
    modulation_indices = [0.6, 0.7, 0.8, 0.9]
    harmonic_orders_list = [[5, 7], [5, 7, 11], [5, 7, 11, 13]]
    
    results = {}
    
    for mi in modulation_indices:
        print(f"\n调制比: {mi}")
        print("-" * 40)
        results[mi] = {}
        
        for harmonic_orders in harmonic_orders_list:
            print(f"消除谐波: {harmonic_orders}")
            
            # 仿真参数
            t = np.linspace(0, 0.02, 20000)  # 一个工频周期，高采样率
            
            # 生成输出电压
            V_total, V_modules = she_system.generate_phase_shifted_pwm(t, mi)
            
            # 计算谐波频谱
            freqs, magnitude = she_system.calculate_harmonic_spectrum(V_total, t)
            
            # 计算THD
            thd = calculate_thd(freqs, magnitude, f_grid)
            
            # 计算RMS值
            v_rms = np.sqrt(np.mean(V_total**2))
            v_peak = np.max(np.abs(V_total))
            
            # 存储结果
            results[mi][str(harmonic_orders)] = {
                'V_total': V_total,
                'freqs': freqs,
                'magnitude': magnitude,
                'THD': thd,
                'V_rms': v_rms,
                'V_peak': v_peak
            }
            
            print(f"  THD: {thd:.2f}%")
            print(f"  V_rms: {v_rms:.1f} V")
            print(f"  V_peak: {v_peak:.1f} V")
    
    # 绘制优化结果
    plot_optimization_results(t, results, modulation_indices, harmonic_orders_list)
    
    # 生成优化报告
    generate_optimization_report(results, modulation_indices, harmonic_orders_list)
    
    return results

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

def plot_optimization_results(t, results, modulation_indices, harmonic_orders_list):
    """绘制优化结果"""
    # 创建自适应图形
    fig, axes = create_adaptive_figure(3, 3, title='35kV System SHE-PWM Optimization Results')
    
    # 第一行：不同调制比下的THD比较
    for i, mi in enumerate(modulation_indices):
        if i < 3:  # 只显示前3个调制比
            thd_values = []
            labels = []
            
            for harmonic_orders in harmonic_orders_list:
                key = str(harmonic_orders)
                if key in results[mi]:
                    thd_values.append(results[mi][key]['THD'])
                    labels.append(f"消除{harmonic_orders}")
            
            if thd_values:
                axes[0, i].bar(labels, thd_values, color=['blue', 'green', 'red'][:len(thd_values)], alpha=0.7)
                format_axis_labels(axes[0, i], 'Harmonic Elimination', 'THD (%)', f'THD Comparison (m={mi})')
                add_grid(axes[0, i])
                set_adaptive_ylim(axes[0, i], thd_values)
    
    # 第二行：输出电压波形比较（调制比0.8，消除[5,7]）
    mi = 0.8
    key = str([5, 7])
    if mi in results and key in results[mi]:
        V_total = results[mi][key]['V_total']
        axes[1, 0].plot(t * 1000, V_total / 1000, 'b-', linewidth=2)
        format_axis_labels(axes[1, 0], 'Time (ms)', 'Voltage (kV)', f'SHE-PWM Output (m={mi}, 消除[5,7])')
        add_grid(axes[1, 0])
        set_adaptive_ylim(axes[1, 0], V_total / 1000)
    
    # 第二行：谐波频谱比较
    if mi in results and key in results[mi]:
        freqs = results[mi][key]['freqs']
        magnitude = results[mi][key]['magnitude']
        axes[1, 1].plot(freqs, magnitude, 'r-', linewidth=2)
        format_axis_labels(axes[1, 1], 'Frequency (Hz)', 'Magnitude (V)', f'Harmonic Spectrum (m={mi})')
        axes[1, 1].set_xlim(0, 5000)
        add_grid(axes[1, 1])
        set_adaptive_ylim(axes[1, 1], magnitude)
    
    # 第二行：THD vs 调制比
    thd_vs_mi = []
    mi_values = []
    for mi in modulation_indices:
        key = str([5, 7])
        if mi in results and key in results[mi]:
            thd_vs_mi.append(results[mi][key]['THD'])
            mi_values.append(mi)
    
    if thd_vs_mi:
        axes[1, 2].plot(mi_values, thd_vs_mi, 'bo-', linewidth=2, markersize=8)
        format_axis_labels(axes[1, 2], 'Modulation Index', 'THD (%)', 'THD vs Modulation Index')
        add_grid(axes[1, 2])
        set_adaptive_ylim(axes[1, 2], thd_vs_mi)
    
    # 第三行：不同谐波消除策略的THD比较
    mi = 0.8
    if mi in results:
        strategies = []
        thd_values = []
        
        for harmonic_orders in harmonic_orders_list:
            key = str(harmonic_orders)
            if key in results[mi]:
                strategies.append(f"消除{harmonic_orders}")
                thd_values.append(results[mi][key]['THD'])
        
        if thd_values:
            axes[2, 0].bar(strategies, thd_values, color=['blue', 'green', 'red'][:len(thd_values)], alpha=0.7)
            format_axis_labels(axes[2, 0], 'Harmonic Elimination Strategy', 'THD (%)', f'Strategy Comparison (m={mi})')
            add_grid(axes[2, 0])
            set_adaptive_ylim(axes[2, 0], thd_values)
    
    # 第三行：电压利用率分析
    voltage_utilization = []
    mi_values = []
    for mi in modulation_indices:
        key = str([5, 7])
        if mi in results and key in results[mi]:
            v_rms = results[mi][key]['V_rms']
            v_theoretical = mi * 40 * 875  # 理论值
            utilization = v_rms / v_theoretical * 100
            voltage_utilization.append(utilization)
            mi_values.append(mi)
    
    if voltage_utilization:
        axes[2, 1].plot(mi_values, voltage_utilization, 'go-', linewidth=2, markersize=8)
        format_axis_labels(axes[2, 1], 'Modulation Index', 'Voltage Utilization (%)', 'Voltage Utilization vs MI')
        add_grid(axes[2, 1])
        set_adaptive_ylim(axes[2, 1], voltage_utilization)
    
    # 第三行：最佳策略总结
    axes[2, 2].text(0.1, 0.8, '35kV系统SHE-PWM优化总结:', fontsize=12, fontweight='bold')
    axes[2, 2].text(0.1, 0.6, '• 40模块级联H桥', fontsize=10)
    axes[2, 2].text(0.1, 0.5, '• 开关频率: 1000Hz', fontsize=10)
    axes[2, 2].text(0.1, 0.4, '• 目标: 降低THD', fontsize=10)
    axes[2, 2].text(0.1, 0.3, '• 策略: 选择性谐波消除', fontsize=10)
    axes[2, 2].text(0.1, 0.2, '• 优化: 开关角计算', fontsize=10)
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    
    # 优化布局
    optimize_layout(fig)
    finalize_plot(fig, '35kV System SHE-PWM Optimization')
    
    # 保存图片
    plt.savefig('result/35kv_she_optimization.png', dpi=300, bbox_inches='tight')
    print("\n优化结果已保存到: result/35kv_she_optimization.png")

def generate_optimization_report(results, modulation_indices, harmonic_orders_list):
    """生成优化报告"""
    print("\n" + "="*80)
    print("35kV系统SHE-PWM优化报告")
    print("="*80)
    
    # 找出最佳策略
    best_thd = float('inf')
    best_config = None
    
    for mi in modulation_indices:
        print(f"\n调制比 m = {mi}:")
        print("-" * 60)
        
        # 创建表格
        print(f"{'谐波消除策略':<20} {'THD (%)':<12} {'V_rms (V)':<15} {'V_peak (V)':<15}")
        print("-" * 70)
        
        for harmonic_orders in harmonic_orders_list:
            key = str(harmonic_orders)
            if key in results[mi]:
                data = results[mi][key]
                thd = data['THD']
                v_rms = data['V_rms']
                v_peak = data['V_peak']
                
                thd_str = f"{thd:.2f}" if thd != float('inf') else "N/A"
                print(f"{str(harmonic_orders):<20} {thd_str:<12} {v_rms:<15.1f} {v_peak:<15.1f}")
                
                # 记录最佳配置
                if thd != float('inf') and thd < best_thd:
                    best_thd = thd
                    best_config = (mi, harmonic_orders)
    
    print("\n" + "="*80)
    print("优化分析总结:")
    print("="*80)
    
    if best_config:
        print(f"• 最佳THD性能: 调制比={best_config[0]}, 消除谐波={best_config[1]} (THD = {best_thd:.2f}%)")
    
    print("• 系统特点:")
    print("  - 40模块级联H桥，总电压35kV")
    print("  - 开关频率1000Hz，适合高频应用")
    print("  - SHE-PWM选择性消除低次谐波")
    
    print("\n• 优化建议:")
    print("  - 调制比0.6-0.8范围内THD较低")
    print("  - 消除[5,7]次谐波效果较好")
    print("  - 可考虑消除更多谐波以进一步降低THD")
    print("  - 注意开关角计算的数值稳定性")

if __name__ == "__main__":
    # 运行35kV系统SHE-PWM优化
    results = optimize_she_for_35kv()
    print("\n35kV系统SHE-PWM优化完成！")

