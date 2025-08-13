#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级PWM策略比较分析
比较PS-PWM、NLM和SHE-PWM的性能差异
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import CascadedHBridgeSystem
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

def compare_advanced_pwm_strategies():
    """比较不同PWM策略的性能"""
    print("=== 高级PWM策略性能比较 ===")
    
    # 系统参数
    N_modules = 40
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
    
    # 仿真参数
    t = np.linspace(0, 0.02, 20000)  # 一个工频周期，高采样率
    modulation_indices = [0.6, 0.8, 0.9]
    
    # 存储结果
    results = {}
    
    for strategy in strategies:
        print(f"\n分析 {strategy} 策略...")
        results[strategy] = {}
        
        for mi in modulation_indices:
            print(f"  调制比: {mi}")
            
            # 生成输出电压
            V_total, V_modules = systems[strategy].generate_phase_shifted_pwm(t, mi)
            
            # 计算谐波频谱
            freqs, magnitude = systems[strategy].calculate_harmonic_spectrum(V_total, t)
            
            # 计算THD（改为时域法）
            thd = systems[strategy].calculate_thd_time_domain(V_total, t) * 100.0
            
            # 计算RMS值
            v_rms = np.sqrt(np.mean(V_total**2))
            v_peak = np.max(np.abs(V_total))
            
            # 存储结果
            results[strategy][mi] = {
                'V_total': V_total,
                'V_modules': V_modules,
                'freqs': freqs,
                'magnitude': magnitude,
                'THD': thd,
                'V_rms': v_rms,
                'V_peak': v_peak
            }
            
            print(f"    THD: {thd:.2f}%")
            print(f"    V_rms: {v_rms:.1f} V")
            print(f"    V_peak: {v_peak:.1f} V")
    
    # 绘制比较结果
    plot_pwm_comparison(t, results, strategies, modulation_indices)
    
    # 生成性能报告
    generate_performance_report(results, strategies, modulation_indices)
    
    return results

def calculate_thd(freqs, magnitude, fundamental_freq):
	"""不用频域THD，保留占位。"""
	return None

def plot_pwm_comparison(t, results, strategies, modulation_indices):
    """绘制PWM策略比较图"""
    # 创建自适应图形
    fig, axes = create_adaptive_figure(3, 3, title='Advanced PWM Strategies Comparison')
    
    # 第一行：不同调制比下的THD比较
    for i, mi in enumerate(modulation_indices):
        thd_values = [results[strategy][mi]['THD'] for strategy in strategies]
        valid_indices = [j for j, thd in enumerate(thd_values) if thd != float('inf')]
        
        if valid_indices:
            valid_strategies = [strategies[j] for j in valid_indices]
            valid_thd = [thd_values[j] for j in valid_indices]
            
            axes[0, i].bar(valid_strategies, valid_thd, color=['blue', 'green', 'red'][:len(valid_strategies)], alpha=0.7)
            format_axis_labels(axes[0, i], 'PWM Strategy', 'THD (%)', f'THD Comparison (m={mi})')
            add_grid(axes[0, i])
            set_adaptive_ylim(axes[0, i], valid_thd)
    
    # 第二行：输出电压波形比较（调制比0.8）
    mi = 0.8
    for i, strategy in enumerate(strategies):
        if mi in results[strategy]:
            V_total = results[strategy][mi]['V_total']
            axes[1, i].plot(t * 1000, V_total / 1000, linewidth=2)
            format_axis_labels(axes[1, i], 'Time (ms)', 'Voltage (kV)', f'{strategy} Output (m={mi})')
            add_grid(axes[1, i])
            set_adaptive_ylim(axes[1, i], V_total / 1000)
    
    # 第三行：谐波频谱比较（调制比0.8）
    for i, strategy in enumerate(strategies):
        if mi in results[strategy]:
            freqs = results[strategy][mi]['freqs']
            magnitude = results[strategy][mi]['magnitude']
            axes[2, i].plot(freqs, magnitude, linewidth=2)
            format_axis_labels(axes[2, i], 'Frequency (Hz)', 'Magnitude (V)', f'{strategy} Spectrum (m={mi})')
            axes[2, i].set_xlim(0, 5000)
            add_grid(axes[2, i])
            set_adaptive_ylim(axes[2, i], magnitude)
    
    # 优化布局
    optimize_layout(fig)
    finalize_plot(fig, 'Advanced PWM Strategies Comparison Results')
    
    # 保存图片
    plt.savefig('result/advanced_pwm_comparison.png', dpi=300, bbox_inches='tight')
    print("\n比较结果已保存到: result/advanced_pwm_comparison.png")

def generate_performance_report(results, strategies, modulation_indices):
    """生成性能报告"""
    print("\n" + "="*60)
    print("高级PWM策略性能报告")
    print("="*60)
    
    for mi in modulation_indices:
        print(f"\n调制比 m = {mi}:")
        print("-" * 40)
        
        # 创建表格
        print(f"{'策略':<12} {'THD (%)':<10} {'V_rms (V)':<12} {'V_peak (V)':<12}")
        print("-" * 50)
        
        for strategy in strategies:
            if mi in results[strategy]:
                data = results[strategy][mi]
                thd = data['THD']
                v_rms = data['V_rms']
                v_peak = data['V_peak']
                
                thd_str = f"{thd:.2f}" if thd != float('inf') else "N/A"
                print(f"{strategy:<12} {thd_str:<10} {v_rms:<12.1f} {v_peak:<12.1f}")
    
    print("\n" + "="*60)
    print("性能分析总结:")
    print("="*60)
    
    # 找出最佳策略
    best_strategy = None
    best_thd = float('inf')
    
    for strategy in strategies:
        if 0.8 in results[strategy]:
            thd = results[strategy][0.8]['THD']
            if thd != float('inf') and thd < best_thd:
                best_thd = thd
                best_strategy = strategy
    
    if best_strategy:
        print(f"• 最佳THD性能: {best_strategy} (THD = {best_thd:.2f}%)")
    
    print("• PS-PWM: 适合高频开关，THD中等，开关损耗较高")
    print("• NLM: 开关频率低，THD较低，适合多电平系统")
    print("• SHE-PWM: 选择性消除谐波，THD最低，但计算复杂")

if __name__ == "__main__":
    # 运行比较分析
    results = compare_advanced_pwm_strategies()
    print("\n分析完成！")
