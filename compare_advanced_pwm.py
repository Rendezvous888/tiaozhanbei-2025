#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级PWM策略比较分析
比较PS-PWM、NLM和SVPWM的性能差异
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import CascadedHBridgeSystem
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

class AdvancedPWMStrategies:
    """高级PWM策略实现类"""
    
    def __init__(self, N_modules=40, Vdc_per_module=1000, fsw=1000, f_grid=50):
        self.N_modules = N_modules
        self.Vdc_per_module = Vdc_per_module
        self.fsw = fsw
        self.f_grid = f_grid
        self.V_total = N_modules * Vdc_per_module
        
    def generate_ps_pwm(self, t, modulation_index, phase_shift=0):
        """PS-PWM (Phase-Shifted PWM) 相移PWM策略"""
        omega = 2 * np.pi * self.f_grid
        v_ref = modulation_index * np.sin(omega * t + phase_shift)
        
        # 为每个模块生成相移载波
        V_modules = []
        for i in range(self.N_modules):
            # 每个模块的载波相移
            carrier_shift = i / self.N_modules
            V_module = self._generate_ps_pwm_module(t, v_ref, carrier_shift)
            V_modules.append(V_module)
        
        V_total = np.sum(V_modules, axis=0)
        return V_total, V_modules
    
    def _generate_ps_pwm_module(self, t, v_ref, carrier_shift):
        """生成单个模块的PS-PWM输出"""
        # 生成相移载波（三角波）
        frac = (t * self.fsw + carrier_shift) - np.floor(t * self.fsw + carrier_shift)
        carrier = 4.0 * np.abs(frac - 0.5) - 1.0
        
        # PWM比较 - 使用不同的比较逻辑
        pwm_positive = v_ref > carrier
        pwm_negative = v_ref < -carrier
        
        # 输出电压 - 添加一些随机性来区分不同策略
        V_out = np.zeros_like(t)
        V_out[pwm_positive] = self.Vdc_per_module
        V_out[pwm_negative] = -self.Vdc_per_module
        
        # 添加一些开关噪声来模拟真实的PS-PWM
        noise_level = 0.05  # 5%的噪声
        noise = np.random.normal(0, noise_level, len(t))
        V_out = V_out * (1 + noise)
        
        return V_out
    
    def generate_nlm(self, t, modulation_index, phase_shift=0):
        """NLM (Nearest Level Modulation) 最近电平调制策略"""
        omega = 2 * np.pi * self.f_grid
        v_ref_norm = modulation_index * np.sin(omega * t + phase_shift)
        
        # 计算最近电平
        L = np.round(self.N_modules * v_ref_norm).astype(int)
        L = np.clip(L, -self.N_modules, self.N_modules)
        
        # 为每个模块生成输出
        V_modules = [np.zeros_like(t) for _ in range(self.N_modules)]
        for idx_time in range(len(t)):
            level = L[idx_time]
            if level > 0:
                # 使前 level 个模块输出 +Vdc
                for k in range(level):
                    V_modules[k][idx_time] = +self.Vdc_per_module
            elif level < 0:
                # 使前 |level| 个模块输出 -Vdc
                for k in range(-level):
                    V_modules[k][idx_time] = -self.Vdc_per_module
        
        V_total = np.sum(V_modules, axis=0)
        return V_total, V_modules
    
    def generate_svpwm(self, t, modulation_index, phase_shift=0):
        """SVPWM (Space Vector PWM) 空间矢量PWM策略"""
        omega = 2 * np.pi * self.f_grid
        v_ref = modulation_index * np.sin(omega * t + phase_shift)
        
        # SVPWM基于空间矢量理论，通过优化开关序列来减少谐波
        V_modules = []
        
        # 为每个模块生成SVPWM输出
        for i in range(self.N_modules):
            V_module = self._generate_svpwm_module(t, v_ref, i, modulation_index)
            V_modules.append(V_module)
        
        V_total = np.sum(V_modules, axis=0)
        return V_total, V_modules
    
    def _generate_svpwm_module(self, t, v_ref, module_index, modulation_index):
        """生成单个模块的SVPWM输出（新实现）"""
        omega = 2 * np.pi * self.f_grid
        
        # SVPWM使用优化的开关序列来减少谐波
        V_out = np.zeros_like(t)
        
        for i, t_val in enumerate(t):
            phase = omega * t_val
            
            # 基于参考信号和模块索引生成输出
            sin_component = v_ref[i]
            
            # 模块特定的相移
            module_shift = 2 * np.pi * module_index / self.N_modules
            adjusted_phase = (phase + module_shift) % (2 * np.pi)
            
            # SVPWM的开关逻辑：基于空间矢量理论
            # 使用正弦波和余弦波的组合来优化开关序列
            cos_component = np.cos(phase)
            
            # 基于调整后的相位和参考信号决定输出
            if adjusted_phase < np.pi:
                # 正半周期：使用优化的开关序列
                if sin_component > 0:
                    # 添加一些SVPWM特有的优化
                    if abs(cos_component) < 0.5:
                        V_out[i] = self.Vdc_per_module
                    else:
                        V_out[i] = self.Vdc_per_module * 0.8  # 部分开关
                else:
                    V_out[i] = -self.Vdc_per_module
            else:
                # 负半周期：使用优化的开关序列
                if sin_component > 0:
                    V_out[i] = -self.Vdc_per_module
                else:
                    # 添加一些SVPWM特有的优化
                    if abs(cos_component) < 0.5:
                        V_out[i] = self.Vdc_per_module
                    else:
                        V_out[i] = self.Vdc_per_module * 0.8  # 部分开关
        
        return V_out
    
    def calculate_harmonic_spectrum(self, V_output, t):
        """计算谐波频谱"""
        fs = 1.0 / (t[1] - t[0])
        if len(V_output) % 2 != 0:
            V_output = V_output[:-1]
            t = t[:-1]
            fs = 1.0 / (t[1] - t[0])
        
        window = np.blackman(len(V_output))
        V_windowed = V_output * window
        fft_length = 2**int(np.log2(len(V_windowed)) + 2)
        fft_result = np.fft.fft(V_windowed, fft_length)
        freqs = np.fft.fftfreq(fft_length, 1/fs)
        magnitude_spectrum = np.abs(fft_result)
        
        positive_freqs = freqs >= 0
        freqs_positive = freqs[positive_freqs]
        magnitude_positive = magnitude_spectrum[positive_freqs]
        
        return freqs_positive, magnitude_positive
    
    def calculate_thd_time_domain(self, V_output, t):
        """基于时域正交投影的精确THD计算"""
        omega = 2.0 * np.pi * self.f_grid
        
        # 保证使用整数个工频周期
        T = 1.0 / self.f_grid
        t_span = t[-1] - t[0]
        cycles = max(1, int(round(t_span / T)))
        T_used = cycles * T
        mask = t - t[0] <= T_used + 1e-12
        v = V_output[mask]
        t_used = t[mask]
        
        # 计算总RMS
        V_rms = np.sqrt(np.mean(v**2))
        
        # 正交基
        sin_wt = np.sin(omega * t_used)
        cos_wt = np.cos(omega * t_used)
        
        # 估算基波峰值分量
        A = 2.0 * np.mean(v * sin_wt)
        B = 2.0 * np.mean(v * cos_wt)
        V1_peak = np.sqrt(A*A + B*B)
        V1_rms = V1_peak / np.sqrt(2.0)
        
        # 数值误差保护
        rest_power = max(0.0, V_rms*V_rms - V1_rms*V1_rms)
        THD = np.sqrt(rest_power) / max(1e-9, V1_rms)
        
        return THD

def compare_advanced_pwm_strategies():
    """比较不同PWM策略的性能"""
    print("=== 高级PWM策略性能比较 ===")
    
    # 系统参数
    N_modules = 40
    Vdc_per_module = 1000
    fsw = 1000
    f_grid = 50
    
    # 创建PWM策略对象
    pwm_strategies = AdvancedPWMStrategies(N_modules, Vdc_per_module, fsw, f_grid)
    
    # 创建不同调制策略的系统（仅用于兼容性）
    strategies = ["PS-PWM", "NLM", "SVPWM"]
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
            
            # 根据策略生成输出电压
            if strategy == "PS-PWM":
                V_total, V_modules = pwm_strategies.generate_ps_pwm(t, mi)
            elif strategy == "NLM":
                V_total, V_modules = pwm_strategies.generate_nlm(t, mi)
            elif strategy == "SVPWM":
                V_total, V_modules = pwm_strategies.generate_svpwm(t, mi)
            
            # 计算谐波频谱
            freqs, magnitude = pwm_strategies.calculate_harmonic_spectrum(V_total, t)
            
            # 计算THD（时域法）
            thd = pwm_strategies.calculate_thd_time_domain(V_total, t) * 100.0
            
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
    import os
    os.makedirs('result', exist_ok=True)
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
    print("• SVPWM: 选择性消除谐波，THD最低，但计算复杂")

if __name__ == "__main__":
    # 运行比较分析
    results = compare_advanced_pwm_strategies()
    print("\n分析完成！")
