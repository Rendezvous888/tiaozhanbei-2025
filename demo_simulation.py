#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示仿真模块
提供PCS系统的快速演示和验证功能
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy import signal
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PCSParameters:
    """PCS System Parameters"""
    def __init__(self):
        # System parameters
        self.V_grid = 35e3          # Grid voltage (V)
        self.P_rated = 25e6         # Rated power (W)
        self.I_rated = self.P_rated / (np.sqrt(3) * self.V_grid)  # Rated current (A)
        self.f_grid = 50            # Grid frequency (Hz)
        
        # 导入设备参数
        from device_parameters import get_optimized_parameters
        device_params = get_optimized_parameters()
        
        # H-bridge parameters - 使用device_parameters.py中的参数
        self.N_modules_per_phase = device_params['system'].cascaded_power_modules  # H-bridge modules per phase
        self.Vdc_per_module = self.V_grid / (np.sqrt(3) * self.N_modules_per_phase)  # DC voltage per module (V)
        self.Cdc_per_module = device_params['system'].module_dc_bus_capacitance_mF * 1e-3  # DC capacitor per module (F)
        
        # IGBT parameters - 使用device_parameters.py中的参数
        self.fsw = device_params['system'].module_switching_frequency_Hz  # IGBT switching frequency (Hz)
        self.Vce_sat = device_params['igbt'].Vce_sat_V["25C"][0]  # IGBT saturation voltage (V)
        self.Vf = device_params['igbt'].diode_Vf_V[0]  # Diode forward voltage (V)
        self.Tj_max = device_params['igbt'].junction_temperature_C[1]  # Maximum junction temperature (°C)
        
        # Thermal parameters - 使用device_parameters.py中的参数
        self.Rth_jc = device_params['thermal'].Rth_jc  # Junction to case thermal resistance (K/W)
        self.Rth_ca = device_params['thermal'].Rth_ca  # Case to ambient thermal resistance (K/W)
        self.Cth_jc = device_params['thermal'].Cth_jc  # Junction to case thermal capacitance (J/K)
        self.Cth_ca = device_params['thermal'].Cth_ca  # Case to ambient thermal capacitance (J/K)
        self.T_amb = device_params['thermal'].T_amb  # Ambient temperature (°C)

class HBridgeUnit:
    """Single H-bridge Unit Model"""
    def __init__(self, Vdc=1000, fsw=1000, f_grid=50):
        self.Vdc = Vdc
        self.fsw = fsw
        self.f_grid = f_grid
        
    def generate_pwm_voltage(self, t, modulation_index, phase_shift=0):
        """Generate PWM output voltage"""
        omega = 2 * np.pi * self.f_grid
        reference = modulation_index * np.sin(omega * t + phase_shift)
        
        # Simple PWM implementation
        carrier = 2 * (t * self.fsw - np.floor(t * self.fsw + 0.5))
        
        # PWM comparison
        pwm_positive = reference > carrier
        pwm_negative = reference < -carrier
        
        # Output voltage
        V_out = np.zeros_like(t)
        V_out[pwm_positive] = self.Vdc
        V_out[pwm_negative] = -self.Vdc
        
        return V_out
    
    def calculate_losses(self, I_rms, duty_cycle=0.5):
        """Calculate switching and conduction losses"""
        # 导入设备参数
        from device_parameters import get_optimized_parameters
        device_params = get_optimized_parameters()
        
        # IGBT parameters - 使用device_parameters.py中的参数
        Vce_sat = device_params['igbt'].Vce_sat_V["25C"][0]  # IGBT saturation voltage (V)
        Vf = device_params['igbt'].diode_Vf_V[0]  # Diode forward voltage (V)
        
        # Switching losses - 使用device_parameters.py中的开关损耗
        E_on = device_params['igbt'].switching_energy_mJ['Eon'][0] * 1e-3  # Turn-on loss (J)
        E_off = device_params['igbt'].switching_energy_mJ['Eoff'][0] * 1e-3  # Turn-off loss (J)
        E_rec = device_params['igbt'].diode_Erec_mJ[0] * 1e-3  # Reverse recovery loss (J)
        
        P_sw = (E_on + E_off + E_rec) * self.fsw  # Switching loss power (W)
        
        # Conduction losses
        P_cond_igbt = Vce_sat * I_rms * duty_cycle
        P_cond_diode = Vf * I_rms * (1 - duty_cycle)
        P_cond = P_cond_igbt + P_cond_diode
        
        return P_sw, P_cond

class CascadedHBridgeSystem:
    """Cascaded H-bridge System"""
    def __init__(self, N_modules=40, Vdc_per_module=1000, fsw=1000, f_grid=50):
        self.N_modules = N_modules
        self.Vdc_per_module = Vdc_per_module
        self.fsw = fsw
        self.f_grid = f_grid
        
        # Create H-bridge units
        self.hbridge_units = [HBridgeUnit(Vdc_per_module, fsw, f_grid) for _ in range(N_modules)]
        
    def generate_output_voltage(self, t, modulation_index):
        """Generate cascaded output voltage"""
        # Phase-shifted PWM
        phase_shifts = np.linspace(0, 2*np.pi, self.N_modules, endpoint=False)
        
        V_total = np.zeros_like(t)
        for i, hbridge in enumerate(self.hbridge_units):
            V_module = hbridge.generate_pwm_voltage(t, modulation_index, phase_shifts[i])
            V_total += V_module
        
        return V_total
    
    def calculate_total_losses(self, I_rms):
        """Calculate total system losses"""
        total_switching_loss = 0
        total_conduction_loss = 0
        
        for hbridge in self.hbridge_units:
            I_module = I_rms / self.N_modules
            P_sw, P_cond = hbridge.calculate_losses(I_module)
            total_switching_loss += P_sw
            total_conduction_loss += P_cond
        
        return total_switching_loss, total_conduction_loss

class ThermalModel:
    """IGBT Thermal Model"""
    def __init__(self, params):
        self.params = params
        self.Tj = params.T_amb  # Junction temperature
        self.Tc = params.T_amb  # Case temperature
        
    def update_temperature(self, P_loss, dt):
        """Update temperature state"""
        # Thermal network model
        dTj_dt = (P_loss - (self.Tj - self.Tc) / self.params.Rth_jc) / self.params.Cth_jc
        self.Tj += dTj_dt * dt
        
        dTc_dt = ((self.Tj - self.Tc) / self.params.Rth_jc - (self.Tc - self.params.T_amb) / self.params.Rth_ca) / self.params.Cth_ca
        self.Tc += dTc_dt * dt
        
        # Temperature limits
        self.Tj = np.clip(self.Tj, self.params.T_amb, self.params.Tj_max)
        self.Tc = np.clip(self.Tc, self.params.T_amb - 20, self.params.T_amb + 80)
        
        return self.Tj, self.Tc

def run_demo_simulation():
    """Run the demo simulation"""
    print("=" * 60)
    print("35 kV/25 MW PCS Simulation Demo")
    print("=" * 60)
    
    # Create system parameters
    params = PCSParameters()
    
    print(f"System Parameters:")
    print(f"- Rated Power: {params.P_rated/1e6:.1f} MW")
    print(f"- Grid Voltage: {params.V_grid/1e3:.1f} kV")
    print(f"- Modules per Phase: {params.N_modules_per_phase}")
    print(f"- DC Voltage per Module: {params.Vdc_per_module:.1f} V")
    print(f"- Switching Frequency: {params.fsw} Hz")
    
    # Create cascaded H-bridge system
    cascaded_system = CascadedHBridgeSystem(
        N_modules=params.N_modules_per_phase,
        Vdc_per_module=params.Vdc_per_module,
        fsw=params.fsw,
        f_grid=params.f_grid
    )
    
    # Simulation time
    t = np.linspace(0, 0.02, 1000)  # One grid cycle
    
    # Generate output voltage
    modulation_index = 0.8
    V_output = cascaded_system.generate_output_voltage(t, modulation_index)
    
    # Calculate losses
    I_rms = 100  # Assume RMS current
    P_sw, P_cond = cascaded_system.calculate_total_losses(I_rms)
    P_total = P_sw + P_cond
    
    print(f"\nLoss Analysis:")
    print(f"- Total Loss: {P_total/1e3:.2f} kW")
    print(f"- Switching Loss: {P_sw/1e3:.2f} kW")
    print(f"- Conduction Loss: {P_cond/1e3:.2f} kW")
    
    # Thermal simulation
    thermal_model = ThermalModel(params)
    dt = t[1] - t[0]
    
    Tj_history = []
    Tc_history = []
    
    for i in range(len(t)):
        # Assume constant loss for simplicity
        Tj, Tc = thermal_model.update_temperature(P_total, dt)
        Tj_history.append(Tj)
        Tc_history.append(Tc)
    
    # Plot results
    plot_demo_results(t, V_output, Tj_history, Tc_history, params, cascaded_system)
    
    return params, cascaded_system, V_output, P_total

def plot_demo_results(t, V_output, Tj_history, Tc_history, params, system):
    """Plot demo results"""
    # 使用自适应绘图工具创建图形
    fig, axes = create_adaptive_figure(2, 2, title='PCS演示仿真结果', title_size=14)
    
    # 子图1: 输出电压
    ax1 = axes[0, 0]
    ax1.plot(t * 1000, V_output / 1000, 'b-', linewidth=2)
    format_axis_labels(ax1, '时间 (ms)', '输出电压 (kV)', '输出电压波形')
    add_grid(ax1)
    set_adaptive_ylim(ax1, V_output / 1000)
    
    # 子图2: 温度响应
    ax2 = axes[0, 1]
    ax2.plot(t * 1000, Tj_history, 'r-', label='Junction Temp', linewidth=2)
    ax2.plot(t * 1000, Tc_history, 'g-', label='Case Temp', linewidth=2)
    format_axis_labels(ax2, '时间 (ms)', '温度 (°C)', '温度响应')
    ax2.legend(fontsize=8, loc='best')
    add_grid(ax2)
    set_adaptive_ylim(ax2, np.concatenate([Tj_history, Tc_history]))
    
    # 子图3: 频谱分析
    ax3 = axes[1, 0]
    # 计算FFT
    freqs = np.fft.fftfreq(len(V_output), t[1] - t[0])
    fft_vals = np.abs(np.fft.fft(V_output))
    
    # 只显示正频率部分
    freqs_positive = freqs[freqs > 0]
    magnitude_positive = fft_vals[freqs > 0]
    
    ax3.plot(freqs_positive, magnitude_positive, 'purple', linewidth=2)
    format_axis_labels(ax3, '频率 (Hz)', '幅值', '输出电压频谱')
    add_grid(ax3)
    set_adaptive_ylim(ax3, magnitude_positive)
    
    # 子图4: 系统参数信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 系统信息文本
    info_text = f"""System Parameters:
    
Rated Power: {params.P_rated/1e6:.1f} MW
Grid Voltage: {params.V_grid/1e3:.1f} kV
Switching Freq: {params.fsw} Hz
Modules/Phase: {params.N_modules_per_phase}

Performance:
Max Junction Temp: {np.max(Tj_history):.1f}°C
Max Case Temp: {np.max(Tc_history):.1f}°C
Output Voltage RMS: {np.sqrt(np.mean(V_output**2))/1000:.2f} kV

Topology: {system.__class__.__name__}"""
    
    # 使用自适应文本框，避免文字重叠
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             wrap=True)
    
    # 优化布局，避免重叠
    optimize_layout(fig, tight_layout=True, h_pad=1.5, w_pad=1.5)
    
    # 显示图形
    finalize_plot(fig)

def generate_daily_profile():
    """Generate 24-hour power profile"""
    t = np.linspace(0, 24, 1440)  # One point per minute
    P = np.zeros_like(t)
    
    # Typical energy storage operation
    # Charge during low price hours (2-6 AM, 10 PM-12 AM)
    P[(t >= 2) & (t < 6)] = -25e6 * 0.8
    P[(t >= 22) & (t < 24)] = -25e6 * 0.8
    
    # Discharge during high price hours (8-12 AM, 2-6 PM)
    P[(t >= 8) & (t < 12)] = 25e6 * 0.9
    P[(t >= 14) & (t < 18)] = 25e6 * 0.9
    
    return t, P

def run_daily_simulation():
    """Run 24-hour daily simulation"""
    print("\n" + "=" * 60)
    print("24-Hour Daily Operation Simulation")
    print("=" * 60)
    
    # Generate daily profile
    t, P_profile = generate_daily_profile()
    
    # System parameters
    params = PCSParameters()
    cascaded_system = CascadedHBridgeSystem(
        N_modules=params.N_modules_per_phase,
        Vdc_per_module=params.Vdc_per_module,
        fsw=params.fsw,
        f_grid=params.f_grid
    )
    
    # Thermal model
    thermal_model = ThermalModel(params)
    
    # Simulation results
    Tj_history = []
    efficiency_history = []
    dt = t[1] - t[0]
    
    for i, P_out in enumerate(P_profile):
        if P_out != 0:
            # Calculate losses
            I_rms = abs(P_out) / (np.sqrt(3) * params.V_grid)
            P_sw, P_cond = cascaded_system.calculate_total_losses(I_rms)
            P_total_loss = P_sw + P_cond
            
            # Update temperature
            Tj, _ = thermal_model.update_temperature(P_total_loss, dt)
            
            # Calculate efficiency
            efficiency = abs(P_out) / (abs(P_out) + P_total_loss)
        else:
            Tj = thermal_model.Tj
            efficiency = 1.0
        
        Tj_history.append(Tj)
        efficiency_history.append(efficiency)
    
    # Plot daily results
    plot_daily_results(t, P_profile, Tj_history, efficiency_history, params)
    
    # Calculate statistics
    avg_efficiency = np.mean(efficiency_history)
    max_temp = np.max(Tj_history)
    avg_temp = np.mean(Tj_history)
    
    print(f"\nDaily Operation Results:")
    print(f"- Average Efficiency: {avg_efficiency*100:.2f}%")
    print(f"- Maximum Temperature: {max_temp:.1f}°C")
    print(f"- Average Temperature: {avg_temp:.1f}°C")
    print(f"- Charge Hours: 2-6 AM, 10 PM-12 AM")
    print(f"- Discharge Hours: 8-12 AM, 2-6 PM")

def plot_daily_results(t, P_profile, Tj_history, efficiency_history, params):
    """Plot daily operation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('24-Hour Daily Operation Results', fontsize=16, fontweight='bold')
    
    # Power profile
    axes[0, 0].plot(t, P_profile / 1e6, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Power (MW)')
    axes[0, 0].set_title('Daily Power Profile')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Temperature profile
    axes[0, 1].plot(t, Tj_history, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('IGBT Junction Temperature')
    axes[0, 1].grid(True)
    
    # Efficiency profile
    axes[1, 0].plot(t, np.array(efficiency_history) * 100, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Efficiency (%)')
    axes[1, 0].set_title('System Efficiency')
    axes[1, 0].grid(True)
    
    # Statistics
    axes[1, 1].axis('off')
    stats_text = f"""
Daily Operation Statistics:

System Performance:
- Average Efficiency: {np.mean(efficiency_history)*100:.2f}%
- Maximum Efficiency: {np.max(efficiency_history)*100:.2f}%
- Minimum Efficiency: {np.min(efficiency_history)*100:.2f}%

Thermal Performance:
- Maximum Temperature: {np.max(Tj_history):.1f}°C
- Average Temperature: {np.mean(Tj_history):.1f}°C
- Temperature Range: {np.max(Tj_history) - np.min(Tj_history):.1f}°C

Operation Schedule:
- Charge Periods: 2-6 AM, 10 PM-12 AM
- Discharge Periods: 8-12 AM, 2-6 PM
- Idle Periods: 6-8 AM, 12-2 PM, 6-10 PM
    """
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting PCS Simulation Demo...")
    
    # Run basic simulation
    params, system, V_output, P_total = run_demo_simulation()
    
    # Run daily simulation
    run_daily_simulation()
    
    print("\n" + "=" * 60)
    print("Simulation Demo Completed Successfully!")
    print("=" * 60) 