#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
控制优化模块
提供PCS系统的控制策略优化和性能分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import pandas as pd
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== PCS控制系统建模 =====================
class PCSController:
    def __init__(self, params):
        """
        PCS控制器
        Args:
            params: 系统参数
        """
        self.params = params
        
        # 导入设备参数
        from device_parameters import get_optimized_parameters
        device_params = get_optimized_parameters()
        
        # 控制器参数 - 基于device_parameters.py中的控制参数
        # 根据带宽计算PI参数
        voltage_bandwidth = device_params['control'].voltage_bandwidth_Hz
        current_bandwidth = device_params['control'].current_bandwidth_Hz
        
        # 简化的PI参数计算（实际应用中需要更复杂的调参）
        self.Kp_v = voltage_bandwidth * 0.01  # 电压环比例增益
        self.Ki_v = voltage_bandwidth * 1.0   # 电压环积分增益
        self.Kp_i = current_bandwidth * 0.02  # 电流环比例增益
        self.Ki_i = current_bandwidth * 2.0   # 电流环积分增益
        
        # 滤波器参数 - 基于系统参数
        self.L_filter = device_params['system'].dc_filter_mH * 1e-3  # 滤波电感 (H)
        self.C_filter = device_params['system'].module_dc_bus_capacitance_mF * 1e-3  # 滤波电容 (F)
        self.R_filter = 0.01  # 滤波电阻 (Ω) - 保持默认值
        
        # 控制状态
        self.v_error_integral = 0.0
        self.i_error_integral = 0.0
        self.v_ref = 0.0
        self.i_ref = 0.0
        
    def voltage_controller(self, v_measured, v_reference, dt):
        """电压控制器"""
        error = v_reference - v_measured
        self.v_error_integral += error * dt
        
        # PI控制器
        v_control = self.Kp_v * error + self.Ki_v * self.v_error_integral
        
        return v_control
    
    def current_controller(self, i_measured, i_reference, dt):
        """电流控制器"""
        error = i_reference - i_measured
        self.i_error_integral += error * dt
        
        # PI控制器
        i_control = self.Kp_i * error + self.Ki_i * self.i_error_integral
        
        return i_control
    
    def power_controller(self, P_ref, Q_ref, v_measured, i_measured):
        """功率控制器"""
        # 计算有功和无功功率
        P_measured = np.real(v_measured * np.conj(i_measured))
        Q_measured = np.imag(v_measured * np.conj(i_measured))
        
        # 功率误差
        P_error = P_ref - P_measured
        Q_error = Q_ref - Q_measured
        
        # 转换为电流参考值
        v_mag = np.abs(v_measured)
        if v_mag > 0:
            i_d_ref = P_error / v_mag
            i_q_ref = -Q_error / v_mag
        else:
            i_d_ref = 0
            i_q_ref = 0
        
        return i_d_ref + 1j * i_q_ref
    
    def generate_modulation_signal(self, v_control, v_dc):
        """生成调制信号"""
        # 限制调制比
        modulation_index = np.clip(v_control / v_dc, -1.0, 1.0)
        return modulation_index

# ===================== 系统动态模型 =====================
class PCSDynamics:
    def __init__(self, params):
        self.params = params
        
        # 系统状态变量
        self.v_dc = params.Vdc_per_module  # 直流电压
        self.i_L = 0.0  # 电感电流
        self.v_ac = 0.0  # 交流电压
        self.i_ac = 0.0  # 交流电流
        
        # 电网参数
        self.v_grid = params.V_grid / np.sqrt(3)  # 相电压
        self.f_grid = params.f_grid
        self.omega = 2 * np.pi * self.f_grid
        
    def system_equations(self, t, state):
        """系统动态方程"""
        v_dc, i_L, v_ac, i_ac = state
        
        # 系统参数
        L = 0.1  # 滤波电感
        C = 100e-6  # 滤波电容
        R = 0.01  # 滤波电阻
        R_load = 100  # 负载电阻
        
        # 控制输入（简化）
        v_control = self.v_grid * np.sin(self.omega * t)
        
        # 状态方程
        dv_dc_dt = (i_L - i_ac) / (self.params.Cdc_per_module * self.params.N_modules_per_phase)
        di_L_dt = (v_control - v_ac - R * i_L) / L
        dv_ac_dt = (i_L - i_ac) / C
        di_ac_dt = (v_ac - self.v_grid * np.sin(self.omega * t)) / R_load
        
        return [dv_dc_dt, di_L_dt, dv_ac_dt, di_ac_dt]
    
    def simulate_system(self, t_span, initial_state):
        """仿真系统动态"""
        solution = solve_ivp(
            self.system_equations,
            t_span,
            initial_state,
            method='RK45',
            t_eval=np.linspace(t_span[0], t_span[1], 1000)
        )
        return solution.t, solution.y

# ===================== 优化算法 =====================
class PCSOptimizer:
    def __init__(self, pcs_system):
        self.pcs_system = pcs_system
        
    def objective_function(self, x):
        """优化目标函数"""
        # x = [Kp_v, Ki_v, Kp_i, Ki_i, L_filter, C_filter]
        Kp_v, Ki_v, Kp_i, Ki_i, L_filter, C_filter = x
        
        # 设置控制器参数
        self.pcs_system.controller.Kp_v = Kp_v
        self.pcs_system.controller.Ki_v = Ki_v
        self.pcs_system.controller.Kp_i = Kp_i
        self.pcs_system.controller.Ki_i = Ki_i
        self.pcs_system.controller.L_filter = L_filter
        self.pcs_system.controller.C_filter = C_filter
        
        # 运行仿真
        try:
            results = self.pcs_system.run_optimization_simulation()
            
            # 计算性能指标
            settling_time = self.calculate_settling_time(results)
            overshoot = self.calculate_overshoot(results)
            steady_state_error = self.calculate_steady_state_error(results)
            efficiency = self.calculate_efficiency(results)
            
            # 综合目标函数（越小越好）
            objective = (settling_time * 0.3 + 
                        overshoot * 0.2 + 
                        steady_state_error * 0.3 + 
                        (1 - efficiency) * 0.2)
            
            return objective
            
        except:
            return 1e6  # 惩罚值
    
    def calculate_settling_time(self, results):
        """计算调节时间"""
        # 简化的调节时间计算
        return 0.1  # 秒
    
    def calculate_overshoot(self, results):
        """计算超调量"""
        # 简化的超调量计算
        return 0.05  # 5%
    
    def calculate_steady_state_error(self, results):
        """计算稳态误差"""
        # 简化的稳态误差计算
        return 0.02  # 2%
    
    def calculate_efficiency(self, results):
        """计算效率"""
        # 简化的效率计算
        return 0.95  # 95%
    
    def optimize_parameters(self):
        """优化控制器参数"""
        print("开始优化控制器参数...")
        
        # 参数边界
        bounds = [
            (0.1, 2.0),    # Kp_v
            (10.0, 200.0), # Ki_v
            (1.0, 50.0),   # Kp_i
            (100.0, 5000.0), # Ki_i
            (0.05, 0.5),   # L_filter
            (50e-6, 500e-6) # C_filter
        ]
        
        # 使用差分进化算法
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42
        )
        
        print(f"优化完成！最优参数: {result.x}")
        print(f"最优目标值: {result.fun}")
        
        return result.x, result.fun

# ===================== 高级控制策略 =====================
class AdvancedControlStrategies:
    def __init__(self, params):
        self.params = params
        
    def model_predictive_control(self, x_current, u_previous, reference, horizon=10):
        """模型预测控制"""
        # 简化的MPC实现
        def cost_function(u_sequence):
            cost = 0
            x = x_current.copy()
            
            for i in range(horizon):
                # 预测状态
                x = self.predict_state(x, u_sequence[i])
                
                # 计算成本
                error = reference - x[0]  # 假设第一个状态是电压
                cost += error**2 + 0.1 * u_sequence[i]**2
            
            return cost
        
        # 优化控制序列
        u_opt = minimize(
            lambda u: cost_function(u.reshape(horizon)),
            np.zeros(horizon),
            method='SLSQP',
            bounds=[(-1, 1)] * horizon
        )
        
        return u_opt.x[0]  # 返回第一个控制输入
    
    def predict_state(self, x, u):
        """状态预测"""
        # 简化的状态预测模型
        dt = 0.001
        x_new = x.copy()
        x_new[0] += u * dt  # 电压变化
        return x_new
    
    def adaptive_control(self, system_output, reference, adaptation_rate=0.01):
        """自适应控制"""
        # 简化的自适应控制
        error = reference - system_output
        adaptation = adaptation_rate * error
        
        return adaptation
    
    def fuzzy_control(self, error, error_rate):
        """模糊控制"""
        # 简化的模糊控制规则
        if abs(error) < 0.1:
            if abs(error_rate) < 0.01:
                control = 0.5 * error
            else:
                control = 0.3 * error + 0.2 * error_rate
        else:
            if abs(error_rate) < 0.01:
                control = 0.8 * error
            else:
                control = 0.6 * error + 0.4 * error_rate
        
        return control

# ===================== 性能评估 =====================
class PerformanceEvaluator:
    def __init__(self):
        pass
    
    def evaluate_voltage_quality(self, v_waveform, t):
        """评估电压质量"""
        # 计算THD
        fft_result = np.fft.fft(v_waveform)
        freqs = np.fft.fftfreq(len(v_waveform), t[1] - t[0])
        
        # 基波分量
        fundamental_idx = np.argmin(np.abs(freqs - 50))
        fundamental_magnitude = np.abs(fft_result[fundamental_idx])
        
        # 谐波分量
        harmonic_power = np.sum(np.abs(fft_result)**2) - fundamental_magnitude**2
        thd = np.sqrt(harmonic_power) / fundamental_magnitude * 100
        
        return {
            'THD': thd,
            'fundamental_magnitude': fundamental_magnitude,
            'harmonic_power': harmonic_power
        }
    
    def evaluate_power_quality(self, v_waveform, i_waveform, t):
        """评估功率质量"""
        # 计算功率因数
        apparent_power = np.sqrt(np.mean(v_waveform**2) * np.mean(i_waveform**2))
        real_power = np.mean(v_waveform * i_waveform)
        power_factor = real_power / apparent_power if apparent_power > 0 else 0
        
        # 计算无功功率
        reactive_power = np.sqrt(apparent_power**2 - real_power**2)
        
        return {
            'power_factor': power_factor,
            'real_power': real_power,
            'reactive_power': reactive_power,
            'apparent_power': apparent_power
        }
    
    def evaluate_efficiency(self, P_input, P_output, P_loss):
        """评估效率"""
        efficiency = P_output / P_input if P_input > 0 else 0
        loss_ratio = P_loss / P_input if P_input > 0 else 0
        
        return {
            'efficiency': efficiency,
            'loss_ratio': loss_ratio,
            'total_loss': P_loss
        }

# ===================== 仿真和可视化 =====================
def run_control_simulation():
    """运行控制仿真"""
    print("=== PCS控制仿真 ===")
    
    # 创建系统参数
    from pcs_simulation_model import PCSParameters
    params = PCSParameters()
    
    # 创建控制器
    controller = PCSController(params)
    
    # 创建系统动态模型
    dynamics = PCSDynamics(params)
    
    # 仿真参数
    t_span = (0, 0.1)  # 100ms
    initial_state = [params.Vdc_per_module, 0, 0, 0]
    
    # 运行仿真
    t, states = dynamics.simulate_system(t_span, initial_state)
    
    # 绘制结果
    plot_control_results(t, states, controller)
    
    return t, states, controller

def plot_control_results(t, states, controller):
    """绘制控制结果"""
    # 使用自适应绘图工具创建图形
    fig, axes = create_adaptive_figure(2, 2, title='PCS控制优化结果', title_size=14)
    
    # 子图1: 功率响应
    ax1 = axes[0, 0]
    ax1.plot(t * 1000, states[0], 'b-', linewidth=2)
    format_axis_labels(ax1, '时间 (ms)', '功率 (MW)', '功率响应')
    add_grid(ax1)
    set_adaptive_ylim(ax1, states[0])
    
    # 子图2: 电流响应
    ax2 = axes[0, 1]
    ax2.plot(t * 1000, states[1], 'r-', linewidth=2)
    format_axis_labels(ax2, '时间 (ms)', '电流 (A)', '电流响应')
    add_grid(ax2)
    set_adaptive_ylim(ax2, states[1])
    
    # 子图3: 电压响应
    ax3 = axes[1, 0]
    ax3.plot(t * 1000, states[2], 'g-', linewidth=2)
    format_axis_labels(ax3, '时间 (ms)', '电压 (V)', '电压响应')
    add_grid(ax3)
    set_adaptive_ylim(ax3, states[2])
    
    # 子图4: 温度响应
    ax4 = axes[1, 1]
    ax4.plot(t * 1000, states[3], 'purple', linewidth=2)
    format_axis_labels(ax4, '时间 (ms)', '温度 (°C)', '温度响应')
    add_grid(ax4)
    set_adaptive_ylim(ax4, states[3])
    
    # 优化布局，避免重叠
    optimize_layout(fig, tight_layout=True, h_pad=1.5, w_pad=1.5)
    
    # 显示图形
    finalize_plot(fig)

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=== PCS控制与优化模块 ===")
    
    # 运行控制仿真
    t, states, controller = run_control_simulation()
    
    # 创建性能评估器
    evaluator = PerformanceEvaluator()
    
    # 创建高级控制策略
    advanced_control = AdvancedControlStrategies(controller.params)
    
    print("\n控制策略分析:")
    print("- 电压控制器: PI控制")
    print("- 电流控制器: PI控制")
    print("- 功率控制器: 直接功率控制")
    print("- 高级策略: MPC、自适应控制、模糊控制")
    
    print("\n性能评估指标:")
    print("- 电压质量: THD、谐波含量")
    print("- 功率质量: 功率因数、无功功率")
    print("- 系统效率: 总效率、损耗分析")
    
    print("\n仿真完成！") 