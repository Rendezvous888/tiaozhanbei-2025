#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实物理的IGBT热模型
基于35kV/25MW级联储能PCS架构：每相40个H桥级联
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RealisticIGBTThermalModel:
    """真实物理的IGBT热模型"""
    
    def __init__(self):
        # 系统架构参数 - 基于项目要求文档
        self.system_power_MW = 25           # 系统总功率 25MW
        self.system_voltage_kV = 35         # 系统电压 35kV
        self.cascaded_modules_per_phase = 40    # 每相级联40个H桥模块
        self.phases = 3                     # 三相系统
        self.igbt_per_module = 2           # 每个模块2个IGBT (上下桥臂)
        
        # 计算总数量
        self.total_modules = self.cascaded_modules_per_phase * self.phases  # 120个模块
        self.total_igbts = self.total_modules * self.igbt_per_module        # 240个IGBT
        
        # 电气参数计算
        self.rated_current_A = 420          # 额定电流 (项目文档)
        self.module_dc_voltage = self.system_voltage_kV * 1000 / self.cascaded_modules_per_phase  # 875V/模块
        
        # 计算单个IGBT的实际工作参数
        self.power_per_igbt_kW = (self.system_power_MW * 1000) / self.total_igbts  # 104.2kW
        
        print(f"真实系统架构参数:")
        print(f"  每相级联模块数: {self.cascaded_modules_per_phase}")
        print(f"  总模块数: {self.total_modules}")
        print(f"  总IGBT数: {self.total_igbts}")
        print(f"  单模块直流电压: {self.module_dc_voltage:.0f} V")
        print(f"  额定电流: {self.rated_current_A} A")
        print(f"  单个IGBT传输功率: {self.power_per_igbt_kW:.1f} kW")
        
        # IGBT物理参数 - Infineon FF1500R17IP5R
        self.igbt_params = {
            'Vce_sat_25C': 1.75,            # 25°C时饱和压降 (V)
            'Vce_sat_125C': 2.2,            # 125°C时饱和压降 (V)
            'Rce_25C': 1.1e-3,              # 25°C时导通电阻 (Ω)
            'temp_coeff_Vce': 0.004,        # 饱和压降温度系数 (V/K)
            'temp_coeff_Rce': 0.006,        # 导通电阻温度系数 (1/K)
            'Eon_1000A': 25e-3,             # 1000A时开通损耗 (J)
            'Eoff_1000A': 32e-3,            # 1000A时关断损耗 (J)
            'Qrr_diode': 85e-6,             # 二极管反向恢复电荷 (C)
            'Vf_diode': 1.5,                # 二极管正向压降 (V)
        }
        
        # 热网络参数 - 优化散热设计以满足额定工况要求
        self.thermal_params = {
            'Rth_jc': 0.04,                 # 结到壳热阻 (K/W) - 数据手册典型值
            'Rth_cs': 0.005,                # 壳到散热器热阻 (K/W) - 优质导热界面材料
            'Rth_sa': 0.025,                # 散热器到环境热阻 (K/W) - 强化水冷散热器
            'Cth_j': 800,                   # 结热容 (J/K)
            'Cth_c': 3000,                  # 壳热容 (J/K)
            'Cth_s': 15000,                 # 散热器热容 (J/K)
        }
        
        # 温度状态变量
        self.Tj = 25.0      # 结温
        self.Tc = 25.0      # 壳温
        self.Ts = 25.0      # 散热器温度
        self.temperature_history = []
        
        # 验证热设计
        total_Rth = sum(self.thermal_params[k] for k in ['Rth_jc', 'Rth_cs', 'Rth_sa'])
        print(f"\n热网络参数:")
        print(f"  结到壳热阻: {self.thermal_params['Rth_jc']} K/W")
        print(f"  壳到散热器热阻: {self.thermal_params['Rth_cs']} K/W")
        print(f"  散热器到环境热阻: {self.thermal_params['Rth_sa']} K/W")
        print(f"  总热阻: {total_Rth:.2f} K/W")
        print(f"  1kW损耗稳态温升: {total_Rth * 1000:.0f} K")

    def calculate_realistic_losses(self, current_rms: float, switching_freq: float = 1000, 
                                 junction_temp: float = 25.0, load_factor: float = 1.0):
        """
        基于真实物理计算IGBT损耗
        
        Args:
            current_rms: RMS电流 (A)
            switching_freq: 开关频率 (Hz)
            junction_temp: 结温 (°C)
            load_factor: 负载因子 (0-3.0)
        """
        # 温度相关参数修正
        temp_delta = junction_temp - 25.0
        
        # 饱和压降随温度变化
        Vce_sat = self.igbt_params['Vce_sat_25C'] + self.igbt_params['temp_coeff_Vce'] * temp_delta
        
        # 导通电阻随温度变化
        Rce = self.igbt_params['Rce_25C'] * (1 + self.igbt_params['temp_coeff_Rce'] * temp_delta)
        
        # IGBT导通损耗计算
        # 考虑PWM调制，IGBT导通时间占空比
        duty_cycle = 0.5  # 平均占空比
        current_avg = current_rms * duty_cycle * 2 / np.pi  # 平均电流
        
        # 导通损耗 = Vce_sat * I_avg + Rce * I_rms²
        P_cond_igbt = Vce_sat * current_avg + Rce * (current_rms ** 2)
        
        # 二极管导通损耗 (续流期间)
        Vf_diode = self.igbt_params['Vf_diode']
        current_avg_diode = current_rms * (1 - duty_cycle) * 2 / np.pi
        P_cond_diode = Vf_diode * current_avg_diode
        
        # 开关损耗计算
        # 基于数据手册，按电流线性缩放
        ref_current = 1000  # A (参考电流)
        current_factor = current_rms / ref_current
        
        Eon = self.igbt_params['Eon_1000A'] * current_factor
        Eoff = self.igbt_params['Eoff_1000A'] * current_factor
        
        # 开关损耗还要考虑电压和温度影响
        voltage_factor = self.module_dc_voltage / 1200  # 参考电压1200V
        temp_factor = 1 + 0.002 * temp_delta  # 开关损耗温度系数
        
        P_switching = (Eon + Eoff) * switching_freq * voltage_factor * temp_factor
        
        # 反向恢复损耗
        Qrr = self.igbt_params['Qrr_diode'] * current_factor
        P_reverse_recovery = 0.5 * Qrr * self.module_dc_voltage * switching_freq
        
        # 考虑负载变化的附加损耗
        if load_factor > 1.0:
            # 过载时电流密度增加，损耗非线性增加
            overload_factor = load_factor ** 1.5  # 非线性关系
            P_cond_igbt *= overload_factor
            P_cond_diode *= overload_factor
            P_switching *= overload_factor
        
        # 总损耗
        P_total = P_cond_igbt + P_cond_diode + P_switching + P_reverse_recovery
        
        return {
            'current_rms': current_rms,
            'junction_temp': junction_temp,
            'Vce_sat': Vce_sat,
            'P_cond_igbt': P_cond_igbt,
            'P_cond_diode': P_cond_diode,
            'P_switching': P_switching,
            'P_reverse_recovery': P_reverse_recovery,
            'P_total': P_total,
            'transmission_power_kW': self.power_per_igbt_kW * load_factor,
            'efficiency_percent': (self.power_per_igbt_kW * load_factor * 1000 - P_total) / (self.power_per_igbt_kW * load_factor * 1000) * 100
        }

    def update_temperature(self, power_loss: float, ambient_temp: float, dt: float = 60):
        """基于三阶RC网络更新温度"""
        # 热时间常数
        tau_jc = self.thermal_params['Rth_jc'] * self.thermal_params['Cth_j']
        tau_cs = self.thermal_params['Rth_cs'] * self.thermal_params['Cth_c']
        tau_sa = self.thermal_params['Rth_sa'] * self.thermal_params['Cth_s']
        
        # 自适应步长
        min_tau = min(tau_jc, tau_cs, tau_sa)
        internal_dt = min(dt, min_tau / 10)
        num_steps = max(1, int(dt / internal_dt))
        actual_dt = dt / num_steps
        
        for _ in range(num_steps):
            # 热流计算
            q_jc = (self.Tj - self.Tc) / self.thermal_params['Rth_jc']
            q_cs = (self.Tc - self.Ts) / self.thermal_params['Rth_cs']
            q_sa = (self.Ts - ambient_temp) / self.thermal_params['Rth_sa']
            
            # 温度变化率
            dTj_dt = (power_loss - q_jc) / self.thermal_params['Cth_j']
            dTc_dt = (q_jc - q_cs) / self.thermal_params['Cth_c']
            dTs_dt = (q_cs - q_sa) / self.thermal_params['Cth_s']
            
            # 温度更新
            self.Tj += dTj_dt * actual_dt
            self.Tc += dTc_dt * actual_dt
            self.Ts += dTs_dt * actual_dt
        
        self.temperature_history.append(self.Tj)
        return self.Tj, self.Tc, self.Ts

    def reset_state(self, initial_temp: float = 25.0):
        """重置状态"""
        self.Tj = initial_temp
        self.Tc = initial_temp
        self.Ts = initial_temp
        self.temperature_history = []

def test_realistic_scenarios():
    """测试真实工况场景"""
    print("\n" + "=" * 60)
    print("真实物理IGBT热模型测试")
    print("=" * 60)
    
    model = RealisticIGBTThermalModel()
    
    # 基于项目要求的真实工况
    scenarios = {
        'light_load': {
            'load_factor': 0.3,
            'current_rms': model.rated_current_A * 0.3,
            'description': '轻载工况 (30%负载)',
            'duration_hours': 24,
            'switching_freq': 750  # 中等开关频率
        },
        'rated_load': {
            'load_factor': 1.0,
            'current_rms': model.rated_current_A,
            'description': '额定工况 (100%负载)',
            'duration_hours': 24,
            'switching_freq': 1000  # 额定开关频率
        },
        'overload_3x': {
            'load_factor': 3.0,
            'current_rms': model.rated_current_A * 3.0,
            'description': '3倍过载 (10秒)',
            'duration_hours': 10/3600,  # 10秒
            'switching_freq': 500   # 过载时降低开关频率
        }
    }
    
    results = {}
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('真实物理IGBT热模型 - 35kV/25MW级联储能PCS', fontsize=16, fontweight='bold')
    
    for idx, (scenario_name, scenario) in enumerate(scenarios.items()):
        print(f"\n{scenario['description']}分析...")
        
        model.reset_state(25)  # 25°C冷却液温度
        
        # 时间设置
        if scenario['load_factor'] == 3.0:
            time_points = np.linspace(0, 10, 100)  # 10秒，0.1秒间隔
            time_label = '时间 (秒)'
            dt = 0.1
        else:
            time_points = np.linspace(0, 24, 24*6)  # 24小时，10分钟间隔
            time_label = '时间 (小时)'
            dt = 600  # 10分钟
        
        # 环境温度变化 (考虑20-40°C环境温度要求)
        if scenario['load_factor'] == 3.0:
            ambient_profile = np.full(len(time_points), 40.0)  # 过载时最高环境温度
        else:
            # 正常工况环境温度变化 (20-40°C)
            ambient_base = 30.0  # 平均30°C
            ambient_daily = 10 * np.sin(2 * np.pi * (time_points - 12) / 24)  # ±10°C日变化
            ambient_noise = 2 * np.random.normal(0, 1, len(time_points))  # ±2°C随机变化
            ambient_profile = ambient_base + ambient_daily + ambient_noise
            ambient_profile = np.clip(ambient_profile, 20, 40)  # 限制在要求范围内
        
        # 负载变化 (考虑储能系统充放电特性)
        if scenario['load_factor'] == 3.0:
            load_factors = np.full(len(time_points), 3.0)  # 过载恒定
        else:
            # 储能充放电循环 (至少一充一放)
            base_cycle = np.sin(2 * np.pi * time_points / 12)  # 12小时一个充放电周期
            load_variation = scenario['load_factor'] * (1 + 0.3 * base_cycle)  # ±30%变化
            # 添加电网调度的随机波动
            random_variation = 0.1 * np.random.normal(0, 1, len(time_points))
            load_factors = np.clip(load_variation + random_variation, 0.1, scenario['load_factor'] * 1.2)
        
        # 运行仿真
        temperatures = []
        power_losses = []
        efficiencies = []
        
        for i, (load_factor, ambient) in enumerate(zip(load_factors, ambient_profile)):
            current_actual = scenario['current_rms'] * (load_factor / scenario['load_factor'])
            
            # 考虑温度反馈的损耗计算
            losses = model.calculate_realistic_losses(
                current_rms=current_actual,
                switching_freq=scenario['switching_freq'],
                junction_temp=model.Tj,
                load_factor=load_factor
            )
            
            power_losses.append(losses['P_total'])
            efficiencies.append(losses['efficiency_percent'])
            
            # 更新温度
            Tj, Tc, Ts = model.update_temperature(losses['P_total'], ambient, dt)
            temperatures.append([Tj, Tc, Ts])
        
        temperatures = np.array(temperatures)
        
        # 分析结果
        temp_range = np.max(temperatures[:, 0]) - np.min(temperatures[:, 0])
        temp_std = np.std(temperatures[:, 0])
        avg_temp = np.mean(temperatures[:, 0])
        max_temp = np.max(temperatures[:, 0])
        avg_loss = np.mean(power_losses)
        avg_efficiency = np.mean(efficiencies)
        
        results[scenario_name] = {
            'load_factor': scenario['load_factor'],
            'current_rms': scenario['current_rms'],
            'avg_temp': avg_temp,
            'max_temp': max_temp,
            'temp_range': temp_range,
            'temp_std': temp_std,
            'avg_loss': avg_loss,
            'avg_efficiency': avg_efficiency,
            'switching_freq': scenario['switching_freq']
        }
        
        print(f"  电流: {scenario['current_rms']:.0f} A")
        print(f"  开关频率: {scenario['switching_freq']} Hz")
        print(f"  平均损耗: {avg_loss:.1f} W")
        print(f"  平均效率: {avg_efficiency:.2f}%")
        print(f"  平均结温: {avg_temp:.1f}°C")
        print(f"  最高结温: {max_temp:.1f}°C")
        print(f"  温度范围: {temp_range:.1f}K")
        
        # 温度评估
        if max_temp > 175:
            print(f"    ❌ 超过IGBT极限温度")
        elif max_temp > 150:
            print(f"    ⚠️ 温度较高，需要监控")
        elif max_temp > 100:
            print(f"    ✅ 温度在合理工作范围")
        else:
            print(f"    ⚠️ 温度可能偏低")
        
        # 绘制结果
        ax1 = axes[idx, 0]
        ax1.plot(time_points, temperatures[:, 0], 'r-', linewidth=2, label='结温')
        ax1.plot(time_points, temperatures[:, 1], 'b-', linewidth=1.5, label='壳温')
        ax1.plot(time_points, temperatures[:, 2], 'g-', linewidth=1.5, label='散热器温度')
        ax1.plot(time_points, ambient_profile, 'k--', linewidth=1, alpha=0.7, label='环境温度')
        
        ax1.axhline(y=175, color='red', linestyle=':', alpha=0.5, label='IGBT极限(175°C)')
        ax1.axhline(y=150, color='orange', linestyle=':', alpha=0.5, label='高温线(150°C)')
        
        ax1.set_xlabel(time_label)
        ax1.set_ylabel('温度 (°C)')
        ax1.set_title(f'{scenario["description"]}\n最高结温: {max_temp:.1f}°C, 平均损耗: {avg_loss:.0f}W')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[idx, 1]
        ax2.plot(time_points, power_losses, 'purple', linewidth=2, label='总损耗')
        if scenario['load_factor'] != 3.0:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(time_points, load_factors, 'orange', linewidth=1.5, alpha=0.7, label='负载因子')
            ax2_twin.set_ylabel('负载因子')
            ax2_twin.legend(loc='upper right')
        
        ax2.set_xlabel(time_label)
        ax2.set_ylabel('损耗功率 (W)')
        ax2.set_title(f'IGBT损耗变化 (效率: {avg_efficiency:.2f}%)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/真实物理IGBT热模型分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def analyze_lifetime_with_realistic_temps():
    """基于真实温度分析寿命"""
    print(f"\n" + "=" * 60)
    print("基于真实温度的寿命分析")
    print("=" * 60)
    
    results = test_realistic_scenarios()
    
    # 寿命计算 - 基于真实IGBT寿命模型
    for scenario_name, data in results.items():
        avg_temp = data['avg_temp']
        max_temp = data['max_temp']
        temp_range = data['temp_range']
        
        # Arrhenius模型 + Coffin-Manson模型
        # 基于125°C额定温度
        Ea = 0.7  # eV，激活能
        k = 8.617e-5  # eV/K，玻尔兹曼常数
        T_ref = 125 + 273.15  # 参考温度(K)
        T_avg = avg_temp + 273.15  # 平均温度(K)
        
        # 温度加速因子
        acceleration_factor = np.exp((Ea/k) * (1/T_ref - 1/T_avg))
        
        # 基础寿命 (125°C下10万小时)
        base_life_hours = 100000
        actual_life_hours = base_life_hours / acceleration_factor
        
        # 温度循环影响 (Coffin-Manson)
        if temp_range > 20:
            cycle_factor = (50.0 / temp_range) ** 2.0  # 温度循环越大，寿命越短
        else:
            cycle_factor = 1.0
        
        final_life_hours = actual_life_hours * cycle_factor
        life_years = final_life_hours / 8760
        
        print(f"\n{scenario_name.upper()}工况寿命分析:")
        print(f"  平均结温: {avg_temp:.1f}°C")
        print(f"  温度范围: {temp_range:.1f}K")
        print(f"  温度加速因子: {acceleration_factor:.1f}")
        print(f"  温度循环因子: {cycle_factor:.2f}")
        print(f"  预期寿命: {life_years:.1f} 年")
        
        data['life_years'] = life_years
    
    # 验证寿命排序
    light_life = results['light_load']['life_years']
    rated_life = results['rated_load']['life_years']
    overload_life = results['overload_3x']['life_years']
    
    print(f"\n寿命排序验证:")
    print(f"  轻载寿命: {light_life:.1f} 年")
    print(f"  额定寿命: {rated_life:.1f} 年")
    print(f"  过载寿命: {overload_life:.1f} 年")
    
    if light_life > rated_life > overload_life:
        print(f"  ✅ 寿命排序正确：轻载 > 额定 > 过载")
    else:
        print(f"  ⚠️ 寿命排序需要调整")
    
    return results

if __name__ == "__main__":
    np.random.seed(42)  # 确保可重复
    
    # 运行真实物理模型分析
    results = analyze_lifetime_with_realistic_temps()
    
    print(f"\n" + "=" * 60)
    print("真实物理IGBT热模型总结")
    print("=" * 60)
    print(f"架构修正:")
    print(f"  ✅ 每相40个级联模块 (总120个模块, 240个IGBT)")
    print(f"  ✅ 单IGBT传输功率: 104.2kW")
    print(f"  ✅ 基于FF1500R17IP5R真实参数")
    print(f"  ✅ 考虑温度反馈的损耗计算")
    print(f"  ✅ 三阶RC热网络模型")
    print(f"  ✅ 储能充放电工况建模")
    print(f"  ✅ 20-40°C环境温度变化")
    print(f"  ✅ 500-1000Hz开关频率范围")
    print("=" * 60)
