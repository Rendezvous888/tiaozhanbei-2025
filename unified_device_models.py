#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一设备建模接口
集成IGBT和母线电容的建模功能，提供统一的API和综合分析能力
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 导入优化的单独模型
from optimized_igbt_model import OptimizedIGBTModel, IGBTPhysicalParams
from optimized_capacitor_model import OptimizedCapacitorModel, CapacitorPhysicalParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class SystemConfiguration:
    """系统配置参数"""
    # 系统级参数
    rated_power_MW: float = 25.0          # 额定功率 (MW)
    rated_voltage_kV: float = 35.0        # 额定电压 (kV)
    modules_per_phase: int = 40           # 每相级联模块数
    phases: int = 3                       # 相数
    
    # 工作条件
    switching_frequency_Hz: float = 1000   # 开关频率 (Hz)
    modulation_index: float = 0.95        # 调制比
    power_factor: float = 1.0             # 功率因数
    ambient_temperature_C: float = 35     # 环境温度 (°C)
    
    # 电容配置
    capacitors_per_module: int = 21       # 每模块电容器数量
    capacitor_manufacturer: str = "Xiamen Farah"
    
    # 冷却系统
    cooling_efficiency: float = 0.85      # 冷却效率

class UnifiedDeviceModel:
    """统一设备建模类"""
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        """
        初始化统一设备模型
        
        Args:
            config: 系统配置参数
        """
        self.config = config or SystemConfiguration()
        
        # 创建子模型实例
        self.igbt_model = OptimizedIGBTModel()
        self.capacitor_model = OptimizedCapacitorModel(self.config.capacitor_manufacturer)
        
        # 计算系统级参数
        self._calculate_system_parameters()
        
        # 初始化状态变量
        self.system_temperature_history = []
        self.system_power_history = []
        self.system_efficiency_history = []
        
    def _calculate_system_parameters(self):
        """计算系统级参数"""
        # 模块级参数
        self.module_power_MW = self.config.rated_power_MW / (self.config.phases * self.config.modules_per_phase)
        self.module_voltage_V = self.config.rated_voltage_kV * 1000 / (np.sqrt(3) * self.config.modules_per_phase)
        self.module_current_A = self.module_power_MW * 1e6 / (self.module_voltage_V * np.sqrt(3))
        
        # 总器件数量
        self.total_igbts = self.config.phases * self.config.modules_per_phase * 6  # 每个H桥6个IGBT
        self.total_capacitors = self.config.phases * self.config.modules_per_phase * self.config.capacitors_per_module
        
        print(f"系统配置计算完成:")
        print(f"  模块功率: {self.module_power_MW:.3f} MW")
        print(f"  模块电压: {self.module_voltage_V:.0f} V")
        print(f"  模块电流: {self.module_current_A:.0f} A")
        print(f"  总IGBT数量: {self.total_igbts}")
        print(f"  总电容器数量: {self.total_capacitors}")
    
    def calculate_system_losses(self, operating_power_MW: float, 
                              operating_mode: str = 'normal') -> Dict[str, float]:
        """
        计算系统总损耗
        
        Args:
            operating_power_MW: 工作功率 (MW)
            operating_mode: 工作模式 ('normal', 'light', 'heavy')
            
        Returns:
            系统损耗字典 (W)
        """
        # 功率分配到模块
        power_per_module_MW = operating_power_MW / (self.config.phases * self.config.modules_per_phase)
        current_per_module_A = power_per_module_MW * 1e6 / (self.module_voltage_V * np.sqrt(3))
        
        # 根据工作模式调整参数
        if operating_mode == 'light':
            duty_cycle = 0.3
            temp_rise_factor = 0.8
        elif operating_mode == 'heavy':
            duty_cycle = 0.8
            temp_rise_factor = 1.2
        else:  # normal
            duty_cycle = 0.5
            temp_rise_factor = 1.0
        
        # IGBT损耗计算
        igbt_temp = self.config.ambient_temperature_C + 40 * temp_rise_factor  # 估算结温
        igbt_losses_per_module = self.igbt_model.calculate_power_losses(
            current_per_module_A, 
            self.module_voltage_V,
            self.config.switching_frequency_Hz,
            duty_cycle,
            igbt_temp
        )
        
        # 总IGBT损耗 (每个模块6个IGBT)
        total_igbt_losses = igbt_losses_per_module['total'] * 6 * self.config.phases * self.config.modules_per_phase
        
        # 电容器损耗计算
        ripple_current_A = current_per_module_A * 0.15  # 纹波电流约为模块电流的15%
        cap_temp = self.config.ambient_temperature_C + 15 * temp_rise_factor  # 估算电容器温度
        
        cap_losses_per_unit = self.capacitor_model.calculate_power_losses(
            ripple_current_A, 
            self.config.switching_frequency_Hz,
            cap_temp
        )
        
        # 总电容器损耗
        total_cap_losses = cap_losses_per_unit['total'] * self.total_capacitors
        
        # 辅助损耗（冷却系统、控制器等）
        auxiliary_losses = operating_power_MW * 1e6 * 0.005  # 0.5%的辅助损耗
        
        # 总损耗
        total_losses = total_igbt_losses + total_cap_losses + auxiliary_losses
        
        return {
            'total_losses_W': total_losses,
            'igbt_losses_W': total_igbt_losses,
            'capacitor_losses_W': total_cap_losses,
            'auxiliary_losses_W': auxiliary_losses,
            'igbt_losses_per_module_W': igbt_losses_per_module['total'],
            'cap_losses_per_unit_W': cap_losses_per_unit['total'],
            'efficiency_percent': (1 - total_losses / (operating_power_MW * 1e6)) * 100
        }
    
    def calculate_system_thermal_behavior(self, power_profile_MW: List[float], 
                                       time_hours: List[float],
                                       ambient_temp_profile_C: List[float] = None) -> Dict[str, List[float]]:
        """
        计算系统热行为
        
        Args:
            power_profile_MW: 功率曲线 (MW)
            time_hours: 时间序列 (小时)
            ambient_temp_profile_C: 环境温度曲线 (°C)
            
        Returns:
            热行为分析结果
        """
        if ambient_temp_profile_C is None:
            ambient_temp_profile_C = [self.config.ambient_temperature_C] * len(power_profile_MW)
        
        igbt_temps = []
        cap_temps = []
        system_losses = []
        efficiencies = []
        
        for i, (power, ambient_temp) in enumerate(zip(power_profile_MW, ambient_temp_profile_C)):
            # 计算当前功率点的损耗
            losses = self.calculate_system_losses(power)
            
            # 更新IGBT温度
            igbt_power_per_module = losses['igbt_losses_per_module_W']
            dt_s = 3600 if i > 0 else 0  # 1小时时间步长
            igbt_temp, _ = self.igbt_model.update_thermal_state(
                igbt_power_per_module, ambient_temp, dt_s
            )
            
            # 更新电容器温度
            cap_power_per_unit = losses['cap_losses_per_unit_W']
            cap_temp = self.capacitor_model.update_thermal_state(
                cap_power_per_unit, ambient_temp, dt_s
            )
            
            igbt_temps.append(igbt_temp)
            cap_temps.append(cap_temp)
            system_losses.append(losses['total_losses_W'])
            efficiencies.append(losses['efficiency_percent'])
        
        return {
            'time_hours': time_hours,
            'igbt_temperatures_C': igbt_temps,
            'capacitor_temperatures_C': cap_temps,
            'system_losses_W': system_losses,
            'system_efficiency_percent': efficiencies,
            'power_profile_MW': power_profile_MW,
            'ambient_temperature_C': ambient_temp_profile_C
        }
    
    def calculate_system_lifetime(self, operating_scenario: Dict, 
                                analysis_years: int = 10) -> Dict[str, pd.DataFrame]:
        """
        计算系统寿命
        
        Args:
            operating_scenario: 工作场景
            analysis_years: 分析年限
            
        Returns:
            系统寿命分析结果
        """
        # IGBT寿命分析
        igbt_operating_profile = {
            'load_factor': operating_scenario.get('load_factor', 0.7),
            'base_current_A': self.module_current_A * operating_scenario.get('load_factor', 0.7),
            'switching_freq_Hz': self.config.switching_frequency_Hz,
            'ambient_temp_C': operating_scenario.get('ambient_temp_C', self.config.ambient_temperature_C)
        }
        
        igbt_life_results = self.igbt_model.predict_lifetime(igbt_operating_profile, analysis_years)
        
        # 电容器寿命分析
        cap_operating_profile = {
            'ripple_current_A': self.module_current_A * 0.15 * operating_scenario.get('load_factor', 0.7),
            'applied_voltage_V': self.module_voltage_V * 0.9,  # 90%额定电压
            'ambient_temp_C': operating_scenario.get('ambient_temp_C', self.config.ambient_temperature_C),
            'frequency_Hz': self.config.switching_frequency_Hz
        }
        
        cap_life_results = self.capacitor_model.predict_aging_behavior(cap_operating_profile, analysis_years)
        
        # 系统级寿命分析（取最短寿命）
        system_life_results = []
        for year in range(1, analysis_years + 1):
            igbt_life = igbt_life_results.loc[igbt_life_results['year'] == year, 'remaining_life_percent'].iloc[0]
            cap_life = cap_life_results.loc[cap_life_results['year'] == year, 'remaining_life_percent'].iloc[0]
            
            # 系统寿命取决于最薄弱环节
            system_life = min(igbt_life, cap_life)
            
            system_life_results.append({
                'year': year,
                'system_remaining_life_percent': system_life,
                'igbt_remaining_life_percent': igbt_life,
                'capacitor_remaining_life_percent': cap_life,
                'limiting_component': 'IGBT' if igbt_life < cap_life else 'Capacitor'
            })
        
        system_life_df = pd.DataFrame(system_life_results)
        
        return {
            'system_life': system_life_df,
            'igbt_life': igbt_life_results,
            'capacitor_life': cap_life_results
        }
    
    def optimize_system_parameters(self, target_lifetime_years: int = 20,
                                 target_efficiency_percent: float = 97.0) -> Dict[str, float]:
        """
        优化系统参数
        
        Args:
            target_lifetime_years: 目标寿命 (年)
            target_efficiency_percent: 目标效率 (%)
            
        Returns:
            优化参数建议
        """
        from scipy.optimize import minimize
        
        def objective_function(params):
            """优化目标函数"""
            switching_freq, ambient_temp, load_factor = params
            
            # 更新临时配置
            temp_config = SystemConfiguration(
                switching_frequency_Hz=switching_freq,
                ambient_temperature_C=ambient_temp
            )
            
            # 创建临时模型
            temp_model = UnifiedDeviceModel(temp_config)
            
            # 计算寿命和效率
            scenario = {'load_factor': load_factor, 'ambient_temp_C': ambient_temp}
            life_results = temp_model.calculate_system_lifetime(scenario, target_lifetime_years)
            
            final_life = life_results['system_life'].iloc[-1]['system_remaining_life_percent']
            
            # 计算效率
            losses = temp_model.calculate_system_losses(25.0 * load_factor)  # 25MW系统
            efficiency = losses['efficiency_percent']
            
            # 目标函数：最小化与目标的偏差
            life_penalty = abs(final_life - 80) / 80  # 目标80%剩余寿命
            efficiency_penalty = abs(efficiency - target_efficiency_percent) / target_efficiency_percent
            
            return life_penalty + efficiency_penalty
        
        # 参数边界：[开关频率(Hz), 环境温度(°C), 负载因子]
        bounds = [(500, 2000), (25, 45), (0.5, 1.0)]
        initial_guess = [1000, 35, 0.8]
        
        # 优化
        result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_freq, optimal_temp, optimal_load = result.x
            
            # 计算优化结果的性能
            temp_config = SystemConfiguration(
                switching_frequency_Hz=optimal_freq,
                ambient_temperature_C=optimal_temp
            )
            temp_model = UnifiedDeviceModel(temp_config)
            
            scenario = {'load_factor': optimal_load, 'ambient_temp_C': optimal_temp}
            life_results = temp_model.calculate_system_lifetime(scenario, target_lifetime_years)
            losses = temp_model.calculate_system_losses(25.0 * optimal_load)
            
            return {
                'optimal_switching_frequency_Hz': optimal_freq,
                'optimal_ambient_temperature_C': optimal_temp,
                'optimal_load_factor': optimal_load,
                'predicted_lifetime_years': target_lifetime_years,
                'remaining_life_percent': life_results['system_life'].iloc[-1]['system_remaining_life_percent'],
                'predicted_efficiency_percent': losses['efficiency_percent'],
                'optimization_success': True
            }
        else:
            return {
                'optimization_success': False,
                'message': '优化失败，使用默认参数'
            }
    
    def generate_comprehensive_report(self, analysis_results: Dict) -> str:
        """生成综合分析报告"""
        report = []
        report.append("=" * 80)
        report.append("35kV/25MW级联储能PCS设备建模综合分析报告")
        report.append("=" * 80)
        
        # 系统配置
        report.append(f"\n系统配置:")
        report.append(f"  额定功率: {self.config.rated_power_MW} MW")
        report.append(f"  额定电压: {self.config.rated_voltage_kV} kV")
        report.append(f"  级联模块数: {self.config.modules_per_phase}/相")
        report.append(f"  开关频率: {self.config.switching_frequency_Hz} Hz")
        report.append(f"  环境温度: {self.config.ambient_temperature_C} °C")
        
        # 设备统计
        report.append(f"\n设备统计:")
        report.append(f"  总IGBT数量: {self.total_igbts}")
        report.append(f"  总电容器数量: {self.total_capacitors}")
        report.append(f"  IGBT型号: {self.igbt_model.params.model}")
        report.append(f"  电容器制造商: {self.capacitor_model.params.manufacturer}")
        
        # 性能分析
        if 'losses' in analysis_results:
            losses = analysis_results['losses']
            report.append(f"\n性能分析 (额定工况):")
            report.append(f"  系统效率: {losses['efficiency_percent']:.2f}%")
            report.append(f"  总损耗: {losses['total_losses_W']/1e3:.1f} kW")
            report.append(f"    IGBT损耗: {losses['igbt_losses_W']/1e3:.1f} kW")
            report.append(f"    电容器损耗: {losses['capacitor_losses_W']/1e3:.1f} kW")
            report.append(f"    辅助损耗: {losses['auxiliary_losses_W']/1e3:.1f} kW")
        
        # 寿命预测
        if 'lifetime' in analysis_results:
            lifetime = analysis_results['lifetime']
            system_life = lifetime['system_life']
            final_year = system_life.iloc[-1]
            
            report.append(f"\n寿命预测 (10年分析):")
            report.append(f"  系统剩余寿命: {final_year['system_remaining_life_percent']:.1f}%")
            report.append(f"  IGBT剩余寿命: {final_year['igbt_remaining_life_percent']:.1f}%")
            report.append(f"  电容器剩余寿命: {final_year['capacitor_remaining_life_percent']:.1f}%")
            report.append(f"  限制性器件: {final_year['limiting_component']}")
        
        # 优化建议
        if 'optimization' in analysis_results:
            opt = analysis_results['optimization']
            if opt['optimization_success']:
                report.append(f"\n优化建议:")
                report.append(f"  建议开关频率: {opt['optimal_switching_frequency_Hz']:.0f} Hz")
                report.append(f"  建议环境温度: {opt['optimal_ambient_temperature_C']:.1f} °C")
                report.append(f"  建议负载因子: {opt['optimal_load_factor']:.2f}")
                report.append(f"  预期效率: {opt['predicted_efficiency_percent']:.2f}%")
        
        # 维护建议
        report.append(f"\n维护建议:")
        if 'lifetime' in analysis_results:
            final_life = lifetime['system_life'].iloc[-1]['system_remaining_life_percent']
            if final_life > 80:
                report.append("  ✓ 系统状态良好，继续正常运行")
            elif final_life > 60:
                report.append("  ⚠ 建议加强监测，考虑预防性维护")
            else:
                report.append("  ⚠ 建议计划设备更换或大修")
        
        report.append("  • 定期检查冷却系统性能")
        report.append("  • 监测关键器件温度")
        report.append("  • 记录负载变化和环境条件")
        report.append("  • 建立预测性维护程序")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def plot_comprehensive_analysis(self, save_path: str = None):
        """绘制综合分析图表"""
        try:
            from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot
        except ImportError:
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle('35kV/25MW级联储能PCS设备综合分析', fontsize=16, fontweight='bold')
        else:
            fig, axes = create_adaptive_figure(3, 3, title='35kV/25MW级联储能PCS设备综合分析')
        
        # 1. 功率-效率特性
        power_range = np.linspace(5, 25, 20)
        efficiencies = []
        
        for power in power_range:
            losses = self.calculate_system_losses(power)
            efficiencies.append(losses['efficiency_percent'])
        
        axes[0, 0].plot(power_range, efficiencies, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('功率 (MW)')
        axes[0, 0].set_ylabel('效率 (%)')
        axes[0, 0].set_title('系统效率特性')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 损耗分布
        nominal_losses = self.calculate_system_losses(25.0)  # 额定功率损耗
        loss_labels = ['IGBT损耗', '电容器损耗', '辅助损耗']
        loss_values = [
            nominal_losses['igbt_losses_W'] / 1e3,
            nominal_losses['capacitor_losses_W'] / 1e3,
            nominal_losses['auxiliary_losses_W'] / 1e3
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        axes[0, 1].pie(loss_values, labels=loss_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('额定功率损耗分布')
        
        # 3. 热特性仿真 (24小时)
        time_24h = np.linspace(0, 24, 25)
        power_profile = 15 + 8 * np.sin(2 * np.pi * time_24h / 24)  # 日负载变化
        ambient_profile = 35 + 5 * np.sin(2 * np.pi * (time_24h - 6) / 24)  # 日温度变化
        
        thermal_results = self.calculate_system_thermal_behavior(power_profile, time_24h, ambient_profile)
        
        axes[0, 2].plot(time_24h, thermal_results['igbt_temperatures_C'], 'r-', label='IGBT温度', linewidth=2)
        axes[0, 2].plot(time_24h, thermal_results['capacitor_temperatures_C'], 'b-', label='电容器温度', linewidth=2)
        axes[0, 2].plot(time_24h, ambient_profile, 'g--', label='环境温度', linewidth=2)
        axes[0, 2].set_xlabel('时间 (小时)')
        axes[0, 2].set_ylabel('温度 (°C)')
        axes[0, 2].set_title('24小时热特性仿真')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 寿命预测对比
        scenarios = {
            '轻载': {'load_factor': 0.5, 'ambient_temp_C': 30},
            '正常': {'load_factor': 0.7, 'ambient_temp_C': 35},
            '重载': {'load_factor': 0.9, 'ambient_temp_C': 40}
        }
        
        for scenario_name, scenario in scenarios.items():
            life_results = self.calculate_system_lifetime(scenario, 10)
            system_life = life_results['system_life']
            axes[1, 0].plot(system_life['year'], system_life['system_remaining_life_percent'], 
                          'o-', linewidth=2, markersize=5, label=scenario_name)
        
        axes[1, 0].set_xlabel('运行年数')
        axes[1, 0].set_ylabel('剩余寿命 (%)')
        axes[1, 0].set_title('不同工况下的寿命预测')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. IGBT vs 电容器寿命对比
        normal_life = self.calculate_system_lifetime(scenarios['正常'], 10)
        axes[1, 1].plot(normal_life['igbt_life']['year'], normal_life['igbt_life']['remaining_life_percent'], 
                       'r-', linewidth=2, marker='s', markersize=5, label='IGBT')
        axes[1, 1].plot(normal_life['capacitor_life']['year'], normal_life['capacitor_life']['remaining_life_percent'], 
                       'b-', linewidth=2, marker='o', markersize=5, label='电容器')
        axes[1, 1].plot(normal_life['system_life']['year'], normal_life['system_life']['system_remaining_life_percent'], 
                       'k--', linewidth=2, marker='^', markersize=5, label='系统')
        
        axes[1, 1].set_xlabel('运行年数')
        axes[1, 1].set_ylabel('剩余寿命 (%)')
        axes[1, 1].set_title('器件寿命对比 (正常工况)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 温度分布统计
        temp_data = {
            'IGBT': thermal_results['igbt_temperatures_C'],
            '电容器': thermal_results['capacitor_temperatures_C']
        }
        
        for i, (component, temps) in enumerate(temp_data.items()):
            axes[1, 2].hist(temps, bins=10, alpha=0.7, label=component, density=True)
        
        axes[1, 2].set_xlabel('温度 (°C)')
        axes[1, 2].set_ylabel('概率密度')
        axes[1, 2].set_title('24小时温度分布')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 开关频率优化
        freq_range = np.linspace(500, 2000, 20)
        freq_efficiencies = []
        freq_igbt_temps = []
        
        for freq in freq_range:
            # 临时修改开关频率
            original_freq = self.config.switching_frequency_Hz
            self.config.switching_frequency_Hz = freq
            
            losses = self.calculate_system_losses(20.0)  # 20MW
            freq_efficiencies.append(losses['efficiency_percent'])
            
            # 简化的温度估算
            temp_rise = losses['igbt_losses_W'] / 1e6  # 简化计算
            freq_igbt_temps.append(self.config.ambient_temperature_C + temp_rise)
            
            # 恢复原始频率
            self.config.switching_frequency_Hz = original_freq
        
        ax_temp = axes[2, 0].twinx()
        line1 = axes[2, 0].plot(freq_range, freq_efficiencies, 'b-', linewidth=2, label='效率')
        line2 = ax_temp.plot(freq_range, freq_igbt_temps, 'r--', linewidth=2, label='IGBT温度')
        
        axes[2, 0].set_xlabel('开关频率 (Hz)')
        axes[2, 0].set_ylabel('效率 (%)', color='b')
        ax_temp.set_ylabel('温度 (°C)', color='r')
        axes[2, 0].set_title('开关频率优化')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[2, 0].legend(lines, labels, loc='upper right')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. 成本效益分析 (简化)
        years = np.arange(1, 11)
        maintenance_costs = years * 50000  # 年度维护成本 (元)
        efficiency_savings = (np.array(freq_efficiencies[:10]) - 95) * 2e6  # 效率提升节约 (元/年)
        cumulative_savings = np.cumsum(efficiency_savings) - maintenance_costs
        
        axes[2, 1].plot(years, cumulative_savings / 1e6, 'g-', linewidth=2, marker='o', markersize=5)
        axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[2, 1].set_xlabel('运行年数')
        axes[2, 1].set_ylabel('累积收益 (百万元)')
        axes[2, 1].set_title('系统经济效益分析')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. 可靠性雷达图
        reliability_metrics = {
            'IGBT寿命': normal_life['igbt_life'].iloc[-1]['remaining_life_percent'],
            '电容器寿命': normal_life['capacitor_life'].iloc[-1]['remaining_life_percent'],
            '系统效率': nominal_losses['efficiency_percent'],
            '热管理': 100 - max(thermal_results['igbt_temperatures_C']) / 1.75,  # 175°C为最大值
            '电气性能': 95,  # 假设值
            '维护性': 90    # 假设值
        }
        
        # 雷达图数据准备
        angles = np.linspace(0, 2*np.pi, len(reliability_metrics), endpoint=False)
        values = list(reliability_metrics.values())
        values += values[:1]  # 闭合图形
        angles = np.concatenate((angles, [angles[0]]))
        
        axes[2, 2].plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.8)
        axes[2, 2].fill(angles, values, alpha=0.25, color='blue')
        axes[2, 2].set_xticks(angles[:-1])
        axes[2, 2].set_xticklabels(list(reliability_metrics.keys()), fontsize=9)
        axes[2, 2].set_ylim(0, 100)
        axes[2, 2].set_title('系统可靠性评估')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # [[memory:6155470]] 保存各子图到pic文件夹
            for i, ax in enumerate(axes.flat):
                fig_single = plt.figure(figsize=(8, 6))
                ax_new = fig_single.add_subplot(111)
                
                # 复制子图内容（简化处理）
                if ax.get_title():  # 只有有标题的子图才保存
                    ax_new.set_title(ax.get_title())
                    ax_new.text(0.5, 0.5, f'子图 {i+1}\n{ax.get_title()}', 
                              ha='center', va='center', transform=ax_new.transAxes, fontsize=12)
                
                subplot_path = save_path.replace('.png', f'_subplot_{i+1}.png')
                plt.savefig(subplot_path, dpi=300, bbox_inches='tight')
                plt.close(fig_single)
        
        plt.show()

def run_comprehensive_analysis():
    """运行综合分析"""
    print("开始35kV/25MW级联储能PCS设备综合建模分析...")
    print("=" * 70)
    
    # 创建系统配置
    config = SystemConfiguration(
        rated_power_MW=25.0,
        rated_voltage_kV=35.0,
        modules_per_phase=40,
        switching_frequency_Hz=1000,
        ambient_temperature_C=35
    )
    
    # 创建统一模型
    unified_model = UnifiedDeviceModel(config)
    
    # 分析结果字典
    analysis_results = {}
    
    # 1. 性能分析
    print("\n1. 系统性能分析...")
    losses = unified_model.calculate_system_losses(25.0)  # 额定功率
    analysis_results['losses'] = losses
    
    print(f"  系统效率: {losses['efficiency_percent']:.2f}%")
    print(f"  总损耗: {losses['total_losses_W']/1e3:.1f} kW")
    
    # 2. 寿命预测
    print("\n2. 系统寿命预测...")
    scenario = {'load_factor': 0.7, 'ambient_temp_C': 35}
    lifetime_results = unified_model.calculate_system_lifetime(scenario, 10)
    analysis_results['lifetime'] = lifetime_results
    
    final_life = lifetime_results['system_life'].iloc[-1]['system_remaining_life_percent']
    print(f"  10年后系统剩余寿命: {final_life:.1f}%")
    
    # 3. 参数优化
    print("\n3. 系统参数优化...")
    optimization_results = unified_model.optimize_system_parameters(20, 97.0)
    analysis_results['optimization'] = optimization_results
    
    if optimization_results['optimization_success']:
        print(f"  优化开关频率: {optimization_results['optimal_switching_frequency_Hz']:.0f} Hz")
        print(f"  优化后效率: {optimization_results['predicted_efficiency_percent']:.2f}%")
    
    # 4. 生成报告
    print("\n4. 生成综合分析报告...")
    report = unified_model.generate_comprehensive_report(analysis_results)
    print(report)
    
    # 5. 绘制综合分析图表
    print("\n5. 生成综合分析图表...")
    unified_model.plot_comprehensive_analysis('pic/35kV25MW级联储能PCS设备综合分析.png')
    
    print("\n✓ 综合分析完成！")
    
    return unified_model, analysis_results

if __name__ == "__main__":
    run_comprehensive_analysis()
