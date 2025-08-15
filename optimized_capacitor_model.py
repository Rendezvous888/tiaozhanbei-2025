#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的母线电容建模模块
集成了多个源文件的功能，提供统一、高效的母线电容建模接口
基于Xiamen Farah和Nantong Jianghai薄膜电容器的完整特性建模
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class CapacitorPhysicalParams:
    """电容器物理参数数据类"""
    # 基本参数
    manufacturer: str = "Xiamen Farah"
    model: str = "DC-Link Film Capacitor"
    capacitance_uF: float = 720  # 电容值 (μF)
    rated_voltage_V: int = 1200  # 额定电压 (V)
    max_current_A: int = 80      # 最大纹波电流 (A)
    
    # 电气特性
    ESR_base_mOhm: float = 1.2   # 基准ESR (mΩ)
    ESL_nH: float = 45           # 等效串联电感 (nH)
    dielectric_loss: float = 0.0002  # 介电损耗
    breakdown_strength_V_per_um: float = 800  # 击穿强度
    
    # 热参数
    thermal_resistance_K_per_W: float = 0.5    # 热阻 (K/W)
    thermal_capacity_J_per_K: float = 2000     # 热容 (J/K)
    max_temperature_C: int = 85                # 最高工作温度 (°C)
    min_temperature_C: int = -40               # 最低工作温度 (°C)
    
    # 寿命参数
    base_lifetime_h: int = 100000              # 基准寿命 (小时)
    reference_temperature_C: int = 70          # 参考温度 (°C)
    activation_energy_eV: float = 0.12         # 激活能 (eV)
    voltage_stress_exponent: float = 2.5       # 电压应力指数
    current_stress_exponent: float = 1.8       # 电流应力指数
    
    # 材料参数
    dielectric_constant: float = 4.2           # 介电常数
    density_kg_per_m3: float = 1800           # 密度 (kg/m³)
    thermal_expansion_coeff: float = 1.5e-5   # 热膨胀系数 (1/K)

class OptimizedCapacitorModel:
    """优化的母线电容建模类"""
    
    def __init__(self, manufacturer: str = "Xiamen Farah", params: Optional[CapacitorPhysicalParams] = None):
        """
        初始化电容器模型
        
        Args:
            manufacturer: 制造商名称
            params: 电容器物理参数，默认使用标准参数
        """
        self.manufacturer = manufacturer
        self.params = params or CapacitorPhysicalParams(manufacturer=manufacturer)
        
        # 创建特性查找表和插值器
        self._create_characteristic_tables()
        self._create_interpolators()
        
        # 初始化状态变量
        self.case_temperature_C = 25.0
        self.ripple_current_rms_A = 0.0
        self.temperature_history = []
        self.stress_history = []
        
    def _create_characteristic_tables(self):
        """创建电容器特性查找表"""
        # 频率范围 (Hz)
        self.frequency_range_Hz = np.array([50, 100, 1000, 5000, 10000, 50000, 100000])
        
        # ESR频率特性（基于薄膜电容器典型特性）
        self.ESR_freq_factors = np.array([2.5, 2.0, 1.0, 0.8, 0.7, 0.6, 0.55])
        
        # 温度范围 (°C)
        self.temperature_range_C = np.array([-40, -20, 0, 25, 50, 70, 85])
        
        # 电容值温度系数（薄膜电容器）
        self.capacitance_temp_factors = np.array([0.92, 0.95, 0.98, 1.0, 0.99, 0.97, 0.94])
        
        # ESR温度系数
        self.ESR_temp_factors = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.15, 1.25])
        
        # 电压vs电容特性（非线性效应）
        self.voltage_ratio_range = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.1])
        self.capacitance_voltage_factors = np.array([1.02, 1.01, 1.005, 1.0, 0.998, 0.995])
        
        # 电流vs损耗特性
        self.current_ratio_range = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.loss_current_factors = np.array([1.0, 1.05, 1.12, 1.22, 1.35, 1.5])
    
    def _create_interpolators(self):
        """创建插值器"""
        # ESR频率特性插值器
        self.ESR_freq_interp = interp1d(
            self.frequency_range_Hz, self.ESR_freq_factors,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        # 电容值温度特性插值器
        self.cap_temp_interp = interp1d(
            self.temperature_range_C, self.capacitance_temp_factors,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        # ESR温度特性插值器
        self.ESR_temp_interp = interp1d(
            self.temperature_range_C, self.ESR_temp_factors,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        # 电容值电压特性插值器
        self.cap_voltage_interp = interp1d(
            self.voltage_ratio_range, self.capacitance_voltage_factors,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        # 损耗电流特性插值器
        self.loss_current_interp = interp1d(
            self.current_ratio_range, self.loss_current_factors,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
    
    def get_capacitance(self, temperature_C: float = None, voltage_ratio: float = 1.0) -> float:
        """
        获取电容值
        
        Args:
            temperature_C: 工作温度 (°C)
            voltage_ratio: 电压比（实际电压/额定电压）
            
        Returns:
            电容值 (F)
        """
        if temperature_C is None:
            temperature_C = self.case_temperature_C
            
        # 基准电容值
        C_base = self.params.capacitance_uF * 1e-6  # 转换为F
        
        # 温度补偿
        temp_factor = self.cap_temp_interp(temperature_C)
        
        # 电压补偿
        voltage_factor = self.cap_voltage_interp(np.clip(voltage_ratio, 0.1, 1.2))
        
        return C_base * temp_factor * voltage_factor
    
    def get_ESR(self, frequency_Hz: float = 1000, temperature_C: float = None) -> float:
        """
        获取等效串联电阻
        
        Args:
            frequency_Hz: 工作频率 (Hz)
            temperature_C: 工作温度 (°C)
            
        Returns:
            ESR (Ω)
        """
        if temperature_C is None:
            temperature_C = self.case_temperature_C
            
        # 基准ESR
        ESR_base = self.params.ESR_base_mOhm * 1e-3  # 转换为Ω
        
        # 频率补偿
        freq_factor = self.ESR_freq_interp(frequency_Hz)
        
        # 温度补偿
        temp_factor = self.ESR_temp_interp(temperature_C)
        
        return ESR_base * freq_factor * temp_factor
    
    def get_ESL(self) -> float:
        """
        获取等效串联电感
        
        Returns:
            ESL (H)
        """
        return self.params.ESL_nH * 1e-9
    
    def calculate_impedance(self, frequency_Hz: float, temperature_C: float = None) -> complex:
        """
        计算电容器阻抗
        
        Args:
            frequency_Hz: 频率 (Hz)
            temperature_C: 温度 (°C)
            
        Returns:
            复阻抗 (Ω)
        """
        C = self.get_capacitance(temperature_C)
        ESR = self.get_ESR(frequency_Hz, temperature_C)
        ESL = self.get_ESL()
        
        omega = 2 * np.pi * frequency_Hz
        
        # Z = ESR + j(ωL - 1/(ωC))
        Z_reactive = omega * ESL - 1 / (omega * C)
        Z = complex(ESR, Z_reactive)
        
        return Z
    
    def calculate_ripple_current_capability(self, frequency_Hz: float, 
                                         temperature_C: float = None) -> float:
        """
        计算纹波电流承载能力
        
        Args:
            frequency_Hz: 纹波频率 (Hz)
            temperature_C: 工作温度 (°C)
            
        Returns:
            最大纹波电流 (Arms)
        """
        if temperature_C is None:
            temperature_C = self.case_temperature_C
            
        # 基准纹波电流（在参考频率和温度下）
        I_base = self.params.max_current_A
        
        # 频率补偿：高频时纹波电流降低
        if frequency_Hz <= 1000:
            freq_factor = 1.0
        elif frequency_Hz <= 10000:
            freq_factor = 1.0 - 0.1 * np.log10(frequency_Hz / 1000)
        else:
            freq_factor = 0.8
        
        # 温度降额
        temp_factor = 1.0 - max(0, (temperature_C - 70) / (85 - 70)) * 0.3
        
        return I_base * freq_factor * temp_factor
    
    def calculate_power_losses(self, ripple_current_rms_A: float, 
                             frequency_Hz: float = 1000,
                             temperature_C: float = None) -> Dict[str, float]:
        """
        计算功率损耗
        
        Args:
            ripple_current_rms_A: 纹波电流RMS值 (A)
            frequency_Hz: 纹波频率 (Hz)
            temperature_C: 工作温度 (°C)
            
        Returns:
            功率损耗字典 (W)
        """
        if temperature_C is None:
            temperature_C = self.case_temperature_C
            
        # ESR损耗
        ESR = self.get_ESR(frequency_Hz, temperature_C)
        P_ESR = ripple_current_rms_A ** 2 * ESR
        
        # 介电损耗
        C = self.get_capacitance(temperature_C)
        omega = 2 * np.pi * frequency_Hz
        voltage_rms = ripple_current_rms_A * abs(self.calculate_impedance(frequency_Hz, temperature_C))
        P_dielectric = omega * C * voltage_rms ** 2 * self.params.dielectric_loss
        
        # 电流依赖的损耗增加
        current_ratio = ripple_current_rms_A / self.params.max_current_A
        loss_factor = self.loss_current_interp(np.clip(current_ratio, 0, 1))
        
        P_total = (P_ESR + P_dielectric) * loss_factor
        
        return {
            'total': P_total,
            'ESR_loss': P_ESR * loss_factor,
            'dielectric_loss': P_dielectric * loss_factor,
            'loss_factor': loss_factor
        }
    
    def update_thermal_state(self, power_loss_W: float, ambient_temp_C: float = 25,
                           dt_s: float = 1.0) -> float:
        """
        更新热状态
        
        Args:
            power_loss_W: 功率损耗 (W)
            ambient_temp_C: 环境温度 (°C)
            dt_s: 时间步长 (s)
            
        Returns:
            外壳温度 (°C)
        """
        # 热网络模型
        Rth = self.params.thermal_resistance_K_per_W
        Cth = self.params.thermal_capacity_J_per_K
        
        # 稳态温升
        temp_rise_ss = power_loss_W * Rth
        target_temp = ambient_temp_C + temp_rise_ss
        
        # 一阶热响应
        tau = Rth * Cth
        alpha = dt_s / tau
        self.case_temperature_C += alpha * (target_temp - self.case_temperature_C)
        
        # 温度限制
        self.case_temperature_C = np.clip(
            self.case_temperature_C,
            self.params.min_temperature_C,
            self.params.max_temperature_C + 10  # 允许轻微超温
        )
        
        # 记录温度历史
        self.temperature_history.append(self.case_temperature_C)
        
        return self.case_temperature_C
    
    def calculate_voltage_stress(self, applied_voltage_V: float) -> Dict[str, float]:
        """
        计算电压应力
        
        Args:
            applied_voltage_V: 施加电压 (V)
            
        Returns:
            电压应力分析结果
        """
        voltage_ratio = applied_voltage_V / self.params.rated_voltage_V
        
        # 电场强度（简化计算）
        # 假设介质厚度与额定电压成正比
        dielectric_thickness_um = self.params.rated_voltage_V / self.params.breakdown_strength_V_per_um
        electric_field_V_per_um = applied_voltage_V / dielectric_thickness_um
        
        # 安全裕度
        safety_margin = self.params.breakdown_strength_V_per_um / electric_field_V_per_um
        
        # 应力水平评估
        if voltage_ratio <= 0.8:
            stress_level = "低"
        elif voltage_ratio <= 1.0:
            stress_level = "正常"
        elif voltage_ratio <= 1.1:
            stress_level = "偏高"
        else:
            stress_level = "危险"
        
        return {
            'voltage_ratio': voltage_ratio,
            'electric_field_V_per_um': electric_field_V_per_um,
            'safety_margin': safety_margin,
            'stress_level': stress_level
        }
    
    def calculate_lifetime(self, operating_conditions: Dict, operating_hours: float = 8760) -> Dict[str, float]:
        """
        计算电容器寿命
        
        Args:
            operating_conditions: 工作条件字典
            operating_hours: 运行小时数
            
        Returns:
            寿命分析结果
        """
        # 获取工作条件
        ripple_current_A = operating_conditions.get('ripple_current_A', 20)
        applied_voltage_V = operating_conditions.get('applied_voltage_V', 1000)
        ambient_temp_C = operating_conditions.get('ambient_temp_C', 40)
        frequency_Hz = operating_conditions.get('frequency_Hz', 1000)
        
        # 计算功率损耗和温升
        losses = self.calculate_power_losses(ripple_current_A, frequency_Hz)
        case_temp = ambient_temp_C + losses['total'] * self.params.thermal_resistance_K_per_W
        
        # Arrhenius温度加速模型
        k_boltzmann = 8.617e-5  # eV/K
        T_ref_K = self.params.reference_temperature_C + 273
        T_case_K = case_temp + 273
        
        temp_acceleration = np.exp(
            self.params.activation_energy_eV / k_boltzmann * (1/T_case_K - 1/T_ref_K)
        )
        
        # 电压应力模型
        voltage_ratio = applied_voltage_V / self.params.rated_voltage_V
        voltage_stress = (1.0 / voltage_ratio) ** self.params.voltage_stress_exponent
        
        # 电流应力模型
        current_ratio = ripple_current_A / self.params.max_current_A
        current_stress = (1.0 / current_ratio) ** self.params.current_stress_exponent if current_ratio > 0 else 1.0
        
        # 综合寿命计算
        lifetime_hours = (self.params.base_lifetime_h * 
                         temp_acceleration * 
                         voltage_stress * 
                         current_stress)
        
        # 寿命消耗
        life_consumption = operating_hours / lifetime_hours
        remaining_life = max(0, 1 - life_consumption)
        
        return {
            'lifetime_hours': lifetime_hours,
            'remaining_life': remaining_life,
            'life_consumption': life_consumption,
            'case_temperature_C': case_temp,
            'temp_acceleration': temp_acceleration,
            'voltage_stress': voltage_stress,
            'current_stress': current_stress,
            'power_loss_W': losses['total']
        }
    
    def optimize_operating_point(self, target_lifetime_years: float = 20,
                               constraints: Dict = None) -> Dict[str, float]:
        """
        优化工作点以满足寿命要求
        
        Args:
            target_lifetime_years: 目标寿命 (年)
            constraints: 约束条件
            
        Returns:
            优化的工作点
        """
        if constraints is None:
            constraints = {
                'max_ripple_current_A': self.params.max_current_A * 0.8,
                'max_voltage_V': self.params.rated_voltage_V * 1.0,
                'max_ambient_temp_C': 50
            }
        
        target_lifetime_hours = target_lifetime_years * 8760
        
        def objective(voltage_ratio):
            """优化目标函数"""
            operating_conditions = {
                'ripple_current_A': constraints['max_ripple_current_A'],
                'applied_voltage_V': voltage_ratio * self.params.rated_voltage_V,
                'ambient_temp_C': constraints['max_ambient_temp_C'],
                'frequency_Hz': 1000
            }
            
            result = self.calculate_lifetime(operating_conditions)
            # 最小化与目标寿命的差异
            return abs(result['lifetime_hours'] - target_lifetime_hours)
        
        # 优化电压比
        result = minimize_scalar(objective, bounds=(0.5, 1.0), method='bounded')
        
        optimal_voltage = result.x * self.params.rated_voltage_V
        
        # 返回优化结果
        optimal_conditions = {
            'ripple_current_A': constraints['max_ripple_current_A'],
            'applied_voltage_V': optimal_voltage,
            'ambient_temp_C': constraints['max_ambient_temp_C'],
            'frequency_Hz': 1000
        }
        
        life_result = self.calculate_lifetime(optimal_conditions)
        
        return {
            'optimal_voltage_V': optimal_voltage,
            'voltage_ratio': optimal_voltage / self.params.rated_voltage_V,
            'expected_lifetime_years': life_result['lifetime_hours'] / 8760,
            'case_temperature_C': life_result['case_temperature_C'],
            'power_loss_W': life_result['power_loss_W']
        }
    
    def predict_aging_behavior(self, operating_profile: Dict, years: int = 10) -> pd.DataFrame:
        """
        预测老化行为
        
        Args:
            operating_profile: 工作曲线
            years: 预测年数
            
        Returns:
            老化预测结果DataFrame
        """
        results = []
        cumulative_damage = 0.0
        
        for year in range(1, years + 1):
            # 年度参数变化（考虑老化效应）
            capacitance_degradation = 1 - year * 0.005  # 每年下降0.5%
            ESR_increase = 1 + year * 0.02  # 每年增加2%
            
            # 工作条件
            base_conditions = {
                'ripple_current_A': operating_profile.get('ripple_current_A', 40),
                'applied_voltage_V': operating_profile.get('applied_voltage_V', 1000),
                'ambient_temp_C': operating_profile.get('ambient_temp_C', 40) + year * 0.2,  # 环境温度年度变化
                'frequency_Hz': operating_profile.get('frequency_Hz', 1000)
            }
            
            # 考虑老化的工作条件
            aged_conditions = base_conditions.copy()
            aged_conditions['ripple_current_A'] *= 1.02 ** year  # 电流随负载增长
            
            # 计算年度寿命消耗
            life_result = self.calculate_lifetime(aged_conditions, 8760)
            
            # 累积损伤
            annual_damage = life_result['life_consumption']
            cumulative_damage += annual_damage
            remaining_life = max(0, 1 - cumulative_damage)
            
            # 性能参数变化
            current_capacitance = self.params.capacitance_uF * capacitance_degradation
            current_ESR = self.params.ESR_base_mOhm * ESR_increase
            
            results.append({
                'year': year,
                'remaining_life_percent': remaining_life * 100,
                'cumulative_damage_percent': cumulative_damage * 100,
                'annual_damage_percent': annual_damage * 100,
                'capacitance_uF': current_capacitance,
                'ESR_mOhm': current_ESR,
                'case_temperature_C': life_result['case_temperature_C'],
                'power_loss_W': life_result['power_loss_W'],
                'ripple_current_A': aged_conditions['ripple_current_A']
            })
        
        return pd.DataFrame(results)
    
    def plot_characteristics(self, save_path: str = None):
        """绘制电容器特性曲线"""
        try:
            from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot
        except ImportError:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('厦门法拉母线电容模型特性分析', fontsize=14, fontweight='bold')
        else:
            fig, axes = create_adaptive_figure(2, 3, title='厦门法拉母线电容模型特性分析')
        
        # ESR频率特性
        freq_range = np.logspace(2, 5, 100)
        temps = [25, 50, 75]
        
        for temp in temps:
            ESR_values = [self.get_ESR(f, temp) * 1e3 for f in freq_range]  # 转换为mΩ
            axes[0, 0].semilogx(freq_range, ESR_values, label=f'T={temp}°C', linewidth=2)
        
        axes[0, 0].set_xlabel('频率 (Hz)')
        axes[0, 0].set_ylabel('ESR (mΩ)')
        axes[0, 0].set_title('ESR频率特性')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 电容值温度特性
        temp_range = np.linspace(-40, 85, 100)
        cap_values = [self.get_capacitance(t) * 1e6 for t in temp_range]  # 转换为μF
        
        axes[0, 1].plot(temp_range, cap_values, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('温度 (°C)')
        axes[0, 1].set_ylabel('电容值 (μF)')
        axes[0, 1].set_title('电容值温度特性')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 阻抗频率特性
        impedance_values = [abs(self.calculate_impedance(f)) * 1e3 for f in freq_range]  # 转换为mΩ
        
        axes[0, 2].loglog(freq_range, impedance_values, 'r-', linewidth=2)
        axes[0, 2].set_xlabel('频率 (Hz)')
        axes[0, 2].set_ylabel('阻抗幅值 (mΩ)')
        axes[0, 2].set_title('阻抗频率特性')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 功率损耗vs纹波电流
        current_range = np.linspace(5, 80, 50)
        frequencies = [1000, 5000, 10000]
        
        for freq in frequencies:
            loss_values = []
            for current in current_range:
                losses = self.calculate_power_losses(current, freq)
                loss_values.append(losses['total'])
            axes[1, 0].plot(current_range, loss_values, label=f'f={freq}Hz', linewidth=2)
        
        axes[1, 0].set_xlabel('纹波电流 (Arms)')
        axes[1, 0].set_ylabel('功率损耗 (W)')
        axes[1, 0].set_title('功率损耗vs纹波电流')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 寿命vs工作条件
        voltage_ratios = np.linspace(0.6, 1.0, 20)
        current_levels = [30, 50, 70]  # A
        
        for current in current_levels:
            lifetimes = []
            for v_ratio in voltage_ratios:
                conditions = {
                    'ripple_current_A': current,
                    'applied_voltage_V': v_ratio * self.params.rated_voltage_V,
                    'ambient_temp_C': 40,
                    'frequency_Hz': 1000
                }
                result = self.calculate_lifetime(conditions)
                lifetimes.append(result['lifetime_hours'] / 8760)  # 转换为年
            axes[1, 1].plot(voltage_ratios, lifetimes, label=f'I={current}A', linewidth=2)
        
        axes[1, 1].set_xlabel('电压比')
        axes[1, 1].set_ylabel('寿命 (年)')
        axes[1, 1].set_title('寿命vs工作电压')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 老化行为预测
        operating_profiles = {
            'Light': {'ripple_current_A': 30, 'applied_voltage_V': 1000, 'ambient_temp_C': 35},
            'Medium': {'ripple_current_A': 50, 'applied_voltage_V': 1100, 'ambient_temp_C': 40},
            'Heavy': {'ripple_current_A': 70, 'applied_voltage_V': 1150, 'ambient_temp_C': 45}
        }
        
        for profile_name, profile in operating_profiles.items():
            results = self.predict_aging_behavior(profile, years=10)
            axes[1, 2].plot(results['year'], results['remaining_life_percent'], 
                          'o-', linewidth=2, markersize=5, label=profile_name)
        
        axes[1, 2].set_xlabel('运行年数')
        axes[1, 2].set_ylabel('剩余寿命 (%)')
        axes[1, 2].set_title('不同负载下的老化预测')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # [[memory:6155470]] 保存各子图到pic文件夹
            for i, ax in enumerate(axes.flat):
                fig_single = plt.figure(figsize=(8, 6))
                ax_new = fig_single.add_subplot(111)
                
                # 复制子图内容
                for line in ax.get_lines():
                    ax_new.plot(line.get_xdata(), line.get_ydata(), 
                              label=line.get_label(), linewidth=line.get_linewidth(), 
                              color=line.get_color(), marker=line.get_marker())
                
                ax_new.set_xlabel(ax.get_xlabel())
                ax_new.set_ylabel(ax.get_ylabel())
                ax_new.set_title(ax.get_title())
                if ax.get_legend():
                    ax_new.legend()
                ax_new.grid(True, alpha=0.3)
                
                subplot_path = save_path.replace('.png', f'_subplot_{i+1}.png')
                plt.savefig(subplot_path, dpi=300, bbox_inches='tight')
                plt.close(fig_single)
        
        plt.show()

def test_optimized_capacitor_model():
    """测试优化的电容器模型"""
    print("=" * 60)
    print("测试优化的母线电容模型")
    print("=" * 60)
    
    # 创建模型实例
    cap_xiamen = OptimizedCapacitorModel("Xiamen Farah")
    cap_jianghai = OptimizedCapacitorModel("Nantong Jianghai")
    
    print(f"\n厦门法拉电容器参数:")
    print(f"  制造商: {cap_xiamen.params.manufacturer}")
    print(f"  电容值: {cap_xiamen.params.capacitance_uF} μF")
    print(f"  额定电压: {cap_xiamen.params.rated_voltage_V} V")
    print(f"  最大电流: {cap_xiamen.params.max_current_A} A")
    print(f"  基准ESR: {cap_xiamen.params.ESR_base_mOhm} mΩ")
    
    # 基本特性测试
    test_freq = 1000  # Hz
    test_temp = 50   # °C
    test_current = 40  # A
    
    print(f"\n基本特性测试 (f={test_freq}Hz, T={test_temp}°C, I={test_current}A):")
    print("-" * 60)
    
    # 电容值
    capacitance = cap_xiamen.get_capacitance(test_temp)
    print(f"电容值: {capacitance*1e6:.1f} μF")
    
    # ESR
    ESR = cap_xiamen.get_ESR(test_freq, test_temp)
    print(f"ESR: {ESR*1e3:.2f} mΩ")
    
    # 阻抗
    impedance = cap_xiamen.calculate_impedance(test_freq, test_temp)
    print(f"阻抗: {abs(impedance)*1e3:.2f} mΩ ∠{np.angle(impedance, deg=True):.1f}°")
    
    # 功率损耗
    losses = cap_xiamen.calculate_power_losses(test_current, test_freq, test_temp)
    print(f"功率损耗: {losses['total']:.2f} W")
    print(f"  ESR损耗: {losses['ESR_loss']:.2f} W")
    print(f"  介电损耗: {losses['dielectric_loss']:.3f} W")
    
    # 寿命分析
    print(f"\n寿命分析:")
    print("-" * 40)
    
    operating_conditions = {
        'ripple_current_A': 50,
        'applied_voltage_V': 1100,
        'ambient_temp_C': 40,
        'frequency_Hz': 1000
    }
    
    life_result = cap_xiamen.calculate_lifetime(operating_conditions)
    print(f"工况: 50A纹波电流, 1100V工作电压, 40°C环境温度")
    print(f"预期寿命: {life_result['lifetime_hours']/8760:.1f} 年")
    print(f"外壳温度: {life_result['case_temperature_C']:.1f} °C")
    print(f"功率损耗: {life_result['power_loss_W']:.2f} W")
    
    # 工作点优化
    print(f"\n工作点优化 (目标寿命: 20年):")
    print("-" * 40)
    
    optimal = cap_xiamen.optimize_operating_point(20)
    print(f"最优工作电压: {optimal['optimal_voltage_V']:.0f} V")
    print(f"电压比: {optimal['voltage_ratio']:.3f}")
    print(f"预期寿命: {optimal['expected_lifetime_years']:.1f} 年")
    print(f"外壳温度: {optimal['case_temperature_C']:.1f} °C")
    
    # 老化预测
    print(f"\n老化预测 (中等负载):")
    print("-" * 40)
    
    aging_profile = {
        'ripple_current_A': 45,
        'applied_voltage_V': 1050,
        'ambient_temp_C': 35,
        'frequency_Hz': 1000
    }
    
    aging_results = cap_xiamen.predict_aging_behavior(aging_profile, years=5)
    print(f"5年后剩余寿命: {aging_results.iloc[-1]['remaining_life_percent']:.1f}%")
    print(f"5年累积损伤: {aging_results.iloc[-1]['cumulative_damage_percent']:.2f}%")
    print(f"电容值退化: {aging_results.iloc[-1]['capacitance_uF']:.1f} μF")
    print(f"ESR增长: {aging_results.iloc[-1]['ESR_mOhm']:.2f} mΩ")
    
    # 绘制特性曲线
    print(f"\n正在生成特性曲线...")
    cap_xiamen.plot_characteristics('pic/优化母线电容模型特性分析.png')
    
    print(f"\n✓ 优化母线电容模型测试完成！")
    
    return cap_xiamen

if __name__ == "__main__":
    test_optimized_capacitor_model()
