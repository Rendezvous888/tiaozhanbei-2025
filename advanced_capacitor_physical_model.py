#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级母线电容物理建模模块
基于真实物理特性的高精度电容器建模，更接近实际物理世界的表现
包含多层介质模型、非线性特性、寄生参数、热动力学和材料物理特性

主要改进:
1. 多层介质电容器物理模型
2. 频率相关的复阻抗特性
3. 非线性电容和ESR特性
4. 真实的热动力学建模
5. 材料特性和应力耦合
6. 高精度老化和失效机制建模
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import minimize_scalar, curve_fit
from scipy.integrate import solve_ivp
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class AdvancedCapacitorPhysics:
    """高级电容器物理参数"""
    # 基本参数
    manufacturer: str = "Xiamen Farah"
    model: str = "DC-Link Film Capacitor"
    technology: str = "Metallized Polypropylene"  # 金属化聚丙烯
    capacitance_uF: float = 720
    rated_voltage_V: int = 1200
    max_current_A: int = 80
    
    # 介质物理特性
    dielectric_layers: int = 4                    # 介质层数
    dielectric_thickness_um: float = 8.5          # 单层厚度 (μm)
    electrode_thickness_nm: float = 30            # 电极厚度 (nm)
    dielectric_constant: float = 2.2             # 相对介电常数
    dielectric_loss_tan_delta: float = 0.0002    # 损耗角正切
    breakdown_field_strength_MV_per_m: float = 650  # 击穿场强 (MV/m)
    
    # 材料物理特性
    pp_density_kg_per_m3: float = 910            # 聚丙烯密度
    pp_specific_heat_J_per_kgK: float = 2000     # 比热容
    pp_thermal_conductivity_W_per_mK: float = 0.22  # 导热系数
    pp_thermal_expansion_coeff: float = 1.5e-4   # 热膨胀系数
    pp_youngs_modulus_Pa: float = 1.5e9          # 杨氏模量
    pp_poisson_ratio: float = 0.4               # 泊松比
    
    # 金属电极特性
    metal_resistivity_ohm_m: float = 2.8e-8      # 铝电阻率
    metal_thermal_expansion: float = 2.3e-5      # 铝热膨胀系数
    metal_thermal_conductivity: float = 237      # 铝导热系数
    
    # 几何结构
    winding_diameter_mm: float = 110             # 卷绕直径
    winding_height_mm: float = 170               # 高度
    effective_area_m2: float = 0.25              # 有效面积
    winding_length_m: float = 1500               # 总卷绕长度
    
    # 热参数 (修正为多层热网络)
    thermal_resistance_jc_K_per_W: float = 0.3   # 芯子到外壳
    thermal_resistance_ca_K_per_W: float = 0.8   # 外壳到环境
    thermal_capacity_core_J_per_K: float = 1800  # 芯子热容
    thermal_capacity_case_J_per_K: float = 3500  # 外壳热容
    
    # 寄生参数
    ESL_series_nH: float = 25                    # 串联电感
    ESL_parallel_nH: float = 150                 # 并联电感
    mutual_inductance_nH: float = 10             # 互感
    internal_resistance_mohm: float = 0.8        # 内阻
    
    # 老化参数
    base_lifetime_h: int = 120000                # 基准寿命
    reference_temperature_C: int = 70            # 参考温度
    activation_energy_eV: float = 0.15           # 激活能
    voltage_stress_exponent: float = 2.8         # 电压应力指数
    thermal_stress_exponent: float = 1.2         # 热应力指数
    mechanical_stress_factor: float = 0.05       # 机械应力因子

class AdvancedCapacitorPhysicalModel:
    """高级电容器物理建模类"""
    
    def __init__(self, params: Optional[AdvancedCapacitorPhysics] = None):
        """
        初始化高级电容器物理模型
        
        Args:
            params: 高级电容器物理参数
        """
        self.params = params or AdvancedCapacitorPhysics()
        
        # 物理常数
        self.k_boltzmann = 8.617e-5  # eV/K
        self.epsilon_0 = 8.854e-12   # F/m
        
        # 创建物理模型
        self._create_dielectric_model()
        self._create_frequency_model()
        self._create_thermal_network()
        self._create_mechanical_model()
        self._create_aging_model()
        
        # 状态变量
        self.core_temperature_C = 25.0
        self.case_temperature_C = 25.0
        self.dielectric_stress_MPa = 0.0
        self.accumulated_damage = 0.0
        self.capacitance_drift_percent = 0.0
        self.ESR_drift_percent = 0.0
        
        # 历史记录
        self.temperature_history = []
        self.stress_history = []
        self.damage_history = []
        
    def _create_dielectric_model(self):
        """创建多层介质模型"""
        # 频率相关的介电常数 (Debye模型)
        self.debye_params = {
            'epsilon_s': self.params.dielectric_constant,     # 静态介电常数
            'epsilon_inf': self.params.dielectric_constant * 0.9,  # 高频介电常数
            'tau_s': 1e-6,                                    # 弛豫时间 (s)
        }
        
        # 非线性介电特性 (场强依赖)
        self.nonlinear_dielectric = {
            'field_points_MV_per_m': np.array([0, 50, 100, 200, 400, 600]),
            'epsilon_factors': np.array([1.0, 0.998, 0.995, 0.988, 0.975, 0.960])
        }
        
        # 多层模型参数
        self.layer_model = {
            'num_layers': self.params.dielectric_layers,
            'layer_thickness': self.params.dielectric_thickness_um * 1e-6,
            'total_thickness': self.params.dielectric_layers * self.params.dielectric_thickness_um * 1e-6
        }
        
    def _create_frequency_model(self):
        """创建频率相关模型"""
        # 频率范围 (Hz)
        self.freq_range = np.logspace(1, 6, 100)
        
        # ESR频率特性 (基于实际测量数据)
        self.esr_freq_model = {
            'frequencies_Hz': np.array([50, 100, 1000, 10000, 50000, 100000]),
            'esr_factors': np.array([3.5, 2.8, 1.0, 0.6, 0.4, 0.35]),
            'skin_depth_factor': True  # 考虑趋肤效应
        }
        
        # 复阻抗模型参数
        self.impedance_model = {
            'R0': self.params.internal_resistance_mohm * 1e-3,  # 直流电阻
            'L_series': self.params.ESL_series_nH * 1e-9,       # 串联电感
            'R_parallel': 1e6,                                  # 并联电阻 (泄漏)
            'C_parallel': 50e-12                                # 并联电容 (杂散)
        }
        
        # 谐振特性
        self.resonance_model = {
            'fundamental_freq_Hz': None,  # 将在初始化时计算
            'Q_factor': 50,               # 品质因数
            'harmonic_factors': [1, 3, 5, 7, 9]  # 谐波分量
        }
        
        # 计算基本谐振频率
        L_total = self.impedance_model['L_series']
        C_nominal = self.params.capacitance_uF * 1e-6
        self.resonance_model['fundamental_freq_Hz'] = 1 / (2 * np.pi * np.sqrt(L_total * C_nominal))
        
    def _create_thermal_network(self):
        """创建多节点热网络模型"""
        # 三节点热网络: 内芯 - 外壳 - 环境
        self.thermal_network = {
            # 热阻矩阵 (K/W)
            'Rth_matrix': np.array([
                [self.params.thermal_resistance_jc_K_per_W],    # 内芯-外壳
                [self.params.thermal_resistance_ca_K_per_W]     # 外壳-环境
            ]),
            
            # 热容矩阵 (J/K)
            'Cth_matrix': np.array([
                self.params.thermal_capacity_core_J_per_K,      # 内芯热容
                self.params.thermal_capacity_case_J_per_K       # 外壳热容
            ]),
            
            # 对流换热系数 (W/m²K)
            'h_conv': 25.0,  # 自然对流
            'surface_area': 0.15  # 换热面积 (m²)
        }
        
        # 温度相关的材料特性
        self.thermal_properties = {
            'temperature_points_C': np.array([-40, 0, 25, 50, 70, 85, 100]),
            'thermal_conductivity_factors': np.array([1.15, 1.08, 1.0, 0.92, 0.88, 0.85, 0.82]),
            'specific_heat_factors': np.array([0.85, 0.92, 1.0, 1.05, 1.08, 1.10, 1.12])
        }
        
    def _create_mechanical_model(self):
        """创建机械应力模型"""
        # 热膨胀应力
        self.mechanical_model = {
            'thermal_expansion': {
                'pp_coeff': self.params.pp_thermal_expansion_coeff,
                'metal_coeff': self.params.metal_thermal_expansion,
                'reference_temp_C': 25.0
            },
            
            # 电致伸缩应力
            'electrostriction': {
                'coeff_m2_per_V2': 1e-18,  # 电致伸缩系数
                'max_field_V_per_m': self.params.breakdown_field_strength_MV_per_m * 1e6
            },
            
            # 机械特性
            'mechanical_properties': {
                'youngs_modulus': self.params.pp_youngs_modulus_Pa,
                'poisson_ratio': self.params.pp_poisson_ratio,
                'yield_strength_Pa': 30e6,  # 屈服强度
                'fatigue_limit_Pa': 15e6    # 疲劳极限
            }
        }
        
    def _create_aging_model(self):
        """创建高级老化模型"""
        # 多物理场耦合老化模型
        self.aging_model = {
            # Arrhenius温度模型
            'thermal_aging': {
                'activation_energy': self.params.activation_energy_eV,
                'pre_exponential': 1e10,
                'reference_temp_K': self.params.reference_temperature_C + 273.15
            },
            
            # Eyring电压应力模型
            'voltage_aging': {
                'stress_exponent': self.params.voltage_stress_exponent,
                'characteristic_field': 400e6,  # V/m
                'threshold_field': 100e6        # V/m
            },
            
            # 机械疲劳模型
            'mechanical_aging': {
                'paris_law_C': 2e-12,          # Paris定律常数
                'paris_law_m': 3.0,            # Paris定律指数
                'stress_intensity_factor': 1.2 # 应力强度因子
            },
            
            # 多应力耦合因子
            'coupling_factors': {
                'thermal_voltage': 1.2,        # 热-电耦合
                'thermal_mechanical': 1.5,     # 热-机械耦合
                'voltage_mechanical': 1.1      # 电-机械耦合
            }
        }
        
    def get_complex_permittivity(self, frequency_Hz: float, temperature_C: float = None, 
                               electric_field_V_per_m: float = 0) -> complex:
        """
        获取复介电常数
        
        Args:
            frequency_Hz: 频率 (Hz)
            temperature_C: 温度 (°C)
            electric_field_V_per_m: 电场强度 (V/m)
            
        Returns:
            复介电常数
        """
        if temperature_C is None:
            temperature_C = self.core_temperature_C
            
        # Debye弛豫模型
        omega = 2 * np.pi * frequency_Hz
        tau = self.debye_params['tau_s'] * np.exp(0.1 * (temperature_C - 25) / 25)  # 温度相关
        
        epsilon_s = self.debye_params['epsilon_s']
        epsilon_inf = self.debye_params['epsilon_inf']
        
        # 频率相关的介电常数
        epsilon_real = epsilon_inf + (epsilon_s - epsilon_inf) / (1 + (omega * tau)**2)
        epsilon_imag = (epsilon_s - epsilon_inf) * omega * tau / (1 + (omega * tau)**2)
        
        # 电场强度影响 (非线性)
        if electric_field_V_per_m > 0:
            field_factor = np.interp(
                electric_field_V_per_m / 1e6,
                self.nonlinear_dielectric['field_points_MV_per_m'],
                self.nonlinear_dielectric['epsilon_factors']
            )
            epsilon_real *= field_factor
        
        # 温度影响
        temp_factor = 1 - 0.0004 * (temperature_C - 25)
        epsilon_real *= temp_factor
        
        # 损耗角修正
        tan_delta = self.params.dielectric_loss_tan_delta * (1 + 0.002 * (temperature_C - 25))
        epsilon_imag += epsilon_real * tan_delta
        
        return complex(epsilon_real, epsilon_imag)
    
    def get_capacitance(self, frequency_Hz: float = 1000, temperature_C: float = None,
                       voltage_V: float = 0) -> float:
        """
        获取频率和温度相关的电容值
        
        Args:
            frequency_Hz: 频率 (Hz)
            temperature_C: 温度 (°C)
            voltage_V: 施加电压 (V)
            
        Returns:
            电容值 (F)
        """
        if temperature_C is None:
            temperature_C = self.core_temperature_C
            
        # 计算电场强度
        total_thickness = self.layer_model['total_thickness']
        electric_field = voltage_V / total_thickness if total_thickness > 0 else 0
        
        # 获取复介电常数
        epsilon_complex = self.get_complex_permittivity(frequency_Hz, temperature_C, electric_field)
        epsilon_r = epsilon_complex.real
        
        # 基本电容计算 (平行板电容器)
        epsilon_0 = self.epsilon_0
        A = self.params.effective_area_m2
        d = total_thickness
        
        C_base = epsilon_0 * epsilon_r * A / d
        
        # 多层修正
        layer_factor = 1 + 0.02 * (self.params.dielectric_layers - 1)  # 层间耦合效应
        C_corrected = C_base * layer_factor
        
        # 老化影响
        aging_factor = 1 - self.capacitance_drift_percent / 100
        
        return C_corrected * aging_factor
    
    def get_complex_impedance(self, frequency_Hz: float, temperature_C: float = None,
                            voltage_V: float = 0) -> complex:
        """
        获取复阻抗
        
        Args:
            frequency_Hz: 频率 (Hz)
            temperature_C: 温度 (°C)
            voltage_V: 施加电压 (V)
            
        Returns:
            复阻抗 (Ω)
        """
        if temperature_C is None:
            temperature_C = self.core_temperature_C
            
        omega = 2 * np.pi * frequency_Hz
        
        # 电容阻抗
        C = self.get_capacitance(frequency_Hz, temperature_C, voltage_V)
        Z_C = -1j / (omega * C)
        
        # ESR (频率和温度相关)
        ESR_base = self.params.internal_resistance_mohm * 1e-3
        
        # 频率影响 (趋肤效应)
        freq_factor = np.interp(
            frequency_Hz,
            self.esr_freq_model['frequencies_Hz'],
            self.esr_freq_model['esr_factors']
        )
        
        # 温度影响
        temp_factor = 1 + 0.004 * (temperature_C - 25)
        
        # 老化影响
        aging_factor = 1 + self.ESR_drift_percent / 100
        
        ESR = ESR_base * freq_factor * temp_factor * aging_factor
        
        # 串联电感
        L_series = self.impedance_model['L_series']
        Z_L = 1j * omega * L_series
        
        # 寄生并联阻抗
        R_parallel = self.impedance_model['R_parallel']
        C_parallel = self.impedance_model['C_parallel']
        Z_parallel = 1 / (1/R_parallel + 1j*omega*C_parallel)
        
        # 串联部分
        Z_series = ESR + Z_L + Z_C
        
        # 总阻抗 (串联部分与并联部分并联)
        Z_total = 1 / (1/Z_series + 1/Z_parallel)
        
        return Z_total
    
    def calculate_power_losses(self, current_rms_A: float, frequency_Hz: float = 1000,
                             temperature_C: float = None, voltage_V: float = 0) -> Dict[str, float]:
        """
        计算功率损耗 (包含多种损耗机制)
        
        Args:
            current_rms_A: RMS电流 (A)
            frequency_Hz: 频率 (Hz)
            temperature_C: 温度 (°C)
            voltage_V: 施加电压 (V)
            
        Returns:
            损耗分析结果
        """
        if temperature_C is None:
            temperature_C = self.core_temperature_C
            
        # 1. ESR损耗
        Z_complex = self.get_complex_impedance(frequency_Hz, temperature_C, voltage_V)
        ESR = Z_complex.real
        P_ESR = current_rms_A**2 * ESR
        
        # 2. 介电损耗
        C = self.get_capacitance(frequency_Hz, temperature_C, voltage_V)
        epsilon_complex = self.get_complex_permittivity(frequency_Hz, temperature_C)
        tan_delta = epsilon_complex.imag / epsilon_complex.real
        
        voltage_rms = abs(Z_complex) * current_rms_A
        P_dielectric = 2 * np.pi * frequency_Hz * C * voltage_rms**2 * tan_delta
        
        # 3. 涡流损耗 (金属电极中)
        # 简化模型：基于频率和电流
        P_eddy = 1e-6 * frequency_Hz * current_rms_A**2 * (frequency_Hz / 1000)**0.5
        
        # 4. 磁滞损耗 (非常小，可忽略)
        P_hysteresis = 1e-8 * frequency_Hz * current_rms_A**2
        
        # 5. 泄漏损耗
        R_leakage = self.impedance_model['R_parallel']
        P_leakage = voltage_rms**2 / R_leakage
        
        # 总损耗
        P_total = P_ESR + P_dielectric + P_eddy + P_hysteresis + P_leakage
        
        return {
            'total_W': P_total,
            'ESR_loss_W': P_ESR,
            'dielectric_loss_W': P_dielectric,
            'eddy_loss_W': P_eddy,
            'hysteresis_loss_W': P_hysteresis,
            'leakage_loss_W': P_leakage,
            'loss_breakdown_percent': {
                'ESR': P_ESR / P_total * 100,
                'dielectric': P_dielectric / P_total * 100,
                'eddy': P_eddy / P_total * 100,
                'other': (P_hysteresis + P_leakage) / P_total * 100
            }
        }
    
    def update_thermal_state(self, power_loss_W: float, ambient_temp_C: float = 25,
                           dt_s: float = 1.0) -> Tuple[float, float]:
        """
        更新热状态 (多节点热网络)
        
        Args:
            power_loss_W: 功率损耗 (W)
            ambient_temp_C: 环境温度 (°C)
            dt_s: 时间步长 (s)
            
        Returns:
            (内芯温度, 外壳温度) (°C)
        """
        # 热网络参数
        Rth_jc = self.thermal_network['Rth_matrix'][0, 0]
        Rth_ca = self.thermal_network['Rth_matrix'][1, 0]
        Cth_core = self.thermal_network['Cth_matrix'][0]
        Cth_case = self.thermal_network['Cth_matrix'][1]
        
        # 当前温度
        T_core = self.core_temperature_C
        T_case = self.case_temperature_C
        T_amb = ambient_temp_C
        
        # 热流计算
        Q_core_to_case = (T_core - T_case) / Rth_jc
        Q_case_to_amb = (T_case - T_amb) / Rth_ca
        
        # 对流换热修正
        h_conv = self.thermal_network['h_conv']
        A_surface = self.thermal_network['surface_area']
        Q_conv = h_conv * A_surface * (T_case - T_amb)
        
        # 温度变化率
        dT_core_dt = (power_loss_W - Q_core_to_case) / Cth_core
        dT_case_dt = (Q_core_to_case - Q_case_to_amb - Q_conv) / Cth_case
        
        # 更新温度
        self.core_temperature_C += dT_core_dt * dt_s
        self.case_temperature_C += dT_case_dt * dt_s
        
        # 温度限制
        self.core_temperature_C = np.clip(self.core_temperature_C, -40, 150)
        self.case_temperature_C = np.clip(self.case_temperature_C, -40, 120)
        
        # 记录历史
        self.temperature_history.append({
            'time_s': len(self.temperature_history) * dt_s,
            'core_temp_C': self.core_temperature_C,
            'case_temp_C': self.case_temperature_C,
            'power_loss_W': power_loss_W
        })
        
        return self.core_temperature_C, self.case_temperature_C
    
    def calculate_mechanical_stress(self, temperature_C: float = None, 
                                  voltage_V: float = 0) -> Dict[str, float]:
        """
        计算机械应力
        
        Args:
            temperature_C: 温度 (°C)
            voltage_V: 施加电压 (V)
            
        Returns:
            机械应力分析结果
        """
        if temperature_C is None:
            temperature_C = self.core_temperature_C
            
        # 热应力
        delta_T = temperature_C - self.mechanical_model['thermal_expansion']['reference_temp_C']
        alpha_pp = self.mechanical_model['thermal_expansion']['pp_coeff']
        alpha_metal = self.mechanical_model['thermal_expansion']['metal_coeff']
        
        # 差异热膨胀应力
        delta_alpha = alpha_metal - alpha_pp
        E = self.mechanical_model['mechanical_properties']['youngs_modulus']
        thermal_stress_Pa = E * delta_alpha * delta_T
        
        # 电致伸缩应力
        electric_field = voltage_V / (self.layer_model['total_thickness'])
        electrostrictive_coeff = self.mechanical_model['electrostriction']['coeff_m2_per_V2']
        electrostrictive_stress_Pa = electrostrictive_coeff * electric_field**2 * E
        
        # 内部压力应力 (卷绕张力)
        winding_stress_Pa = 5e6  # 估算值 (Pa)
        
        # 总应力
        total_stress_Pa = abs(thermal_stress_Pa) + abs(electrostrictive_stress_Pa) + winding_stress_Pa
        
        # 应力水平评估
        yield_strength = self.mechanical_model['mechanical_properties']['yield_strength_Pa']
        stress_ratio = total_stress_Pa / yield_strength
        
        if stress_ratio < 0.3:
            stress_level = "低"
        elif stress_ratio < 0.6:
            stress_level = "中"
        elif stress_ratio < 0.8:
            stress_level = "高"
        else:
            stress_level = "危险"
        
        self.dielectric_stress_MPa = total_stress_Pa / 1e6
        
        return {
            'total_stress_MPa': total_stress_Pa / 1e6,
            'thermal_stress_MPa': thermal_stress_Pa / 1e6,
            'electrostrictive_stress_MPa': electrostrictive_stress_Pa / 1e6,
            'winding_stress_MPa': winding_stress_Pa / 1e6,
            'stress_ratio': stress_ratio,
            'stress_level': stress_level,
            'safety_margin': (1 - stress_ratio) * 100
        }
    
    def calculate_lifetime_consumption(self, operating_conditions: Dict, 
                                     time_hours: float = 8760) -> Dict[str, float]:
        """
        计算寿命消耗 (多物理场耦合)
        
        Args:
            operating_conditions: 工作条件
            time_hours: 运行时间 (小时)
            
        Returns:
            寿命分析结果
        """
        # 提取工作条件
        temp_C = operating_conditions.get('temperature_C', 40)
        voltage_V = operating_conditions.get('voltage_V', 1000)
        current_A = operating_conditions.get('current_A', 40)
        frequency_Hz = operating_conditions.get('frequency_Hz', 1000)
        
        # 1. 热老化 (Arrhenius模型)
        T_K = temp_C + 273.15
        T_ref_K = self.aging_model['thermal_aging']['reference_temp_K']
        Ea = self.aging_model['thermal_aging']['activation_energy']
        
        thermal_acceleration = np.exp(
            Ea / self.k_boltzmann * (1/T_K - 1/T_ref_K)
        )
        
        # 2. 电压老化 (Eyring模型)
        electric_field = voltage_V / self.layer_model['total_thickness']
        field_threshold = self.aging_model['voltage_aging']['threshold_field']
        field_char = self.aging_model['voltage_aging']['characteristic_field']
        stress_exp = self.aging_model['voltage_aging']['stress_exponent']
        
        if electric_field > field_threshold:
            voltage_acceleration = (field_char / electric_field)**stress_exp
        else:
            voltage_acceleration = 1.0
        
        # 3. 机械疲劳老化
        stress_analysis = self.calculate_mechanical_stress(temp_C, voltage_V)
        stress_amplitude_MPa = stress_analysis['total_stress_MPa']
        
        # Paris定律参数
        C_paris = self.aging_model['mechanical_aging']['paris_law_C']
        m_paris = self.aging_model['mechanical_aging']['paris_law_m']
        
        # 疲劳损伤率
        if stress_amplitude_MPa > 5:  # 疲劳门槛值
            fatigue_damage_rate = C_paris * (stress_amplitude_MPa)**m_paris
        else:
            fatigue_damage_rate = 0
        
        # 4. 多应力耦合
        coupling = self.aging_model['coupling_factors']
        
        # 热-电耦合
        thermal_voltage_coupling = coupling['thermal_voltage'] if temp_C > 60 and voltage_V > 1000 else 1.0
        
        # 热-机械耦合
        thermal_mechanical_coupling = coupling['thermal_mechanical'] if temp_C > 70 and stress_amplitude_MPa > 10 else 1.0
        
        # 综合加速因子
        total_acceleration = (1 / thermal_acceleration) * (1 / voltage_acceleration) * \
                           thermal_voltage_coupling * thermal_mechanical_coupling
        
        # 基准寿命
        base_lifetime_h = self.params.base_lifetime_h
        
        # 实际寿命
        actual_lifetime_h = base_lifetime_h * total_acceleration
        
        # 寿命消耗
        life_consumption = time_hours / actual_lifetime_h
        
        # 疲劳累积损伤
        fatigue_damage = fatigue_damage_rate * time_hours
        
        # 总损伤
        total_damage = life_consumption + fatigue_damage
        self.accumulated_damage += total_damage
        
        # 剩余寿命
        remaining_life = max(0, 1 - self.accumulated_damage)
        
        return {
            'lifetime_hours': actual_lifetime_h,
            'life_consumption': life_consumption,
            'fatigue_damage': fatigue_damage,
            'total_damage': total_damage,
            'accumulated_damage': self.accumulated_damage,
            'remaining_life_percent': remaining_life * 100,
            'thermal_acceleration': thermal_acceleration,
            'voltage_acceleration': voltage_acceleration,
            'coupling_factor': thermal_voltage_coupling * thermal_mechanical_coupling,
            'stress_level_MPa': stress_amplitude_MPa
        }
    
    def predict_parameter_drift(self, years: int = 10, 
                              operating_profile: Dict = None) -> pd.DataFrame:
        """
        预测参数漂移
        
        Args:
            years: 预测年数
            operating_profile: 工作曲线
            
        Returns:
            参数漂移预测结果
        """
        if operating_profile is None:
            operating_profile = {
                'temperature_C': 50,
                'voltage_V': 1100,
                'current_A': 50,
                'frequency_Hz': 1000
            }
        
        results = []
        
        for year in range(1, years + 1):
            # 累积老化时间
            accumulated_hours = year * 8760
            
            # 计算寿命消耗
            lifetime_analysis = self.calculate_lifetime_consumption(operating_profile, 8760)
            
            # 电容值漂移 (通常下降)
            cap_drift_rate = 0.5  # %/年
            thermal_factor = max(1, operating_profile['temperature_C'] / 70)
            voltage_factor = max(1, operating_profile['voltage_V'] / 1200)
            
            capacitance_drift = -cap_drift_rate * year * thermal_factor * voltage_factor
            
            # ESR漂移 (通常上升)
            esr_drift_rate = 2.0  # %/年
            ESR_drift = esr_drift_rate * year * thermal_factor
            
            # 损耗角漂移
            tan_delta_drift_rate = 5.0  # %/年
            tan_delta_drift = tan_delta_drift_rate * year
            
            # 绝缘电阻下降
            insulation_resistance_base = 1e12  # Ω
            ir_degradation_factor = np.exp(-year * 0.1)
            insulation_resistance = insulation_resistance_base * ir_degradation_factor
            
            results.append({
                'year': year,
                'accumulated_hours': accumulated_hours,
                'capacitance_drift_percent': capacitance_drift,
                'ESR_drift_percent': ESR_drift,
                'tan_delta_drift_percent': tan_delta_drift,
                'insulation_resistance_ohm': insulation_resistance,
                'remaining_life_percent': lifetime_analysis['remaining_life_percent'],
                'thermal_stress_MPa': lifetime_analysis['stress_level_MPa'],
                'damage_accumulation': lifetime_analysis['accumulated_damage']
            })
        
        return pd.DataFrame(results)

def test_advanced_capacitor_model():
    """测试高级电容器物理模型"""
    print("=" * 70)
    print("测试高级母线电容物理模型")
    print("=" * 70)
    
    # 创建模型实例
    model = AdvancedCapacitorPhysicalModel()
    
    print(f"\n模型基本信息:")
    print(f"  制造商: {model.params.manufacturer}")
    print(f"  技术: {model.params.technology}")
    print(f"  额定电容: {model.params.capacitance_uF} μF")
    print(f"  额定电压: {model.params.rated_voltage_V} V")
    print(f"  介质层数: {model.params.dielectric_layers}")
    print(f"  总厚度: {model.layer_model['total_thickness']*1e6:.1f} μm")
    
    # 基本特性测试
    test_freq = 1000  # Hz
    test_temp = 60   # °C
    test_voltage = 1100  # V
    test_current = 50  # A
    
    print(f"\n基本特性测试 (f={test_freq}Hz, T={test_temp}°C, V={test_voltage}V):")
    print("-" * 60)
    
    # 复介电常数
    epsilon_complex = model.get_complex_permittivity(test_freq, test_temp)
    print(f"复介电常数: {epsilon_complex.real:.3f} - j{abs(epsilon_complex.imag):.6f}")
    print(f"损耗角: {np.arctan(epsilon_complex.imag/epsilon_complex.real)*180/np.pi:.4f}°")
    
    # 电容值
    capacitance = model.get_capacitance(test_freq, test_temp, test_voltage)
    print(f"电容值: {capacitance*1e6:.1f} μF")
    
    # 复阻抗
    impedance = model.get_complex_impedance(test_freq, test_temp, test_voltage)
    print(f"复阻抗: {abs(impedance)*1e3:.2f} mΩ ∠{np.angle(impedance, deg=True):.1f}°")
    print(f"ESR: {impedance.real*1e3:.2f} mΩ")
    
    # 功率损耗
    losses = model.calculate_power_losses(test_current, test_freq, test_temp, test_voltage)
    print(f"\n功率损耗分析:")
    print(f"  总损耗: {losses['total_W']:.3f} W")
    print(f"  ESR损耗: {losses['ESR_loss_W']:.3f} W ({losses['loss_breakdown_percent']['ESR']:.1f}%)")
    print(f"  介电损耗: {losses['dielectric_loss_W']:.3f} W ({losses['loss_breakdown_percent']['dielectric']:.1f}%)")
    print(f"  涡流损耗: {losses['eddy_loss_W']:.3f} W ({losses['loss_breakdown_percent']['eddy']:.1f}%)")
    
    # 机械应力分析
    stress_analysis = model.calculate_mechanical_stress(test_temp, test_voltage)
    print(f"\n机械应力分析:")
    print(f"  总应力: {stress_analysis['total_stress_MPa']:.2f} MPa")
    print(f"  热应力: {stress_analysis['thermal_stress_MPa']:.2f} MPa")
    print(f"  电致伸缩应力: {stress_analysis['electrostrictive_stress_MPa']:.2f} MPa")
    print(f"  应力水平: {stress_analysis['stress_level']}")
    print(f"  安全裕度: {stress_analysis['safety_margin']:.1f}%")
    
    # 寿命分析
    operating_conditions = {
        'temperature_C': test_temp,
        'voltage_V': test_voltage,
        'current_A': test_current,
        'frequency_Hz': test_freq
    }
    
    lifetime_analysis = model.calculate_lifetime_consumption(operating_conditions, 8760)
    print(f"\n寿命分析 (年度):")
    print(f"  预期寿命: {lifetime_analysis['lifetime_hours']/8760:.1f} 年")
    print(f"  年度消耗: {lifetime_analysis['life_consumption']*100:.3f}%")
    print(f"  疲劳损伤: {lifetime_analysis['fatigue_damage']*100:.3f}%")
    print(f"  剩余寿命: {lifetime_analysis['remaining_life_percent']:.1f}%")
    
    # 参数漂移预测
    drift_prediction = model.predict_parameter_drift(10, operating_conditions)
    print(f"\n10年参数漂移预测:")
    final_year = drift_prediction.iloc[-1]
    print(f"  电容值漂移: {final_year['capacitance_drift_percent']:.2f}%")
    print(f"  ESR漂移: {final_year['ESR_drift_percent']:.2f}%")
    print(f"  损耗角漂移: {final_year['tan_delta_drift_percent']:.2f}%")
    print(f"  绝缘电阻: {final_year['insulation_resistance_ohm']:.2e} Ω")
    
    print(f"\n✓ 高级电容器物理模型测试完成！")
    
    return model

if __name__ == "__main__":
    test_advanced_capacitor_model()
