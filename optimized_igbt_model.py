#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的IGBT建模模块
集成了多个源文件的功能，提供统一、高效的IGBT建模接口
基于Infineon FF1500R17IP5R的完整特性建模
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.signal import find_peaks
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class IGBTPhysicalParams:
    """IGBT物理参数数据类"""
    # 基本参数
    model: str = "Infineon FF1500R17IP5R"
    rated_voltage_V: int = 1700
    rated_current_A: int = 1500
    max_current_A: int = 3000
    junction_temp_range_C: Tuple[int, int] = (-40, 200)  # 放宽到200°C
    
    # 电气特性
    Vce_sat_25C: Tuple[float, float] = (1.75, 2.30)
    Vce_sat_125C: Tuple[float, float] = (2.20, 2.90)
    Vge_th_V: Tuple[float, float] = (5.35, 6.25)
    
    # 开关特性
    switching_energy_mJ: Dict[str, List[float]] = None
    switching_times_us: Dict[str, List[float]] = None
    
    # 热参数 - 修正为更合理的值
    Rth_jc_K_per_W: float = 0.08      # 结到壳热阻，增加到0.08
    Rth_ca_K_per_W: float = 0.4       # 壳到环境热阻，增加到0.4
    Cth_jc_J_per_K: float = 300       # 结热容，增加到300
    Cth_ca_J_per_K: float = 1500      # 壳热容，增加到1500
    
    # 材料参数
    thermal_expansion_coeff: float = 2.6e-6
    youngs_modulus_Pa: float = 130e9
    thermal_conductivity_W_per_mK: float = 120
    density_kg_per_m3: float = 2330
    
    # 寿命模型参数
    base_life_cycles: float = 1.2e6
    activation_energy_eV: float = 0.12
    boltzmann_constant_eV_per_K: float = 8.617e-5
    reference_temperature_K: float = 398  # 125°C

    def __post_init__(self):
        """初始化后处理"""
        if self.switching_energy_mJ is None:
            self.switching_energy_mJ = {
                "Eon": [335, 595],
                "Eoff": [330, 545]
            }
        if self.switching_times_us is None:
            self.switching_times_us = {
                "td_on": [0.30, 0.32],
                "tr": [0.15, 0.16],
                "td_off": [0.66, 0.80],
                "tf": [0.11, 0.17]
            }

class OptimizedIGBTModel:
    """优化的IGBT建模类"""
    
    def __init__(self, params: Optional[IGBTPhysicalParams] = None):
        """
        初始化IGBT模型
        
        Args:
            params: IGBT物理参数，默认使用FF1500R17IP5R参数
        """
        self.params = params or IGBTPhysicalParams()
        
        # 创建特性查找表
        self._create_characteristic_tables()
        self._create_interpolators()
        
        # 初始化状态变量
        self.junction_temperature_C = 25.0
        self.case_temperature_C = 25.0
        self.temperature_history = []
        
    def _create_characteristic_tables(self):
        """创建IGBT特性查找表"""
        # 电流范围
        self.current_range_A = np.array([0, 100, 500, 1000, 1500, 2000, 2500])
        
        # 饱和压降特性表（基于数据手册典型曲线）
        self.Vce_sat_table = {
            'current_A': self.current_range_A,
            'Vce_25C': np.array([0.8, 1.75, 2.0, 2.2, 2.3, 2.6, 2.9]),
            'Vce_125C': np.array([1.0, 2.2, 2.5, 2.7, 2.9, 3.1, 3.4])
        }
        
        # 开关损耗特性表（2D：电流 × 电压）
        self.voltage_range_V = np.array([600, 900, 1200, 1500, 1700])
        
        # 开通损耗查找表 (mJ)
        self.Eon_table = np.array([
            [0, 50, 200, 335, 400, 500, 600],      # 600V
            [0, 60, 250, 400, 480, 600, 720],      # 900V
            [0, 80, 300, 465, 558, 700, 840],      # 1200V
            [0, 100, 375, 580, 696, 875, 1050],    # 1500V
            [0, 120, 450, 695, 834, 1050, 1260]    # 1700V
        ])
        
        # 关断损耗查找表 (mJ)
        self.Eoff_table = np.array([
            [0, 45, 180, 330, 380, 470, 560],      # 600V
            [0, 55, 225, 395, 456, 564, 672],      # 900V
            [0, 75, 270, 460, 532, 658, 784],      # 1200V
            [0, 95, 338, 575, 665, 823, 980],      # 1500V
            [0, 115, 405, 690, 798, 988, 1176]     # 1700V
        ])
        
        # 二极管特性表
        self.diode_table = {
            'current_A': self.current_range_A,
            'Vf_25C': np.array([0.6, 1.75, 1.85, 1.95, 2.10, 2.25, 2.4]),
            'Vf_125C': np.array([0.8, 1.6, 1.7, 1.8, 1.95, 2.1, 2.25])
        }
    
    def _create_interpolators(self):
        """创建插值器以提高计算效率"""
        # 饱和压降插值器
        self.Vce_interp_25C = interp1d(
            self.Vce_sat_table['current_A'], 
            self.Vce_sat_table['Vce_25C'],
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        self.Vce_interp_125C = interp1d(
            self.Vce_sat_table['current_A'], 
            self.Vce_sat_table['Vce_125C'],
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        # 开关损耗2D插值器
        self.Eon_interp = RegularGridInterpolator(
            (self.voltage_range_V, self.current_range_A), self.Eon_table,
            bounds_error=False, fill_value=None
        )
        self.Eoff_interp = RegularGridInterpolator(
            (self.voltage_range_V, self.current_range_A), self.Eoff_table,
            bounds_error=False, fill_value=None
        )
        
        # 二极管特性插值器
        self.Vf_interp_25C = interp1d(
            self.diode_table['current_A'], 
            self.diode_table['Vf_25C'],
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        self.Vf_interp_125C = interp1d(
            self.diode_table['current_A'], 
            self.diode_table['Vf_125C'],
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
    
    def get_saturation_voltage(self, current_A: Union[float, np.ndarray], 
                             temperature_C: Union[float, np.ndarray] = None) -> Union[float, np.ndarray]:
        """
        获取IGBT饱和压降
        
        Args:
            current_A: 集电极电流 (A)
            temperature_C: 结温 (°C)，默认使用当前结温
            
        Returns:
            饱和压降 (V)
        """
        if temperature_C is None:
            temperature_C = self.junction_temperature_C
            
        current_A = np.clip(current_A, 0, self.params.max_current_A)
        
        # 获取25°C和125°C时的饱和压降
        Vce_25 = self.Vce_interp_25C(current_A)
        Vce_125 = self.Vce_interp_125C(current_A)
        
        # 温度线性插值
        if np.isscalar(temperature_C):
            if temperature_C <= 25:
                Vce_sat = Vce_25
            elif temperature_C >= 125:
                Vce_sat = Vce_125
            else:
                ratio = (temperature_C - 25) / 100
                Vce_sat = Vce_25 + ratio * (Vce_125 - Vce_25)
        else:
            temperature_C = np.array(temperature_C)
            Vce_sat = np.where(
                temperature_C <= 25, Vce_25,
                np.where(
                    temperature_C >= 125, Vce_125,
                    Vce_25 + (temperature_C - 25) / 100 * (Vce_125 - Vce_25)
                )
            )
        
        return np.clip(Vce_sat, 0.5, 5.0)  # 合理范围限制
    
    def get_switching_losses(self, current_A: float, voltage_V: float, 
                           temperature_C: float = None) -> Tuple[float, float]:
        """
        获取开关损耗
        
        Args:
            current_A: 集电极电流 (A)
            voltage_V: 直流母线电压 (V)
            temperature_C: 结温 (°C)
            
        Returns:
            (开通损耗, 关断损耗) (J)
        """
        if temperature_C is None:
            temperature_C = self.junction_temperature_C
            
        # 限制输入范围
        current_A = np.clip(current_A, 0, self.current_range_A[-1])
        voltage_V = np.clip(voltage_V, self.voltage_range_V[0], self.voltage_range_V[-1])
        
        # 使用2D插值获取基准损耗
        Eon_base = float(self.Eon_interp((voltage_V, current_A))) * 1e-3  # 转换为J
        Eoff_base = float(self.Eoff_interp((voltage_V, current_A))) * 1e-3
        
        # 温度补偿（基于IGBT5技术特性）
        temp_factor = 1 + 0.004 * (temperature_C - 25)  # 每°C增加0.4%
        
        # 电压补偿（考虑非线性特性）
        voltage_factor = (voltage_V / 1200) ** 1.2  # 基准1200V
        
        Eon = Eon_base * temp_factor * voltage_factor
        Eoff = Eoff_base * temp_factor * voltage_factor
        
        return np.clip(Eon, 0, 0.02), np.clip(Eoff, 0, 0.02)
    
    def get_diode_forward_voltage(self, current_A: Union[float, np.ndarray], 
                                temperature_C: Union[float, np.ndarray] = None) -> Union[float, np.ndarray]:
        """
        获取二极管正向压降
        
        Args:
            current_A: 正向电流 (A)
            temperature_C: 结温 (°C)
            
        Returns:
            正向压降 (V)
        """
        if temperature_C is None:
            temperature_C = self.junction_temperature_C
            
        current_A = np.clip(current_A, 0, self.params.max_current_A)
        
        # 获取25°C和125°C时的正向压降
        Vf_25 = self.Vf_interp_25C(current_A)
        Vf_125 = self.Vf_interp_125C(current_A)
        
        # 温度线性插值
        if np.isscalar(temperature_C):
            if temperature_C <= 25:
                Vf = Vf_25
            elif temperature_C >= 125:
                Vf = Vf_125
            else:
                ratio = (temperature_C - 25) / 100
                Vf = Vf_25 + ratio * (Vf_125 - Vf_25)
        else:
            temperature_C = np.array(temperature_C)
            Vf = np.where(
                temperature_C <= 25, Vf_25,
                np.where(
                    temperature_C >= 125, Vf_125,
                    Vf_25 + (temperature_C - 25) / 100 * (Vf_125 - Vf_25)
                )
            )
        
        return np.clip(Vf, 0.3, 3.0)
    
    def calculate_power_losses(self, current_rms_A: float, voltage_dc_V: float, 
                             switching_freq_Hz: float, duty_cycle: float = 0.5,
                             temperature_C: float = None) -> Dict[str, float]:
        """
        计算功率损耗
        
        Args:
            current_rms_A: RMS电流 (A)
            voltage_dc_V: 直流电压 (V)
            switching_freq_Hz: 开关频率 (Hz)
            duty_cycle: 占空比
            temperature_C: 结温 (°C)
            
        Returns:
            功率损耗字典 (W)
        """
        if temperature_C is None:
            temperature_C = self.junction_temperature_C
            
        # IGBT导通损耗
        Vce_sat = self.get_saturation_voltage(current_rms_A, temperature_C)
        P_cond_igbt = Vce_sat * current_rms_A * duty_cycle
        
        # IGBT开关损耗
        Eon, Eoff = self.get_switching_losses(current_rms_A, voltage_dc_V, temperature_C)
        P_sw_igbt = (Eon + Eoff) * switching_freq_Hz
        
        # 二极管导通损耗
        Vf = self.get_diode_forward_voltage(current_rms_A, temperature_C)
        P_cond_diode = Vf * current_rms_A * (1 - duty_cycle)
        
        # 总损耗
        P_total = P_cond_igbt + P_sw_igbt + P_cond_diode
        
        return {
            'total': P_total,
            'igbt_conduction': P_cond_igbt,
            'igbt_switching': P_sw_igbt,
            'diode_conduction': P_cond_diode
        }
    
    def update_thermal_state(self, power_loss_W: float, ambient_temp_C: float = 25,
                           dt_s: float = 1.0) -> Tuple[float, float]:
        """
        更新热状态 - 基于修复的三阶热网络模型
        
        Args:
            power_loss_W: 功率损耗 (W)
            ambient_temp_C: 环境温度 (°C)
            dt_s: 时间步长 (s)
            
        Returns:
            (结温, 壳温) (°C)
        """
        # 三阶热网络参数
        Rth_jc = 0.05   # 结到壳热阻 (K/W)
        Rth_ch = 0.02   # 壳到散热器热阻 (K/W)
        Rth_ha = 0.4    # 散热器到环境热阻 (K/W)
        
        Cth_j = 1000    # 结热容 (J/K)
        Cth_c = 5000    # 壳热容 (J/K)
        Cth_h = 20000   # 散热器热容 (J/K)
        
        # 初始化散热器温度（如果第一次调用）
        if not hasattr(self, 'heatsink_temperature_C'):
            self.heatsink_temperature_C = ambient_temp_C
        
        # 使用数值稳定的小步长积分
        tau_min = min(Rth_jc * Cth_j, Rth_ch * Cth_c, Rth_ha * Cth_h)
        internal_dt = min(dt_s, tau_min / 10)  # 内部步长为最小时间常数的1/10
        num_steps = max(1, int(dt_s / internal_dt))
        actual_dt = dt_s / num_steps
        
        for _ in range(num_steps):
            # 热流计算
            q_jc = (self.junction_temperature_C - self.case_temperature_C) / Rth_jc
            q_ch = (self.case_temperature_C - self.heatsink_temperature_C) / Rth_ch
            q_ha = (self.heatsink_temperature_C - ambient_temp_C) / Rth_ha
            
            # 温度变化率
            dTj_dt = (power_loss_W - q_jc) / Cth_j
            dTc_dt = (q_jc - q_ch) / Cth_c  
            dTh_dt = (q_ch - q_ha) / Cth_h
            
            # 欧拉积分更新
            self.junction_temperature_C += dTj_dt * actual_dt
            self.case_temperature_C += dTc_dt * actual_dt
            self.heatsink_temperature_C += dTh_dt * actual_dt
            
            # 物理合理性检查（只在极端情况下限制）
            max_temp = 200.0  # 合理的最高温度
            min_temp = ambient_temp_C - 5
            
            self.junction_temperature_C = max(min_temp, min(self.junction_temperature_C, max_temp))
            self.case_temperature_C = max(min_temp, min(self.case_temperature_C, max_temp - 10))
            self.heatsink_temperature_C = max(min_temp, min(self.heatsink_temperature_C, max_temp - 20))
        
        # 记录温度历史
        self.temperature_history.append(self.junction_temperature_C)
        
        return self.junction_temperature_C, self.case_temperature_C
    
    def rainflow_counting(self, temperature_history: List[float] = None) -> List[float]:
        """
        雨流计数算法分析温度循环
        
        Args:
            temperature_history: 温度历史数据
            
        Returns:
            温度变化幅度列表
        """
        if temperature_history is None:
            temperature_history = self.temperature_history
            
        if len(temperature_history) < 3:
            return []
        
        temp_array = np.array(temperature_history)
        
        # 寻找峰值和谷值
        peaks, _ = find_peaks(temp_array, height=np.mean(temp_array))
        valleys, _ = find_peaks(-temp_array, height=-np.mean(temp_array))
        
        # 合并并排序极值点
        extrema_indices = np.sort(np.concatenate([peaks, valleys]))
        
        if len(extrema_indices) < 2:
            return []
        
        # 简化的雨流计数
        cycles = []
        for i in range(len(extrema_indices) - 1):
            for j in range(i + 1, len(extrema_indices)):
                delta_T = abs(temp_array[extrema_indices[j]] - temp_array[extrema_indices[i]])
                if delta_T > 5.0:  # 最小温度变化阈值
                    cycles.append(delta_T)
        
        return cycles
    
    def calculate_life_consumption(self, temperature_history: List[float] = None,
                                 operating_hours: float = 8760) -> Dict[str, float]:
        """
        计算寿命消耗
        
        Args:
            temperature_history: 温度历史数据
            operating_hours: 运行小时数
            
        Returns:
            寿命分析结果
        """
        if temperature_history is None:
            temperature_history = self.temperature_history
            
        if len(temperature_history) < 2:
            return {
                'remaining_life': 1.0,
                'total_damage': 0.0,
                'thermal_cycles': 0,
                'avg_temperature': 25.0
            }
        
        temp_array = np.array(temperature_history)
        avg_temp = np.mean(temp_array)
        
        # 温度循环分析
        temp_cycles = self.rainflow_counting(temperature_history)
        
        # 改进的Coffin-Manson模型
        cycle_damage = 0.0
        for delta_T in temp_cycles:
            if delta_T > 5.0:  # 只考虑有意义的温度循环
                # 更合理的寿命公式，考虑负载强度
                cycles_to_failure = self.params.base_life_cycles * (40.0 / max(delta_T, 10)) ** 4.0
                damage = 1.0 / max(cycles_to_failure, 1000)  # 避免除零
                cycle_damage += damage
        
        # Arrhenius温度加速模型 - 修正计算
        if avg_temp > 25:
            # 更合理的激活能和参考温度
            temp_acceleration = np.exp(
                self.params.activation_energy_eV / self.params.boltzmann_constant_eV_per_K * 
                (1 / (avg_temp + 273) - 1 / 298)  # 参考温度25°C
            )
            # 基于温度的年化损伤，考虑高温的累积效应
            base_annual_damage = 0.01  # 基准年损伤1%
            temp_damage = base_annual_damage * temp_acceleration * (operating_hours / 8760)
        else:
            temp_damage = 0.005 * (operating_hours / 8760)  # 低温时的基础损伤
        
        # 综合损伤，增加温度权重
        if avg_temp > 150:  # 高温严重损伤
            temp_penalty = 1.5 * ((avg_temp - 150) / 25) ** 2  # 温度超过150°C时指数增长
        elif avg_temp > 125:  # 中高温
            temp_penalty = 0.5 * ((avg_temp - 125) / 25)
        else:
            temp_penalty = 0.0
        
        total_damage = cycle_damage + temp_damage + temp_penalty
        total_damage = min(total_damage, 0.99)  # 限制最大损伤
        remaining_life = max(0.01, 1.0 - total_damage)  # 保留至少1%
        
        return {
            'remaining_life': remaining_life,
            'total_damage': total_damage,
            'thermal_cycles': len(temp_cycles),
            'avg_temperature': avg_temp,
            'max_temperature': np.max(temp_array),
            'min_temperature': np.min(temp_array)
        }
    
    def predict_lifetime(self, operating_profile: Dict, years: int = 10) -> pd.DataFrame:
        """
        预测IGBT寿命
        
        Args:
            operating_profile: 工作曲线配置
            years: 预测年数
            
        Returns:
            寿命预测结果DataFrame
        """
        results = []
        cumulative_damage = 0.0
        
        for year in range(1, years + 1):
            # 重置模型状态
            self.junction_temperature_C = 25.0
            self.temperature_history = []
            
            # 生成年度工况
            hours_per_year = 8760
            time_steps = np.linspace(0, hours_per_year, 1000)  # 1000个时间点
            
            # 根据工况类型设置参数
            load_factor = operating_profile.get('load_factor', 0.7)
            base_current = operating_profile.get('base_current_A', 500)
            switching_freq = operating_profile.get('switching_freq_Hz', 1000)
            ambient_temp = operating_profile.get('ambient_temp_C', 25)
            
            # 年度老化：参数逐年变化
            current_rms = base_current * load_factor * (1 + year * 0.02)  # 每年增加2%负载
            ambient_temp += year * 0.3  # 每年环境温度升高0.3°C
            
            # 温度历史仿真（增加变化以避免直线）
            for i, t in enumerate(time_steps):
                # 更复杂的环境温度变化
                daily_variation = 8 * np.sin(2 * np.pi * t / 24)  # 日温度变化±8°C
                weekly_variation = 3 * np.sin(2 * np.pi * t / (24 * 7))  # 周变化±3°C
                seasonal_variation = 5 * np.sin(2 * np.pi * t / (24 * 365))  # 季节变化±5°C
                random_variation = np.random.normal(0, 1.5)  # 随机天气波动
                temp_ambient = ambient_temp + daily_variation + weekly_variation + seasonal_variation + random_variation
                
                # 更复杂的负载变化
                load_variation = 0.8 + 0.4 * np.sin(2 * np.pi * t / 24)  # 日负载变化
                load_variation *= (1 + 0.1 * np.sin(2 * np.pi * t / (24 * 7)))  # 周负载变化
                load_variation *= (1 + 0.1 * np.random.normal(0, 1))  # 随机负载波动
                current_actual = current_rms * np.clip(load_variation, 0.3, 1.5)
                
                # 开关频率也可能变化
                freq_variation = switching_freq * (1 + 0.05 * np.random.normal(0, 1))
                freq_actual = np.clip(freq_variation, switching_freq * 0.8, switching_freq * 1.2)
                
                # 计算功率损耗（考虑温度反馈）
                losses = self.calculate_power_losses(
                    current_actual, 1200, freq_actual, 0.5, 
                    self.junction_temperature_C
                )
                
                # 更新温度（使用较小的时间步长以提高精度）
                self.update_thermal_state(losses['total'], temp_ambient, 3600)  # 1小时步长
            
            # 计算年度寿命消耗
            life_result = self.calculate_life_consumption(
                self.temperature_history, hours_per_year
            )
            
            # 累积损伤
            cumulative_damage += life_result['total_damage']
            remaining_life = max(0.0, 1.0 - cumulative_damage)
            
            results.append({
                'year': year,
                'remaining_life_percent': remaining_life * 100,
                'cumulative_damage_percent': cumulative_damage * 100,
                'annual_damage_percent': life_result['total_damage'] * 100,
                'avg_temperature_C': life_result['avg_temperature'],
                'max_temperature_C': life_result['max_temperature'],
                'thermal_cycles': life_result['thermal_cycles'],
                'current_rms_A': current_rms
            })
        
        return pd.DataFrame(results)
    
    def plot_characteristics(self, save_path: str = None):
        """绘制IGBT特性曲线"""
        try:
            from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot
        except ImportError:
            # 使用基本的matplotlib
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('优化IGBT模型特性分析', fontsize=14, fontweight='bold')
        else:
            fig, axes = create_adaptive_figure(2, 3, title='优化IGBT模型特性分析')
        
        # 饱和压降特性
        current_range = np.linspace(100, 2000, 100)
        temps = [25, 75, 125]
        
        for temp in temps:
            Vce_sat = self.get_saturation_voltage(current_range, temp)
            axes[0, 0].plot(current_range, Vce_sat, label=f'Tj={temp}°C', linewidth=2)
        
        axes[0, 0].set_xlabel('集电极电流 (A)')
        axes[0, 0].set_ylabel('饱和压降 (V)')
        axes[0, 0].set_title('IGBT饱和压降特性')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 开关损耗特性
        voltage_range = np.linspace(600, 1700, 100)
        test_current = 1000
        
        Eon_list = []
        Eoff_list = []
        for voltage in voltage_range:
            Eon, Eoff = self.get_switching_losses(test_current, voltage)
            Eon_list.append(Eon * 1e3)  # 转换为mJ
            Eoff_list.append(Eoff * 1e3)
        
        axes[0, 1].plot(voltage_range, Eon_list, 'b-', label='开通损耗', linewidth=2)
        axes[0, 1].plot(voltage_range, Eoff_list, 'r-', label='关断损耗', linewidth=2)
        axes[0, 1].set_xlabel('直流电压 (V)')
        axes[0, 1].set_ylabel('开关损耗 (mJ)')
        axes[0, 1].set_title(f'开关损耗特性 (Ic={test_current}A)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 二极管特性
        for temp in temps:
            Vf = self.get_diode_forward_voltage(current_range, temp)
            axes[0, 2].plot(current_range, Vf, label=f'Tj={temp}°C', linewidth=2)
        
        axes[0, 2].set_xlabel('正向电流 (A)')
        axes[0, 2].set_ylabel('正向压降 (V)')
        axes[0, 2].set_title('二极管正向压降特性')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 功率损耗分析
        power_range = np.linspace(1e6, 10e6, 50)  # 1-10 MW
        switching_freqs = [500, 1000, 2000]
        
        for fsw in switching_freqs:
            total_losses = []
            for power in power_range:
                current = power / (1200 * np.sqrt(3))  # 简化电流计算
                losses = self.calculate_power_losses(current, 1200, fsw)
                total_losses.append(losses['total'] / 1e3)  # 转换为kW
            axes[1, 0].plot(power_range / 1e6, total_losses, label=f'fsw={fsw}Hz', linewidth=2)
        
        axes[1, 0].set_xlabel('功率 (MW)')
        axes[1, 0].set_ylabel('总损耗 (kW)')
        axes[1, 0].set_title('功率损耗vs开关频率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 热特性仿真
        time_hours = np.linspace(0, 24, 100)
        power_profile = 5000 + 2000 * np.sin(2 * np.pi * time_hours / 24)  # 日负载变化
        
        # 重置温度状态
        original_temp = self.junction_temperature_C
        self.junction_temperature_C = 25.0
        temp_history = []
        
        for power in power_profile:
            self.update_thermal_state(power, 25, 3600)  # 1小时步长
            temp_history.append(self.junction_temperature_C)
        
        axes[1, 1].plot(time_hours, temp_history, 'r-', linewidth=2, label='结温')
        axes[1, 1].plot(time_hours, power_profile / 100, 'b--', linewidth=2, label='功率/100')
        axes[1, 1].set_xlabel('时间 (小时)')
        axes[1, 1].set_ylabel('温度 (°C) / 功率 (100×W)')
        axes[1, 1].set_title('24小时热特性仿真')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 恢复原始温度
        self.junction_temperature_C = original_temp
        
        # 寿命分析示例
        operating_profiles = {
            'Light': {'load_factor': 0.3, 'base_current_A': 300},
            'Medium': {'load_factor': 0.6, 'base_current_A': 600},
            'Heavy': {'load_factor': 0.9, 'base_current_A': 1000}
        }
        
        for profile_name, profile in operating_profiles.items():
            results = self.predict_lifetime(profile, years=5)
            axes[1, 2].plot(results['year'], results['remaining_life_percent'], 
                          'o-', linewidth=2, markersize=6, label=profile_name)
        
        axes[1, 2].set_xlabel('运行年数')
        axes[1, 2].set_ylabel('剩余寿命 (%)')
        axes[1, 2].set_title('不同负载下的寿命预测')
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

def test_optimized_igbt_model():
    """测试优化的IGBT模型"""
    print("=" * 60)
    print("测试优化的IGBT模型")
    print("=" * 60)
    
    # 创建模型实例
    igbt = OptimizedIGBTModel()
    
    # 测试基本特性
    test_current = 1000  # A
    test_voltage = 1200  # V
    test_temp = 75  # °C
    
    print(f"\n基本特性测试 (Ic={test_current}A, Vdc={test_voltage}V, Tj={test_temp}°C):")
    print("-" * 60)
    
    # 饱和压降
    Vce_sat = igbt.get_saturation_voltage(test_current, test_temp)
    print(f"饱和压降: {Vce_sat:.3f} V")
    
    # 开关损耗
    Eon, Eoff = igbt.get_switching_losses(test_current, test_voltage, test_temp)
    print(f"开通损耗: {Eon*1e3:.2f} mJ")
    print(f"关断损耗: {Eoff*1e3:.2f} mJ")
    
    # 二极管特性
    Vf = igbt.get_diode_forward_voltage(test_current, test_temp)
    print(f"二极管压降: {Vf:.3f} V")
    
    # 功率损耗计算
    losses = igbt.calculate_power_losses(test_current, test_voltage, 1000, 0.5, test_temp)
    print(f"\n功率损耗分析 (fsw=1000Hz, D=0.5):")
    print(f"  总损耗: {losses['total']/1e3:.2f} kW")
    print(f"  IGBT导通损耗: {losses['igbt_conduction']/1e3:.2f} kW")
    print(f"  IGBT开关损耗: {losses['igbt_switching']/1e3:.2f} kW")
    print(f"  二极管导通损耗: {losses['diode_conduction']/1e3:.2f} kW")
    
    # 寿命预测测试
    print(f"\n寿命预测测试:")
    print("-" * 40)
    
    operating_profile = {
        'load_factor': 0.7,
        'base_current_A': 800,
        'switching_freq_Hz': 1000,
        'ambient_temp_C': 35
    }
    
    results = igbt.predict_lifetime(operating_profile, years=3)
    print(f"工况: 70%负载, 800A基础电流, 35°C环境温度")
    print(f"3年后剩余寿命: {results.iloc[-1]['remaining_life_percent']:.1f}%")
    print(f"3年累积损伤: {results.iloc[-1]['cumulative_damage_percent']:.2f}%")
    print(f"平均工作温度: {results.iloc[-1]['avg_temperature_C']:.1f}°C")
    
    # 绘制特性曲线
    print(f"\n正在生成特性曲线...")
    igbt.plot_characteristics('pic/优化IGBT模型特性分析.png')
    
    print(f"\n✓ 优化IGBT模型测试完成！")
    
    return igbt

if __name__ == "__main__":
    test_optimized_igbt_model()
