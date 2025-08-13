#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级器件建模模块
提供IGBT、电容器和热模型的详细建模功能
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from device_parameters import IGBTParameters, CapacitorParameters
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedIGBTModel:
    """高级IGBT建模 - 基于Infineon FF1500R17IP5R"""
    
    def __init__(self):
        # 导入设备参数
        self.params = IGBTParameters()
        
        # 创建查找表和插值器
        self._create_lookup_tables()
        self._create_interpolators()
    
    def _create_lookup_tables(self):
        """创建IGBT特性查找表 - 基于device_parameters.py中的真实参数"""
        # 饱和压降查找表 - 使用device_parameters.py中的真实数据
        # 基于Infineon FF1500R17IP5R数据手册的典型特性曲线
        self.Vce_sat_table = {
            'Ic': np.array([0, 100, 500, 1000, 1500, 2000]),  # 集电极电流 (A)
            'Tj_25': np.array([0.8, self.params.Vce_sat_V["25C"][0], 2.0, 2.3, self.params.Vce_sat_V["25C"][1], 2.8]),  # 25°C饱和压降 (V)
            'Tj_125': np.array([1.0, self.params.Vce_sat_V["125C"][0], 2.5, 2.9, self.params.Vce_sat_V["125C"][1], 3.2])   # 125°C饱和压降 (V)
        }
        
        # 开关损耗查找表 - 基于device_parameters.py中的真实switching_energy_mJ数据
        # 使用数据手册中的典型工作点
        self.switching_loss_table = {
            'Ic': np.array([0, 100, 500, 1000, 1500, 2000]),  # 集电极电流 (A)
            'Vdc': np.array([600, 900, 1200, 1500, 1700]),  # 直流电压 (V)
            'Eon': np.array([
                [0, 50, 200, self.params.switching_energy_mJ['Eon'][0], 400, 600],
                [0, 60, 250, 350, 500, 700],
                [0, 80, 300, 450, 600, 800],
                [0, 100, 400, self.params.switching_energy_mJ['Eon'][1], 650, 900],
                [0, 120, 500, 550, 750, 1000]
            ]),    # 开通损耗 (mJ) - 2D查找表 [Ic][Vdc]
            'Eoff': np.array([
                [0, 50, 200, self.params.switching_energy_mJ['Eoff'][0], 400, 600],
                [0, 60, 250, 350, 500, 700],
                [0, 80, 300, 450, 600, 800],
                [0, 100, 400, self.params.switching_energy_mJ['Eoff'][1], 650, 900],
                [0, 120, 500, 550, 750, 1000]
            ])    # 关断损耗 (mJ) - 2D查找表 [Ic][Vdc]
        }
        
        # 二极管特性查找表 - 基于device_parameters.py中的真实diode_Vf_V数据
        # 使用数据手册中的典型特性曲线
        self.diode_table = {
            'If': np.array([0, 100, 500, 1000, 1500, 2000]),  # 正向电流 (A)
            'Tj_25': np.array([0.6, self.params.diode_Vf_V[0], 1.85, 2.0, self.params.diode_Vf_V[1], 2.3]),  # 25°C正向压降 (V)
            'Tj_125': np.array([0.8, 1.6, 1.7, 1.85, 1.95, 2.1])  # 125°C正向压降 (V)
        }
    
    def _create_interpolators(self):
        """创建插值器以提高性能"""
        # 饱和压降插值器
        self.Vce_interp_25 = interp1d(self.Vce_sat_table['Ic'], self.Vce_sat_table['Tj_25'], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
        self.Vce_interp_125 = interp1d(self.Vce_sat_table['Ic'], self.Vce_sat_table['Tj_125'], 
                                      kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # 二极管插值器
        self.diode_interp_25 = interp1d(self.diode_table['If'], self.diode_table['Tj_25'], 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
        self.diode_interp_125 = interp1d(self.diode_table['If'], self.diode_table['Tj_125'], 
                                        kind='linear', bounds_error=False, fill_value='extrapolate')
    
    def _validate_inputs(self, Ic, Tj):
        """验证输入参数"""
        if not isinstance(Ic, (int, float, np.ndarray)) or not isinstance(Tj, (int, float, np.ndarray)):
            raise TypeError("Ic和Tj必须是数值类型")
        
        if np.any(Ic < 0) or np.any(Tj < -40):
            raise ValueError("Ic必须大于等于0，Tj必须大于等于-40°C")
        
        return True
    
    def get_Vce_sat(self, Ic, Tj):
        """获取饱和压降 - 优化版本"""
        try:
            self._validate_inputs(Ic, Tj)
            
            # 使用插值器获取25°C和125°C的压降
            Vce_25 = self.Vce_interp_25(Ic)
            Vce_125 = self.Vce_interp_125(Ic)
            
            # 温度插值
            if Tj <= 25:
                Vce_sat = Vce_25
            elif Tj >= 125:
                Vce_sat = Vce_125
            else:
                ratio = (Tj - 25) / 100
                Vce_sat = Vce_25 + ratio * (Vce_125 - Vce_25)
            
            return np.clip(Vce_sat, 0, 10)  # 限制在合理范围内
            
        except Exception as e:
            print(f"获取饱和压降时出错: {e}")
            return np.nan
    
    def get_switching_losses(self, Ic, Vdc, Tj=25):
        """获取开关损耗 - 基于2D查找表的改进版本"""
        try:
            if not isinstance(Ic, (int, float)) or not isinstance(Vdc, (int, float)):
                raise TypeError("Ic和Vdc必须是数值类型")
            
            if Ic < 0 or Vdc < 0:
                raise ValueError("Ic和Vdc必须大于等于0")
            
            # 限制输入范围到查找表范围内
            Ic = np.clip(Ic, 0, self.switching_loss_table['Ic'][-1])
            Vdc = np.clip(Vdc, self.switching_loss_table['Vdc'][0], self.switching_loss_table['Vdc'][-1])
            
            # 查找最接近的工作点
            Ic_idx = np.argmin(np.abs(self.switching_loss_table['Ic'] - Ic))
            Vdc_idx = np.argmin(np.abs(self.switching_loss_table['Vdc'] - Vdc))
            
            # 获取基准损耗 (2D查找表)
            Eon_base = self.switching_loss_table['Eon'][Ic_idx][Vdc_idx] * 1e-3  # 转换为J
            Eoff_base = self.switching_loss_table['Eoff'][Ic_idx][Vdc_idx] * 1e-3  # 转换为J
            
            # 改进的温度补偿模型 - 基于IGBT5技术特性
            # 温度每升高1°C，损耗增加0.4% (IGBT5的典型值)
            temp_factor = 1 + 0.004 * (Tj - 25)
            
            # 电压补偿 - 使用平方关系更准确
            V_ratio = (Vdc / self.switching_loss_table['Vdc'][Vdc_idx]) ** 1.5
            
            # 电流补偿 - 考虑IGBT的电流特性
            I_ratio = (Ic / max(self.switching_loss_table['Ic'][Ic_idx], 1)) ** 0.8
            
            Eon = Eon_base * V_ratio * I_ratio * temp_factor
            Eoff = Eoff_base * V_ratio * I_ratio * temp_factor
            
            # 限制在合理范围内
            return np.clip(Eon, 0, 15), np.clip(Eoff, 0, 15)
            
        except Exception as e:
            print(f"获取开关损耗时出错: {e}")
            return np.nan, np.nan
    
    def get_diode_Vf(self, If, Tj):
        """获取二极管正向压降 - 优化版本"""
        try:
            self._validate_inputs(If, Tj)
            
            # 使用插值器获取25°C和125°C的压降
            Vf_25 = self.diode_interp_25(If)
            Vf_125 = self.diode_interp_125(If)
            
            # 温度插值
            if Tj <= 25:
                Vf = Vf_25
            elif Tj >= 125:
                Vf = Vf_125
            else:
                ratio = (Tj - 25) / 100
                Vf = Vf_25 + ratio * (Vf_125 - Vf_25)
            
            return np.clip(Vf, 0, 5)  # 限制在合理范围内
            
        except Exception as e:
            print(f"获取二极管压降时出错: {e}")
            return np.nan

class AdvancedCapacitorModel:
    """高级电容器建模 - 基于Xiamen Farah/Nantong Jianghai"""
    
    def __init__(self, manufacturer="Xiamen Farah"):
        self.params = CapacitorParameters(manufacturer)
        
        # 创建电容器模型和插值器
        self._create_capacitor_model()
        self._create_interpolators()
    
    def _create_capacitor_model(self):
        """创建电容器模型 - 基于真实器件参数"""
        # ESR频率特性 - 基于Xiamen Farah/Nantong Jianghai电容器数据手册
        self.freq_ESR = {
            'freq': np.array([50, 100, 1000, 10000, 100000]),  # 频率 (Hz)
            'ESR': np.array([1.5, 1.3, self.params.get_ESR()*1e3, 0.8, 0.6])  # ESR (mΩ)
        }
        
        # 电容值温度特性 - 基于薄膜电容器典型特性
        self.temp_cap = {
            'temp': np.array([-40, -20, 0, 25, 50, 70, 85]),  # 温度 (°C)
            'cap_ratio': np.array([0.92, 0.95, 0.98, 1.0, 0.99, 0.97, 0.94])  # 电容值比例
        }
        
        # 改进的寿命模型参数 - 基于薄膜电容器实际数据
        self.life_params = {
            'L0': self.params.get_lifetime(),  # 基准寿命 (小时) - 从参数文件获取
            'T0': 70,      # 基准温度 (°C)
            'Ea': 0.12,    # 激活能 (eV) - 薄膜电容器典型值
            'k': 8.617e-5, # 玻尔兹曼常数 (eV/K)
            'n': 2.5,      # 电压应力指数 - 薄膜电容器典型值
            'm': 1.8       # 电流应力指数 - 薄膜电容器典型值
        }
    
    def _create_interpolators(self):
        """创建插值器"""
        self.ESR_interp = interp1d(self.freq_ESR['freq'], self.freq_ESR['ESR'], 
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
        self.cap_interp = interp1d(self.temp_cap['temp'], self.temp_cap['cap_ratio'], 
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
    
    def get_ESR(self, freq=1000, temp=25):
        """获取ESR值 - 优化版本"""
        try:
            if not isinstance(freq, (int, float)) or not isinstance(temp, (int, float)):
                raise TypeError("freq和temp必须是数值类型")
            
            if freq < 0 or temp < -40:
                raise ValueError("freq必须大于等于0，temp必须大于等于-40°C")
            
            # 使用插值器获取ESR
            ESR = self.ESR_interp(freq)
            
            # 温度补偿
            temp_factor = 1 + 0.002 * (temp - 25)  # 温度每升高1°C，ESR增加0.2%
            ESR *= temp_factor
            
            return np.clip(ESR * 1e-3, 0, 1)  # 转换为Ω并限制范围
            
        except Exception as e:
            print(f"获取ESR时出错: {e}")
            return np.nan
    
    def get_capacitance(self, temp=25):
        """获取电容值 - 优化版本"""
        try:
            if not isinstance(temp, (int, float)):
                raise TypeError("temp必须是数值类型")
            
            if temp < -40:
                raise ValueError("temp必须大于等于-40°C")
            
            # 使用插值器获取电容比例
            cap_ratio = self.cap_interp(temp)
            
            return self.params.get_capacitance() * np.clip(cap_ratio, 0.5, 1.5)
            
        except Exception as e:
            print(f"获取电容值时出错: {e}")
            return np.nan
    
    def calculate_lifetime(self, Ir_rms, V_ratio, Tamb, time_hours):
        """计算电容器寿命 - 优化版本"""
        try:
            if not all(isinstance(x, (int, float)) for x in [Ir_rms, V_ratio, Tamb, time_hours]):
                raise TypeError("所有参数必须是数值类型")
            
            if Ir_rms < 0 or V_ratio <= 0 or time_hours < 0:
                raise ValueError("Ir_rms、V_ratio和time_hours必须大于0")
            
            # 温升计算
            ESR = self.get_ESR(temp=Tamb)
            if np.isnan(ESR):
                return np.nan, np.nan
                
            P_loss = Ir_rms**2 * ESR
            Rth = 0.5  # 热阻 (K/W)
            delta_T = P_loss * Rth
            Thot = Tamb + delta_T
            
            # Arrhenius寿命模型
            T0_K = 273 + self.life_params['T0']
            Thot_K = 273 + Thot
            
            # 温度应力
            temp_stress = np.exp(self.life_params['Ea'] / self.life_params['k'] * (1/Thot_K - 1/T0_K))
            
            # 电压应力
            voltage_stress = (1.0 / V_ratio)**self.life_params['n'] if V_ratio > 0 else 1
            
            # 电流应力
            current_stress = (self.params.current_params['max_current_A'] / Ir_rms)**self.life_params['m'] if Ir_rms > 0 else 1
            
            # 综合寿命
            L = self.life_params['L0'] * temp_stress * voltage_stress * current_stress
            
            # 寿命消耗
            life_consumption = time_hours / L
            life_remaining = max(1 - life_consumption, 0)
            
            return life_remaining, life_consumption
            
        except Exception as e:
            print(f"计算寿命时出错: {e}")
            return np.nan, np.nan

class AdvancedThermalModel:
    """高级热模型"""
    
    def __init__(self):
        from device_parameters import ThermalParameters
        self.params = ThermalParameters()
        
        # 创建热网络模型
        self._create_thermal_network()
    
    def _create_thermal_network(self):
        """创建热网络模型 - 基于Infineon FF1500R17IP5R真实参数"""
        # 热阻网络 - 基于IGBT数据手册的真实值
        # 注意：这些值需要根据实际散热器配置调整
        self.thermal_network = {
            'Rth_jc': self.params.Rth_jc,  # 结到壳热阻 (K/W) - 从参数文件获取
            'Rth_ca': self.params.Rth_ca,  # 壳到环境热阻 (K/W) - 从参数文件获取
            'Cth_jc': self.params.Cth_jc,  # 结到壳热容 (J/K) - 从参数文件获取
            'Cth_ca': self.params.Cth_ca   # 壳到环境热容 (J/K) - 从参数文件获取
        }
        
        # 改进的热网络 - 考虑IGBT模块的实际结构
        # 添加中间节点的热阻和热容
        self.thermal_network.update({
            'Rth_jb': 0.03,  # 结到基板热阻 (K/W) - IGBT5典型值
            'Rth_ba': 0.12,  # 基板到环境热阻 (K/W) - 考虑散热器
            'Cth_jb': 80,    # 结到基板热容 (J/K)
            'Cth_ba': 800    # 基板到环境热容 (J/K)
        })
        
        # 温度状态
        self.Tj = self.params.T_amb  # 结温
        self.Tc = self.params.T_amb  # 壳温
        self.Tb = self.params.T_amb  # 基板温度
        self.Ta = self.params.T_amb  # 环境温度
    
    def update_temperature(self, P_loss, dt, cooling_efficiency=0.85):
        """更新温度状态 - 基于改进热网络的精确模型"""
        try:
            if not isinstance(P_loss, (int, float)) or not isinstance(dt, (int, float)):
                raise TypeError("P_loss和dt必须是数值类型")
            
            if P_loss < 0 or dt <= 0:
                raise ValueError("P_loss必须大于等于0，dt必须大于0")
            
            # 有效热阻 (考虑冷却效率)
            Rth_eff = self.thermal_network['Rth_ba'] / max(cooling_efficiency, 0.1)
            
            # 改进的热传递模型 - 考虑结-基板-环境的热路径
            # 结到基板热传递
            dTj_dt = (P_loss - (self.Tj - self.Tb) / self.thermal_network['Rth_jb']) / self.thermal_network['Cth_jb']
            self.Tj += dTj_dt * dt
            
            # 基板到环境热传递
            dTb_dt = ((self.Tj - self.Tb) / self.thermal_network['Rth_jb'] - 
                      (self.Tb - self.Ta) / Rth_eff) / self.thermal_network['Cth_ba']
            self.Tb += dTb_dt * dt
            
            # 壳温计算 (基于基板温度)
            self.Tc = self.Tb + (self.Tj - self.Tb) * 0.1  # 壳温略高于基板温度
            
            # 温度限制 - 基于IGBT数据手册
            self.Tj = np.clip(self.Tj, self.params.Tj_min, self.params.Tj_max)
            self.Tb = np.clip(self.Tb, self.Ta - 10, self.Ta + 100)
            self.Tc = np.clip(self.Tc, self.Tb - 5, self.Tb + 15)
            
            return self.Tj, self.Tc, self.Tb
            
        except Exception as e:
            print(f"更新温度时出错: {e}")
            return self.Tj, self.Tc, self.Tb
    
    def get_thermal_stress(self):
        """获取热应力因子"""
        # 基于结温的热应力
        if self.Tj <= 100:
            stress = 1.0
        elif self.Tj <= 125:
            stress = 1.2
        elif self.Tj <= 150:
            stress = 1.5
        else:
            stress = 2.0
        
        return stress

class AdvancedPowerLossModel:
    """高级功率损耗模型"""
    
    def __init__(self):
        self.igbt_model = AdvancedIGBTModel()
        self.cap_model = AdvancedCapacitorModel()
        self.thermal_model = AdvancedThermalModel()
    
    def calculate_total_losses(self, P_out, Vdc, fsw, mode='discharge', Tj=25):
        """计算总功率损耗 - 基于35kV系统特性的改进版本"""
        try:
            if not all(isinstance(x, (int, float)) for x in [P_out, Vdc, fsw]):
                raise TypeError("P_out、Vdc和fsw必须是数值类型")
            
            if P_out < 0 or Vdc < 0 or fsw < 0:
                raise ValueError("P_out、Vdc和fsw必须大于等于0")
            
            # 改进的电流计算 - 考虑级联H桥拓扑
            # 基于device_parameters.py中的系统参数
            from device_parameters import SystemParameters
            sys_params = SystemParameters()
            
            # 每相级联模块数
            modules_per_phase = sys_params.cascaded_power_modules
            
            # 模块级电流计算
            if mode == 'discharge':
                # 放电模式：电池向电网供电
                I_rms_per_module = P_out / (np.sqrt(3) * Vdc * modules_per_phase)
            else:
                # 充电模式：电网向电池充电
                I_rms_per_module = P_out / (np.sqrt(3) * Vdc * modules_per_phase)
            
            # IGBT损耗计算
            Eon, Eoff = self.igbt_model.get_switching_losses(I_rms_per_module, Vdc, Tj)
            if np.isnan(Eon) or np.isnan(Eoff):
                return {'total': np.nan, 'switching': np.nan, 'conduction_igbt': np.nan, 
                       'conduction_diode': np.nan, 'capacitor': np.nan}
            
            # 开关损耗 - 考虑每相6个IGBT (3个H桥)
            P_sw_igbt = (Eon + Eoff) * fsw * 6
            
            # 导通损耗计算 - 改进的占空比模型
            Vce_sat = self.igbt_model.get_Vce_sat(I_rms_per_module, Tj)
            Vf = self.igbt_model.get_diode_Vf(I_rms_per_module, Tj)
            if np.isnan(Vce_sat) or np.isnan(Vf):
                return {'total': np.nan, 'switching': np.nan, 'conduction_igbt': np.nan, 
                       'conduction_diode': np.nan, 'capacitor': np.nan}
            
            # 基于调制比的占空比计算
            modulation_index = 0.95  # 典型值
            duty_cycle_igbt = modulation_index * 0.5
            duty_cycle_diode = (1 - modulation_index) * 0.5
            
            P_cond_igbt = Vce_sat * I_rms_per_module * duty_cycle_igbt * 6
            P_cond_diode = Vf * I_rms_per_module * duty_cycle_diode * 6
            
            # 电容器损耗 - 基于实际电容器参数
            ESR = self.cap_model.get_ESR(freq=fsw, temp=Tj)
            if np.isnan(ESR):
                return {'total': np.nan, 'switching': np.nan, 'conduction_igbt': np.nan, 
                       'conduction_diode': np.nan, 'capacitor': np.nan}
            
            # 电容器电流 - 基于开关频率和纹波电流
            # 纹波电流约为模块电流的15-20%
            ripple_factor = 0.18
            I_cap_rms = I_rms_per_module * ripple_factor
            P_cap = I_cap_rms**2 * ESR * modules_per_phase
            
            # 总损耗
            P_total = P_sw_igbt + P_cond_igbt + P_cond_diode + P_cap
            
            return {
                'total': P_total,
                'switching': P_sw_igbt,
                'conduction_igbt': P_cond_igbt,
                'conduction_diode': P_cond_diode,
                'capacitor': P_cap
            }
            
        except Exception as e:
            print(f"计算总损耗时出错: {e}")
            return {'total': np.nan, 'switching': np.nan, 'conduction_igbt': np.nan, 
                   'conduction_diode': np.nan, 'capacitor': np.nan}
    
    def optimize_switching_frequency(self, P_out, Vdc, Tj=25):
        """优化开关频率 - 优化版本"""
        try:
            if not all(isinstance(x, (int, float)) for x in [P_out, Vdc]):
                raise TypeError("P_out和Vdc必须是数值类型")
            
            if P_out < 0 or Vdc < 0:
                raise ValueError("P_out和Vdc必须大于等于0")
            
            def total_loss(fsw):
                try:
                    losses = self.calculate_total_losses(P_out, Vdc, fsw[0], Tj=Tj)
                    return losses['total'] if not np.isnan(losses['total']) else 1e6
                except:
                    return 1e6
            
            # 优化开关频率 (100Hz - 10kHz)
            result = minimize(total_loss, x0=[1000], bounds=[(100, 10000)], 
                            method='L-BFGS-B', options={'maxiter': 100})
            
            if result.success:
                return result.x[0], total_loss([result.x[0]])
            else:
                print("开关频率优化失败，使用默认值")
                return 1000, total_loss([1000])
                
        except Exception as e:
            print(f"优化开关频率时出错: {e}")
            return 1000, np.nan

def plot_device_characteristics():
    """绘制器件特性曲线 - 优化版本"""
    try:
        igbt_model = AdvancedIGBTModel()
        cap_model = AdvancedCapacitorModel()
        
        # 使用自适应绘图工具创建图形
        fig, axes = create_adaptive_figure(2, 3, title='35 kV/25 MW PCS器件特性分析', title_size=16)
        
        # IGBT饱和压降特性
        Ic_range = np.linspace(100, 1500, 50)
        Tj_range = [25, 75, 125]
        
        for Tj in Tj_range:
            Vce_sat = [igbt_model.get_Vce_sat(Ic, Tj) for Ic in Ic_range]
            if not any(np.isnan(x) for x in Vce_sat):
                axes[0, 0].plot(Ic_range, Vce_sat, label=f'Tj={Tj}°C', linewidth=2)
        
        format_axis_labels(axes[0, 0], '集电极电流 (A)', '饱和压降 (V)', 'IGBT饱和压降特性')
        axes[0, 0].legend(fontsize=8, loc='best')
        add_grid(axes[0, 0])
        set_adaptive_ylim(axes[0, 0], [0, None])  # 压降从0开始
        
        # 开关损耗特性
        Vdc_range = np.linspace(600, 1500, 50)
        Ic_test = 1000
        
        Eon_list = []
        Eoff_list = []
        for Vdc in Vdc_range:
            Eon, Eoff = igbt_model.get_switching_losses(Ic_test, Vdc)
            if not (np.isnan(Eon) or np.isnan(Eoff)):
                Eon_list.append(Eon * 1e3)  # 转换为mJ
                Eoff_list.append(Eoff * 1e3)
            else:
                Eon_list.append(0)
                Eoff_list.append(0)
        
        if Eon_list and Eoff_list:
            axes[0, 1].plot(Vdc_range, Eon_list, 'b-', label='开通损耗', linewidth=2)
            axes[0, 1].plot(Vdc_range, Eoff_list, 'r-', label='关断损耗', linewidth=2)
            format_axis_labels(axes[0, 1], '直流电压 (V)', '开关损耗 (mJ)', f'开关损耗特性 (Ic={Ic_test}A)')
            axes[0, 1].legend(fontsize=8, loc='best')
            add_grid(axes[0, 1])
            set_adaptive_ylim(axes[0, 1], np.concatenate([Eon_list, Eoff_list]))
        
        # 二极管正向压降特性
        If_range = np.linspace(100, 1500, 50)
        
        for Tj in Tj_range:
            Vf = [igbt_model.get_diode_Vf(If, Tj) for If in If_range]
            if not any(np.isnan(x) for x in Vf):
                axes[0, 2].plot(If_range, Vf, label=f'Tj={Tj}°C', linewidth=2)
        
        format_axis_labels(axes[0, 2], '正向电流 (A)', '正向压降 (V)', '二极管正向压降特性')
        axes[0, 2].legend(fontsize=8, loc='best')
        add_grid(axes[0, 2])
        set_adaptive_ylim(axes[0, 2], [0, None])  # 压降从0开始
        
        # 电容器ESR频率特性
        freq_range = np.logspace(2, 5, 50)
        temp_range = [25, 50, 70]
        
        for temp in temp_range:
            ESR = [cap_model.get_ESR(freq, temp) * 1e3 for freq in freq_range]  # 转换为mΩ
            if not any(np.isnan(x) for x in ESR):
                axes[1, 0].semilogx(freq_range, ESR, label=f'T={temp}°C', linewidth=2)
        
        format_axis_labels(axes[1, 0], '频率 (Hz)', 'ESR (mΩ)', '电容器ESR频率特性')
        axes[1, 0].legend(fontsize=8, loc='best')
        add_grid(axes[1, 0])
        set_adaptive_ylim(axes[1, 0], [0, None])  # ESR从0开始
        
        # 电容器电容值温度特性
        temp_range = np.linspace(-40, 85, 50)
        cap_ratio = [cap_model.get_capacitance(temp) / cap_model.params.get_capacitance() for temp in temp_range]
        
        if not any(np.isnan(x) for x in cap_ratio):
            axes[1, 1].plot(temp_range, cap_ratio, 'g-', linewidth=2)
            format_axis_labels(axes[1, 1], '温度 (°C)', '电容值比例', '电容器电容值温度特性')
            add_grid(axes[1, 1])
            set_adaptive_ylim(axes[1, 1], cap_ratio)
        
        # 功率损耗优化
        P_range = np.linspace(1e6, 25e6, 20)  # 减少点数以提高性能
        loss_model = AdvancedPowerLossModel()
        
        optimal_fsw = []
        min_losses = []
        
        for P in P_range:
            try:
                fsw_opt, loss_min = loss_model.optimize_switching_frequency(P, 1000)
                if not np.isnan(fsw_opt) and not np.isnan(loss_min):
                    optimal_fsw.append(fsw_opt)
                    min_losses.append(loss_min / 1e3)  # 转换为kW
                else:
                    optimal_fsw.append(1000)
                    min_losses.append(0)
            except:
                optimal_fsw.append(1000)
                min_losses.append(0)
        
        if optimal_fsw and min_losses:
            axes[1, 2].plot(P_range / 1e6, optimal_fsw, 'b-', label='最优开关频率', linewidth=2)
            format_axis_labels(axes[1, 2], '功率 (MW)', '开关频率 (Hz)', '功率损耗优化')
            axes[1, 2].legend(fontsize=8, loc='best')
            add_grid(axes[1, 2])
            set_adaptive_ylim(axes[1, 2], optimal_fsw)
            
            # 添加损耗曲线
            ax_twin = axes[1, 2].twinx()
            ax_twin.plot(P_range / 1e6, min_losses, 'r--', label='最小损耗', linewidth=2)
            ax_twin.set_ylabel('损耗 (kW)', color='r', fontsize=9)
            ax_twin.tick_params(axis='y', labelcolor='r')
            ax_twin.legend(fontsize=8, loc='upper right')
            set_adaptive_ylim(ax_twin, min_losses)
        
        # 优化布局，避免重叠
        optimize_layout(fig, tight_layout=True, h_pad=1.5, w_pad=1.5)
        
        # 显示图形
        finalize_plot(fig)
        
    except Exception as e:
        print(f"绘制器件特性曲线时出错: {e}")

if __name__ == "__main__":
    print("35 kV/25 MW级联储能PCS高级器件建模")
    print("=" * 60)
    
    try:
        # 测试器件模型
        igbt_model = AdvancedIGBTModel()
        cap_model = AdvancedCapacitorModel()
        loss_model = AdvancedPowerLossModel()
        
        print(f"IGBT型号: {igbt_model.params.model}")
        print(f"电容器型号: {cap_model.params.manufacturer}")
        
        # 测试损耗计算
        P_test = 10e6  # 10 MW
        Vdc_test = 1000  # 1000 V
        fsw_test = 1000  # 1000 Hz
        
        losses = loss_model.calculate_total_losses(P_test, Vdc_test, fsw_test)
        
        if not any(np.isnan(v) for v in losses.values()):
            print(f"\n功率损耗分析 (P={P_test/1e6:.1f}MW):")
            print(f"  总损耗: {losses['total']/1e3:.2f} kW")
            print(f"  开关损耗: {losses['switching']/1e3:.2f} kW")
            print(f"  IGBT导通损耗: {losses['conduction_igbt']/1e3:.2f} kW")
            print(f"  二极管导通损耗: {losses['conduction_diode']/1e3:.2f} kW")
            print(f"  电容器损耗: {losses['capacitor']/1e3:.2f} kW")
        else:
            print("\n功率损耗计算失败，请检查参数")
        
        # 绘制特性曲线
        plot_device_characteristics()
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc() 