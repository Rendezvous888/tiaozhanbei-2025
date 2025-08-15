#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
35 kV/25 MW级联储能PCS设备参数配置
基于Infineon FF1500R17IP5R IGBT和Xiamen Farah/Nantong Jianghai电容器的详细参数
"""

import numpy as np

class SystemParameters:
    """系统级参数配置"""
    def __init__(self):
        # 系统电压和频率范围
        self.system_voltage_kV = [30, 40.5]  # 系统电压范围 (kV)
        self.system_frequency_Hz = [47.5, 52.5]  # 系统频率范围 (Hz)
        # 全局时间步长（秒）
        self.time_step_seconds = 60
        
        # 级联H桥配置（比赛方案口径）：按比赛方案要求每相40个H桥级联
        # 注：严格按照比赛方案"每相H桥级联40个"的要求
        self.cascaded_power_modules = 40  # 每相级联模块数（比赛方案要求）
        self.module_switching_frequency_Hz = 1000  # 模块开关频率 (Hz)
        self.rated_current_A = 420  # 额定电流 (A)
        self.overload_capability_pu = "3 pu / 10s"  # 过载能力
        
        # 电池配置
        self.battery_capacity_Ah = 314  # 电池容量 (Ah)
        self.battery_series_cells = 312  # 串联电池数
        self.battery_note = "Ningde or other domestic/foreign suppliers"
        # 目标系统能量（以小时计），用于推导并联倍数（恢复为2h系统，保持合理的储能容量）
        self.energy_hours = 2.0  # 恢复为2小时系统
        # 额外系统损耗（辅助、滤波、变压等）按输出功率比例估计
        self.misc_loss_fraction = 0.03  # ~3%
        # 固定辅助损耗（风机、控制器等），W
        self.aux_loss_w = 100_000.0
        
        # 模块配置
        self.module_dc_bus_capacitance_mF = 15  # 模块直流母线电容 (mF)
        self.dc_filter_mH = 0.5  # 直流滤波器电感 (mH)
        
        # 冷却和环境参数
        self.water_cooling_inlet_temperature_C = 25  # 水冷入口温度 (°C)
        # 项目要求：环境工作温度20-40°C，设计验证需覆盖全范围
        self.ambient_temperature_C = [20, 40]  # 环境温度范围 (°C) - 符合项目要求
        self.power_device = "1700V IGBT"  # 功率器件类型

class IGBTParameters:
    """Infineon FF1500R17IP5R IGBT详细参数"""
    def __init__(self):
        # 基本信息
        self.model = "Infineon FF1500R17IP5R"
        self.type = "PrimePACK3 module (Trench/Fieldstop IGBT5 + EC diode + NTC)"
        self.per_module_quantity = 2  # 每模块IGBT数量
        
        # 额定参数
        self.Vces_V = 1700  # 集电极-发射极电压 (V)
        self.Ic_dc_A = 1500  # 直流集电极电流 (A)
        self.Icrm_A = 3000  # 重复峰值集电极电流 (A)
        self.Vges_V = "±20"  # 栅极-发射极电压 (V)
        self.short_circuit_current_A = 6000  # 短路电流 (A)
        self.junction_temperature_C = [-40, 175]  # 结温范围 (°C)
        
        # 电气特性 - 饱和压降
        self.Vce_sat_V = {
            "25C": [1.75, 2.30],  # 25°C时的饱和压降范围 (V)
            "125C": [2.20, 2.90]  # 125°C时的饱和压降范围 (V)
        }
        
        # 栅极特性
        self.Vge_th_V = [5.35, 6.25]  # 栅极阈值电压 (V)
        self.Qg_uC = 7.5  # 栅极电荷 (μC)
        self.Rg_internal_Ohm = 1.0  # 内部栅极电阻 (Ω)
        self.Cies_nF = 88  # 输入电容 (nF)
        self.Cres_nF = 2.7  # 反向传输电容 (nF)
        self.Ices_mA = 10  # 栅极-发射极漏电流 (mA)
        
        # 开关特性
        self.switching_times_us = {
            "td_on": [0.30, 0.32],  # 开通延迟时间 (μs)
            "tr": [0.15, 0.16],     # 上升时间 (μs)
            "td_off": [0.66, 0.80], # 关断延迟时间 (μs)
            "tf": [0.11, 0.17]      # 下降时间 (μs)
        }
        
        # 开关损耗
        self.switching_energy_mJ = {
            "Eon": [335, 595],   # 开通损耗 (mJ)
            "Eoff": [330, 545]   # 关断损耗 (mJ)
        }
        
        # 二极管特性
        self.diode_Vrrm_V = 1700  # 反向重复峰值电压 (V)
        self.diode_If_dc_A = 1500  # 直流正向电流 (A)
        self.diode_Ifrm_A = 3000  # 重复峰值正向电流 (A)
        self.diode_I2t_kA2s = [485, 580]  # I²t值 (kA²s)
        self.diode_Vf_V = [1.75, 2.10]  # 正向压降 (V)
        self.diode_Irm_A = [1250, 1550]  # 反向恢复电流 (A)
        self.diode_Qr_uC = [325, 700]  # 反向恢复电荷 (μC)
        self.diode_Erec_mJ = [185, 425]  # 反向恢复损耗 (mJ)
        
        # NTC热敏电阻特性
        self.NTC_resistance_25C_kOhm = 5  # 25°C时电阻 (kΩ)
        self.NTC_deviation_R100_percent = 5  # R100偏差 (%)
        self.NTC_power_dissipation_25C_mW = 20  # 25°C时功耗 (mW)
        self.NTC_B_value_K = {
            "25_50": 3375,   # 25-50°C B值 (K)
            "25_80": 3411,   # 25-80°C B值 (K)
            "25_100": 3433   # 25-100°C B值 (K)
        }
        
        # 机械特性
        self.isolation_voltage_kV = 4.0  # 隔离电压 (kV)
        self.baseplate_material = "Cu"  # 基板材料
        self.creepage_distance_mm = 33  # 爬电距离 (mm)
        self.clearance_mm = 19  # 电气间隙 (mm)
        self.CTI = ">400"  # 相比漏电起痕指数
        self.stray_inductance_nH = 10  # 杂散电感 (nH)
        self.internal_lead_resistance_mOhm = 0.18  # 内部引线电阻 (mΩ)
        self.storage_temperature_C = [-40, 150]  # 存储温度 (°C)
        self.max_baseplate_temperature_C = 150  # 最大基板温度 (°C)
        self.mounting_torque_Nm = {
            "M5": [3.0, 6.0],   # M5螺钉扭矩 (Nm)
            "M4": [1.8, 2.1],   # M4螺钉扭矩 (Nm)
            "M8": [8.0, 10.0]   # M8螺钉扭矩 (Nm)
        }
        self.weight_g = 1200  # 重量 (g)

class CapacitorParameters:
    """直流母线电容器参数"""
    def __init__(self, manufacturer="Xiamen Farah"):
        self.manufacturer = manufacturer
        self.quantity = 21  # 电容器数量
        
        # 电容器选项
        self.options = {
            "Xiamen Farah": {
                "capacitance_uF": 720,  # 电容值 (μF)
                "voltage_V": 1200,      # 额定电压 (V)
                "max_current_A": 80,    # 最大电流 (A)
                "ESR_mOhm": 1.2,        # 等效串联电阻 (mΩ)
                "LS_nH": 45,            # 等效串联电感 (nH)
                "lifetime_h": 100000,   # 寿命 (小时)
                "operating_temperature_C": [-40, 70],  # 工作温度 (°C)
                "dimensions_mm": "D116×H174"  # 尺寸 (mm)
            },
            "Nantong Jianghai": {
                "capacitance_uF": 720,  # 电容值 (μF)
                "voltage_V": 1200,      # 额定电压 (V)
                "max_current_A": 86,    # 最大电流 (A)
                "ESR_mOhm": 1.2,        # 等效串联电阻 (mΩ)
                "LS_nH": 60,            # 等效串联电感 (nH)
                "lifetime_h": 100000,   # 寿命 (小时)
                "operating_temperature_C": [-40, 85],  # 工作温度 (°C)
                "dimensions_mm": "D116×H175"  # 尺寸 (mm)
            }
        }
        
        # 当前选择的电容器参数
        self.current_params = self.options[manufacturer]
        
    def get_capacitance(self):
        """获取电容值 (F)"""
        return self.current_params["capacitance_uF"] * 1e-6
    
    def get_ESR(self):
        """获取ESR (Ω)"""
        return self.current_params["ESR_mOhm"] * 1e-3
    
    def get_ESL(self):
        """获取ESL (H)"""
        return self.current_params["LS_nH"] * 1e-9
    
    def get_lifetime(self):
        """获取寿命 (小时)"""
        return self.current_params["lifetime_h"]

class ThermalParameters:
    """热管理参数"""
    def __init__(self):
        # 热阻参数 (基于IGBT数据手册)
        self.Rth_jc = 0.05  # 结到壳热阻 (K/W)
        self.Rth_ca = 0.1   # 壳到环境热阻 (K/W)
        self.Rth_ja = self.Rth_jc + self.Rth_ca  # 结到环境热阻 (K/W)
        
        # 热容参数
        self.Cth_jc = 100   # 结到壳热容 (J/K)
        self.Cth_ca = 500   # 壳到环境热容 (J/K)
        
        # 温度参数
        self.T_amb = 25     # 环境温度 (°C)
        self.Tj_max = 175   # 最大结温 (°C) - 基于IGBT规格
        self.Tj_min = -40   # 最小结温 (°C)
        
        # 冷却参数
        self.water_cooling_inlet_temp = 25  # 水冷入口温度 (°C)
        self.cooling_efficiency = 0.85      # 冷却效率

class ControlParameters:
    """控制参数"""
    def __init__(self):
        # PWM控制参数
        self.modulation_index_max = 0.95  # 最大调制比
        self.modulation_index_min = 0.1   # 最小调制比
        self.carrier_frequency_Hz = 1000  # 载波频率 (Hz)
        
        # 电流控制参数
        self.current_bandwidth_Hz = 500   # 电流环带宽 (Hz)
        self.voltage_bandwidth_Hz = 100   # 电压环带宽 (Hz)
        self.power_bandwidth_Hz = 50      # 功率环带宽 (Hz)
        
        # 保护参数
        self.overcurrent_threshold_pu = 3.0  # 过流阈值 (标幺值)
        self.overvoltage_threshold_pu = 1.1  # 过压阈值 (标幺值)
        self.overtemperature_threshold_C = 170  # 过温阈值 (°C)

def get_optimized_parameters():
    """获取优化后的系统参数"""
    system = SystemParameters()
    igbt = IGBTParameters()
    capacitor = CapacitorParameters()
    thermal = ThermalParameters()
    control = ControlParameters()
    
    return {
        'system': system,
        'igbt': igbt,
        'capacitor': capacitor,
        'thermal': thermal,
        'control': control
    }

def print_device_summary():
    """打印设备参数摘要"""
    params = get_optimized_parameters()
    
    print("=" * 80)
    print("35 kV/25 MW级联储能PCS设备参数摘要")
    print("=" * 80)
    
    print(f"\n系统配置:")
    print(f"  • 级联模块数: {params['system'].cascaded_power_modules}")
    print(f"  • 额定电流: {params['system'].rated_current_A} A")
    print(f"  • 开关频率: {params['system'].module_switching_frequency_Hz} Hz")
    print(f"  • 电池容量: {params['system'].battery_capacity_Ah} Ah")
    
    print(f"\nIGBT器件 (Infineon FF1500R17IP5R):")
    print(f"  • 额定电压: {params['igbt'].Vces_V} V")
    print(f"  • 额定电流: {params['igbt'].Ic_dc_A} A")
    print(f"  • 结温范围: {params['igbt'].junction_temperature_C[0]}°C ~ {params['igbt'].junction_temperature_C[1]}°C")
    print(f"  • 开关损耗: Eon={params['igbt'].switching_energy_mJ['Eon'][0]}-{params['igbt'].switching_energy_mJ['Eon'][1]} mJ")
    print(f"  • 关断损耗: Eoff={params['igbt'].switching_energy_mJ['Eoff'][0]}-{params['igbt'].switching_energy_mJ['Eoff'][1]} mJ")
    
    print(f"\n电容器 ({params['capacitor'].manufacturer}):")
    print(f"  • 电容值: {params['capacitor'].current_params['capacitance_uF']} μF")
    print(f"  • 额定电压: {params['capacitor'].current_params['voltage_V']} V")
    print(f"  • ESR: {params['capacitor'].current_params['ESR_mOhm']} mΩ")
    print(f"  • 寿命: {params['capacitor'].current_params['lifetime_h']} 小时")
    
    print(f"\n热管理:")
    print(f"  • 环境温度: {params['thermal'].T_amb}°C")
    print(f"  • 最大结温: {params['thermal'].Tj_max}°C")
    print(f"  • 结到环境热阻: {params['thermal'].Rth_ja:.3f} K/W")
    
    print("=" * 80)

if __name__ == "__main__":
    print_device_summary() 