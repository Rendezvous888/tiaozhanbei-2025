#!/usr/bin/env python3
"""
BatteryModel 功能测试脚本

测试 35 kV/25 MW 级联储能 PCS 电池模型的各项功能：
- 基本参数设置
- SOC 更新和电压计算
- 温度影响
- 过载工况
- 寿命预测
"""

from battery_model import BatteryModel, BatteryModelConfig
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import set_chinese_plot_style, save_chinese_plot

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 创建电池模型
    battery = BatteryModel(initial_soc=0.8, initial_temperature_c=25.0)
    
    print(f"初始SOC: {battery.state_of_charge:.1%}")
    print(f"初始温度: {battery.cell_temperature_c:.1f}°C")
    print(f"额定容量: {battery.config.rated_capacity_ah:.1f} Ah")
    print(f"额定电流: {battery.config.rated_current_a:.1f} A")
    print(f"串联电芯数: {battery.config.series_cells}")
    
    # 测试电压计算
    voltage = battery.get_voltage()
    print(f"初始电压: {voltage:.1f} V")
    
    return battery

def test_soc_update():
    """测试SOC更新"""
    print("\n=== 测试SOC更新 ===")
    
    battery = BatteryModel(initial_soc=0.8, initial_temperature_c=25.0)
    
    # 模拟放电过程
    discharge_current = battery.config.rated_current_a
    dt = 1.0  # 1秒步长
    
    print("时间(min) | SOC(%) | 电压(V) | 温度(°C)")
    print("-" * 40)
    
    for i in range(0, 61, 5):  # 每5分钟记录一次
        if i > 0:
            battery.update_state(discharge_current, i * 60, 25.0)
        
        status = battery.get_battery_status()
        print(f"{i:8d} | {status['soc']:6.1%} | {status['voltage_v']:7.1f} | {status['cell_temperature_c']:8.1f}")

def test_temperature_effects():
    """测试温度影响"""
    print("\n=== 测试温度影响 ===")
    
    battery = BatteryModel(initial_soc=0.5, initial_temperature_c=25.0)
    
    temperatures = [20, 25, 30, 35, 40]
    print("温度(°C) | 容量(Ah) | 内阻(Ω) | OCV(V)")
    print("-" * 40)
    
    for temp in temperatures:
        battery.cell_temperature_c = temp
        battery.ambient_temperature_c = temp
        
        status = battery.get_battery_status()
        print(f"{temp:8.0f} | {status['effective_capacity_ah']:9.1f} | {status['resistance_ohm']:8.3f} | {status['ocv_v']:7.1f}")

def test_overload_conditions():
    """测试过载工况"""
    print("\n=== 测试过载工况 ===")
    
    battery = BatteryModel(initial_soc=0.8, initial_temperature_c=25.0)
    
    # 测试不同过载倍率
    overload_ratios = [1.0, 1.5, 2.0, 2.5, 3.0]
    print("过载倍率 | 电流(A) | 电压(V) | 温度(°C) | 安全状态")
    print("-" * 55)
    
    for ratio in overload_ratios:
        # 重置状态
        battery.state_of_charge = 0.8
        battery.cell_temperature_c = 25.0
        
        # 运行过载工况
        overload_current = battery.config.rated_current_a * ratio
        battery.update_state(overload_current, 300, 30.0)  # 5分钟过载
        
        # 检查安全状态
        safety = battery.check_safety_limits()
        status = battery.get_battery_status()
        
        print(f"{ratio:8.1f} | {overload_current:8.0f} | {status['voltage_v']:8.1f} | "
              f"{status['cell_temperature_c']:8.1f} | {'安全' if safety['is_safe'] else '警告'}")

def test_life_prediction():
    """测试寿命预测"""
    print("\n=== 测试寿命预测 ===")
    
    battery = BatteryModel(initial_soc=0.5, initial_temperature_c=25.0)
    
    # 模拟加速老化测试
    print("循环次数 | 容量衰减(%) | 健康度(%) | 剩余寿命(年)")
    print("-" * 50)
    
    for cycle in range(0, 1001, 100):
        if cycle > 0:
            # 模拟一次完整循环
            battery.update_state(battery.config.rated_current_a, 2 * 3600, 30.0)  # 2小时放电
            battery.update_state(-battery.config.rated_current_a, 2 * 3600, 30.0)  # 2小时充电
        
        life_estimate = battery.estimate_remaining_life()
        status = battery.get_battery_status()
        
        print(f"{cycle:8d} | {status['capacity_fade_percent']:12.2f} | {life_estimate['health_percentage']:10.1f} | "
              f"{life_estimate['remaining_life_years']:14.1f}")

def plot_discharge_curves():
    """绘制放电曲线"""
    print("\n=== 绘制放电曲线 ===")
    
    # 使用绘图工具模块配置中文字体
    set_chinese_plot_style()
    
    battery = BatteryModel(initial_soc=0.8, initial_temperature_c=25.0)
    
    # 不同温度下的放电曲线
    temperatures = [20, 25, 30, 35, 40]
    discharge_data = {}
    
    for temp in temperatures:
        # 重置状态
        battery.state_of_charge = 0.8
        battery.cell_temperature_c = temp
        battery.ambient_temperature_c = temp
        
        # 模拟恒功率放电
        power_w = 500000  # 500 kW
        result = battery.simulate_constant_power_discharge(power_w, temp, 2.0)
        discharge_data[temp] = result
    
    # 绘制SOC-电压曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for temp in temperatures:
        data = discharge_data[temp]
        plt.plot(data['soc'], data['voltage_v'], label=f'{temp}°C', linewidth=2)
    plt.xlabel('SOC')
    plt.ylabel('电压 (V)')
    plt.title('不同温度下的SOC-电压曲线')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for temp in temperatures:
        data = discharge_data[temp]
        plt.plot(data['time_hours'], data['soc'], label=f'{temp}°C', linewidth=2)
    plt.xlabel('时间 (小时)')
    plt.ylabel('SOC')
    plt.title('不同温度下的放电时间曲线')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for temp in temperatures:
        data = discharge_data[temp]
        plt.plot(data['time_hours'], data['temperature_c'], label=f'{temp}°C', linewidth=2)
    plt.xlabel('时间 (小时)')
    plt.ylabel('电池温度 (°C)')
    plt.title('不同温度下的温升曲线')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for temp in temperatures:
        data = discharge_data[temp]
        plt.plot(data['time_hours'], data['current_a'], label=f'{temp}°C', linewidth=2)
    plt.xlabel('时间 (小时)')
    plt.ylabel('电流 (A)')
    plt.title('不同温度下的电流曲线')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 使用新的保存函数
    if save_chinese_plot('battery_discharge_curves.png', dpi=300):
        print("放电曲线图已保存为 'battery_discharge_curves.png'")
    else:
        print("保存图片失败")

def main():
    """主测试函数"""
    print("35 kV/25 MW 级联储能 PCS 电池模型功能测试")
    print("=" * 60)
    
    try:
        # 运行各项测试
        test_basic_functionality()
        test_soc_update()
        test_temperature_effects()
        test_overload_conditions()
        test_life_prediction()
        
        # 绘制图表
        plot_discharge_curves()
        
        print("\n=== 所有测试完成 ===")
        print("电池模型功能验证成功！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
