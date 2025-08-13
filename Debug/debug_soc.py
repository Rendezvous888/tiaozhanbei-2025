#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断SOC计算问题
"""

import numpy as np
from load_profile import generate_profiles
from pcs_simulation_model import PCSSimulation

def debug_soc_calculation():
    """诊断SOC计算问题"""
    print("=== SOC计算问题诊断 ===")
    
    # 创建仿真系统
    pcs_sim = PCSSimulation()
    
    # 检查电池参数
    print(f"\n电池参数:")
    print(f"  电池电压: {pcs_sim.params.V_battery:.1f} V")
    print(f"  电池容量: {pcs_sim.params.C_battery:.1f} Ah")
    print(f"  SOC范围: {pcs_sim.params.SOC_min:.1f} ~ {pcs_sim.params.SOC_max:.1f}")
    
    # 计算电池总能量
    total_energy_wh = pcs_sim.params.V_battery * pcs_sim.params.C_battery
    print(f"  总能量: {total_energy_wh/1e6:.2f} MWh")
    
    # 生成测试数据
    # 读取统一时间步长
    from device_parameters import get_optimized_parameters
    step_seconds = int(get_optimized_parameters()['system'].time_step_seconds)
    P_profile, T_amb = generate_profiles('summer-weekday', step_seconds=step_seconds)
    t = np.arange(len(P_profile)) * (step_seconds / 3600.0)  # 转换为小时
    
    print(f"\n负载曲线分析:")
    print(f"  总点数: {len(P_profile)}")
    print(f"  充电功率范围: {P_profile[P_profile < 0].min()/1e6:.2f} ~ {P_profile[P_profile < 0].max()/1e6:.2f} MW")
    print(f"  放电功率范围: {P_profile[P_profile > 0].min()/1e6:.2f} ~ {P_profile[P_profile > 0].max()/1e6:.2f} MW")
    
    # 计算总充放电能量
    dt_h = step_seconds / 3600.0
    charge_energy = np.sum(P_profile[P_profile < 0]) * dt_h
    discharge_energy = np.sum(P_profile[P_profile > 0]) * dt_h
    
    print(f"  总充电能量: {abs(charge_energy)/1e6:.2f} MWh")
    print(f"  总放电能量: {discharge_energy/1e6:.2f} MWh")
    print(f"  净能量: {(charge_energy + discharge_energy)/1e6:.2f} MWh")
    
    # 检查能量平衡
    if abs(charge_energy + discharge_energy) > 1e6:  # 1 MWh容差
        print(f"  ⚠️  警告: 充放电能量不平衡！")
    
    # 运行仿真并跟踪SOC变化
    print(f"\n运行仿真...")
    results = pcs_sim.run_simulation(t, P_profile, T_amb_profile=T_amb)
    
    # 分析SOC变化
    soc = results['SOC']
    power = results['power']
    
    print(f"\nSOC变化分析:")
    print(f"  初始SOC: {soc[0]:.3f}")
    print(f"  最终SOC: {soc[-1]:.3f}")
    print(f"  SOC变化: {soc[-1] - soc[0]:.3f}")
    print(f"  最小SOC: {soc.min():.3f}")
    print(f"  最大SOC: {soc.max():.3f}")
    
    # 检查SOC为0的时段
    zero_soc_indices = np.where(soc <= 0.01)[0]
    if len(zero_soc_indices) > 0:
        print(f"  ⚠️  SOC接近0的时段: {len(zero_soc_indices)} 个点")
        print(f"      对应时间: {zero_soc_indices[0]/60:.1f} ~ {zero_soc_indices[-1]/60:.1f} 小时")
        print(f"      对应功率: {power[zero_soc_indices[0]]/1e6:.2f} ~ {power[zero_soc_indices[-1]]/1e6:.2f} MW")
    
    # 检查SOC为1的时段
    full_soc_indices = np.where(soc >= 0.99)[0]
    if len(full_soc_indices) > 0:
        print(f"  ⚠️  SOC接近1的时段: {len(full_soc_indices)} 个点")
        print(f"      对应时间: {full_soc_indices[0]/60:.1f} ~ {full_soc_indices[-1]/60:.1f} 小时")
        print(f"      对应功率: {power[full_soc_indices[0]]/1e6:.2f} ~ {power[full_soc_indices[-1]]/1e6:.2f} MW")
    
    # 分析SOC变化率
    soc_changes = np.diff(soc)
    print(f"\nSOC变化率分析:")
    if soc_changes.size > 0:
        print(f"  最大SOC增加: {soc_changes.max():.6f}")
        print(f"  最大SOC减少: {soc_changes.min():.6f}")
        print(f"  平均SOC变化: {np.mean(soc_changes):.6f}")
    else:
        print("  序列过短，无法计算变化率")
    
    # 检查是否有异常的SOC变化
    large_changes = np.where(np.abs(soc_changes) > 0.1)[0]
    if len(large_changes) > 0:
        print(f"  ⚠️  发现异常大的SOC变化: {len(large_changes)} 处")
        for i in large_changes[:5]:  # 显示前5个
            print(f"      时间 {i/60:.1f}h: SOC变化 {soc_changes[i]:.3f}, 功率 {power[i]/1e6:.2f} MW")
    
    return results

def analyze_battery_model():
    """分析详细电池模型"""
    print(f"\n=== 详细电池模型分析 ===")
    
    try:
        from battery_model import BatteryModel
        
        # 创建电池模型
        battery = BatteryModel(initial_soc=0.5, initial_temperature_c=25.0)
        
        print(f"  初始状态:")
        print(f"    SOC: {battery.state_of_charge:.3f}")
        print(f"    温度: {battery.cell_temperature_c:.1f}°C")
        print(f"    端电压: {battery.get_voltage():.1f} V")
        
        # 测试不同电流下的SOC变化
        test_currents = [100, 200, 500, 1000]  # A
        dt = 60.0  # 1分钟
        
        for current in test_currents:
            print(f"\n  测试电流: {current} A (放电)")
            battery.update_state(current, dt, 25.0)
            print(f"    更新后SOC: {battery.state_of_charge:.3f}")
            print(f"    更新后温度: {battery.cell_temperature_c:.1f}°C")
            print(f"    端电压: {battery.get_voltage():.1f} V")
            
            # 重置SOC
            battery.state_of_charge = 0.5
        
        # 测试充电
        print(f"\n  测试充电: -500 A")
        battery.state_of_charge = 0.5
        battery.update_state(-500, dt, 25.0)
        print(f"    更新后SOC: {battery.state_of_charge:.3f}")
        
    except Exception as e:
        print(f"  详细电池模型测试失败: {e}")

if __name__ == "__main__":
    # 运行诊断
    results = debug_soc_calculation()
    
    # 分析详细电池模型
    analyze_battery_model()
    
    print(f"\n=== 诊断完成 ===")
