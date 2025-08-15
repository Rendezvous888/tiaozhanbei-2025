#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试SOC变化问题 - 分析为什么绘图中SOC没有变化
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcs_simulation_model import PCSSimulation
from load_profile import generate_profiles

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def debug_soc_calculation():
    """调试SOC计算过程"""
    print("=" * 60)
    print("调试SOC变化问题")
    print("=" * 60)
    
    # 创建PCS仿真实例
    print("1. 初始化PCS仿真系统...")
    pcs_sim = PCSSimulation()
    
    # 打印关键参数
    print(f"\n系统参数:")
    print(f"- 电池容量: {pcs_sim.params.C_battery:.1f} Ah")
    print(f"- 电池电压: {pcs_sim.params.V_battery:.1f} V")
    print(f"- 总能量容量: {pcs_sim.params.C_battery * pcs_sim.params.V_battery / 1e6:.1f} MWh")
    print(f"- SOC范围: {pcs_sim.params.SOC_min*100:.0f}% - {pcs_sim.params.SOC_max*100:.0f}%")
    print(f"- 初始SOC: {pcs_sim.battery.SOC*100:.1f}%")
    
    # 生成简短的测试功率曲线
    print("\n2. 生成测试功率曲线...")
    step_seconds = 60
    # 只测试4小时，功率变化更明显
    time_hours = 4
    P_profile, T_amb = generate_profiles(day_type="summer-weekday", step_seconds=step_seconds)
    
    # 截取前4小时的数据
    num_points = int(time_hours * 3600 / step_seconds)
    P_profile = P_profile[:num_points]
    T_amb = T_amb[:num_points]
    t = np.arange(len(P_profile)) * (step_seconds / 3600.0)
    
    print(f"- 时间点数: {len(P_profile)}")
    print(f"- 功率范围: {P_profile.min()/1e6:.1f} - {P_profile.max()/1e6:.1f} MW")
    
    # 手动计算预期SOC变化用于对比
    print("\n3. 手动计算预期SOC变化...")
    dt_h = step_seconds / 3600.0
    energy_cumulative_mwh = np.cumsum(P_profile) * dt_h / 1e6  # MWh
    total_energy_capacity_mwh = pcs_sim.params.C_battery * pcs_sim.params.V_battery / 1e6
    
    # 预期SOC变化（假设从50%开始）
    expected_soc = 0.5 - (energy_cumulative_mwh / total_energy_capacity_mwh)
    expected_soc = np.clip(expected_soc, 0, 1)
    expected_soc_range = expected_soc.max() - expected_soc.min()
    
    print(f"- 累积能量变化: {energy_cumulative_mwh.min():.1f} - {energy_cumulative_mwh.max():.1f} MWh")
    print(f"- 总能量容量: {total_energy_capacity_mwh:.1f} MWh")
    print(f"- 预期SOC范围: {expected_soc.min()*100:.1f}% - {expected_soc.max()*100:.1f}%")
    print(f"- 预期SOC变化幅度: {expected_soc_range*100:.1f}%")
    
    # 运行仿真
    print("\n4. 运行仿真...")
    results = pcs_sim.run_simulation(t, P_profile, T_amb_profile=T_amb)
    
    # 分析SOC结果
    soc_actual = results['SOC']
    soc_actual_range = soc_actual.max() - soc_actual.min()
    
    print(f"\n5. 分析SOC结果:")
    print(f"- 实际SOC范围: {soc_actual.min()*100:.1f}% - {soc_actual.max()*100:.1f}%")
    print(f"- 实际SOC变化幅度: {soc_actual_range*100:.1f}%")
    print(f"- SOC是否有变化: {'是' if soc_actual_range > 0.001 else '否'}")
    
    # 检查SOC更新逻辑
    print(f"\n6. 检查SOC更新机制:")
    print(f"- 是否使用battery_module: {pcs_sim.battery_module is not None}")
    print(f"- soc_from_grid_power配置: {pcs_sim.params.soc_from_grid_power}")
    
    # 分析功率有效值
    power_effective = results.get('power_effective', results['power'])
    power_eff_range = power_effective.max() - power_effective.min()
    print(f"- 有效功率范围: {power_effective.min()/1e6:.1f} - {power_effective.max()/1e6:.1f} MW")
    print(f"- 有效功率变化幅度: {power_eff_range/1e6:.1f} MW")
    
    # 绘制详细对比图
    print("\n7. 绘制调试图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SOC变化调试分析', fontsize=16)
    
    # 子图1: 功率曲线
    ax1 = axes[0, 0]
    ax1.plot(t, P_profile / 1e6, 'b-', linewidth=2, label='输入功率')
    ax1.plot(t, power_effective / 1e6, 'r--', linewidth=1, alpha=0.7, label='有效功率')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('功率 (MW)')
    ax1.set_title('功率曲线对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: SOC对比
    ax2 = axes[0, 1]
    ax2.plot(t, expected_soc * 100, 'g-', linewidth=2, label='预期SOC')
    ax2.plot(t, soc_actual * 100, 'r-', linewidth=2, label='实际SOC')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('SOC (%)')
    ax2.set_title('SOC变化对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 子图3: 累积能量
    ax3 = axes[1, 0]
    ax3.plot(t, energy_cumulative_mwh, 'purple', linewidth=2)
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('累积能量 (MWh)')
    ax3.set_title('累积能量变化')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 系统信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    info_text = f"""调试信息:
    
电池参数:
• 容量: {pcs_sim.params.C_battery:.0f} Ah
• 电压: {pcs_sim.params.V_battery:.0f} V
• 总能量: {total_energy_capacity_mwh:.1f} MWh
• SOC范围: {pcs_sim.params.SOC_min*100:.0f}%-{pcs_sim.params.SOC_max*100:.0f}%

功率统计:
• 输入功率范围: {P_profile.min()/1e6:.1f} - {P_profile.max()/1e6:.1f} MW
• 有效功率范围: {power_effective.min()/1e6:.1f} - {power_effective.max()/1e6:.1f} MW

SOC变化分析:
• 预期变化: {expected_soc_range*100:.2f}%
• 实际变化: {soc_actual_range*100:.2f}%
• 问题诊断: {'SOC更新异常' if soc_actual_range < 0.001 else 'SOC正常更新'}

系统配置:
• 使用battery_module: {pcs_sim.battery_module is not None}
• soc_from_grid_power: {pcs_sim.params.soc_from_grid_power}"""
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存调试图表
    os.makedirs('Debug', exist_ok=True)
    plt.savefig('Debug/soc_debug_analysis.png', dpi=300, bbox_inches='tight')
    print("调试图表已保存到: Debug/soc_debug_analysis.png")
    
    plt.show()
    
    # 输出诊断结果
    print(f"\n8. 问题诊断:")
    print("-" * 40)
    
    if soc_actual_range < 0.001:
        print("❌ 发现问题：SOC几乎没有变化")
        
        # 可能的原因分析
        if power_eff_range < 1e6:  # 有效功率变化小于1MW
            print("❌ 可能原因1：有效功率被过度裁剪，导致实际功率变化很小")
        
        if total_energy_capacity_mwh > 1000:  # 电池容量过大
            print("❌ 可能原因2：电池容量过大，相对于功率变化，SOC变化微小")
        
        if not hasattr(pcs_sim, 'battery_module') or pcs_sim.battery_module is None:
            print("❌ 可能原因3：使用简化电池模型，SOC更新逻辑可能有问题")
            
        # 建议解决方案
        print("\n💡 建议解决方案:")
        print("1. 检查电池容量配置是否合理")
        print("2. 确认功率裁剪逻辑是否过于激进")
        print("3. 验证SOC更新计算公式")
        print("4. 考虑调整时间步长或功率幅度")
        
    else:
        print("✅ SOC变化正常")
        print(f"✅ SOC变化幅度: {soc_actual_range*100:.2f}%")

if __name__ == "__main__":
    debug_soc_calculation()
