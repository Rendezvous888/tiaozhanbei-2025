#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试效率和温度异常问题
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

def debug_efficiency_temperature():
    """调试效率和温度计算"""
    print("=" * 60)
    print("调试效率和温度异常问题")
    print("=" * 60)
    
    # 创建PCS仿真实例
    print("1. 初始化PCS仿真系统...")
    pcs_sim = PCSSimulation()
    
    # 检查关键参数
    print(f"\n热模型参数:")
    print(f"- 结到壳热阻: {pcs_sim.params.Rth_jc} K/W")
    print(f"- 壳到环境热阻: {pcs_sim.params.Rth_ca} K/W") 
    print(f"- 结到壳热容: {pcs_sim.params.Cth_jc} J/K")
    print(f"- 壳到环境热容: {pcs_sim.params.Cth_ca} J/K")
    print(f"- 环境温度: {pcs_sim.params.T_amb} °C")
    print(f"- 最大结温: {pcs_sim.params.Tj_max} °C")
    print(f"- 最小结温: {pcs_sim.params.Tj_min} °C")
    
    print(f"\n初始温度:")
    print(f"- 初始结温: {pcs_sim.thermal.Tj} °C")
    print(f"- 初始壳温: {pcs_sim.thermal.Tc} °C")
    
    # 生成测试功率曲线（1小时测试）
    print("\n2. 生成测试功率曲线...")
    step_seconds = 60
    time_hours = 1  # 1小时测试
    P_profile, T_amb = generate_profiles(day_type="summer-weekday", step_seconds=step_seconds)
    
    # 截取前1小时的数据
    num_points = int(time_hours * 3600 / step_seconds)
    P_profile = P_profile[:num_points]
    T_amb = T_amb[:num_points]
    t = np.arange(len(P_profile)) * (step_seconds / 3600.0)
    
    print(f"- 时间点数: {len(P_profile)}")
    print(f"- 功率范围: {P_profile.min()/1e6:.1f} - {P_profile.max()/1e6:.1f} MW")
    
    # 手动测试一步仿真
    print("\n3. 手动测试仿真步骤...")
    
    # 取一个典型功率点进行测试
    P_test = P_profile[30]  # 第30个点
    T_amb_test = T_amb[30]
    dt = t[1] - t[0]  # 时间步长（小时）
    dt_seconds = dt * 3600  # 转换为秒
    
    print(f"测试点功率: {P_test/1e6:.1f} MW")
    print(f"测试点环境温度: {T_amb_test:.1f} °C")
    print(f"时间步长: {dt:.4f} 小时 ({dt_seconds:.0f} 秒)")
    
    # 计算功率器件损耗
    if P_test > 0:  # 放电
        P_loss_conv, P_sw, P_cond, P_cap = pcs_sim.hbridge.calculate_total_losses(P_test, 'discharge')
    else:  # 充电
        P_loss_conv, P_sw, P_cond, P_cap = pcs_sim.hbridge.calculate_total_losses(abs(P_test), 'charge')
    
    # 其它损耗
    P_loss_misc = abs(P_test) * pcs_sim.params.misc_loss_fraction + pcs_sim.params.aux_loss_w
    P_loss_total = P_loss_conv + P_loss_misc
    
    print(f"\n损耗分析:")
    print(f"- 开关损耗: {P_sw/1e3:.1f} kW")
    print(f"- 导通损耗: {P_cond/1e3:.1f} kW")
    print(f"- 电容损耗: {P_cap/1e3:.1f} kW")
    print(f"- 变换器损耗: {P_loss_conv/1e3:.1f} kW")
    print(f"- 其它损耗: {P_loss_misc/1e3:.1f} kW")
    print(f"- 总损耗: {P_loss_total/1e3:.1f} kW")
    
    # 测试温度更新
    print(f"\n温度更新测试:")
    initial_Tj = pcs_sim.thermal.Tj
    initial_Tc = pcs_sim.thermal.Tc
    
    # 更新环境温度
    pcs_sim.params.T_amb = T_amb_test
    Tj_new, Tc_new = pcs_sim.thermal.update_temperature(P_loss_total, dt_seconds)
    
    print(f"- 损耗功率: {P_loss_total/1e3:.1f} kW")
    print(f"- 环境温度: {T_amb_test:.1f} °C")
    print(f"- 结温变化: {initial_Tj:.1f} → {Tj_new:.1f} °C (Δ{Tj_new-initial_Tj:.1f})")
    print(f"- 壳温变化: {initial_Tc:.1f} → {Tc_new:.1f} °C (Δ{Tc_new-initial_Tc:.1f})")
    
    # 测试效率计算
    print(f"\n效率计算测试:")
    P_out = abs(P_test)
    P_batt_abs = P_out + P_loss_total  # 电池侧功率
    efficiency = P_out / P_batt_abs if P_batt_abs > 0 else 0
    
    print(f"- 电网侧功率: {P_out/1e6:.1f} MW")
    print(f"- 电池侧功率: {P_batt_abs/1e6:.1f} MW")
    print(f"- 计算效率: {efficiency*100:.2f}%")
    
    # 运行完整仿真
    print("\n4. 运行完整仿真...")
    results = pcs_sim.run_simulation(t, P_profile, T_amb_profile=T_amb)
    
    # 分析结果
    efficiency_data = results['efficiency']
    Tj_data = results['Tj']
    Tc_data = results['Tc']
    P_loss_data = results['P_loss']
    
    # 过滤有效数据
    valid_eff = efficiency_data[np.isfinite(efficiency_data)]
    
    print(f"\n仿真结果分析:")
    print(f"- 效率数据点数: {len(efficiency_data)}")
    print(f"- 有效效率点数: {len(valid_eff)}")
    if len(valid_eff) > 0:
        print(f"- 效率范围: {valid_eff.min()*100:.2f}% - {valid_eff.max()*100:.2f}%")
        print(f"- 平均效率: {valid_eff.mean()*100:.2f}%")
    else:
        print(f"- 效率数据全为NaN!")
    
    print(f"- 结温范围: {Tj_data.min():.1f} - {Tj_data.max():.1f} °C")
    print(f"- 壳温范围: {Tc_data.min():.1f} - {Tc_data.max():.1f} °C")
    print(f"- 损耗范围: {P_loss_data.min()/1e3:.1f} - {P_loss_data.max()/1e3:.1f} kW")
    
    # 绘制诊断图表
    print("\n5. 绘制诊断图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('效率和温度诊断分析', fontsize=16)
    
    # 子图1: 功率和损耗
    ax1 = axes[0, 0]
    ax1.plot(t, P_profile / 1e6, 'b-', linewidth=2, label='输入功率')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, P_loss_data / 1e3, 'r-', linewidth=1, alpha=0.7, label='系统损耗')
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('功率 (MW)', color='blue')
    ax1_twin.set_ylabel('损耗 (kW)', color='red')
    ax1.set_title('功率与损耗')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 效率
    ax2 = axes[0, 1]
    if len(valid_eff) > 0:
        mask = np.isfinite(efficiency_data)
        ax2.plot(t[mask], efficiency_data[mask] * 100, 'purple', linewidth=2)
        ax2.set_ylim(80, 100)  # 合理的效率范围
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('效率 (%)')
    ax2.set_title('系统效率')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 温度
    ax3 = axes[1, 0]
    ax3.plot(t, Tj_data, 'r-', linewidth=2, label='结温')
    ax3.plot(t, Tc_data, 'g-', linewidth=2, label='壳温')
    ax3.plot(t, results['T_amb'], 'b--', linewidth=1, alpha=0.7, label='环境温度')
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('温度 (°C)')
    ax3.set_title('温度变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 诊断信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 检查异常情况
    issues = []
    if len(valid_eff) == 0:
        issues.append("❌ 效率数据全为NaN")
    elif valid_eff.min() > 0.98:
        issues.append("❌ 效率过高(>98%)")
    elif valid_eff.max() - valid_eff.min() < 0.01:
        issues.append("❌ 效率变化太小")
    
    if Tj_data.max() - Tj_data.min() < 1:
        issues.append("❌ 结温变化太小")
    
    if Tj_data.max() > 150:
        issues.append("❌ 结温过高(>150°C)")
    
    if P_loss_data.max() - P_loss_data.min() < 1000:
        issues.append("❌ 损耗变化太小")
    
    info_text = f"""诊断结果:

热模型参数:
• Rth_jc: {pcs_sim.params.Rth_jc:.3f} K/W
• Rth_ca: {pcs_sim.params.Rth_ca:.3f} K/W
• Cth_jc: {pcs_sim.params.Cth_jc:.0f} J/K
• Cth_ca: {pcs_sim.params.Cth_ca:.0f} J/K

仿真结果:
• 效率范围: {valid_eff.min()*100:.2f}-{valid_eff.max()*100:.2f}% (有效点: {len(valid_eff)})
• 结温范围: {Tj_data.min():.1f}-{Tj_data.max():.1f}°C
• 壳温范围: {Tc_data.min():.1f}-{Tc_data.max():.1f}°C
• 损耗范围: {P_loss_data.min()/1e3:.1f}-{P_loss_data.max()/1e3:.1f}kW

问题诊断:
{chr(10).join(issues) if issues else "✅ 未发现明显异常"}

可能原因:
• 时间常数设置不当
• 热模型参数不合理
• 效率计算逻辑错误
• 温度限制范围问题"""
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存调试图表
    os.makedirs('Debug', exist_ok=True)
    plt.savefig('Debug/efficiency_temperature_debug.png', dpi=300, bbox_inches='tight')
    print("调试图表已保存到: Debug/efficiency_temperature_debug.png")
    
    plt.show()
    
    # 输出建议修复方案
    print(f"\n6. 建议修复方案:")
    print("-" * 40)
    
    if len(issues) == 0:
        print("✅ 未发现明显问题")
    else:
        print("发现以下问题需要修复：")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n💡 建议修复步骤：")
        print("1. 调整热模型时间常数")
        print("2. 修正效率计算逻辑")
        print("3. 优化温度更新算法") 
        print("4. 检查损耗计算公式")

if __name__ == "__main__":
    debug_efficiency_temperature()
