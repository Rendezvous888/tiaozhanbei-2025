#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试动态功率曲线 - 验证0-100MW波动和24小时充放电循环
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_profile import generate_profiles
from main_simulation import main
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_dynamic_power_profile():
    """测试新的动态功率曲线"""
    print("=" * 60)
    print("测试动态功率曲线 - 0-100MW波动和24小时充放电循环")
    print("=" * 60)
    
    # 生成24小时的功率和温度曲线
    print("1. 生成动态功率曲线...")
    step_seconds = 60  # 每分钟一个数据点
    P_profile, T_amb = generate_profiles(day_type="summer-weekday", step_seconds=step_seconds)
    t = np.arange(len(P_profile)) * (step_seconds / 3600.0)  # 小时
    
    # 转换为MW用于分析
    P_mw = P_profile / 1e6
    
    print(f"功率数据点数: {len(P_profile)}")
    print(f"时间范围: {t[0]:.1f} - {t[-1]:.1f} 小时")
    print(f"功率范围: {P_mw.min():.1f} - {P_mw.max():.1f} MW")
    print(f"充电功率(负值): {P_mw[P_mw < 0].min():.1f} MW")
    print(f"放电功率(正值): {P_mw[P_mw > 0].max():.1f} MW")
    
    # 分析充放电循环
    print("\n2. 分析充放电循环...")
    
    # 计算累积能量来分析SOC变化
    dt_h = step_seconds / 3600.0
    energy_cumulative = np.cumsum(P_profile) * dt_h / 1e6  # MWh
    
    # 假设电池容量为100MWh（对应1小时续航）
    battery_capacity_mwh = 100.0
    soc_simulation = 50.0 + (energy_cumulative / battery_capacity_mwh) * 100  # 从50%开始
    soc_simulation = np.clip(soc_simulation, 0, 100)  # 限制在0-100%
    
    soc_range = soc_simulation.max() - soc_simulation.min()
    print(f"SOC变化范围: {soc_simulation.min():.1f}% - {soc_simulation.max():.1f}%")
    print(f"SOC总变化幅度: {soc_range:.1f}%")
    
    # 检查是否完成充放电循环
    charge_periods = []
    discharge_periods = []
    
    for i in range(len(P_mw)):
        if P_mw[i] < -10:  # 充电功率大于10MW
            charge_periods.append(t[i])
        elif P_mw[i] > 10:  # 放电功率大于10MW
            discharge_periods.append(t[i])
    
    if charge_periods and discharge_periods:
        charge_time = len(charge_periods) * dt_h
        discharge_time = len(discharge_periods) * dt_h
        print(f"充电时间: {charge_time:.1f} 小时")
        print(f"放电时间: {discharge_time:.1f} 小时")
        print(f"充放电时间比: {charge_time/(discharge_time+1e-6):.2f}")
    
    # 绘制分析结果
    print("\n3. 绘制分析图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('动态功率曲线测试结果 - 0-100MW波动分析', fontsize=16)
    
    # 子图1: 功率曲线
    ax1 = axes[0, 0]
    ax1.plot(t, P_mw, 'b-', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100MW上限')
    ax1.axhline(y=-100, color='g', linestyle='--', alpha=0.5, label='-100MW下限')
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('功率 (MW)')
    ax1.set_title('24小时功率波动曲线')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-110, 110)
    
    # 子图2: 功率分布直方图
    ax2 = axes[0, 1]
    ax2.hist(P_mw, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('功率 (MW)')
    ax2.set_ylabel('频次')
    ax2.set_title('功率分布统计')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 模拟SOC变化
    ax3 = axes[1, 0]
    ax3.plot(t, soc_simulation, 'orange', linewidth=2)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='0% SOC')
    ax3.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100% SOC')
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('SOC (%)')
    ax3.set_title('模拟电池SOC变化')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(-5, 105)
    
    # 子图4: 充放电分析
    ax4 = axes[1, 1]
    charge_mask = P_mw < -5  # 充电阈值-5MW
    discharge_mask = P_mw > 5  # 放电阈值5MW
    idle_mask = np.abs(P_mw) <= 5  # 待机状态
    
    ax4.fill_between(t, 0, 1, where=charge_mask[:-1] if len(charge_mask) > len(t) else charge_mask, 
                     alpha=0.3, color='green', label=f'充电 ({np.sum(charge_mask)}点)')
    ax4.fill_between(t, 0, 1, where=discharge_mask[:-1] if len(discharge_mask) > len(t) else discharge_mask, 
                     alpha=0.3, color='red', label=f'放电 ({np.sum(discharge_mask)}点)')
    ax4.fill_between(t, 0, 1, where=idle_mask[:-1] if len(idle_mask) > len(t) else idle_mask, 
                     alpha=0.3, color='gray', label=f'待机 ({np.sum(idle_mask)}点)')
    
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('运行状态')
    ax4.set_title('充放电周期分析')
    ax4.legend()
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('Test', exist_ok=True)
    plt.savefig('Test/dynamic_power_profile_test.png', dpi=300, bbox_inches='tight')
    print("图表已保存到: Test/dynamic_power_profile_test.png")
    
    plt.show()
    
    # 验证结果
    print("\n4. 验证测试结果...")
    print("-" * 40)
    
    # 检查功率范围
    power_range_ok = (P_mw.min() >= -100.1) and (P_mw.max() <= 100.1)
    print(f"✓ 功率范围检查: {'通过' if power_range_ok else '失败'}")
    print(f"  范围: {P_mw.min():.1f} ~ {P_mw.max():.1f} MW")
    
    # 检查动态性
    power_std = np.std(P_mw)
    dynamic_ok = power_std > 10  # 标准差大于10MW表示有足够变化
    print(f"✓ 动态波动检查: {'通过' if dynamic_ok else '失败'}")
    print(f"  功率标准差: {power_std:.1f} MW")
    
    # 检查充放电平衡
    total_charge_energy = np.sum(P_mw[P_mw < 0]) * dt_h  # MWh
    total_discharge_energy = np.sum(P_mw[P_mw > 0]) * dt_h  # MWh
    energy_balance = abs(total_charge_energy + total_discharge_energy) / max(abs(total_charge_energy), abs(total_discharge_energy))
    balance_ok = energy_balance < 0.3  # 能量不平衡小于30%
    print(f"✓ 充放电循环检查: {'通过' if balance_ok else '失败'}")
    print(f"  充电能量: {abs(total_charge_energy):.1f} MWh")
    print(f"  放电能量: {total_discharge_energy:.1f} MWh")
    print(f"  能量不平衡: {energy_balance*100:.1f}%")
    
    # 检查SOC变化范围
    soc_range_ok = soc_range > 30  # SOC变化大于30%
    print(f"✓ SOC变化范围检查: {'通过' if soc_range_ok else '失败'}")
    print(f"  SOC变化范围: {soc_range:.1f}%")
    
    # 总体评估
    all_checks = [power_range_ok, dynamic_ok, balance_ok, soc_range_ok]
    overall_ok = all(all_checks)
    
    print("\n5. 测试总结:")
    print("-" * 40)
    if overall_ok:
        print("✅ 所有测试通过！动态功率曲线设计成功。")
        print("✅ 功率已实现0-100MW范围内的连续波动")
        print("✅ 24小时内可完成完整的充放电循环")
    else:
        print("❌ 部分测试失败，需要进一步调整参数")
        failed_tests = []
        if not power_range_ok: failed_tests.append("功率范围")
        if not dynamic_ok: failed_tests.append("动态波动")
        if not balance_ok: failed_tests.append("充放电平衡")
        if not soc_range_ok: failed_tests.append("SOC变化范围")
        print(f"❌ 失败项目: {', '.join(failed_tests)}")

if __name__ == "__main__":
    test_dynamic_power_profile()
