#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试功率范围是否符合25MW系统要求
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_profile import generate_profiles

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_power_range():
    """测试功率范围是否合理"""
    print("=" * 60)
    print("测试25MW系统功率范围")
    print("=" * 60)
    
    # 生成24小时功率曲线
    step_seconds = 60
    P_profile, T_amb = generate_profiles(day_type="summer-weekday", step_seconds=step_seconds)
    t = np.arange(len(P_profile)) * (step_seconds / 3600.0)  # 小时
    
    # 转换为MW
    P_mw = P_profile / 1e6
    
    # 分析功率范围
    max_power = np.max(P_mw)
    min_power = np.min(P_mw)
    max_discharge = np.max(P_mw[P_mw > 0]) if np.any(P_mw > 0) else 0
    max_charge = np.min(P_mw[P_mw < 0]) if np.any(P_mw < 0) else 0
    
    print(f"功率范围分析:")
    print(f"  - 最大功率: {max_power:.2f} MW")
    print(f"  - 最小功率: {min_power:.2f} MW")
    print(f"  - 最大放电功率: {max_discharge:.2f} MW")
    print(f"  - 最大充电功率: {max_charge:.2f} MW")
    print(f"  - 功率变化范围: {max_power - min_power:.2f} MW")
    
    # 检查是否符合25MW系统要求
    print(f"\n合理性检查:")
    if max_power <= 25.0:
        print(f"✅ 最大放电功率 {max_power:.2f} MW ≤ 25 MW (符合要求)")
    else:
        print(f"❌ 最大放电功率 {max_power:.2f} MW > 25 MW (超出系统容量)")
    
    if abs(min_power) <= 25.0:
        print(f"✅ 最大充电功率 {abs(min_power):.2f} MW ≤ 25 MW (符合要求)")
    else:
        print(f"❌ 最大充电功率 {abs(min_power):.2f} MW > 25 MW (超出系统容量)")
        
    # 统计功率分布
    discharge_time = np.sum(P_mw > 0) * step_seconds / 3600  # 放电时间(小时)
    charge_time = np.sum(P_mw < 0) * step_seconds / 3600     # 充电时间(小时)
    idle_time = np.sum(np.abs(P_mw) < 0.1) * step_seconds / 3600  # 空闲时间(小时)
    
    print(f"\n运行模式分析:")
    print(f"  - 放电时间: {discharge_time:.1f} 小时 ({discharge_time/24*100:.1f}%)")
    print(f"  - 充电时间: {charge_time:.1f} 小时 ({charge_time/24*100:.1f}%)")
    print(f"  - 空闲时间: {idle_time:.1f} 小时 ({idle_time/24*100:.1f}%)")
    
    # 能量平衡检查
    total_energy = np.trapz(P_mw, t)  # 总能量积分
    print(f"\n能量平衡检查:")
    print(f"  - 24小时总能量: {total_energy:.2f} MWh")
    if abs(total_energy) < 5:  # 允许5MWh的不平衡
        print(f"✅ 能量基本平衡 (差值 < 5MWh)")
    else:
        print(f"⚠️  能量不平衡较大 (差值 = {total_energy:.2f} MWh)")
    
    # 绘制功率曲线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上子图：功率时间曲线
    ax1.plot(t, P_mw, 'b-', linewidth=1.5, label='系统功率')
    ax1.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='额定功率上限 (+25MW)')
    ax1.axhline(y=-25, color='r', linestyle='--', alpha=0.7, label='额定功率下限 (-25MW)')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('功率 (MW)')
    ax1.set_title('25MW系统24小时功率曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-30, 30)
    
    # 下子图：功率分布直方图
    ax2.hist(P_mw, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=25, color='r', linestyle='--', alpha=0.7, label='额定功率限制')
    ax2.axvline(x=-25, color='r', linestyle='--', alpha=0.7)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    ax2.set_xlabel('功率 (MW)')
    ax2.set_ylabel('频次')
    ax2.set_title('功率分布直方图')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('Test/power_range_test.png', dpi=300, bbox_inches='tight')
    print(f"\n图形已保存: Test/power_range_test.png")
    
    plt.show()
    
    return P_profile, max_power, min_power

if __name__ == "__main__":
    test_power_range()
