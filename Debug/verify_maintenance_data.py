#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证预测性维护数据的完整性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_maintenance import run_predictive_maintenance_optimization
import matplotlib.pyplot as plt
import numpy as np

def verify_data_completeness():
    """验证数据完整性"""
    print("=" * 60)
    print("预测性维护数据完整性验证")
    print("=" * 60)
    
    # 运行预测性维护优化
    optimizer, results = run_predictive_maintenance_optimization()
    
    # 验证年度数据
    print("\n年度数据验证:")
    life_predictions = results['life_predictions']
    print(f"年度预测数据点数: {len(life_predictions)}")
    for year in sorted(life_predictions.keys()):
        igbt_life = life_predictions[year]['igbt']['final_prediction']
        cap_life = life_predictions[year]['capacitor']['final_prediction']
        print(f"  {year}年: IGBT={igbt_life:.1f}%, 电容器={cap_life:.1f}%")
    
    # 验证月度数据
    print("\n月度数据验证:")
    monthly_predictions = results['monthly_predictions']
    print(f"月度预测数据点数: {len(monthly_predictions)}")
    
    # 检查是否有数据缺失
    expected_months = 10 * 12  # 10年 × 12个月
    print(f"期望月度数据点数: {expected_months}")
    print(f"实际月度数据点数: {len(monthly_predictions)}")
    
    if len(monthly_predictions) == expected_months:
        print("✓ 月度数据完整，无缺失")
    else:
        print("✗ 月度数据有缺失")
    
    # 验证成本数据
    print("\n成本数据验证:")
    monthly_economics = results['monthly_economics']
    print(f"月度经济数据点数: {len(monthly_economics)}")
    
    # 检查成本数据的连续性
    time_keys = sorted(monthly_economics.keys())
    
    print("\n前12个月成本数据样本:")
    for i, time_key in enumerate(time_keys[:12]):
        cost = monthly_economics[time_key]['total_cost']
        risk = monthly_economics[time_key]['risk_level']
        print(f"  {time_key:.2f}年: 成本={cost:.0f}元, 风险={risk}")
    
    # 验证年度维护成本是否有空白月份
    print("\n年度成本分布验证:")
    inspection_schedule = results['inspection_schedule']
    for year in range(1, 11):
        if year in inspection_schedule:
            annual_cost = inspection_schedule[year]['total_annual_cost']
            print(f"  {year}年: {annual_cost/10000:.1f}万元")
        else:
            print(f"  {year}年: 0.0万元 (无特殊维护)")
    
    # 绘制数据完整性验证图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 月度成本趋势
    ax1 = axes[0, 0]
    monthly_times = sorted(monthly_economics.keys())
    monthly_costs = [monthly_economics[t]['total_cost'] for t in monthly_times]
    ax1.plot(monthly_times, monthly_costs, 'b.-', linewidth=1, markersize=2)
    ax1.set_title('月度维护成本趋势')
    ax1.set_xlabel('时间 (年)')
    ax1.set_ylabel('月度成本 (元)')
    ax1.grid(True, alpha=0.3)
    
    # 数据点密度
    ax2 = axes[0, 1]
    monthly_years = [int(t) for t in monthly_times]
    year_counts = {y: monthly_years.count(y) for y in range(1, 11)}
    years = list(year_counts.keys())
    counts = list(year_counts.values())
    ax2.bar(years, counts, alpha=0.7)
    ax2.axhline(y=12, color='red', linestyle='--', label='期望值(12个月)')
    ax2.set_title('每年数据点数量')
    ax2.set_xlabel('年份')
    ax2.set_ylabel('数据点数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 风险等级分布
    ax3 = axes[1, 0]
    risk_levels = [monthly_economics[t]['risk_level'] for t in monthly_times]
    unique_risks, counts = np.unique(risk_levels, return_counts=True)
    ax3.pie(counts, labels=unique_risks, autopct='%1.1f%%')
    ax3.set_title('风险等级分布')
    
    # 寿命预测连续性
    ax4 = axes[1, 1]
    monthly_igbt = [monthly_predictions[t]['igbt']['final_prediction'] for t in monthly_times]
    monthly_cap = [monthly_predictions[t]['capacitor']['final_prediction'] for t in monthly_times]
    ax4.plot(monthly_times, monthly_igbt, 'r-', alpha=0.7, linewidth=1, label='IGBT')
    ax4.plot(monthly_times, monthly_cap, 'b-', alpha=0.7, linewidth=1, label='电容器')
    ax4.set_title('月度寿命预测连续性')
    ax4.set_xlabel('时间 (年)')
    ax4.set_ylabel('剩余寿命 (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Debug/maintenance_data_verification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n数据验证完成！验证图表已保存到: Debug/maintenance_data_verification.png")
    
    return True

if __name__ == "__main__":
    verify_data_completeness()
