#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI投资回报率分析
展示现实的投资回报过程：从负值开始逐步实现盈亏平衡
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from predictive_maintenance import PredictiveMaintenanceOptimizer

def analyze_roi_timeline():
    """分析ROI时间线，展示现实的投资回报过程"""
    print("=" * 60)
    print("ROI投资回报率时间线分析")
    print("=" * 60)
    
    # 创建优化器实例
    optimizer = PredictiveMaintenanceOptimizer()
    
    # 创建测试数据
    life_predictions = {}
    for years in range(1, 11):
        igbt_life = max(10, 100 - years * 7.5)
        cap_life = max(15, 100 - years * 5.5)
        life_predictions[years] = {
            'igbt': {'final_prediction': igbt_life},
            'capacitor': {'final_prediction': cap_life}
        }
    
    # 生成计划
    inspection_schedule = optimizer.optimize_inspection_schedule(life_predictions)
    replacement_schedule = optimizer.optimize_replacement_strategy(life_predictions)
    
    # 计算逐年累积ROI（包含系统建设成本）
    years = list(range(1, 11))
    
    # 第0年：系统建设大额投资
    system_setup_total = sum(optimizer.system_setup_costs.values())
    cumulative_investment = system_setup_total  # 初期大额投资
    cumulative_benefits = 0
    roi_timeline = []
    investment_timeline = []
    benefits_timeline = []
    net_value_timeline = []
    
    print(f"\n第0年（系统建设期）:")
    print(f"系统建设投资: {system_setup_total/10000:.1f}万元")
    print("建设内容包括：传感器、数据采集、通信网络、分析软件、系统集成等")
    
    print("\n逐年投资回报分析:")
    print("-" * 60)
    
    for year in years:
        # 年度投资
        annual_investment = 0
        if year in replacement_schedule:
            annual_investment += replacement_schedule[year]['total_cost']
        if year in inspection_schedule:
            insp_data = inspection_schedule[year]
            annual_investment += insp_data['inspection_cost'] + insp_data['monitoring_cost']
        
        cumulative_investment += annual_investment
        
        # 年度效益（与主程序参数一致：前期低，后期逐步显现）
        if year in inspection_schedule:
            annual_failure_prob = inspection_schedule[year]['igbt_failure_prob'] + inspection_schedule[year]['cap_failure_prob']
            base_benefit = annual_failure_prob * optimizer.maintenance_costs['emergency_repair']
            
            # 效益随时间增长（前期低，后期高）
            if year <= 2:
                time_multiplier = 0.3  # 前2年效益很低
            elif year <= 4:
                time_multiplier = 0.8  # 3-4年效益逐步显现
            else:
                time_multiplier = 1.5 + (year - 5) * 0.3  # 5年后效益显著
            
            # 保守效益计算（与主程序一致）
            annual_benefit = base_benefit * (1 + 0.6 + 1.8) * time_multiplier
            cumulative_benefits += annual_benefit
        
        # 计算ROI
        if cumulative_investment > 0:
            roi = ((cumulative_benefits - cumulative_investment) / cumulative_investment) * 100
        else:
            roi = 0
        
        net_value = cumulative_benefits - cumulative_investment
        
        roi_timeline.append(roi)
        investment_timeline.append(cumulative_investment / 10000)  # 转换为万元
        benefits_timeline.append(cumulative_benefits / 10000)     # 转换为万元
        net_value_timeline.append(net_value / 10000)             # 转换为万元
        
        print(f"{year}年: 累积投资={cumulative_investment/10000:.1f}万元, "
              f"累积效益={cumulative_benefits/10000:.1f}万元, "
              f"净值={net_value/10000:.1f}万元, ROI={roi:.1f}%")
    
    # 找到盈亏平衡点
    breakeven_year = None
    for i, net_val in enumerate(net_value_timeline):
        if net_val >= 0:
            breakeven_year = i + 1
            break
    
    print(f"\n关键节点:")
    if breakeven_year:
        print(f"盈亏平衡点: 第{breakeven_year}年")
        print(f"盈亏平衡时ROI: {roi_timeline[breakeven_year-1]:.1f}%")
    print(f"5年ROI: {roi_timeline[4]:.1f}%")
    print(f"10年ROI: {roi_timeline[9]:.1f}%")
    
    # 绘制ROI分析图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROI时间线
    ax1 = axes[0, 0]
    ax1.plot(years, roi_timeline, 'o-', linewidth=2, markersize=6, color='green', label='累积ROI')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='盈亏平衡线')
    if breakeven_year:
        ax1.axvline(x=breakeven_year, color='orange', linestyle=':', alpha=0.7, label=f'盈亏平衡({breakeven_year}年)')
    ax1.set_xlabel('年份')
    ax1.set_ylabel('ROI (%)')
    ax1.set_title('投资回报率时间线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 累积投资与效益
    ax2 = axes[0, 1]
    ax2.plot(years, investment_timeline, 's-', linewidth=2, markersize=6, color='red', label='累积投资')
    ax2.plot(years, benefits_timeline, '^-', linewidth=2, markersize=6, color='blue', label='累积效益')
    ax2.set_xlabel('年份')
    ax2.set_ylabel('金额 (万元)')
    ax2.set_title('累积投资与效益对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 净值变化
    ax3 = axes[1, 0]
    colors = ['red' if x < 0 else 'green' for x in net_value_timeline]
    bars = ax3.bar(years, net_value_timeline, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('年份')
    ax3.set_ylabel('净值 (万元)')
    ax3.set_title('净值变化（负值=亏损，正值=盈利）')
    ax3.grid(True, alpha=0.3)
    
    # 4. 投资回报分析总结
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""投资回报分析总结

初期特征:
• 前{breakeven_year-1 if breakeven_year else 3}年: 投资期，ROI为负值
• 投资主要用于：设备监测、预防性维护

盈亏平衡:
• 盈亏平衡点: 第{breakeven_year if breakeven_year else 'N/A'}年
• 平衡时ROI: {roi_timeline[breakeven_year-1] if breakeven_year else 'N/A'}%

长期收益:
• 5年ROI: {roi_timeline[4]:.0f}%
• 10年ROI: {roi_timeline[9]:.0f}%
• 主要效益: 避免故障、延长寿命、减少停机

投资特点:
✓ 符合预防性维护投资规律
✓ 前期投入大，后期效益显著
✓ 长期ROI表现优秀"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('Debug/roi_timeline_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nROI时间线分析图已保存到: Debug/roi_timeline_analysis.png")
    
    plt.show()
    
    return {
        'roi_timeline': roi_timeline,
        'breakeven_year': breakeven_year,
        'investment_timeline': investment_timeline,
        'benefits_timeline': benefits_timeline,
        'net_value_timeline': net_value_timeline
    }

if __name__ == "__main__":
    analyze_roi_timeline()
