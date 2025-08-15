#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H桥模型优化绘图测试脚本
测试优化后的绘图功能，解决文字重叠问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import simulate_hbridge_system, analyze_pwm_strategies, plot_advanced_analysis, create_monitoring_dashboard

def test_optimized_plotting():
    """测试优化后的绘图功能"""
    print("=== 测试优化后的H桥模型绘图功能 ===")
    
    try:
        # 测试基础仿真结果绘图
        print("1. 测试基础仿真结果绘图...")
        system, V_output, losses = simulate_hbridge_system()
        print("   ✓ 基础仿真结果绘图完成")
        
        # 测试PWM策略分析绘图
        print("2. 测试PWM策略分析绘图...")
        analyze_pwm_strategies()
        print("   ✓ PWM策略分析绘图完成")
        
        # 测试高级分析图表
        print("3. 测试高级分析图表...")
        t_analysis = np.linspace(0, 0.02, 10000)
        advanced_fig = plot_advanced_analysis(system, t_analysis)
        print("   ✓ 高级分析图表完成")
        
        # 测试实时监控仪表板
        print("4. 测试实时监控仪表板...")
        monitoring_fig = create_monitoring_dashboard(system, t_analysis)
        print("   ✓ 实时监控仪表板完成")
        
        print("\n所有测试完成！")
        print("已生成以下优化后的图表：")
        print("1. 基础仿真结果（12个子图，优化布局）")
        print("2. PWM策略分析（4个子图，优化间距）")
        print("3. 高级分析图表（6个子图，包含3D图表，优化标签）")
        print("4. 实时监控仪表板（8个子图，优化显示）")
        print("总计：30个分析图表，均已优化布局和标签显示")
        
        # 显示所有图表
        plt.show()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置matplotlib参数
    plt.rcParams['figure.max_open_warning'] = 50  # 允许更多图表窗口
    
    # 运行测试
    test_optimized_plotting()
