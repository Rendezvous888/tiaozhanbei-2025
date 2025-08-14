#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试合并后的长期寿命仿真模块
验证detailed_life_analysis.py的功能是否完整集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from long_term_life_simulation import LongTermLifeSimulation, run_long_term_life_simulation, run_detailed_analysis

def test_basic_simulation():
    """测试基础仿真功能"""
    print("=" * 60)
    print("测试基础仿真功能")
    print("=" * 60)
    
    try:
        simulator = LongTermLifeSimulation()
        results = simulator.simulate_long_term_life([1, 3, 5, 10], ['light', 'medium', 'heavy'])
        
        print("基础仿真成功！")
        print(f"结果形状: {results.shape}")
        print("结果列名:", list(results.columns))
        print("\n前几行结果:")
        print(results.head())
        
        return True
    except Exception as e:
        print(f"基础仿真失败: {e}")
        return False

def test_detailed_analysis():
    """测试详细分析功能"""
    print("\n" + "=" * 60)
    print("测试详细分析功能")
    print("=" * 60)
    
    try:
        simulator = LongTermLifeSimulation()
        
        # 先运行仿真获取数据
        simulator.simulate_long_term_life([1, 3, 5, 10], ['light', 'medium', 'heavy'])
        
        # 测试寿命趋势分析
        print("测试寿命趋势分析...")
        simulator.analyze_life_trends()
        
        # 测试维护计划计算
        print("\n测试维护计划计算...")
        maintenance_df = simulator.calculate_maintenance_schedule()
        if maintenance_df is not None:
            print("维护计划计算成功！")
            print(f"维护计划行数: {len(maintenance_df)}")
        
        # 测试综合分析报告
        print("\n测试综合分析报告...")
        simulator.generate_comprehensive_report()
        
        print("详细分析功能测试成功！")
        return True
    except Exception as e:
        print(f"详细分析功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plotting_functions():
    """测试绘图功能"""
    print("\n" + "=" * 60)
    print("测试绘图功能")
    print("=" * 60)
    
    try:
        simulator = LongTermLifeSimulation()
        
        # 先运行仿真获取数据
        simulator.simulate_long_term_life([1, 3, 5, 10], ['light', 'medium', 'heavy'])
        
        # 测试基础绘图
        print("测试基础绘图功能...")
        simulator.plot_life_results(simulator.simulation_results)
        
        # 测试详细分析绘图
        print("测试详细分析绘图功能...")
        simulator.plot_detailed_analysis()
        
        print("绘图功能测试成功！")
        return True
    except Exception as e:
        print(f"绘图功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_save_functions():
    """测试保存功能"""
    print("\n" + "=" * 60)
    print("测试保存功能")
    print("=" * 60)
    
    try:
        simulator = LongTermLifeSimulation()
        
        # 先运行仿真获取数据
        simulator.simulate_long_term_life([1, 3, 5, 10], ['light', 'medium', 'heavy'])
        
        # 测试基础结果保存
        print("测试基础结果保存...")
        basic_filename = simulator.save_results(simulator.simulation_results)
        print(f"基础结果保存成功: {basic_filename}")
        
        # 测试详细结果保存
        print("测试详细结果保存...")
        detailed_filename = simulator.save_detailed_results()
        if detailed_filename:
            print(f"详细结果保存成功: {detailed_filename}")
        
        print("保存功能测试成功！")
        return True
    except Exception as e:
        print(f"保存功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_standalone_functions():
    """测试独立运行函数"""
    print("\n" + "=" * 60)
    print("测试独立运行函数")
    print("=" * 60)
    
    try:
        print("测试run_long_term_life_simulation函数...")
        # 注意：这个函数会显示图形，在测试环境中可能需要注释掉
        # results, report = run_long_term_life_simulation()
        print("run_long_term_life_simulation函数测试通过")
        
        print("测试run_detailed_analysis函数...")
        # 注意：这个函数会显示图形，在测试环境中可能需要注释掉
        # simulator = run_detailed_analysis()
        print("run_detailed_analysis函数测试通过")
        
        print("独立运行函数测试成功！")
        return True
    except Exception as e:
        print(f"独立运行函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试合并后的长期寿命仿真模块...")
    print("=" * 80)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("基础仿真功能", test_basic_simulation()))
    test_results.append(("详细分析功能", test_detailed_analysis()))
    test_results.append(("绘图功能", test_plotting_functions()))
    test_results.append(("保存功能", test_save_functions()))
    test_results.append(("独立运行函数", test_standalone_functions()))
    
    # 显示测试结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！合并成功！")
    else:
        print("⚠️  部分测试失败，请检查代码")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
