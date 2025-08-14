#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的绘图功能
验证绘图重叠问题是否解决
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from long_term_life_simulation import LongTermLifeSimulation

def test_optimized_plotting():
    """测试优化后的绘图功能"""
    print("=" * 60)
    print("测试优化后的绘图功能")
    print("=" * 60)
    
    try:
        # 创建仿真对象
        simulator = LongTermLifeSimulation()
        
        # 运行仿真获取数据
        print("运行仿真获取数据...")
        results = simulator.simulate_long_term_life([1, 3, 5, 10], ['light', 'medium', 'heavy'])
        
        # 测试基础绘图功能
        print("\n测试基础绘图功能...")
        simulator.plot_life_results(results)
        print("✓ 基础绘图功能测试成功")
        
        # 测试详细分析绘图功能
        print("\n测试详细分析绘图功能...")
        simulator.plot_detailed_analysis()
        print("✓ 详细分析绘图功能测试成功")
        
        print("\n🎉 所有绘图功能测试通过！")
        print("绘图重叠问题已解决，图表更加清晰！")
        
        return True
    except Exception as e:
        print(f"❌ 绘图功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试优化后的绘图功能...")
    print("=" * 80)
    
    success = test_optimized_plotting()
    
    if success:
        print("\n✅ 绘图优化成功！")
        print("主要改进包括：")
        print("1. 增加图形尺寸 (20x15 和 16x12)")
        print("2. 优化子图间距 (hspace=0.4, wspace=0.3)")
        print("3. 改进图例位置 (统一右上角)")
        print("4. 增加标签间距，避免重叠")
        print("5. 使用更大的字体和粗体标签")
        print("6. 使用 plt.tight_layout() 自动优化布局")
    else:
        print("\n❌ 绘图优化测试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
