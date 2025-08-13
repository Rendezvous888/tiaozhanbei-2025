#!/usr/bin/env python3
"""
测试中文字体显示效果
"""

import matplotlib.pyplot as plt
from plot_utils import set_chinese_plot_style, save_chinese_plot

def test_chinese_font():
    """测试中文字体显示"""
    print("=== 测试中文字体显示 ===")
    
    # 设置中文字体
    set_chinese_plot_style()
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    
    # 绘制数据
    ax.plot(x, y, 'o-', linewidth=2, markersize=8, label='测试数据')
    
    # 设置中文标题和标签
    ax.set_title('中文字体测试图表 - 电池放电曲线', fontsize=16, fontweight='bold')
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('电压 (V)', fontsize=12)
    
    # 添加图例
    ax.legend(fontsize=10)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加一些中文注释
    ax.text(3, 20, '正常工作区域', fontsize=10, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    
    # 保存图表
    if save_chinese_plot('chinese_font_test.png', dpi=300):
        print("中文字体测试图表已保存为 'chinese_font_test.png'")
    else:
        print("保存图表失败")
    
    print("字体测试完成！")

if __name__ == "__main__":
    test_chinese_font()
