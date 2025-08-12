"""
Matplotlib中文字体配置文件
解决matplotlib在Windows系统上中文显示异常的问题
"""

import matplotlib.pyplot as plt
import platform
import os

def configure_chinese_fonts():
    """配置matplotlib支持中文显示"""
    
    # 检测操作系统
    system = platform.system()
    
    if system == "Windows":
        # Windows系统字体配置
        font_list = [
            'SimHei',           # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',           # 宋体
            'KaiTi',            # 楷体
            'FangSong',         # 仿宋
            'DejaVu Sans'       # 备用字体
        ]
    elif system == "Darwin":  # macOS
        # macOS系统字体配置
        font_list = [
            'PingFang SC',      # 苹方
            'Hiragino Sans GB', # 冬青黑体
            'STHeiti',          # 华文黑体
            'DejaVu Sans'       # 备用字体
        ]
    else:  # Linux或其他系统
        # Linux系统字体配置
        font_list = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'DejaVu Sans'           # 备用字体
        ]
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = font_list
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置字体大小
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    print(f"已配置matplotlib中文字体支持 - 系统: {system}")
    print(f"字体列表: {font_list[:3]}...")  # 只显示前3个字体

def test_chinese_display():
    """测试中文字体显示是否正常"""
    try:
        # 创建测试图
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 测试中文标题和标签
        ax.set_title('中文字体测试 - 构网型级联储能PCS')
        ax.set_xlabel('时间 (小时)')
        ax.set_ylabel('功率 (MW)')
        
        # 测试中文图例
        x = [0, 1, 2, 3, 4]
        y = [0, 10, 20, 15, 5]
        ax.plot(x, y, 'b-o', label='系统功率曲线')
        ax.legend()
        
        ax.grid(True)
        plt.tight_layout()
        
        print("✓ 中文字体显示测试成功！")
        plt.show()
        
    except Exception as e:
        print(f"✗ 中文字体显示测试失败: {e}")
        print("请检查系统是否安装了相应的中文字体")

if __name__ == "__main__":
    # 配置字体
    configure_chinese_fonts()
    
    # 测试显示
    test_chinese_display()
