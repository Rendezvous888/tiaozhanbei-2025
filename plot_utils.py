#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应绘图工具
提供通用的绘图自适应功能，确保图形在不同屏幕上都能完整显示
"""

import matplotlib.pyplot as plt
import platform
import numpy as np

class AdaptivePlotter:
    """自适应绘图器"""
    
    def __init__(self):
        self.setup_platform_specific_settings()
    
    def setup_platform_specific_settings(self):
        """根据操作系统设置绘图参数"""
        if platform.system() == 'Windows':
            # Windows系统优化
            plt.rcParams.update({
                'font.size': 9,
                'axes.titlesize': 10,
                'axes.labelsize': 9,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 8,
                'figure.titlesize': 11
            })
            self.default_figsize = (15, 9)
            self.default_dpi = 120
            self.layout_params = {
                'top': 0.90, 'bottom': 0.10, 'left': 0.08, 'right': 0.92,
                'hspace': 0.25, 'wspace': 0.25
            }
        else:
            # 其他系统
            plt.rcParams.update({
                'font.size': 10,
                'axes.titlesize': 11,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 12
            })
            self.default_figsize = (16, 10)
            self.default_dpi = 100
            self.layout_params = {
                'top': 0.92, 'bottom': 0.08, 'left': 0.06, 'right': 0.94,
                'hspace': 0.3, 'wspace': 0.3
            }
    
    def create_adaptive_figure(self, nrows, ncols, figsize=None, dpi=None, 
                              title=None, title_size=None):
        """创建自适应图形"""
        if figsize is None:
            # 根据子图数量自适应调整图形大小
            base_width = self.default_figsize[0]
            base_height = self.default_figsize[1]
            
            # 调整宽度以适应列数
            if ncols > 3:
                width = base_width * (ncols / 3)
            else:
                width = base_width
            
            # 调整高度以适应行数
            if nrows > 2:
                height = base_height * (nrows / 2)
            else:
                height = base_height
            
            figsize = (width, height)
        
        if dpi is None:
            dpi = self.default_dpi
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
        
        # 设置标题
        if title:
            if title_size is None:
                title_size = plt.rcParams['figure.titlesize']
            fig.suptitle(title, fontsize=title_size, fontweight='bold', y=0.95)
        
        return fig, axes
    
    def optimize_layout(self, fig, tight_layout=True, **kwargs):
        """优化图形布局"""
        if tight_layout:
            # 从kwargs中提取tight_layout的参数
            tight_pad = kwargs.get('pad', 1.5)
            tight_h_pad = kwargs.get('h_pad', 1.2)
            tight_w_pad = kwargs.get('w_pad', 1.2)
            plt.tight_layout(pad=tight_pad, h_pad=tight_h_pad, w_pad=tight_w_pad)
        
        # 应用平台特定的布局参数，过滤掉tight_layout的参数
        layout_params = {**self.layout_params}
        for key, value in kwargs.items():
            if key not in ['pad', 'h_pad', 'w_pad']:
                layout_params[key] = value
        
        plt.subplots_adjust(**layout_params)
    
    def set_adaptive_ylim(self, ax, data, margin_factor=0.1, min_margin=0.05):
        """设置自适应的Y轴范围"""
        # 处理特殊输入情况
        if isinstance(data, list) and len(data) == 2:
            if data[0] == 0 and data[1] is None:
                ax.set_ylim(bottom=0)
                return
            elif data[0] is None and data[1] == 0:
                ax.set_ylim(top=0)
                return
        
        # 将数据转换为一维数组并过滤无效值
        try:
            arr = np.asarray(data).astype(float).ravel()
        except Exception:
            # 回退处理
            data_min = data.min()
            data_max = data.max()
        else:
            finite_mask = np.isfinite(arr)
            if not np.any(finite_mask):
                return
            data_min = float(np.min(arr[finite_mask]))
            data_max = float(np.max(arr[finite_mask]))
        
        data_range = data_max - data_min
        
        margin = min_margin if data_range < min_margin else max(data_range * margin_factor, min_margin)
        
        y_min = max(0, data_min - margin) if data_min >= 0 else data_min - margin
        y_max = data_max + margin
        
        ax.set_ylim(y_min, y_max)
    
    def set_adaptive_xlim(self, ax, data, margin_factor=0.1, min_margin=0.05):
        """设置自适应的X轴范围"""
        try:
            arr = np.asarray(data).astype(float).ravel()
        except Exception:
            data_min = data.min()
            data_max = data.max()
        else:
            finite_mask = np.isfinite(arr)
            if not np.any(finite_mask):
                return
            data_min = float(np.min(arr[finite_mask]))
            data_max = float(np.max(arr[finite_mask]))
        
        data_range = data_max - data_min
        margin = min_margin if data_range < min_margin else max(data_range * margin_factor, min_margin)
        
        x_min = data_min - margin
        x_max = data_max + margin
        
        ax.set_xlim(x_min, x_max)
    
    def format_axis_labels(self, ax, xlabel=None, ylabel=None, title=None, 
                          fontsize=None, color=None):
        """格式化坐标轴标签"""
        if fontsize is None:
            fontsize = plt.rcParams['axes.labelsize']
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            if color is not None:
                ax.set_ylabel(ylabel, fontsize=fontsize, color=color)
            else:
                ax.set_ylabel(ylabel, fontsize=fontsize)
        if title:
            ax.set_title(title, fontsize=fontsize, pad=10)
    
    def add_grid(self, ax, alpha=0.3, style='-'):
        """添加网格"""
        ax.grid(True, alpha=alpha, linestyle=style)
    
    def create_legend(self, ax, lines, labels, fontsize=None, loc='best'):
        """创建图例"""
        if fontsize is None:
            fontsize = plt.rcParams['legend.fontsize']
        
        ax.legend(lines, labels, fontsize=fontsize, loc=loc)
    
    def finalize_plot(self, fig, save_path=None, show=True, print_info=True):
        """完成绘图"""
        # 显示图形
        if show:
            plt.show()
        
        # 保存图形
        if save_path:
            fig.savefig(save_path, dpi=fig.dpi, bbox_inches='tight')
        
        # 打印图形信息
        if print_info:
            print(f"图形尺寸: {fig.get_size_inches()}")
            print(f"图形DPI: {fig.dpi}")
            print(f"实际像素尺寸: {fig.get_size_inches() * fig.dpi}")
        
        return fig

# 创建全局实例
adaptive_plotter = AdaptivePlotter()

def get_adaptive_plotter():
    """获取自适应绘图器实例"""
    return adaptive_plotter

def create_adaptive_figure(*args, **kwargs):
    """创建自适应图形的便捷函数"""
    return adaptive_plotter.create_adaptive_figure(*args, **kwargs)

def optimize_layout(*args, **kwargs):
    """优化图形布局的便捷函数"""
    return adaptive_plotter.optimize_layout(*args, **kwargs)

def set_adaptive_ylim(*args, **kwargs):
    """设置自适应Y轴范围的便捷函数"""
    return adaptive_plotter.set_adaptive_ylim(*args, **kwargs)

def set_adaptive_xlim(*args, **kwargs):
    """设置自适应X轴范围的便捷函数"""
    return adaptive_plotter.set_adaptive_xlim(*args, **kwargs)

def finalize_plot(*args, **kwargs):
    """完成绘图的便捷函数"""
    return adaptive_plotter.finalize_plot(*args, **kwargs)

def format_axis_labels(*args, **kwargs):
    """格式化坐标轴标签的便捷函数"""
    return adaptive_plotter.format_axis_labels(*args, **kwargs)

def add_grid(*args, **kwargs):
    """添加网格的便捷函数"""
    return adaptive_plotter.add_grid(*args, **kwargs)

def create_legend(*args, **kwargs):
    """创建图例的便捷函数"""
    return adaptive_plotter.create_legend(*args, **kwargs)

# ==================== 中文字体支持 ====================

def configure_chinese_fonts():
    """配置中文字体支持"""
    import matplotlib.font_manager as fm
    import os
    
    # 根据操作系统选择合适的字体
    if platform.system() == 'Windows':
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif platform.system() == 'Darwin':
        font_names = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
    else:
        font_names = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Liberation Sans', 'Noto Sans CJK SC']
    
    font_set = False
    for font_name in font_names:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path) and font_path != plt.rcParams['font.sans-serif'][0]:
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                font_set = True
                print(f"成功设置中文字体: {font_name}")
                break
        except Exception as e:
            print(f"尝试设置字体 {font_name} 失败: {e}")
            continue
    
    if not font_set:
        print("警告: 未能找到合适的中文字体，图表中的中文可能无法正常显示")
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
        except Exception:
            pass
    
    plt.rcParams['axes.unicode_minus'] = False
    
    return plt.rcParams['font.sans-serif'][0]

def set_chinese_plot_style():
    """设置支持中文的绘图样式"""
    configure_chinese_fonts()
    adaptive_plotter.setup_platform_specific_settings()

def save_chinese_plot(filename, dpi=300):
    """保存支持中文的图表"""
    try:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"图表已保存为: {filename}")
        return True
    except Exception as e:
        print(f"保存图表失败: {e}")
        return False
