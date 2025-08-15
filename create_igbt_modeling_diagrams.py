#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为IGBT建模PPT创建可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_thermal_network_diagram():
    """创建热网络等效电路图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：物理结构
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_aspect('equal')
    
    # 绘制IGBT物理结构
    # 芯片
    chip = Rectangle((4, 6), 2, 1, facecolor='red', alpha=0.7, edgecolor='black')
    ax1.add_patch(chip)
    ax1.text(5, 6.5, 'IGBT芯片\n(结)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 封装
    package = Rectangle((3, 4.5), 4, 1.5, facecolor='gray', alpha=0.7, edgecolor='black')
    ax1.add_patch(package)
    ax1.text(5, 5.2, '封装\n(壳体)', ha='center', va='center', fontsize=10)
    
    # 散热器
    heatsink = Rectangle((2, 2), 6, 2.5, facecolor='lightblue', alpha=0.7, edgecolor='black')
    ax1.add_patch(heatsink)
    ax1.text(5, 3.2, '散热器', ha='center', va='center', fontsize=10)
    
    # 环境
    env = Rectangle((1, 0.5), 8, 1.5, facecolor='lightgreen', alpha=0.5, edgecolor='black')
    ax1.add_patch(env)
    ax1.text(5, 1.2, '环境', ha='center', va='center', fontsize=10)
    
    # 热流箭头
    ax1.arrow(5, 6, 0, -0.4, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax1.arrow(5, 4.5, 0, -0.4, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax1.arrow(5, 2, 0, -0.4, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax1.text(5.5, 5.7, 'Q', fontsize=12, color='red', fontweight='bold')
    ax1.text(5.5, 3.7, 'Q', fontsize=12, color='red', fontweight='bold')
    ax1.text(5.5, 1.7, 'Q', fontsize=12, color='red', fontweight='bold')
    
    ax1.set_title('IGBT热传递物理结构', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 右图：等效热网络
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    
    # 功率源
    power_source = patches.Circle((1, 6.5), 0.3, facecolor='red', edgecolor='black')
    ax2.add_patch(power_source)
    ax2.text(1, 6.5, 'P', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # 节点和连线
    nodes = [(2.5, 6.5), (4.5, 6.5), (6.5, 6.5), (8.5, 6.5)]
    node_labels = ['Tj', 'Tc', 'Ts', 'Ta']
    
    for i, ((x, y), label) in enumerate(zip(nodes, node_labels)):
        circle = patches.Circle((x, y), 0.2, facecolor='yellow', edgecolor='black')
        ax2.add_patch(circle)
        ax2.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(nodes) - 1:
            # 连线
            ax2.plot([x + 0.2, nodes[i+1][0] - 0.2], [y, y], 'k-', linewidth=2)
    
    # 从功率源到第一个节点的连线
    ax2.plot([1.3, 2.3], [6.5, 6.5], 'k-', linewidth=2)
    
    # 热阻
    rth_positions = [(3.5, 5.5), (5.5, 5.5), (7.5, 5.5)]
    rth_labels = ['Rth_jc', 'Rth_cs', 'Rth_sa']
    
    for (x, y), label in zip(rth_positions, rth_labels):
        # 热阻符号（锯齿形）
        rth_box = Rectangle((x-0.3, y-0.2), 0.6, 0.4, facecolor='white', edgecolor='black')
        ax2.add_patch(rth_box)
        ax2.text(x, y, label, ha='center', va='center', fontsize=9)
        
        # 连接到主线的垂直线
        ax2.plot([x, x], [y+0.2, 6.5], 'k-', linewidth=1)
        ax2.plot([x, x], [y-0.2, 4.5], 'k-', linewidth=1)
    
    # 热容（到地）
    cth_positions = [(2.5, 4.5), (4.5, 4.5), (6.5, 4.5)]
    cth_labels = ['Cth_j', 'Cth_c', 'Cth_s']
    
    for (x, y), label in zip(cth_positions, cth_labels):
        # 热容符号（两条平行线）
        ax2.plot([x-0.2, x+0.2], [y, y], 'k-', linewidth=3)
        ax2.plot([x-0.2, x+0.2], [y-0.3, y-0.3], 'k-', linewidth=3)
        ax2.text(x, y-0.6, label, ha='center', va='center', fontsize=9)
        
        # 连接线
        ax2.plot([x, x], [y+0.1, 6.5], 'k-', linewidth=1)
        
        # 接地符号
        for i in range(3):
            ax2.plot([x-0.1+i*0.1, x-0.1+i*0.1], [y-0.3-i*0.1, y-0.4-i*0.1], 'k-', linewidth=2-i*0.5)
    
    ax2.set_title('等效热网络电路', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('pic/PPT_热网络等效电路.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_loss_characteristics():
    """创建IGBT损耗特性曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. I-V特性曲线
    current = np.linspace(0, 1500, 100)
    vce_25 = 1.75 + 1.1e-3 * current
    vce_125 = 2.2 + 1.3e-3 * current
    
    ax1.plot(current, vce_25, 'b-', linewidth=2, label='25°C')
    ax1.plot(current, vce_125, 'r-', linewidth=2, label='125°C')
    ax1.set_xlabel('集电极电流 Ic (A)')
    ax1.set_ylabel('饱和压降 Vce_sat (V)')
    ax1.set_title('IGBT饱和压降特性')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 导通损耗vs电流
    P_cond_25 = vce_25 * current * 0.318  # 假设占空比导致的平均电流系数
    P_cond_125 = vce_125 * current * 0.318
    
    ax2.plot(current, P_cond_25, 'b-', linewidth=2, label='25°C')
    ax2.plot(current, P_cond_125, 'r-', linewidth=2, label='125°C')
    ax2.set_xlabel('集电极电流 Ic (A)')
    ax2.set_ylabel('导通损耗 (W)')
    ax2.set_title('IGBT导通损耗')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 开关损耗vs电流
    current_sw = np.linspace(0, 1500, 100)
    Eon = 25e-3 * (current_sw / 1000)  # 线性缩放
    Eoff = 32e-3 * (current_sw / 1000)
    
    ax3.plot(current_sw, Eon * 1000, 'g-', linewidth=2, label='开通损耗 Eon')
    ax3.plot(current_sw, Eoff * 1000, 'purple', linewidth=2, label='关断损耗 Eoff')
    ax3.plot(current_sw, (Eon + Eoff) * 1000, 'k--', linewidth=2, label='总开关损耗')
    ax3.set_xlabel('集电极电流 Ic (A)')
    ax3.set_ylabel('开关能量 (mJ)')
    ax3.set_title('IGBT开关损耗能量')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 总损耗vs频率
    frequency = np.linspace(100, 2000, 100)
    current_ref = 500  # A
    P_cond_ref = 2.0 * current_ref * 0.318  # 导通损耗（不随频率变化）
    P_sw_ref = (25e-3 + 32e-3) * (current_ref / 1000) * frequency  # 开关损耗（正比于频率）
    
    ax4.plot(frequency, np.full_like(frequency, P_cond_ref), 'b-', linewidth=2, label='导通损耗')
    ax4.plot(frequency, P_sw_ref, 'r-', linewidth=2, label='开关损耗')
    ax4.plot(frequency, P_cond_ref + P_sw_ref, 'k-', linewidth=2, label='总损耗')
    ax4.set_xlabel('开关频率 (Hz)')
    ax4.set_ylabel('功率损耗 (W)')
    ax4.set_title(f'损耗vs开关频率 (Ic={current_ref}A)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/PPT_IGBT损耗特性.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_thermal_response():
    """创建热响应特性"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 阶跃功率响应
    time_min = np.linspace(0, 60, 1000)
    
    # 三阶RC网络响应
    Rth_jc, Rth_cs, Rth_sa = 0.04, 0.005, 0.025
    Cth_j, Cth_c, Cth_s = 800, 3000, 15000
    
    tau1 = Rth_jc * Cth_j / 60  # 转换为分钟
    tau2 = Rth_cs * Cth_c / 60
    tau3 = Rth_sa * Cth_s / 60
    
    P_step = 1000  # 1kW阶跃
    T_ambient = 25
    
    # 简化的多阶响应（近似）
    Tj = T_ambient + P_step * (Rth_jc + Rth_cs + Rth_sa) * (
        1 - 0.6 * np.exp(-time_min / tau1) - 
        0.3 * np.exp(-time_min / tau2) - 
        0.1 * np.exp(-time_min / tau3)
    )
    
    Tc = T_ambient + P_step * (Rth_cs + Rth_sa) * (
        1 - 0.7 * np.exp(-time_min / tau2) - 
        0.3 * np.exp(-time_min / tau3)
    )
    
    Ts = T_ambient + P_step * Rth_sa * (1 - np.exp(-time_min / tau3))
    
    ax1.plot(time_min, Tj, 'r-', linewidth=2, label='结温 Tj')
    ax1.plot(time_min, Tc, 'b-', linewidth=2, label='壳温 Tc')
    ax1.plot(time_min, Ts, 'g-', linewidth=2, label='散热器温度 Ts')
    ax1.axhline(y=T_ambient, color='k', linestyle='--', alpha=0.5, label='环境温度')
    ax1.set_xlabel('时间 (分钟)')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('1kW功率阶跃的热响应')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 不同功率等级的稳态温升
    power_levels = np.array([200, 500, 1000, 1500, 2000])
    temp_rise = power_levels * (Rth_jc + Rth_cs + Rth_sa)
    
    ax2.plot(power_levels, temp_rise, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('功率损耗 (W)')
    ax2.set_ylabel('稳态温升 (K)')
    ax2.set_title('稳态温升vs功率损耗')
    ax2.grid(True, alpha=0.3)
    
    # 添加线性拟合线
    ax2.plot(power_levels, power_levels * (Rth_jc + Rth_cs + Rth_sa), 'b--', alpha=0.7, 
             label=f'斜率 = {Rth_jc + Rth_cs + Rth_sa:.3f} K/W')
    ax2.legend()
    
    # 3. 温度对电参数的影响
    temp_range = np.linspace(25, 175, 100)
    
    # 饱和压降随温度变化
    Vce_sat_temp = 1.75 * (1 + 0.004 * (temp_range - 25))
    # 导通电阻随温度变化
    Rce_temp = 1.1e-3 * (1 + 0.006 * (temp_range - 25))
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(temp_range, Vce_sat_temp, 'r-', linewidth=2, label='饱和压降 Vce_sat')
    line2 = ax3_twin.plot(temp_range, Rce_temp * 1000, 'b-', linewidth=2, label='导通电阻 Rce')
    
    ax3.set_xlabel('结温 (°C)')
    ax3.set_ylabel('饱和压降 (V)', color='r')
    ax3_twin.set_ylabel('导通电阻 (mΩ)', color='b')
    ax3.set_title('电参数的温度特性')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. 电热耦合效应
    time_hours = np.linspace(0, 24, 100)
    
    # 模拟负载变化
    load_factor = 0.8 + 0.3 * np.sin(2 * np.pi * time_hours / 24)
    base_power = 800
    power_profile = base_power * load_factor
    
    # 简化的温度响应（考虑热时间常数）
    temp_response = 25 + power_profile * 0.07  # 简化的温升计算
    
    # 考虑温度反馈的功率修正
    temp_factor = 1 + 0.002 * (temp_response - 25)
    corrected_power = power_profile * temp_factor
    
    ax4.plot(time_hours, power_profile, 'b-', linewidth=2, label='理想功率')
    ax4.plot(time_hours, corrected_power, 'r--', linewidth=2, label='考虑温度反馈的功率')
    ax4_twin2 = ax4.twinx()
    ax4_twin2.plot(time_hours, temp_response, 'g:', linewidth=2, label='结温')
    
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('功率损耗 (W)')
    ax4_twin2.set_ylabel('结温 (°C)', color='g')
    ax4.set_title('电热耦合效应')
    
    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin2.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pic/PPT_热响应特性.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_system_architecture():
    """创建系统架构图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：级联H桥系统架构
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 10)
    
    # 绘制三相系统
    phases = ['A相', 'B相', 'C相']
    colors = ['red', 'blue', 'green']
    
    for phase_idx, (phase, color) in enumerate(zip(phases, colors)):
        y_base = 8 - phase_idx * 2.5
        
        # 相标签
        ax1.text(0.5, y_base, phase, fontsize=12, fontweight='bold', 
                ha='center', va='center', color=color)
        
        # 绘制5个H桥模块（代表40个中的5个）
        for i in range(5):
            x = 2 + i * 2
            
            # H桥模块
            module = Rectangle((x-0.4, y_base-0.4), 0.8, 0.8, 
                             facecolor=color, alpha=0.3, edgecolor='black')
            ax1.add_patch(module)
            
            # IGBT标记
            ax1.text(x, y_base+0.1, 'T1', fontsize=8, ha='center')
            ax1.text(x, y_base-0.1, 'T2', fontsize=8, ha='center')
            
            if i < 4:
                # 连接线
                ax1.plot([x+0.4, x+1.6], [y_base, y_base], color='black', linewidth=1)
            elif i == 4:
                # 省略号
                ax1.text(x+0.8, y_base, '...', fontsize=16, ha='center', va='center')
        
        # 标注40个模块
        ax1.text(6, y_base-0.8, '40个H桥模块', fontsize=10, ha='center', style='italic')
    
    # 系统参数标注
    ax1.text(6, 9.5, '35kV/25MW级联储能PCS', fontsize=14, fontweight='bold', ha='center')
    ax1.text(6, 0.5, '总计：120个模块，240个IGBT', fontsize=12, ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax1.set_title('系统架构图', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 右图：单个H桥模块详细结构
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # 直流母线
    ax2.add_patch(Rectangle((1, 8), 8, 0.5, facecolor='red', alpha=0.7))
    ax2.text(5, 8.25, 'DC+ (875V)', ha='center', va='center', fontweight='bold', color='white')
    
    ax2.add_patch(Rectangle((1, 1), 8, 0.5, facecolor='blue', alpha=0.7))
    ax2.text(5, 1.25, 'DC- (0V)', ha='center', va='center', fontweight='bold', color='white')
    
    # IGBT和二极管
    # 上桥臂
    igbt1 = Rectangle((4, 6.5), 2, 1, facecolor='orange', edgecolor='black')
    ax2.add_patch(igbt1)
    ax2.text(5, 7, 'T1', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 上桥臂二极管
    diode1 = patches.Polygon([(4.2, 6.2), (4.8, 6.2), (4.5, 5.8)], 
                           facecolor='yellow', edgecolor='black')
    ax2.add_patch(diode1)
    ax2.text(4.5, 6, 'D1', ha='center', va='center', fontsize=8)
    
    # 下桥臂
    igbt2 = Rectangle((4, 2.5), 2, 1, facecolor='orange', edgecolor='black')
    ax2.add_patch(igbt2)
    ax2.text(5, 3, 'T2', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 下桥臂二极管
    diode2 = patches.Polygon([(5.2, 3.8), (5.8, 3.8), (5.5, 4.2)], 
                           facecolor='yellow', edgecolor='black')
    ax2.add_patch(diode2)
    ax2.text(5.5, 4, 'D2', ha='center', va='center', fontsize=8)
    
    # 连接线
    ax2.plot([5, 5], [8, 7.5], 'k-', linewidth=2)  # DC+到T1
    ax2.plot([5, 5], [1.5, 2.5], 'k-', linewidth=2)  # T2到DC-
    ax2.plot([5, 5], [3.5, 6.5], 'k-', linewidth=3)  # 中点连接
    
    # 输出
    ax2.plot([5, 7.5], [4.5, 4.5], 'k-', linewidth=3)
    ax2.text(8, 4.5, 'AC输出', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # 母线电容
    cap_positions = [(2, 5), (2.5, 5), (3, 5)]
    for x, y in cap_positions:
        # 电容符号
        ax2.plot([x, x], [y-0.3, y+0.3], 'k-', linewidth=3)
        ax2.plot([x+0.1, x+0.1], [y-0.3, y+0.3], 'k-', linewidth=3)
        ax2.plot([x, x], [y+0.3, 8], 'k-', linewidth=1)
        ax2.plot([x, x], [y-0.3, 1.5], 'k-', linewidth=1)
    
    ax2.text(2.5, 5.8, '21个电容\n15mF', ha='center', va='center', fontsize=9)
    
    # 参数标注
    param_text = """
    模块参数：
    • 直流电压: 875V
    • 额定电流: 420A
    • 开关频率: 1kHz
    • IGBT: FF1500R17IP5R
    """
    ax2.text(0.5, 7, param_text, fontsize=9, va='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax2.set_title('单个H桥模块结构', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('pic/PPT_系统架构图.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_modeling_workflow():
    """创建建模流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    
    # 定义流程步骤
    steps = [
        (2, 10, "参数获取", "数据手册\n测试数据\n文献资料"),
        (7, 10, "电性能建模", "I-V特性\n开关特性\n损耗模型"),
        (12, 10, "热网络建模", "热阻网络\n热容参数\n边界条件"),
        (2, 7, "模型验证", "静态验证\n动态验证\n温度验证"),
        (7, 7, "电热耦合", "温度反馈\n迭代计算\n收敛判断"),
        (12, 7, "系统集成", "多模块\n系统级\n控制策略"),
        (4.5, 4, "应用分析", "效率分析\n热设计\n寿命预测"),
        (9.5, 4, "优化设计", "参数优化\n散热优化\n控制优化"),
        (7, 1, "工程应用", "实际产品\n性能验证\n持续改进")
    ]
    
    # 绘制步骤框
    boxes = []
    for x, y, title, content in steps:
        # 创建圆角矩形
        box = FancyBboxPatch((x-1, y-0.8), 2, 1.6, 
                           boxstyle="round,pad=0.1", 
                           facecolor='lightblue', 
                           edgecolor='navy', 
                           linewidth=2)
        ax.add_patch(box)
        
        # 添加文字
        ax.text(x, y+0.3, title, ha='center', va='center', 
               fontsize=11, fontweight='bold')
        ax.text(x, y-0.3, content, ha='center', va='center', 
               fontsize=9, style='italic')
        
        boxes.append((x, y))
    
    # 绘制箭头连接
    connections = [
        (0, 1), (1, 2), (0, 3), (1, 4), (2, 5),  # 第一行到第二行
        (3, 4), (4, 5),  # 第二行内部
        (3, 6), (4, 7), (5, 7),  # 第二行到第三行
        (6, 7),  # 第三行内部
        (6, 8), (7, 8)  # 第三行到第四行
    ]
    
    for start_idx, end_idx in connections:
        x1, y1 = boxes[start_idx]
        x2, y2 = boxes[end_idx]
        
        # 计算箭头方向
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) > abs(dy):  # 水平方向为主
            if dx > 0:  # 向右
                start_x, start_y = x1 + 1, y1
                end_x, end_y = x2 - 1, y2
            else:  # 向左
                start_x, start_y = x1 - 1, y1
                end_x, end_y = x2 + 1, y2
        else:  # 垂直方向为主
            if dy > 0:  # 向上
                start_x, start_y = x1, y1 + 0.8
                end_x, end_y = x2, y2 - 0.8
            else:  # 向下
                start_x, start_y = x1, y1 - 0.8
                end_x, end_y = x2, y2 + 0.8
        
        # 绘制箭头
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))
    
    # 添加反馈回路
    ax.annotate('', xy=(3, 9.2), xytext=(11, 7.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='green', 
                             connectionstyle="arc3,rad=0.3"))
    ax.text(7, 8.5, '反馈优化', ha='center', va='center', 
           fontsize=10, color='green', fontweight='bold')
    
    ax.set_title('IGBT电热建模完整流程', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('pic/PPT_建模流程图.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 创建所有PPT用图表
    print("正在生成PPT用图表...")
    
    print("1. 创建热网络等效电路图...")
    create_thermal_network_diagram()
    
    print("2. 创建IGBT损耗特性图...")
    create_loss_characteristics()
    
    print("3. 创建热响应特性图...")
    create_thermal_response()
    
    print("4. 创建系统架构图...")
    create_system_architecture()
    
    print("5. 创建建模流程图...")
    create_modeling_workflow()
    
    print("\n所有PPT图表已生成完成！")
    print("图表保存在 pic/ 文件夹中：")
    print("- PPT_热网络等效电路.png")
    print("- PPT_IGBT损耗特性.png") 
    print("- PPT_热响应特性.png")
    print("- PPT_系统架构图.png")
    print("- PPT_建模流程图.png")
