#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
35 kV/25 MW PCS主仿真程序
集成所有模块的完整仿真流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 导入自定义模块
from pcs_simulation_model import PCSSimulation
from load_profile import generate_profiles
from h_bridge_model import CascadedHBridgeSystem
from enhanced_igbt_life_model import EnhancedIGBTLifeModel
from detailed_life_analysis import DetailedLifeAnalysis
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主程序"""
    print("=" * 60)
    print("35 kV/25 MW级联储能PCS仿真系统")
    print("=" * 60)
    print(f"仿真开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 导入各个模块
        print("\n1. 导入仿真模块...")
        from pcs_simulation_model import PCSSimulation
        from h_bridge_model import CascadedHBridgeSystem, simulate_hbridge_system
        from control_optimization import PCSController, AdvancedControlStrategies, PerformanceEvaluator
        
        # 创建主仿真系统
        print("\n2. 初始化PCS仿真系统...")
        pcs_sim = PCSSimulation()
        
        # 显示系统参数
        print_system_parameters(pcs_sim.params)
        
        # 运行H桥系统仿真
        print("\n3. 运行H桥系统仿真...")
        hbridge_system, V_output, hbridge_losses = simulate_hbridge_system()
        
        # 运行主仿真
        print("\n4. 运行24小时运行仿真...")
        # 使用真实化负载与温度曲线（逐分钟，24h）
        # 时间步长读取自全局参数
        from device_parameters import get_optimized_parameters
        sys_params = get_optimized_parameters()['system']
        step_seconds = int(getattr(sys_params, 'time_step_seconds', 60))
        P_profile, T_amb = generate_profiles(day_type="summer-weekday", step_seconds=step_seconds)
        t = np.arange(len(P_profile)) * (step_seconds / 3600.0)  # 小时
        results = pcs_sim.run_simulation(t, P_profile, T_amb_profile=T_amb)
        
        # 分析结果
        print("\n5. 分析仿真结果...")
        analysis = pcs_sim.analyze_results(results)
        
        # 显示结果
        print_simulation_results(analysis)
        
        # 创建控制器和性能评估器
        print("\n6. 创建控制系统...")
        controller = PCSController(pcs_sim.params)
        advanced_control = AdvancedControlStrategies(pcs_sim.params)
        evaluator = PerformanceEvaluator()
        
        # 绘制综合结果
        print("\n7. 生成仿真图表...")
        plot_comprehensive_results(results, analysis, pcs_sim, hbridge_system)
        
        # 生成报告
        print("\n8. 生成仿真报告...")
        generate_simulation_report(results, analysis, pcs_sim.params)
        
        print(f"\n仿真完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
    except ImportError as e:
        print(f"导入模块错误: {e}")
        print("请确保所有依赖模块都已正确安装")
    except Exception as e:
        print(f"仿真过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def print_system_parameters(params):
    """打印系统参数"""
    print("\n系统配置参数:")
    print("-" * 40)
    print(f"额定功率: {params.P_rated/1e6:.1f} MW")
    print(f"并网电压: {params.V_grid/1e3:.1f} kV")
    print(f"额定电流: {params.I_rated:.1f} A")
    print(f"每相H桥模块数: {params.N_modules_per_phase}")
    print(f"每模块直流电压: {params.Vdc_per_module:.1f} V")
    print(f"IGBT开关频率: {params.fsw} Hz")
    print(f"电网频率: {params.f_grid} Hz")
    print(f"电池容量: {params.C_battery} Ah")
    print(f"环境温度: {params.T_amb} °C")
    
    # 显示详细的器件参数
    from device_parameters import get_optimized_parameters
    device_params = get_optimized_parameters()
    
    print(f"\n器件参数详情:")
    print("-" * 40)
    print(f"IGBT型号: {device_params['igbt'].model}")
    print(f"IGBT额定电压: {device_params['igbt'].Vces_V} V")
    print(f"IGBT额定电流: {device_params['igbt'].Ic_dc_A} A")
    print(f"电容器型号: {device_params['capacitor'].manufacturer}")
    print(f"电容器电容值: {device_params['capacitor'].current_params['capacitance_uF']} μF")
    print(f"电容器ESR: {device_params['capacitor'].current_params['ESR_mOhm']} mΩ")
    print(f"电容器寿命: {device_params['capacitor'].current_params['lifetime_h']} 小时")

def print_simulation_results(analysis):
    """打印仿真结果"""
    print("\n仿真结果分析:")
    print("-" * 40)
    print(f"IGBT寿命剩余: {analysis['igbt_life_remaining']*100:.2f}%")
    print(f"电容寿命剩余: {analysis['capacitor_life_remaining']*100:.2f}%")
    print(f"平均效率: {analysis['avg_efficiency']*100:.2f}%")
    print(f"最大效率: {analysis['max_efficiency']*100:.2f}%")
    print(f"最小效率: {analysis['min_efficiency']*100:.2f}%")
    print(f"最大结温: {analysis['max_Tj']:.1f} °C")
    print(f"平均结温: {analysis['avg_Tj']:.1f} °C")

def plot_comprehensive_results(results, analysis, pcs_sim, hbridge_system):
    """绘制综合仿真结果"""
    # 使用自适应绘图工具创建图形
    fig, axes = create_adaptive_figure(3, 3, title='35 kV/25 MW PCS综合仿真结果', title_size=16)
    
    # 子图1: 功率曲线
    ax1 = axes[0, 0]
    ax1.plot(results['time'], results['power'] / 1e6, 'b-', linewidth=2)
    format_axis_labels(ax1, '时间 (小时)', '功率 (MW)', '24小时功率曲线')
    add_grid(ax1)
    set_adaptive_ylim(ax1, results['power'] / 1e6)
    
    # 子图2: 温度曲线
    ax2 = axes[0, 1]
    # 适配可能出现的标量退化（若模型返回常数）
    Tj = results['Tj']
    Tc = results['Tc']
    if np.isscalar(Tj) or (isinstance(Tj, np.ndarray) and Tj.shape == (1,)):
        Tj = np.full_like(results['time'], float(np.atleast_1d(Tj)[0]))
    if np.isscalar(Tc) or (isinstance(Tc, np.ndarray) and Tc.shape == (1,)):
        Tc = np.full_like(results['time'], float(np.atleast_1d(Tc)[0]))
    ax2.plot(results['time'], Tj, 'r-', label='结温', linewidth=2)
    ax2.plot(results['time'], Tc, 'g-', label='壳温', linewidth=2)
    format_axis_labels(ax2, '时间 (小时)', '温度 (°C)', '温度响应')
    ax2.legend(fontsize=8, loc='best')
    add_grid(ax2)
    set_adaptive_ylim(ax2, np.concatenate([results['Tj'], results['Tc']]))
    
    # 子图3: 效率曲线
    ax3 = axes[0, 2]
    # 过滤NaN效率，避免绘图报错
    eff = results['efficiency']
    if isinstance(eff, np.ndarray):
        mask = np.isfinite(eff)
        eff_plot = eff[mask]
        time_plot = results['time'][mask]
    else:
        eff_plot = eff
        time_plot = results['time']
    if isinstance(eff_plot, np.ndarray) and eff_plot.size > 0:
        ax3.plot(time_plot, eff_plot * 100, 'purple', linewidth=2)
    format_axis_labels(ax3, '时间 (小时)', '效率 (%)', '系统效率')
    add_grid(ax3)
    if isinstance(eff_plot, np.ndarray) and eff_plot.size > 0:
        set_adaptive_ylim(ax3, eff_plot * 100)
    
    # 子图4: SOC曲线
    ax4 = axes[1, 0]
    ax4.plot(results['time'], results['SOC'] * 100, 'orange', linewidth=2)
    format_axis_labels(ax4, '时间 (小时)', 'SOC (%)', '电池荷电状态')
    add_grid(ax4)
    set_adaptive_ylim(ax4, results['SOC'] * 100)
    
    # 子图5: 损耗曲线
    ax5 = axes[1, 1]
    ax5.plot(results['time'], results['P_loss'] / 1e3, 'brown', linewidth=2)
    format_axis_labels(ax5, '时间 (小时)', '损耗 (kW)', '系统损耗')
    add_grid(ax5)
    set_adaptive_ylim(ax5, results['P_loss'] / 1e3)
    
    # 子图6: 功率分布直方图
    ax6 = axes[1, 2]
    ax6.hist(results.get('power_effective', results['power']) / 1e6, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    format_axis_labels(ax6, '功率 (MW)', '频次', '功率分布')
    add_grid(ax6)
    set_adaptive_ylim(ax6, [0, None])  # 直方图从0开始
    
    # 子图7: 温度分布直方图
    ax7 = axes[2, 0]
    ax7.hist(results['Tj'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    format_axis_labels(ax7, '结温 (°C)', '频次', '温度分布')
    add_grid(ax7)
    set_adaptive_ylim(ax7, [0, None])  # 直方图从0开始
    
    # 子图8: 效率分布直方图
    ax8 = axes[2, 1]
    eff_hist = results['efficiency']
    eff_hist = eff_hist[np.isfinite(eff_hist)] if isinstance(eff_hist, np.ndarray) else eff_hist
    if isinstance(eff_hist, np.ndarray) and eff_hist.size > 0:
        ax8.hist(eff_hist * 100, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    format_axis_labels(ax8, '效率 (%)', '频次', '效率分布')
    add_grid(ax8)
    set_adaptive_ylim(ax8, [0, None])  # 直方图从0开始
    
    # 子图9: 系统信息
    ax9 = axes[2, 2]
    ax9.axis('off')
    info_text = f"""System Performance Summary:

Basic Parameters:
• Rated Power: {pcs_sim.params.P_rated/1e6:.1f} MW
• Grid Voltage: {pcs_sim.params.V_grid/1e3:.1f} kV
• Modules per Phase: {pcs_sim.params.N_modules_per_phase}

Performance Metrics:
• IGBT Life: {analysis['igbt_life_remaining']*100:.1f}%
• Capacitor Life: {analysis['capacitor_life_remaining']*100:.1f}%
• Avg Efficiency: {analysis['avg_efficiency']*100:.1f}%
• Max Junction Temp: {analysis['max_Tj']:.1f}°C

Operation Status:
• Charge Period: 2-6h, 22-24h
• Discharge Period: 8-12h, 14-18h
• Switching Frequency: {pcs_sim.params.fsw} Hz"""
    
    # 使用自适应文本框，避免文字重叠
    ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             wrap=True)
    
    # 优化布局，避免重叠
    optimize_layout(fig, tight_layout=True, h_pad=1.5, w_pad=1.5)
    
    # 显示图形
    finalize_plot(fig)

def generate_simulation_report(results, analysis, params):
    """生成仿真报告"""
    print("\n生成仿真报告...")
    
    # 创建报告数据
    report_data = {
        'Parameter': [
            'Rated Power (MW)',
            'Grid Voltage (kV)',
            'Modules per Phase',
            'IGBT Switching Frequency (Hz)',
            'IGBT Life Remaining (%)',
            'Capacitor Life Remaining (%)',
            'Average Efficiency (%)',
            'Max Junction Temp (°C)',
            'Average Junction Temp (°C)',
            'Max Efficiency (%)',
            'Min Efficiency (%)'
        ],
        'Value': [
            f"{params.P_rated/1e6:.1f}",
            f"{params.V_grid/1e3:.1f}",
            f"{params.N_modules_per_phase}",
            f"{params.fsw}",
            f"{analysis['igbt_life_remaining']*100:.2f}",
            f"{analysis['capacitor_life_remaining']*100:.2f}",
            f"{analysis['avg_efficiency']*100:.2f}",
            f"{analysis['max_Tj']:.1f}",
            f"{analysis['avg_Tj']:.1f}",
            f"{analysis['max_efficiency']*100:.2f}",
            f"{analysis['min_efficiency']*100:.2f}"
        ]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(report_data)
    
    # 确保result目录存在
    import os
    os.makedirs('result', exist_ok=True)
    
    # 保存报告
    report_filename = f"result/PCS_仿真报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_filename, index=False, encoding='utf-8-sig')
    
    print(f"仿真报告已保存: {report_filename}")
    
    # 显示报告摘要
    print("\n仿真报告摘要:")
    print("-" * 50)
    print(df.to_string(index=False))
    
    return df

def run_quick_test():
    """快速测试功能"""
    print("\n运行快速测试...")
    
    try:
        # 测试基本功能
        from pcs_simulation_model import PCSParameters
        params = PCSParameters()
        
        print("✓ 系统参数初始化成功")
        print(f"  - 额定功率: {params.P_rated/1e6:.1f} MW")
        print(f"  - 模块数: {params.N_modules_per_phase}")
        
        # 测试H桥功能
        from h_bridge_model import HBridgeUnit
        hbridge = HBridgeUnit()
        print("✓ H桥单元建模成功")
        
        # 测试控制器
        from control_optimization import PCSController
        controller = PCSController(params)
        print("✓ 控制器初始化成功")
        
        print("\n所有模块测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    # 运行快速测试
    if run_quick_test():
        # 运行主仿真
        main()
    else:
        print("模块测试失败，请检查依赖项") 