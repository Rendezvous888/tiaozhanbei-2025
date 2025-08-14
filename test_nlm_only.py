#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试NLM调制策略的性能
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import CascadedHBridgeSystem

def test_nlm_performance():
    """测试NLM调制策略的性能"""
    print("=== NLM调制策略性能测试 ===")
    
    # 系统参数 - 35kV系统
    N_modules = 40
    Vdc_per_module = 875  # 35kV / 40 = 875V
    fsw = 1000  # Hz
    f_grid = 50  # Hz
    
    # 创建级联H桥系统（仅NLM）
    system = CascadedHBridgeSystem(N_modules, Vdc_per_module, fsw, f_grid)
    
    print(f"系统配置:")
    print(f"- 模块数: {system.N_modules}")
    print(f"- 每模块直流电压: {system.Vdc_per_module} V")
    print(f"- 总输出电压: {system.V_total/1000:.1f} kV")
    print(f"- 开关频率: {system.fsw} Hz")
    print(f"- 调制策略: {system.modulation_strategy}")
    
    # 仿真时间 - 多个工频周期以确保THD计算准确
    t = np.linspace(0, 0.1, 5000)  # 5个工频周期
    
    # 测试不同调制比
    modulation_indices = [0.6, 0.7, 0.8, 0.9]
    
    results = {}
    
    for mi in modulation_indices:
        print(f"\n测试调制比: {mi}")
        
        # 生成输出电压
        V_total, V_modules = system.generate_phase_shifted_pwm(t, mi)
        
        # 计算THD
        thd = system.calculate_thd_time_domain(V_total, t) * 100.0
        
        # 计算RMS值
        V_rms = np.sqrt(np.mean(V_total**2))
        
        # 计算峰值
        V_peak = np.max(np.abs(V_total))
        
        # 计算电平数
        unique_levels = len(np.unique(V_total))
        
        results[mi] = {
            'thd': thd,
            'v_rms': V_rms,
            'v_peak': V_peak,
            'levels': unique_levels
        }
        
        print(f"  THD: {thd:.3f}%")
        print(f"  RMS电压: {V_rms/1000:.3f} kV")
        print(f"  峰值电压: {V_peak/1000:.3f} kV")
        print(f"  输出电平数: {unique_levels}")
    
    # 绘制结果
    plot_nlm_results(t, system, modulation_indices, results)
    
    return system, results

def plot_nlm_results(t, system, modulation_indices, results):
    """绘制NLM调制结果"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NLM调制策略性能分析', fontsize=16)
    
    # 第一行：波形分析
    # 选择调制比0.8的波形进行详细分析
    mi_test = 0.8
    V_total, V_modules = system.generate_phase_shifted_pwm(t, mi_test)
    
    # 输出电压波形
    axes[0, 0].plot(t * 1000, V_total / 1000, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('时间 (ms)')
    axes[0, 0].set_ylabel('输出电压 (kV)')
    axes[0, 0].set_title(f'输出电压波形 (m={mi_test})')
    axes[0, 0].grid(True)
    axes[0, 0].set_xlim(0, 40)  # 显示前2个周期
    
    # 单个模块输出（显示前5个模块）
    for i in range(min(5, len(V_modules))):
        axes[0, 1].plot(t * 1000, V_modules[i] / 1000, alpha=0.7, label=f'模块 {i+1}')
    axes[0, 1].set_xlabel('时间 (ms)')
    axes[0, 1].set_ylabel('模块电压 (kV)')
    axes[0, 1].set_title('单个模块输出电压')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_xlim(0, 40)
    
    # 第二行：性能分析
    # THD vs 调制比
    thd_values = [results[mi]['thd'] for mi in modulation_indices]
    axes[1, 0].plot(modulation_indices, thd_values, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('调制比')
    axes[1, 0].set_ylabel('THD (%)')
    axes[1, 0].set_title('THD vs 调制比')
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(0, max(thd_values) * 1.1)
    
    # 在THD图上添加目标线（5%）
    axes[1, 0].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='目标THD: 5%')
    axes[1, 0].legend()
    
    # 输出电平数 vs 调制比
    level_values = [results[mi]['levels'] for mi in modulation_indices]
    axes[1, 1].plot(modulation_indices, level_values, 'go-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('调制比')
    axes[1, 1].set_ylabel('输出电平数')
    axes[1, 1].set_title('输出电平数 vs 调制比')
    axes[1, 1].grid(True)
    
    # 添加理论电平数参考线
    theoretical_levels = [2 * system.N_modules + 1] * len(modulation_indices)
    axes[1, 1].plot(modulation_indices, theoretical_levels, 'k--', alpha=0.7, label=f'理论最大电平数: {2*system.N_modules+1}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('NLM_Modulation_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n图表已保存为: NLM_Modulation_Analysis.png")

def generate_performance_report(results):
    """生成性能报告"""
    print("\n=== NLM调制性能报告 ===")
    
    # 找到最佳性能
    best_thd = min(results.values(), key=lambda x: x['thd'])
    best_mi = [mi for mi, res in results.items() if res == best_thd][0]
    
    print(f"最佳THD性能:")
    print(f"- 调制比: {best_mi}")
    print(f"- THD: {best_thd['thd']:.3f}%")
    print(f"- RMS电压: {best_thd['v_rms']/1000:.3f} kV")
    
    # 检查是否满足THD < 5%的要求
    all_thd_below_5 = all(res['thd'] < 5.0 for res in results.values())
    
    if all_thd_below_5:
        print(f"\n✅ 所有调制比下的THD都小于5%，满足要求！")
    else:
        print(f"\n⚠️  部分调制比下的THD超过5%:")
        for mi, res in results.items():
            if res['thd'] >= 5.0:
                print(f"  - 调制比 {mi}: THD = {res['thd']:.3f}%")
    
    # 输出电平数分析
    print(f"\n输出电平数分析:")
    for mi, res in results.items():
        print(f"- 调制比 {mi}: {res['levels']} 个电平")
    
    return all_thd_below_5

if __name__ == "__main__":
    # 运行测试
    system, results = test_nlm_performance()
    
    # 生成报告
    success = generate_performance_report(results)
    
    if success:
        print(f"\n🎉 NLM调制策略测试成功！THD性能满足要求。")
    else:
        print(f"\n⚠️  NLM调制策略需要进一步优化以降低THD。")

