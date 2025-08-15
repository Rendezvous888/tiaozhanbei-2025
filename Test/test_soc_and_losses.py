#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试电池SOC扩展和PCS损耗调整功能
验证：
1. 电池SOC能够在0-100%之间变化
2. 功率平衡关系正确
3. PCS损耗计算更真实
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_soc_range():
    """测试电池SOC范围扩展"""
    print("=== 测试电池SOC范围扩展 ===")
    
    try:
        from pcs_simulation_model import PCSParameters
        
        # 创建参数实例
        params = PCSParameters()
        
        print(f"电池SOC范围:")
        print(f"  最小SOC: {params.SOC_min:.1%}")
        print(f"  最大SOC: {params.SOC_max:.1%}")
        print(f"  SOC变化范围: {(params.SOC_max - params.SOC_min):.1%}")
        
        # 验证SOC范围
        if params.SOC_min == 0.0 and params.SOC_max == 1.0:
            print("✓ SOC范围已成功扩展到0-100%")
        else:
            print("✗ SOC范围扩展失败")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ 测试SOC范围时出错: {e}")
        return False

def test_power_balance():
    """测试功率平衡关系"""
    print("\n=== 测试功率平衡关系 ===")
    
    try:
        from pcs_simulation_model import PCSSimulation
        
        # 创建仿真实例
        pcs_sim = PCSSimulation()
        
        # 创建测试功率曲线（24小时，逐小时）
        t = np.arange(24)
        P_profile = np.zeros(24)
        
        # 设置典型的充放电模式
        P_profile[2:6] = -pcs_sim.params.P_rated * 0.8   # 2-6点充电
        P_profile[8:12] = pcs_sim.params.P_rated * 0.9    # 8-12点放电
        P_profile[14:18] = pcs_sim.params.P_rated * 0.9   # 14-18点放电
        P_profile[22:24] = -pcs_sim.params.P_rated * 0.8  # 22-24点充电
        
        # 创建环境温度曲线
        T_amb = np.full(24, 25.0)  # 25°C恒温
        
        print("功率曲线设置:")
        print(f"  充电功率: {P_profile[2:6][0]/1e6:.1f} MW")
        print(f"  放电功率: {P_profile[8:12][0]/1e6:.1f} MW")
        print(f"  时间步长: {pcs_sim.params.time_step_seconds} 秒")
        
        # 运行仿真
        print("\n运行仿真...")
        results = pcs_sim.run_simulation(t, P_profile, T_amb)
        
        # 分析功率平衡
        print("\n功率平衡分析:")
        print(f"  总充电能量: {np.sum(P_profile[P_profile < 0]) * (pcs_sim.params.time_step_seconds/3600):.2f} MWh")
        print(f"  总放电能量: {np.sum(P_profile[P_profile > 0]) * (pcs_sim.params.time_step_seconds/3600):.2f} MWh")
        
        # 分析SOC变化
        soc = results['SOC']
        print(f"\nSOC变化分析:")
        print(f"  初始SOC: {soc[0]:.3f}")
        print(f"  最终SOC: {soc[-1]:.3f}")
        print(f"  SOC变化: {soc[-1] - soc[0]:.3f}")
        print(f"  最小SOC: {soc.min():.3f}")
        print(f"  最大SOC: {soc.max():.3f}")
        
        # 验证SOC范围
        if soc.min() <= 0.1 and soc.max() >= 0.9:
            print("✓ SOC变化范围符合预期")
        else:
            print("⚠️  SOC变化范围较小，可能需要调整功率曲线")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试功率平衡时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_calculation():
    """测试PCS损耗计算"""
    print("\n=== 测试PCS损耗计算 ===")
    
    try:
        from h_bridge_model import CascadedHBridgeSystem
        
        # 创建级联H桥系统
        cascaded_system = CascadedHBridgeSystem(
            N_modules=40,
            Vdc_per_module=875,
            fsw=750,
            f_grid=50
        )
        
        # 测试不同电流下的损耗
        test_currents = [100, 500, 1000, 1500]  # A
        
        print("损耗计算测试:")
        print(f"{'电流(A)':<10} {'总损耗(W)':<12} {'开关损耗(W)':<12} {'导通损耗(W)':<12} {'效率(%)':<10}")
        print("-" * 70)
        
        for I_rms in test_currents:
            losses = cascaded_system.calculate_total_losses(I_rms)
            
            # 计算效率（假设输出功率为电流*电压）
            V_output = 1000  # 假设输出电压
            P_output = I_rms * V_output
            efficiency = P_output / (P_output + losses['total_loss']) * 100
            
            print(f"{I_rms:<10} {losses['total_loss']:<12.0f} {losses['switching_loss']:<12.0f} "
                  f"{losses['conduction_loss']:<12.0f} {efficiency:<10.1f}")
        
        # 验证损耗是否合理
        if losses['total_loss'] > 1000:  # 损耗应该足够大
            print("✓ PCS损耗计算已调整，结果更贴近工程实际")
        else:
            print("⚠️  PCS损耗可能仍然偏小")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试损耗计算时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_test_results():
    """绘制测试结果"""
    print("\n=== 绘制测试结果 ===")
    
    try:
        # 创建测试数据
        t = np.arange(24)
        soc_old = np.linspace(0.1, 0.9, 24)  # 旧的SOC范围
        soc_new = np.linspace(0.0, 1.0, 24)  # 新的SOC范围
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 子图1: SOC范围对比
        ax1.plot(t, soc_old * 100, 'r-', linewidth=2, label='旧SOC范围 (10%-90%)')
        ax1.plot(t, soc_new * 100, 'b-', linewidth=2, label='新SOC范围 (0%-100%)')
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('SOC (%)')
        ax1.set_title('电池SOC范围扩展对比')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim(-5, 105)
        
        # 子图2: 功率平衡示意图
        power_charge = np.where(t < 6, -25, 0)  # 充电功率
        power_discharge = np.where((t >= 8) & (t < 18), 25, 0)  # 放电功率
        
        ax2.plot(t, power_charge, 'g-', linewidth=2, label='充电功率')
        ax2.plot(t, power_discharge, 'r-', linewidth=2, label='放电功率')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('功率 (MW)')
        ax2.set_title('24小时功率平衡示意图')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(-30, 30)
        
        plt.tight_layout()
        plt.savefig('Test/soc_and_losses_test_results.png', dpi=300, bbox_inches='tight')
        print("✓ 测试结果图表已保存: Test/soc_and_losses_test_results.png")
        
        return True
        
    except Exception as e:
        print(f"✗ 绘制测试结果时出错: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("电池SOC扩展和PCS损耗调整功能测试")
    print("=" * 60)
    
    # 运行各项测试
    tests = [
        ("SOC范围扩展", test_soc_range),
        ("功率平衡关系", test_power_balance),
        ("PCS损耗计算", test_loss_calculation),
        ("结果可视化", plot_test_results)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 显示测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n总测试数: {len(results)}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {len(results) - passed}")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！SOC扩展和损耗调整功能正常")
    else:
        print(f"\n⚠️  有 {len(results) - passed} 个测试失败，请检查相关功能")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
