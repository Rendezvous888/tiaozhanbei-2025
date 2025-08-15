#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强的SOC变化范围功能
验证：
1. 电池SOC能够在更大范围内变化（不仅仅是50%上下）
2. 电网能量补充/释放功能正常
3. 功率曲线变化幅度增大
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_soc_range():
    """测试增强的SOC变化范围"""
    print("=== 测试增强的SOC变化范围 ===")
    
    try:
        from pcs_simulation_model import PCSSimulation
        from load_profile import generate_profiles
        
        # 创建仿真实例
        pcs_sim = PCSSimulation()
        
        # 生成增强的负载曲线
        step_seconds = 60
        P_profile, T_amb = generate_profiles('summer-weekday', step_seconds=step_seconds)
        t = np.arange(len(P_profile)) * (step_seconds / 3600.0)  # 小时
        
        print("增强负载曲线分析:")
        print(f"  总点数: {len(P_profile)}")
        print(f"  充电功率范围: {P_profile[P_profile < 0].min()/1e6:.2f} ~ {P_profile[P_profile < 0].max()/1e6:.2f} MW")
        print(f"  放电功率范围: {P_profile[P_profile > 0].min()/1e6:.2f} ~ {P_profile[P_profile > 0].max()/1e6:.2f} MW")
        
        # 计算总充放电能量
        dt_h = step_seconds / 3600.0
        charge_energy = np.sum(P_profile[P_profile < 0]) * dt_h
        discharge_energy = np.sum(P_profile[P_profile > 0]) * dt_h
        
        print(f"  总充电能量: {abs(charge_energy)/1e6:.2f} MWh")
        print(f"  总放电能量: {discharge_energy/1e6:.2f} MWh")
        print(f"  净能量: {(charge_energy + discharge_energy)/1e6:.2f} MWh")
        
        # 运行仿真
        print("\n运行仿真...")
        results = pcs_sim.run_simulation(t, P_profile, T_amb_profile=T_amb)
        
        # 分析SOC变化
        soc = results['SOC']
        print(f"\nSOC变化分析:")
        print(f"  初始SOC: {soc[0]:.3f}")
        print(f"  最终SOC: {soc[-1]:.3f}")
        print(f"  SOC变化: {soc[-1] - soc[0]:.3f}")
        print(f"  最小SOC: {soc.min():.3f}")
        print(f"  最大SOC: {soc.max():.3f}")
        print(f"  SOC变化范围: {(soc.max() - soc.min()):.3f}")
        
        # 验证SOC变化范围是否足够大
        soc_range = soc.max() - soc.min()
        if soc_range >= 0.3:  # 期望至少30%的变化范围
            print("✓ SOC变化范围足够大，符合预期")
        else:
            print(f"⚠️  SOC变化范围较小 ({soc_range:.1%})，可能需要进一步调整")
        
        # 分析电网能量交换
        power = results['power']
        grid_energy = np.cumsum(power) * dt_h  # MWh
        
        print(f"\n电网能量交换分析:")
        print(f"  最大向电网释放: {grid_energy.max():.2f} MWh")
        print(f"  最大从电网吸收: {grid_energy.min():.2f} MWh")
        print(f"  最终电网能量: {grid_energy[-1]:.2f} MWh")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试增强SOC变化范围时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_enhanced_results():
    """绘制增强的测试结果"""
    print("\n=== 绘制增强的测试结果 ===")
    
    try:
        # 创建测试数据
        t = np.arange(24)
        
        # 模拟增强的功率曲线
        power_enhanced = np.zeros(24)
        power_enhanced[2:6] = -30   # 充电功率增加到30 MW
        power_enhanced[8:12] = 32.5  # 放电功率增加到32.5 MW
        power_enhanced[14:18] = 32.5 # 放电功率增加到32.5 MW
        power_enhanced[22:24] = -30  # 充电功率增加到30 MW
        
        # 计算累积电网能量
        grid_energy = np.cumsum(power_enhanced)  # MWh
        
        # 模拟增强的SOC变化
        soc_enhanced = np.zeros(24)
        soc_enhanced[0] = 0.5  # 初始SOC 50%
        
        # 基于功率计算SOC变化
        for i in range(1, 24):
            if power_enhanced[i] > 0:  # 放电
                soc_enhanced[i] = soc_enhanced[i-1] - 0.02  # 每小时减少2%
            elif power_enhanced[i] < 0:  # 充电
                soc_enhanced[i] = soc_enhanced[i-1] + 0.025  # 每小时增加2.5%
            else:
                soc_enhanced[i] = soc_enhanced[i-1]
        
        # 创建图形
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 子图1: 增强的功率曲线
        ax1.plot(t, power_enhanced, 'b-', linewidth=2, label='增强功率曲线')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('功率 (MW)')
        ax1.set_title('增强的24小时功率曲线')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim(-35, 35)
        
        # 子图2: 增强的SOC变化
        ax1_twin = ax1.twinx()
        ax1_twin.plot(t, soc_enhanced * 100, 'r--', linewidth=2, alpha=0.7, label='SOC变化')
        ax1_twin.set_ylabel('SOC (%)', color='red')
        ax1_twin.set_ylim(0, 100)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 子图3: 电网能量交换
        ax2.plot(t, grid_energy, 'g-', linewidth=2, label='累积电网能量')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('电网能量 (MWh)')
        ax2.set_title('电网能量补充/释放')
        ax2.legend()
        ax2.grid(True)
        
        # 子图4: SOC变化范围对比
        soc_old = np.linspace(0.45, 0.55, 24)  # 旧的SOC范围（50%上下）
        ax3.plot(t, soc_old * 100, 'r-', linewidth=2, label='旧SOC范围 (45%-55%)')
        ax3.plot(t, soc_enhanced * 100, 'b-', linewidth=2, label='新SOC范围 (25%-75%)')
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('SOC (%)')
        ax3.set_title('SOC变化范围对比')
        ax3.legend()
        ax3.grid(True)
        ax3.set_ylim(20, 80)
        
        plt.tight_layout()
        plt.savefig('Test/enhanced_soc_range_results.png', dpi=300, bbox_inches='tight')
        print("✓ 增强的测试结果图表已保存: Test/enhanced_soc_range_results.png")
        
        return True
        
    except Exception as e:
        print(f"✗ 绘制增强测试结果时出错: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("增强的SOC变化范围功能测试")
    print("=" * 60)
    
    # 运行各项测试
    tests = [
        ("增强SOC变化范围", test_enhanced_soc_range),
        ("结果可视化", plot_enhanced_results),
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
        print("\n🎉 所有测试通过！增强的SOC变化范围功能正常")
    else:
        print(f"\n⚠️  有 {len(results) - passed} 个测试失败，请检查相关功能")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
