#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后IGBT和母线电容建模测试脚本
验证模型的准确性、稳定性和性能
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from typing import Dict, List

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_igbt_model_accuracy():
    """测试IGBT模型精度"""
    print("=" * 60)
    print("测试IGBT模型精度")
    print("=" * 60)
    
    from optimized_igbt_model import OptimizedIGBTModel
    
    igbt = OptimizedIGBTModel()
    
    # 测试数据点（基于Infineon FF1500R17IP5R数据手册典型值）
    test_cases = [
        # (电流A, 电压V, 温度°C, 期望Vce_sat范围V, 期望Eon范围mJ, 期望Eoff范围mJ)
        (500, 1200, 25, (1.8, 2.2), (200, 400), (200, 400)),
        (1000, 1200, 25, (2.0, 2.5), (400, 600), (400, 600)),
        (1500, 1200, 25, (2.2, 2.8), (500, 800), (500, 800)),
        (1000, 1200, 125, (2.5, 3.2), (450, 700), (450, 700)),
    ]
    
    accuracy_results = []
    
    for current, voltage, temp, vce_range, eon_range, eoff_range in test_cases:
        # 测试饱和压降
        vce_sat = igbt.get_saturation_voltage(current, temp)
        vce_accuracy = vce_range[0] <= vce_sat <= vce_range[1]
        
        # 测试开关损耗
        eon, eoff = igbt.get_switching_losses(current, voltage, temp)
        eon_mJ = eon * 1e3
        eoff_mJ = eoff * 1e3
        eon_accuracy = eon_range[0] <= eon_mJ <= eon_range[1]
        eoff_accuracy = eoff_range[0] <= eoff_mJ <= eoff_range[1]
        
        accuracy_results.append({
            'test_case': f'{current}A_{voltage}V_{temp}°C',
            'vce_sat_V': vce_sat,
            'vce_expected': vce_range,
            'vce_accurate': vce_accuracy,
            'eon_mJ': eon_mJ,
            'eon_expected': eon_range,
            'eon_accurate': eon_accuracy,
            'eoff_mJ': eoff_mJ,
            'eoff_expected': eoff_range,
            'eoff_accurate': eoff_accuracy
        })
        
        print(f"测试案例: {current}A, {voltage}V, {temp}°C")
        print(f"  Vce_sat: {vce_sat:.3f}V {'✓' if vce_accuracy else '✗'} (期望: {vce_range})")
        print(f"  Eon: {eon_mJ:.1f}mJ {'✓' if eon_accuracy else '✗'} (期望: {eon_range})")
        print(f"  Eoff: {eoff_mJ:.1f}mJ {'✓' if eoff_accuracy else '✗'} (期望: {eoff_range})")
    
    # 计算总体精度
    total_tests = len(accuracy_results) * 3  # 每个案例3个参数
    accurate_tests = sum([
        result['vce_accurate'] + result['eon_accurate'] + result['eoff_accurate'] 
        for result in accuracy_results
    ])
    
    accuracy_percentage = (accurate_tests / total_tests) * 100
    print(f"\nIGBT模型总体精度: {accuracy_percentage:.1f}% ({accurate_tests}/{total_tests})")
    
    return accuracy_percentage >= 80  # 80%以上认为合格

def test_capacitor_model_accuracy():
    """测试电容器模型精度"""
    print("\n" + "=" * 60)
    print("测试电容器模型精度")
    print("=" * 60)
    
    from optimized_capacitor_model import OptimizedCapacitorModel
    
    cap = OptimizedCapacitorModel("Xiamen Farah")
    
    # 测试数据点（基于薄膜电容器典型特性）
    test_cases = [
        # (频率Hz, 温度°C, 期望ESR倍数范围, 期望电容倍数范围)
        (1000, 25, (0.8, 1.2), (0.98, 1.02)),
        (10000, 25, (0.6, 0.8), (0.98, 1.02)),
        (1000, 70, (1.1, 1.3), (0.95, 0.99)),
        (1000, -20, (0.8, 1.0), (0.93, 0.97)),
    ]
    
    accuracy_results = []
    base_ESR = cap.params.ESR_base_mOhm * 1e-3
    base_cap = cap.params.capacitance_uF * 1e-6
    
    for freq, temp, esr_range, cap_range in test_cases:
        # 测试ESR
        esr = cap.get_ESR(freq, temp)
        esr_factor = esr / base_ESR
        esr_accuracy = esr_range[0] <= esr_factor <= esr_range[1]
        
        # 测试电容值
        capacitance = cap.get_capacitance(temp)
        cap_factor = capacitance / base_cap
        cap_accuracy = cap_range[0] <= cap_factor <= cap_range[1]
        
        accuracy_results.append({
            'test_case': f'{freq}Hz_{temp}°C',
            'esr_factor': esr_factor,
            'esr_expected': esr_range,
            'esr_accurate': esr_accuracy,
            'cap_factor': cap_factor,
            'cap_expected': cap_range,
            'cap_accurate': cap_accuracy
        })
        
        print(f"测试案例: {freq}Hz, {temp}°C")
        print(f"  ESR倍数: {esr_factor:.3f} {'✓' if esr_accuracy else '✗'} (期望: {esr_range})")
        print(f"  电容倍数: {cap_factor:.3f} {'✓' if cap_accuracy else '✗'} (期望: {cap_range})")
    
    # 计算总体精度
    total_tests = len(accuracy_results) * 2  # 每个案例2个参数
    accurate_tests = sum([
        result['esr_accurate'] + result['cap_accurate'] 
        for result in accuracy_results
    ])
    
    accuracy_percentage = (accurate_tests / total_tests) * 100
    print(f"\n电容器模型总体精度: {accuracy_percentage:.1f}% ({accurate_tests}/{total_tests})")
    
    return accuracy_percentage >= 80

def test_model_performance():
    """测试模型性能"""
    print("\n" + "=" * 60)
    print("测试模型计算性能")
    print("=" * 60)
    
    from optimized_igbt_model import OptimizedIGBTModel
    from optimized_capacitor_model import OptimizedCapacitorModel
    
    igbt = OptimizedIGBTModel()
    cap = OptimizedCapacitorModel()
    
    # 性能测试参数
    test_iterations = 1000
    test_arrays = np.random.uniform(100, 1500, test_iterations)
    
    # IGBT性能测试
    print("IGBT模型性能测试...")
    
    # 单点计算性能
    start_time = time.time()
    for current in test_arrays:
        vce = igbt.get_saturation_voltage(current, 75)
        eon, eoff = igbt.get_switching_losses(current, 1200, 75)
    single_time = time.time() - start_time
    
    # 向量化计算性能
    start_time = time.time()
    vce_array = igbt.get_saturation_voltage(test_arrays, 75)
    vector_time = time.time() - start_time
    
    print(f"  单点计算: {single_time:.3f}s ({test_iterations}次)")
    print(f"  向量计算: {vector_time:.3f}s ({test_iterations}个点)")
    print(f"  性能提升: {single_time/vector_time:.1f}倍")
    
    # 电容器性能测试
    print("\n电容器模型性能测试...")
    
    start_time = time.time()
    for i in range(test_iterations):
        freq = 1000 + i
        esr = cap.get_ESR(freq, 50)
        capacitance = cap.get_capacitance(50)
    cap_time = time.time() - start_time
    
    print(f"  计算时间: {cap_time:.3f}s ({test_iterations}次)")
    print(f"  平均单次: {cap_time/test_iterations*1e6:.1f}μs")
    
    # 内存使用测试
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建大量模型实例
    models = []
    for i in range(100):
        models.append(OptimizedIGBTModel())
        models.append(OptimizedCapacitorModel())
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_per_model = (memory_after - memory_before) / 200  # 200个模型
    
    print(f"\n内存使用测试:")
    print(f"  200个模型实例内存增加: {memory_after - memory_before:.1f} MB")
    print(f"  平均每个模型: {memory_per_model:.3f} MB")
    
    # 性能基准
    igbt_performance_ok = single_time < 1.0  # 1000次计算应在1秒内
    cap_performance_ok = cap_time < 0.5      # 1000次计算应在0.5秒内
    memory_ok = memory_per_model < 1.0       # 每个模型内存应小于1MB
    
    return igbt_performance_ok and cap_performance_ok and memory_ok

def test_model_stability():
    """测试模型稳定性"""
    print("\n" + "=" * 60)
    print("测试模型稳定性")
    print("=" * 60)
    
    from optimized_igbt_model import OptimizedIGBTModel
    from optimized_capacitor_model import OptimizedCapacitorModel
    
    igbt = OptimizedIGBTModel()
    cap = OptimizedCapacitorModel()
    
    # 边界条件测试
    boundary_tests = [
        ("最小电流", 0.1, 600, -40),
        ("最大电流", 3000, 1700, 175),
        ("负电流", -100, 1200, 25),  # 应该被限制
        ("零电压", 1000, 0, 25),     # 应该被限制
        ("超高温度", 1000, 1200, 300), # 应该被限制
    ]
    
    stability_results = []
    
    for test_name, current, voltage, temperature in boundary_tests:
        try:
            # IGBT测试
            vce = igbt.get_saturation_voltage(current, temperature)
            eon, eoff = igbt.get_switching_losses(current, voltage, temperature)
            
            # 检查结果是否合理
            vce_ok = 0 <= vce <= 10  # 饱和压降应在合理范围
            eon_ok = 0 <= eon <= 0.02  # 开关损耗应在合理范围
            eoff_ok = 0 <= eoff <= 0.02
            
            igbt_stable = vce_ok and eon_ok and eoff_ok and not np.isnan(vce) and not np.isnan(eon) and not np.isnan(eoff)
            
            # 电容器测试
            esr = cap.get_ESR(abs(voltage), temperature)  # 使用电压绝对值作为频率
            capacitance = cap.get_capacitance(temperature)
            
            # 检查结果是否合理
            esr_ok = 0 <= esr <= 1  # ESR应在合理范围
            cap_ok = 0 <= capacitance <= 10e-3  # 电容值应在合理范围
            
            cap_stable = esr_ok and cap_ok and not np.isnan(esr) and not np.isnan(capacitance)
            
            stability_results.append({
                'test': test_name,
                'igbt_stable': igbt_stable,
                'cap_stable': cap_stable,
                'vce': vce,
                'eon_mJ': eon * 1e3,
                'esr_mOhm': esr * 1e3,
                'cap_uF': capacitance * 1e6
            })
            
            print(f"{test_name}: IGBT {'✓' if igbt_stable else '✗'}, 电容器 {'✓' if cap_stable else '✗'}")
            
        except Exception as e:
            print(f"{test_name}: 异常 - {e}")
            stability_results.append({
                'test': test_name,
                'igbt_stable': False,
                'cap_stable': False,
                'error': str(e)
            })
    
    # 连续运行测试
    print("\n连续运行稳定性测试...")
    continuous_ok = True
    
    try:
        for i in range(1000):
            current = np.random.uniform(100, 2000)
            voltage = np.random.uniform(600, 1700)
            temp = np.random.uniform(25, 125)
            
            vce = igbt.get_saturation_voltage(current, temp)
            eon, eoff = igbt.get_switching_losses(current, voltage, temp)
            
            if np.isnan(vce) or np.isnan(eon) or np.isnan(eoff):
                continuous_ok = False
                break
                
            if i % 100 == 0:
                print(f"  完成 {i+1}/1000 次测试")
                
    except Exception as e:
        print(f"连续运行测试失败: {e}")
        continuous_ok = False
    
    stable_count = sum([r['igbt_stable'] and r['cap_stable'] for r in stability_results if 'error' not in r])
    total_tests = len([r for r in stability_results if 'error' not in r])
    stability_percentage = (stable_count / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n边界条件稳定性: {stability_percentage:.1f}% ({stable_count}/{total_tests})")
    print(f"连续运行稳定性: {'✓' if continuous_ok else '✗'}")
    
    return stability_percentage >= 80 and continuous_ok

def test_unified_model():
    """测试统一模型"""
    print("\n" + "=" * 60)
    print("测试统一设备模型")
    print("=" * 60)
    
    try:
        from unified_device_models import UnifiedDeviceModel, SystemConfiguration
        
        # 创建系统配置
        config = SystemConfiguration(
            rated_power_MW=25.0,
            rated_voltage_kV=35.0,
            modules_per_phase=40
        )
        
        # 创建统一模型
        unified = UnifiedDeviceModel(config)
        
        # 测试系统损耗计算
        print("测试系统损耗计算...")
        losses = unified.calculate_system_losses(20.0)  # 20MW
        
        if losses['efficiency_percent'] > 95 and losses['efficiency_percent'] < 99:
            print(f"  ✓ 系统效率合理: {losses['efficiency_percent']:.2f}%")
            efficiency_ok = True
        else:
            print(f"  ✗ 系统效率异常: {losses['efficiency_percent']:.2f}%")
            efficiency_ok = False
        
        # 测试热行为计算
        print("测试热行为计算...")
        power_profile = [15, 20, 25, 20, 15]  # MW
        time_profile = [0, 6, 12, 18, 24]    # hours
        
        thermal_results = unified.calculate_system_thermal_behavior(power_profile, time_profile)
        
        max_igbt_temp = max(thermal_results['igbt_temperatures_C'])
        max_cap_temp = max(thermal_results['capacitor_temperatures_C'])
        
        if max_igbt_temp < 150 and max_cap_temp < 80:
            print(f"  ✓ 温度计算合理: IGBT {max_igbt_temp:.1f}°C, 电容器 {max_cap_temp:.1f}°C")
            thermal_ok = True
        else:
            print(f"  ✗ 温度计算异常: IGBT {max_igbt_temp:.1f}°C, 电容器 {max_cap_temp:.1f}°C")
            thermal_ok = False
        
        # 测试寿命预测
        print("测试寿命预测...")
        scenario = {'load_factor': 0.7, 'ambient_temp_C': 35}
        life_results = unified.calculate_system_lifetime(scenario, 5)
        
        final_life = life_results['system_life'].iloc[-1]['system_remaining_life_percent']
        
        if 50 <= final_life <= 100:
            print(f"  ✓ 寿命预测合理: {final_life:.1f}%")
            lifetime_ok = True
        else:
            print(f"  ✗ 寿命预测异常: {final_life:.1f}%")
            lifetime_ok = False
        
        return efficiency_ok and thermal_ok and lifetime_ok
        
    except Exception as e:
        print(f"统一模型测试失败: {e}")
        return False

def generate_test_report(test_results: Dict[str, bool]):
    """生成测试报告"""
    print("\n" + "=" * 80)
    print("优化IGBT和母线电容建模测试报告")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\n测试概要:")
    print(f"  总测试项: {total_tests}")
    print(f"  通过测试: {passed_tests}")
    print(f"  失败测试: {total_tests - passed_tests}")
    print(f"  通过率: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\n详细结果:")
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name:<20}: {status}")
    
    # 总体评估
    if passed_tests == total_tests:
        overall_status = "优秀 - 所有测试通过"
    elif passed_tests >= total_tests * 0.8:
        overall_status = "良好 - 大部分测试通过"
    elif passed_tests >= total_tests * 0.6:
        overall_status = "一般 - 部分测试通过"
    else:
        overall_status = "需要改进 - 多项测试失败"
    
    print(f"\n总体评估: {overall_status}")
    
    # 建议
    print(f"\n改进建议:")
    if not test_results.get('IGBT模型精度', True):
        print("  • 检查IGBT模型参数和插值算法")
    if not test_results.get('电容器模型精度', True):
        print("  • 检查电容器模型温度和频率特性")
    if not test_results.get('模型性能', True):
        print("  • 优化计算算法，减少不必要的计算")
    if not test_results.get('模型稳定性', True):
        print("  • 增强边界条件处理和异常捕获")
    if not test_results.get('统一模型', True):
        print("  • 检查统一模型的集成逻辑和参数传递")
    
    if passed_tests == total_tests:
        print("  • 所有测试通过，模型已准备投入使用")
    
    print("=" * 80)
    
    return passed_tests / total_tests

def main():
    """主测试函数"""
    print("开始优化IGBT和母线电容建模验证测试...")
    
    # 执行各项测试
    test_results = {}
    
    try:
        test_results['IGBT模型精度'] = test_igbt_model_accuracy()
    except Exception as e:
        print(f"IGBT模型精度测试失败: {e}")
        test_results['IGBT模型精度'] = False
    
    try:
        test_results['电容器模型精度'] = test_capacitor_model_accuracy()
    except Exception as e:
        print(f"电容器模型精度测试失败: {e}")
        test_results['电容器模型精度'] = False
    
    try:
        test_results['模型性能'] = test_model_performance()
    except Exception as e:
        print(f"模型性能测试失败: {e}")
        test_results['模型性能'] = False
    
    try:
        test_results['模型稳定性'] = test_model_stability()
    except Exception as e:
        print(f"模型稳定性测试失败: {e}")
        test_results['模型稳定性'] = False
    
    try:
        test_results['统一模型'] = test_unified_model()
    except Exception as e:
        print(f"统一模型测试失败: {e}")
        test_results['统一模型'] = False
    
    # 生成测试报告
    pass_rate = generate_test_report(test_results)
    
    # 保存测试结果
    test_summary = {
        'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'),
        'pass_rate': pass_rate,
        'detailed_results': test_results
    }
    
    # 保存到文件
    import json
    with open(f'result/优化建模测试结果_{test_summary["timestamp"]}.json', 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试结果已保存到: result/优化建模测试结果_{test_summary['timestamp']}.json")
    
    return pass_rate >= 0.8  # 80%通过率认为成功

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
