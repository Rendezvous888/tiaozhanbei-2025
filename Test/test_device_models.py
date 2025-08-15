#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Device Models Test Script
测试IGBT和电容器建模脚本

作者: AI Assistant
日期: 2025
描述: 测试和验证IGBT.py和Bus_Capacitor.py的功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IGBT import IGBTModel
from Bus_Capacitor import BusCapacitorModel

def test_igbt_model():
    """测试IGBT模型"""
    print("=" * 50)
    print("测试IGBT模型...")
    print("=" * 50)
    
    try:
        # 创建IGBT模型实例
        igbt = IGBTModel()
        print("✓ IGBT模型创建成功")
        
        # 测试基本参数
        assert igbt.VCES == 1700.0, f"VCES错误: {igbt.VCES}"
        assert igbt.IC_nom == 1500.0, f"IC_nom错误: {igbt.IC_nom}"
        print("✓ 基本参数正确")
        
        # 测试饱和电压计算
        vce_sat_25 = igbt.calculate_vce_sat(1500, 25)
        vce_sat_125 = igbt.calculate_vce_sat(1500, 125)
        assert 1.5 <= vce_sat_25 <= 2.0, f"25°C饱和电压错误: {vce_sat_25}"
        assert 2.0 <= vce_sat_125 <= 2.5, f"125°C饱和电压错误: {vce_sat_125}"
        print("✓ 饱和电压计算正确")
        
        # 测试开关损耗计算
        switching_losses = igbt.calculate_switching_losses(1000, 900, 15, 1000)
        assert 'Psw_total' in switching_losses, "开关损耗计算失败"
        assert switching_losses['Psw_total'] > 0, "开关损耗应为正值"
        print("✓ 开关损耗计算正确")
        
        # 测试总损耗计算
        total_losses = igbt.calculate_total_losses(1000, 900, 15, 1000, 0.5, 125)
        assert 'Ptotal' in total_losses, "总损耗计算失败"
        assert total_losses['Ptotal'] > 0, "总损耗应为正值"
        print("✓ 总损耗计算正确")
        
        # 测试热行为计算
        thermal = igbt.calculate_thermal_behavior(500, 25)
        assert 'Tj' in thermal, "热行为计算失败"
        assert thermal['Tj'] > 25, "结温应高于环境温度"
        print("✓ 热行为计算正确")
        
        print("✓ IGBT模型所有测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ IGBT模型测试失败: {e}")
        return False

def test_capacitor_model():
    """测试电容器模型"""
    print("\n" + "=" * 50)
    print("测试电容器模型...")
    print("=" * 50)
    
    try:
        # 创建电容器模型实例
        capacitor = BusCapacitorModel()
        print("✓ 电容器模型创建成功")
        
        # 测试基本参数
        assert capacitor.C_nom == 1000e-6, f"C_nom错误: {capacitor.C_nom}"
        assert capacitor.V_rated == 400.0, f"V_rated错误: {capacitor.V_rated}"
        print("✓ 基本参数正确")
        
        # 测试电容值计算
        C_25 = capacitor.calculate_capacitance(25)
        C_85 = capacitor.calculate_capacitance(85)
        assert 800e-6 <= C_25 <= 1200e-6, f"25°C电容值错误: {C_25}"
        assert C_85 < C_25, "高温时电容值应减小"
        print("✓ 电容值计算正确")
        
        # 测试ESR计算
        ESR_25 = capacitor.calculate_ESR(25)
        ESR_85 = capacitor.calculate_ESR(85)
        assert ESR_25 > 0, "ESR应为正值"
        assert ESR_85 > ESR_25, "高温时ESR应增加"
        print("✓ ESR计算正确")
        
        # 测试损耗计算
        losses = capacitor.calculate_conduction_losses(5, 25)
        assert 'P_total' in losses, "损耗计算失败"
        assert losses['P_total'] > 0, "损耗应为正值"
        print("✓ 损耗计算正确")
        
        # 测试纹波电压计算
        ripple = capacitor.calculate_ripple_voltage(5, 1000, 25)
        assert 'V_ripple_total' in ripple, "纹波电压计算失败"
        assert ripple['V_ripple_total'] > 0, "纹波电压应为正值"
        print("✓ 纹波电压计算正确")
        
        # 测试热行为计算
        thermal = capacitor.calculate_thermal_behavior(1.0, 25)
        assert 'T_case' in thermal, "热行为计算失败"
        assert thermal['T_case'] > 25, "外壳温度应高于环境温度"
        print("✓ 热行为计算正确")
        
        # 测试寿命计算
        life = capacitor.calculate_lifetime(25, 1.0)
        assert 'life_expected' in life, "寿命计算失败"
        assert life['life_expected'] > 0, "预期寿命应为正值"
        print("✓ 寿命计算正确")
        
        # 测试阻抗频谱计算
        freq_range = np.logspace(1, 4, 10)
        impedance = capacitor.calculate_impedance_spectrum(freq_range, 25)
        assert 'Z_magnitude' in impedance, "阻抗频谱计算失败"
        assert len(impedance['Z_magnitude']) == len(freq_range), "阻抗数组长度错误"
        print("✓ 阻抗频谱计算正确")
        
        print("✓ 电容器模型所有测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ 电容器模型测试失败: {e}")
        return False

def test_integration():
    """测试两个模型的集成"""
    print("\n" + "=" * 50)
    print("测试模型集成...")
    print("=" * 50)
    
    try:
        # 创建两个模型实例
        igbt = IGBTModel()
        capacitor = BusCapacitorModel()
        
        # 模拟一个简单的功率转换器
        # IGBT工作在1000A, 900V, 1000Hz
        # 电容器提供滤波和储能
        
        # IGBT损耗
        igbt_losses = igbt.calculate_total_losses(1000, 900, 15, 1000, 0.5, 125)
        igbt_power = igbt_losses['Ptotal']
        
        # 电容器纹波电流 (假设为IGBT电流的10%)
        ripple_current = 1000 * 0.1  # 100A RMS
        
        # 电容器损耗
        cap_losses = capacitor.calculate_conduction_losses(ripple_current, 50)
        cap_power = cap_losses['P_total']
        
        # 电容器纹波电压
        ripple_voltage = capacitor.calculate_ripple_voltage(ripple_current, 1000, 50)
        
        # 总系统损耗
        total_system_loss = igbt_power + cap_power
        
        print(f"IGBT损耗: {igbt_power:.1f} W")
        print(f"电容器损耗: {cap_power:.3f} W")
        print(f"总系统损耗: {total_system_loss:.1f} W")
        print(f"电容器纹波电压: {ripple_voltage['V_ripple_total']:.3f} V")
        
        # 验证结果合理性
        assert total_system_loss > 0, "总系统损耗应为正值"
        assert ripple_voltage['V_ripple_total'] < 50, "纹波电压应小于50V"
        
        print("✓ 模型集成测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ 模型集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始器件模型测试...")
    
    # 运行所有测试
    tests = [
        test_igbt_model,
        test_capacitor_model,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过! 器件模型工作正常。")
        return True
    else:
        print("❌ 部分测试失败，请检查模型实现。")
        return False

if __name__ == "__main__":
    # 导入numpy用于电容器测试
    import numpy as np
    
    success = main()
    sys.exit(0 if success else 1)
