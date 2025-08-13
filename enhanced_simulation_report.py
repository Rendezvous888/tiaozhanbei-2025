#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
35 kV/25 MW级联储能PCS增强仿真报告生成器
包含详细的器件分析、寿命预测和优化建议
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedSimulationReport:
    """增强仿真报告生成器"""
    
    def __init__(self):
        from device_parameters import get_optimized_parameters
        self.device_params = get_optimized_parameters()
        
    def generate_comprehensive_report(self, results, analysis, params):
        """生成综合仿真报告"""
        print("\n" + "="*80)
        print("35 kV/25 MW级联储能PCS综合仿真报告")
        print("="*80)
        print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 系统配置分析
        self._analyze_system_configuration(params)
        
        # 2. 器件性能分析
        self._analyze_device_performance(results, analysis)
        
        # 3. 热管理分析
        self._analyze_thermal_management(results, analysis)
        
        # 4. 寿命预测分析
        self._analyze_life_prediction(results, analysis)
        
        # 5. 效率分析
        self._analyze_efficiency_performance(results, analysis)
        
        # 6. 优化建议
        self._generate_optimization_recommendations(results, analysis, params)
        
        # 7. 生成详细报告文件
        self._save_detailed_report(results, analysis, params)
        
        print("\n" + "="*80)
        print("仿真报告生成完成")
        print("="*80)
    
    def _analyze_system_configuration(self, params):
        """分析系统配置"""
        print(f"\n1. 系统配置分析")
        print("-" * 50)
        
        # 基本参数
        print(f"✓ 额定功率: {params.P_rated/1e6:.1f} MW")
        print(f"✓ 并网电压: {params.V_grid/1e3:.1f} kV")
        print(f"✓ 额定电流: {params.I_rated:.1f} A")
        print(f"✓ 级联模块数: {params.N_modules_per_phase}")
        
        # 器件配置
        print(f"\n器件配置:")
        print(f"  • IGBT型号: {self.device_params['igbt'].model}")
        print(f"  • 每模块IGBT数: {self.device_params['igbt'].per_module_quantity}")
        print(f"  • 电容器型号: {self.device_params['capacitor'].manufacturer}")
        print(f"  • 电容器数量: {self.device_params['capacitor'].quantity}")
        
        # 配置评估
        voltage_margin = (self.device_params['igbt'].Vces_V - params.Vdc_per_module) / self.device_params['igbt'].Vces_V
        current_margin = (self.device_params['igbt'].Ic_dc_A - params.I_rated/params.N_modules_per_phase) / self.device_params['igbt'].Ic_dc_A
        
        print(f"\n配置评估:")
        print(f"  • 电压裕量: {voltage_margin*100:.1f}% {'✓' if voltage_margin > 0.2 else '⚠'}")
        print(f"  • 电流裕量: {current_margin*100:.1f}% {'✓' if current_margin > 0.3 else '⚠'}")
        
        if voltage_margin < 0.2 or current_margin < 0.3:
            print(f"  ⚠ 建议: 考虑增加器件裕量以提高可靠性")
    
    def _analyze_device_performance(self, results, analysis):
        """分析器件性能"""
        print(f"\n2. 器件性能分析")
        print("-" * 50)
        
        # IGBT性能
        max_current = np.max(results['power']) / (np.sqrt(3) * 35e3) / 58  # 每模块最大电流
        max_voltage = 35e3 / (np.sqrt(3) * 58)  # 每模块最大电压
        
        print(f"IGBT性能:")
        print(f"  • 最大模块电流: {max_current:.1f} A")
        print(f"  • 最大模块电压: {max_voltage:.1f} V")
        print(f"  • 电流利用率: {max_current/self.device_params['igbt'].Ic_dc_A*100:.1f}%")
        print(f"  • 电压利用率: {max_voltage/self.device_params['igbt'].Vces_V*100:.1f}%")
        
        # 电容器性能
        cap_current_rms = max_current * 0.1  # 假设电容器电流为模块电流的10%
        cap_voltage = max_voltage
        
        print(f"\n电容器性能:")
        print(f"  • 最大纹波电流: {cap_current_rms:.1f} A")
        print(f"  • 最大电压: {cap_voltage:.1f} V")
        print(f"  • 电流利用率: {cap_current_rms/self.device_params['capacitor'].current_params['max_current_A']*100:.1f}%")
        print(f"  • 电压利用率: {cap_voltage/self.device_params['capacitor'].current_params['voltage_V']*100:.1f}%")
    
    def _analyze_thermal_management(self, results, analysis):
        """分析热管理"""
        print(f"\n3. 热管理分析")
        print("-" * 50)
        
        max_temp = analysis['max_Tj']
        avg_temp = analysis['avg_Tj']
        temp_rise = max_temp - 25  # 相对于环境温度的温升
        
        print(f"温度分析:")
        print(f"  • 最大结温: {max_temp:.1f}°C")
        print(f"  • 平均结温: {avg_temp:.1f}°C")
        print(f"  • 最大温升: {temp_rise:.1f}°C")
        print(f"  • 温度裕量: {self.device_params['igbt'].junction_temperature_C[1] - max_temp:.1f}°C")
        
        # 热管理评估
        if max_temp < 125:
            thermal_rating = "优秀"
        elif max_temp < 150:
            thermal_rating = "良好"
        elif max_temp < 170:
            thermal_rating = "可接受"
        else:
            thermal_rating = "需要改进"
        
        print(f"  • 热管理评级: {thermal_rating}")
        
        if max_temp > 150:
            print(f"  ⚠ 建议: 考虑改进散热设计或降低功率密度")
    
    def _analyze_life_prediction(self, results, analysis):
        """分析寿命预测"""
        print(f"\n4. 寿命预测分析")
        print("-" * 50)
        
        igbt_life = analysis['igbt_life_remaining']
        cap_life = analysis['capacitor_life_remaining']
        
        print(f"寿命预测:")
        print(f"  • IGBT寿命剩余: {igbt_life*100:.1f}%")
        print(f"  • 电容器寿命剩余: {cap_life*100:.1f}%")
        
        # 寿命评估
        if igbt_life > 0.8:
            igbt_rating = "优秀"
        elif igbt_life > 0.6:
            igbt_rating = "良好"
        elif igbt_life > 0.4:
            igbt_rating = "可接受"
        else:
            igbt_rating = "需要关注"
        
        if cap_life > 0.8:
            cap_rating = "优秀"
        elif cap_life > 0.6:
            cap_rating = "良好"
        elif cap_life > 0.4:
            cap_rating = "可接受"
        else:
            cap_rating = "需要关注"
        
        print(f"  • IGBT寿命评级: {igbt_rating}")
        print(f"  • 电容器寿命评级: {cap_rating}")
        
        # 寿命建议
        if igbt_life < 0.6 or cap_life < 0.6:
            print(f"  ⚠ 建议: 考虑优化工作条件或更换更高等级器件")
    
    def _analyze_efficiency_performance(self, results, analysis):
        """分析效率性能"""
        print(f"\n5. 效率性能分析")
        print("-" * 50)
        
        avg_eff = analysis['avg_efficiency']
        max_eff = analysis['max_efficiency']
        min_eff = analysis['min_efficiency']
        eff_range = max_eff - min_eff
        
        print(f"效率分析:")
        print(f"  • 平均效率: {avg_eff*100:.2f}%")
        print(f"  • 最大效率: {max_eff*100:.2f}%")
        print(f"  • 最小效率: {min_eff*100:.2f}%")
        print(f"  • 效率变化范围: {eff_range*100:.2f}%")
        
        # 效率评估
        if avg_eff > 0.98:
            eff_rating = "优秀"
        elif avg_eff > 0.96:
            eff_rating = "良好"
        elif avg_eff > 0.94:
            eff_rating = "可接受"
        else:
            eff_rating = "需要改进"
        
        print(f"  • 效率评级: {eff_rating}")
        
        if avg_eff < 0.96:
            print(f"  ⚠ 建议: 考虑优化控制策略或改进器件选型")
    
    def _generate_optimization_recommendations(self, results, analysis, params):
        """生成优化建议"""
        print(f"\n6. 优化建议")
        print("-" * 50)
        
        recommendations = []
        
        # 基于温度的建议
        if analysis['max_Tj'] > 150:
            recommendations.append("• 改进散热设计，考虑增加散热器面积或使用更高效的冷却系统")
            recommendations.append("• 优化开关频率，在效率和温度之间找到平衡点")
        
        # 基于寿命的建议
        if analysis['igbt_life_remaining'] < 0.6:
            recommendations.append("• 考虑使用更高等级的IGBT器件")
            recommendations.append("• 优化工作条件，减少温度循环应力")
        
        if analysis['capacitor_life_remaining'] < 0.6:
            recommendations.append("• 考虑使用更高寿命的电容器")
            recommendations.append("• 优化电容器工作条件，减少纹波电流")
        
        # 基于效率的建议
        if analysis['avg_efficiency'] < 0.96:
            recommendations.append("• 优化PWM控制策略，减少开关损耗")
            recommendations.append("• 考虑使用更高效的IGBT器件")
        
        # 基于配置的建议
        voltage_margin = (self.device_params['igbt'].Vces_V - params.Vdc_per_module) / self.device_params['igbt'].Vces_V
        if voltage_margin < 0.2:
            recommendations.append("• 考虑增加电压裕量，提高系统可靠性")
        
        # 显示建议
        if recommendations:
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print(f"  ✓ 当前配置良好，无需重大改进")
    
    def _save_detailed_report(self, results, analysis, params):
        """保存详细报告"""
        # 创建详细报告数据
        report_data = {
            'Category': [
                'System Configuration',
                'System Configuration',
                'System Configuration',
                'System Configuration',
                'Device Performance',
                'Device Performance',
                'Device Performance',
                'Thermal Management',
                'Thermal Management',
                'Thermal Management',
                'Life Prediction',
                'Life Prediction',
                'Efficiency Performance',
                'Efficiency Performance',
                'Efficiency Performance'
            ],
            'Parameter': [
                'Rated Power (MW)',
                'Grid Voltage (kV)',
                'Rated Current (A)',
                'Modules per Phase',
                'IGBT Life Remaining (%)',
                'Capacitor Life Remaining (%)',
                'Max Junction Temperature (°C)',
                'Average Junction Temperature (°C)',
                'Temperature Rise (°C)',
                'Thermal Margin (°C)',
                'IGBT Life Rating',
                'Capacitor Life Rating',
                'Average Efficiency (%)',
                'Maximum Efficiency (%)',
                'Efficiency Rating'
            ],
            'Value': [
                f"{params.P_rated/1e6:.1f}",
                f"{params.V_grid/1e3:.1f}",
                f"{params.I_rated:.1f}",
                f"{params.N_modules_per_phase}",
                f"{analysis['igbt_life_remaining']*100:.1f}",
                f"{analysis['capacitor_life_remaining']*100:.1f}",
                f"{analysis['max_Tj']:.1f}",
                f"{analysis['avg_Tj']:.1f}",
                f"{analysis['max_Tj'] - 25:.1f}",
                f"{self.device_params['igbt'].junction_temperature_C[1] - analysis['max_Tj']:.1f}",
                self._get_life_rating(analysis['igbt_life_remaining']),
                self._get_life_rating(analysis['capacitor_life_remaining']),
                f"{analysis['avg_efficiency']*100:.2f}",
                f"{analysis['max_efficiency']*100:.2f}",
                self._get_efficiency_rating(analysis['avg_efficiency'])
            ]
        }
        
        # 创建DataFrame
        df = pd.DataFrame(report_data)
        
        # 确保result目录存在
        import os
        os.makedirs('result', exist_ok=True)
        
        # 保存报告
        report_filename = f"result/增强仿真报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(report_filename, index=False, encoding='utf-8-sig')
        
        print(f"\n详细报告已保存: {report_filename}")
    
    def _get_life_rating(self, life_remaining):
        """获取寿命评级"""
        if life_remaining > 0.8:
            return "优秀"
        elif life_remaining > 0.6:
            return "良好"
        elif life_remaining > 0.4:
            return "可接受"
        else:
            return "需要关注"
    
    def _get_efficiency_rating(self, efficiency):
        """获取效率评级"""
        if efficiency > 0.98:
            return "优秀"
        elif efficiency > 0.96:
            return "良好"
        elif efficiency > 0.94:
            return "可接受"
        else:
            return "需要改进"

def generate_device_comparison_report():
    """生成器件对比报告"""
    print("\n" + "="*80)
    print("器件对比分析报告")
    print("="*80)
    
    from device_parameters import get_optimized_parameters
    params = get_optimized_parameters()
    
    # IGBT对比
    print(f"\nIGBT器件对比:")
    print("-" * 50)
    print(f"当前选型: {params['igbt'].model}")
    print(f"  • 额定电压: {params['igbt'].Vces_V} V")
    print(f"  • 额定电流: {params['igbt'].Ic_dc_A} A")
    print(f"  • 开关损耗: Eon={params['igbt'].switching_energy_mJ['Eon'][0]}-{params['igbt'].switching_energy_mJ['Eon'][1]} mJ")
    print(f"  • 关断损耗: Eoff={params['igbt'].switching_energy_mJ['Eoff'][0]}-{params['igbt'].switching_energy_mJ['Eoff'][1]} mJ")
    
    # 电容器对比
    print(f"\n电容器对比:")
    print("-" * 50)
    for manufacturer in ["Xiamen Farah", "Nantong Jianghai"]:
        cap_params = params['capacitor'].options[manufacturer]
        print(f"{manufacturer}:")
        print(f"  • 电容值: {cap_params['capacitance_uF']} μF")
        print(f"  • 额定电压: {cap_params['voltage_V']} V")
        print(f"  • 最大电流: {cap_params['max_current_A']} A")
        print(f"  • ESR: {cap_params['ESR_mOhm']} mΩ")
        print(f"  • 寿命: {cap_params['lifetime_h']} 小时")
        print(f"  • 工作温度: {cap_params['operating_temperature_C'][0]}°C ~ {cap_params['operating_temperature_C'][1]}°C")

if __name__ == "__main__":
    # 测试报告生成器
    print("35 kV/25 MW级联储能PCS增强仿真报告生成器")
    
    # 生成器件对比报告
    generate_device_comparison_report() 