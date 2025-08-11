"""
长期寿命仿真模块
用于分析IGBT和母线电容在长期运行（5年、10年等）后的剩余寿命
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class LongTermLifeSimulation:
    """长期寿命仿真类"""
    
    def __init__(self):
        # 系统参数
        self.system_params = {
            'rated_power': 25e6,  # 25 MW
            'rated_voltage': 35e3,  # 35 kV
            'h_bridge_units_per_phase': 40,
            'switching_frequency': 750,  # Hz
            'ambient_temperature': 25,  # °C
            'operating_hours_per_year': 8760,  # 24小时/天，365天/年
        }
        
        # IGBT参数
        self.igbt_params = {
            'junction_to_case_resistance': 0.1,  # K/W
            'case_to_ambient_resistance': 0.5,   # K/W
            'thermal_capacity': 0.1,             # J/K
            'base_life_cycles': 1e6,             # 基础寿命循环数
            'activation_energy': 0.1,            # eV
            'boltzmann_constant': 8.617e-5,      # eV/K
            'reference_temperature': 273 + 125,  # K (125°C)
        }
        
        # 导入设备参数
        from device_parameters import get_optimized_parameters
        device_params = get_optimized_parameters()
        
        # 电容参数 - 使用device_parameters.py中的参数
        self.capacitor_params = {
            'rated_voltage': device_params['capacitor'].current_params['voltage_V'],  # V
            'rated_capacitance': device_params['capacitor'].get_capacitance(),  # F
            'esr': device_params['capacitor'].get_ESR(),  # 等效串联电阻
            'base_life_hours': device_params['capacitor'].get_lifetime(),  # 基础寿命小时数
            'temperature_coefficient': 2,  # 温度系数
            'voltage_coefficient': 3,      # 电压系数
            'ripple_coefficient': 2,       # 纹波系数
        }
        
        # 负载工况参数
        self.load_profiles = {
            'light_load': {'power_factor': 0.3, 'duty_cycle': 0.2},
            'medium_load': {'power_factor': 0.6, 'duty_cycle': 0.5},
            'heavy_load': {'power_factor': 0.9, 'duty_cycle': 0.8},
            'peak_load': {'power_factor': 0.95, 'duty_cycle': 0.95},
        }
        
    def generate_daily_load_profile(self, load_type='medium'):
        """生成日负载曲线"""
        hours = np.arange(24)
        
        if load_type == 'light':
            # 轻负载：夜间低，白天中等
            base_load = 0.2
            peak_factor = 0.3
        elif load_type == 'medium':
            # 中等负载：双峰曲线
            base_load = 0.4
            peak_factor = 0.6
        elif load_type == 'heavy':
            # 重负载：高负载运行
            base_load = 0.6
            peak_factor = 0.8
        else:
            base_load = 0.5
            peak_factor = 0.7
        
        # 生成双峰负载曲线
        morning_peak = np.exp(-((hours - 9) / 2)**2)
        evening_peak = np.exp(-((hours - 19) / 2)**2)
        load_profile = base_load + peak_factor * (morning_peak + evening_peak)
        
        return load_profile
    
    def calculate_igbt_temperature_cycles(self, load_profile, years):
        """计算IGBT温度循环"""
        # 计算每个负载水平下的结温
        junction_temps = []
        for load in load_profile:
            # 修正的温度计算模型
            power_loss_per_unit = load * 0.02 * self.system_params['rated_power'] / (3 * self.system_params['h_bridge_units_per_phase'])  # 每单元损耗
            temp_rise = power_loss_per_unit * self.igbt_params['junction_to_case_resistance']
            junction_temp = self.system_params['ambient_temperature'] + temp_rise
            junction_temps.append(junction_temp)
        
        # 生成多年的温度历史
        total_hours = years * self.system_params['operating_hours_per_year']
        daily_cycles = years * 365
        
        # 添加温度波动
        temp_history = []
        for day in range(int(daily_cycles)):
            daily_temps = np.array(junction_temps) + np.random.normal(0, 2, len(junction_temps))
            temp_history.extend(daily_temps)
        
        return np.array(temp_history)
    
    def calculate_capacitor_stress(self, load_profile, years):
        """计算电容应力"""
        # 计算电压纹波和温度
        voltage_ripples = []
        capacitor_temps = []
        
        for load in load_profile:
            # 电压纹波计算
            ripple_current = load * self.system_params['rated_power'] / (self.system_params['rated_voltage'] * np.sqrt(3))
            ripple_voltage = ripple_current * self.capacitor_params['esr'] / self.system_params['h_bridge_units_per_phase']
            voltage_ripples.append(ripple_voltage)
            
            # 电容温度计算
            power_loss = ripple_current**2 * self.capacitor_params['esr'] / self.system_params['h_bridge_units_per_phase']
            temp_rise = power_loss * 0.05  # 修正的热阻
            capacitor_temp = self.system_params['ambient_temperature'] + temp_rise
            capacitor_temps.append(capacitor_temp)
        
        # 生成多年的应力历史
        total_hours = years * self.system_params['operating_hours_per_year']
        daily_cycles = years * 365
        
        ripple_history = []
        temp_history = []
        
        for day in range(int(daily_cycles)):
            daily_ripples = np.array(voltage_ripples) + np.random.normal(0, 0.05, len(voltage_ripples))
            daily_temps = np.array(capacitor_temps) + np.random.normal(0, 1, len(capacitor_temps))
            ripple_history.extend(daily_ripples)
            temp_history.extend(daily_temps)
        
        return np.array(ripple_history), np.array(temp_history)
    
    def calculate_igbt_life_consumption(self, temp_history):
        """计算IGBT寿命消耗"""
        # 使用Coffin-Manson模型计算疲劳寿命
        temp_cycles = []
        
        # 简化的雨流计数（温度循环识别）
        for i in range(1, len(temp_history)):
            if abs(temp_history[i] - temp_history[i-1]) > 5:  # 温度变化超过5°C
                temp_cycles.append(abs(temp_history[i] - temp_history[i-1]))
        
        if not temp_cycles:
            return 0
        
        # 计算等效温度循环
        total_damage = 0
        for delta_temp in temp_cycles:
            # Coffin-Manson模型
            cycles_to_failure = self.igbt_params['base_life_cycles'] * (delta_temp / 50)**(-2)
            damage = 1 / cycles_to_failure
            total_damage += damage
        
        # 考虑温度加速因子
        avg_temp = np.mean(temp_history)
        if avg_temp > 0:  # 避免负温度
            temp_acceleration = np.exp(self.igbt_params['activation_energy'] / 
                                     self.igbt_params['boltzmann_constant'] * 
                                     (1/self.igbt_params['reference_temperature'] - 1/(avg_temp + 273)))
        else:
            temp_acceleration = 1
        
        life_consumption = total_damage * temp_acceleration
        return min(life_consumption, 1.0)  # 限制最大消耗为100%
    
    def calculate_capacitor_life_consumption(self, ripple_history, temp_history):
        """计算电容寿命消耗"""
        # 使用Arrhenius模型计算寿命
        total_hours = len(temp_history)
        
        # 计算等效工作小时数
        equivalent_hours = 0
        
        for i in range(len(temp_history)):
            temp = max(temp_history[i], 25)  # 最低温度25°C
            ripple = max(ripple_history[i], 1)  # 最小纹波1V
            
            # 温度加速因子
            temp_acceleration = 2**((temp - 85) / 10)  # 每10°C温度升高，寿命减半
            
            # 电压加速因子
            voltage_ratio = min(ripple / self.capacitor_params['rated_voltage'], 1.0)
            voltage_acceleration = (voltage_ratio)**self.capacitor_params['voltage_coefficient']
            
            # 纹波加速因子
            ripple_acceleration = (ripple / 100)**self.capacitor_params['ripple_coefficient']
            
            # 综合加速因子
            total_acceleration = max(temp_acceleration * voltage_acceleration * ripple_acceleration, 0.1)
            
            equivalent_hours += 1 / total_acceleration
        
        life_consumption = equivalent_hours / self.capacitor_params['base_life_hours']
        return min(life_consumption, 1.0)  # 限制最大消耗为100%
    
    def simulate_long_term_life(self, years_list=[1, 3, 5, 10], load_types=['light', 'medium', 'heavy']):
        """仿真长期寿命"""
        results = []
        
        for years in years_list:
            for load_type in load_types:
                print(f"仿真 {years}年 {load_type}负载工况...")
                
                # 生成负载曲线
                daily_load = self.generate_daily_load_profile(load_type)
                
                # 计算IGBT温度循环
                temp_history = self.calculate_igbt_temperature_cycles(daily_load, years)
                
                # 计算电容应力
                ripple_history, cap_temp_history = self.calculate_capacitor_stress(daily_load, years)
                
                # 计算寿命消耗
                igbt_life_consumption = self.calculate_igbt_life_consumption(temp_history)
                capacitor_life_consumption = self.calculate_capacitor_life_consumption(ripple_history, cap_temp_history)
                
                # 计算剩余寿命
                igbt_remaining_life = max(0, 1 - igbt_life_consumption) * 100
                capacitor_remaining_life = max(0, 1 - capacitor_life_consumption) * 100
                
                # 计算平均温度
                avg_igbt_temp = np.mean(temp_history)
                avg_cap_temp = np.mean(cap_temp_history)
                max_igbt_temp = np.max(temp_history)
                max_cap_temp = np.max(cap_temp_history)
                
                results.append({
                    'years': years,
                    'load_type': load_type,
                    'igbt_remaining_life': igbt_remaining_life,
                    'capacitor_remaining_life': capacitor_remaining_life,
                    'avg_igbt_temp': avg_igbt_temp,
                    'avg_cap_temp': avg_cap_temp,
                    'max_igbt_temp': max_igbt_temp,
                    'max_cap_temp': max_cap_temp,
                    'igbt_life_consumption': igbt_life_consumption,
                    'capacitor_life_consumption': capacitor_life_consumption
                })
        
        return pd.DataFrame(results)
    
    def plot_life_results(self, results_df):
        """绘制寿命结果"""
        # 使用自适应绘图工具创建图形
        fig, axes = create_adaptive_figure(2, 2, title='长期运行寿命分析结果', title_size=16)
        
        # IGBT剩余寿命
        ax1 = axes[0, 0]
        for load_type in results_df['load_type'].unique():
            data = results_df[results_df['load_type'] == load_type]
            ax1.plot(data['years'], data['igbt_remaining_life'], 'o-', 
                    label=f'{load_type}负载', linewidth=2, markersize=8)
        format_axis_labels(ax1, '运行年数', 'IGBT剩余寿命 (%)', 'IGBT剩余寿命 vs 运行时间')
        add_grid(ax1, alpha=0.3)
        ax1.legend(fontsize=8, loc='best')
        set_adaptive_ylim(ax1, results_df['igbt_remaining_life'])
        
        # 电容剩余寿命
        ax2 = axes[0, 1]
        for load_type in results_df['load_type'].unique():
            data = results_df[results_df['load_type'] == load_type]
            ax2.plot(data['years'], data['capacitor_remaining_life'], 's-', 
                    label=f'{load_type}负载', linewidth=2, markersize=8)
        format_axis_labels(ax2, '运行年数', '电容剩余寿命 (%)', '母线电容剩余寿命 vs 运行时间')
        add_grid(ax2, alpha=0.3)
        ax2.legend(fontsize=8, loc='best')
        set_adaptive_ylim(ax2, results_df['capacitor_remaining_life'])
        
        # 平均温度
        ax3 = axes[1, 0]
        for load_type in results_df['load_type'].unique():
            data = results_df[results_df['load_type'] == load_type]
            ax3.plot(data['years'], data['avg_igbt_temp'], 'o-', 
                    label=f'IGBT-{load_type}', linewidth=2, markersize=6)
            ax3.plot(data['years'], data['avg_cap_temp'], 's--', 
                    label=f'电容-{load_type}', linewidth=2, markersize=6)
        format_axis_labels(ax3, '运行年数', '平均温度 (°C)', '平均工作温度 vs 运行时间')
        add_grid(ax3, alpha=0.3)
        ax3.legend(fontsize=8, loc='best')
        set_adaptive_ylim(ax3, np.concatenate([
            results_df['avg_igbt_temp'], 
            results_df['avg_cap_temp']
        ]))
        
        # 寿命消耗对比
        ax4 = axes[1, 1]
        years = results_df['years'].unique()
        x = np.arange(len(years))
        width = 0.35
        
        igbt_consumption = results_df[results_df['load_type'] == 'medium']['igbt_life_consumption'].values
        cap_consumption = results_df[results_df['load_type'] == 'medium']['capacitor_life_consumption'].values
        
        bars1 = ax4.bar(x - width/2, igbt_consumption, width, label='IGBT寿命消耗', alpha=0.8)
        bars2 = ax4.bar(x + width/2, cap_consumption, width, label='电容寿命消耗', alpha=0.8)
        format_axis_labels(ax4, '运行年数', '寿命消耗率', '中等负载下寿命消耗对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(years)
        ax4.legend(fontsize=8, loc='best')
        add_grid(ax4, alpha=0.3)
        
        # 添加数值标签，避免重叠
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 优化布局，避免重叠
        optimize_layout(fig, tight_layout=True, h_pad=1.5, w_pad=1.5)
        
        # 显示图形
        finalize_plot(fig)
    
    def generate_life_report(self, results_df):
        """生成寿命分析报告"""
        report = []
        report.append("=" * 60)
        report.append("35kV/25MW级联储能PCS长期寿命分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 系统参数
        report.append("系统参数:")
        report.append(f"  额定功率: {self.system_params['rated_power']/1e6:.1f} MW")
        report.append(f"  额定电压: {self.system_params['rated_voltage']/1e3:.1f} kV")
        report.append(f"  每相H桥单元数: {self.system_params['h_bridge_units_per_phase']}")
        report.append(f"  开关频率: {self.system_params['switching_frequency']} Hz")
        report.append("")
        
        # 关键发现
        report.append("关键发现:")
        
        # 找出最严重的情况
        worst_igbt = results_df.loc[results_df['igbt_remaining_life'].idxmin()]
        worst_cap = results_df.loc[results_df['capacitor_remaining_life'].idxmin()]
        
        report.append(f"  IGBT最严重工况: {worst_igbt['years']}年 {worst_igbt['load_type']}负载")
        report.append(f"    剩余寿命: {worst_igbt['igbt_remaining_life']:.1f}%")
        report.append(f"    平均温度: {worst_igbt['avg_igbt_temp']:.1f}°C")
        report.append(f"    最高温度: {worst_igbt['max_igbt_temp']:.1f}°C")
        report.append("")
        
        report.append(f"  电容最严重工况: {worst_cap['years']}年 {worst_cap['load_type']}负载")
        report.append(f"    剩余寿命: {worst_cap['capacitor_remaining_life']:.1f}%")
        report.append(f"    平均温度: {worst_cap['avg_cap_temp']:.1f}°C")
        report.append(f"    最高温度: {worst_cap['max_cap_temp']:.1f}°C")
        report.append("")
        
        # 10年预测
        ten_year_data = results_df[results_df['years'] == 10]
        if not ten_year_data.empty:
            report.append("10年运行预测:")
            for _, row in ten_year_data.iterrows():
                report.append(f"  {row['load_type']}负载:")
                report.append(f"    IGBT剩余寿命: {row['igbt_remaining_life']:.1f}%")
                report.append(f"    电容剩余寿命: {row['capacitor_remaining_life']:.1f}%")
            report.append("")
        
        # 维护建议
        report.append("维护建议:")
        report.append("  1. 定期监测IGBT结温和电容温度")
        report.append("  2. 根据负载工况调整维护周期")
        report.append("  3. 考虑采用主动冷却技术延长寿命")
        report.append("  4. 建立预测性维护体系")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, results_df, filename=None):
        """保存结果到CSV文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'result/长期寿命分析结果_{timestamp}.csv'
        
        results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {filename}")
        return filename

def run_long_term_life_simulation():
    """运行长期寿命仿真"""
    print("开始长期寿命仿真分析...")
    print("=" * 50)
    
    # 创建仿真对象
    simulator = LongTermLifeSimulation()
    
    # 运行仿真
    years_list = [1, 3, 5, 10]
    load_types = ['light', 'medium', 'heavy']
    
    results = simulator.simulate_long_term_life(years_list, load_types)
    
    # 显示结果
    print("\n仿真结果摘要:")
    print(results.round(2))
    
    # 绘制图表
    print("\n生成分析图表...")
    simulator.plot_life_results(results)
    
    # 生成报告
    print("\n生成分析报告...")
    report = simulator.generate_life_report(results)
    print(report)
    
    # 保存结果
    filename = simulator.save_results(results)
    
    print(f"\n长期寿命仿真完成！")
    print(f"详细结果已保存到: {filename}")
    
    return results, report

if __name__ == "__main__":
    run_long_term_life_simulation() 