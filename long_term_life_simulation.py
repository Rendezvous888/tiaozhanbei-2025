"""
长期寿命仿真模块
用于分析IGBT和母线电容在长期运行（5年、10年等）后的剩余寿命
包含详细寿命分析功能
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
    """长期寿命仿真类，包含详细寿命分析功能"""
    
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
        
        # 仿真结果存储
        self.simulation_results = None
        
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
        
        self.simulation_results = pd.DataFrame(results)
        return self.simulation_results
    
    def plot_life_results(self, results_df):
        """绘制寿命结果"""
        # 创建更大的图形以避免重叠
        fig = plt.figure(figsize=(16, 12), dpi=100)
        fig.suptitle('长期运行寿命分析结果', fontsize=16, fontweight='bold', y=0.98)
        
        # 创建2x2子图网格，增加间距
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        axes = []
        
        for i in range(2):
            for j in range(2):
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)
        
        # 将axes重新组织为2D数组
        axes = np.array(axes).reshape(2, 2)
        
        # IGBT剩余寿命
        ax1 = axes[0, 0]
        for load_type in results_df['load_type'].unique():
            data = results_df[results_df['load_type'] == load_type]
            ax1.plot(data['years'], data['igbt_remaining_life'], 'o-', 
                    label=f'{load_type}负载', linewidth=2, markersize=8)
        format_axis_labels(ax1, '运行年数', 'IGBT剩余寿命 (%)', 'IGBT剩余寿命 vs 运行时间')
        add_grid(ax1, alpha=0.3)
        ax1.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        set_adaptive_ylim(ax1, results_df['igbt_remaining_life'])
        
        # 电容剩余寿命
        ax2 = axes[0, 1]
        for load_type in results_df['load_type'].unique():
            data = results_df[results_df['load_type'] == load_type]
            ax2.plot(data['years'], data['capacitor_remaining_life'], 's-', 
                    label=f'{load_type}负载', linewidth=2, markersize=8)
        format_axis_labels(ax2, '运行年数', '电容剩余寿命 (%)', '母线电容剩余寿命 vs 运行时间')
        add_grid(ax2, alpha=0.3)
        ax2.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
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
        ax3.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
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
        ax4.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        add_grid(ax4, alpha=0.3)
        
        # 添加数值标签，避免重叠，增加间距
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 优化布局，避免重叠
        plt.tight_layout()
        
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
        
        # 确保result目录存在
        import os
        os.makedirs('result', exist_ok=True)
        
        results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {filename}")
        return filename

    # ==================== 详细寿命分析功能 ====================
    
    def analyze_life_trends(self):
        """分析寿命趋势"""
        if self.simulation_results is None:
            print("请先运行仿真获取数据")
            return
            
        print("=" * 60)
        print("详细寿命趋势分析")
        print("=" * 60)
        
        # 按负载类型分组分析
        load_types = self.simulation_results['load_type'].unique()
        
        for load_type in load_types:
            data = self.simulation_results[self.simulation_results['load_type'] == load_type]
            print(f"\n{load_type}负载工况分析:")
            print(f"  1年运行: IGBT剩余寿命 {data[data['years']==1]['igbt_remaining_life'].iloc[0]:.1f}%")
            print(f"  5年运行: IGBT剩余寿命 {data[data['years']==5]['igbt_remaining_life'].iloc[0]:.1f}%")
            print(f"  10年运行: IGBT剩余寿命 {data[data['years']==10]['igbt_remaining_life'].iloc[0]:.1f}%")
            
            # 计算年化寿命消耗率
            if len(data) > 1:
                first_year = data[data['years']==1]['igbt_life_consumption'].iloc[0]
                ten_year = data[data['years']==10]['igbt_life_consumption'].iloc[0]
                annual_rate = (ten_year - first_year) / 9  # 9年间的平均年化率
                print(f"  年化寿命消耗率: {annual_rate*100:.2f}%/年")
    
    def calculate_maintenance_schedule(self):
        """计算维护计划"""
        if self.simulation_results is None:
            print("请先运行仿真获取数据")
            return None
            
        print("\n" + "=" * 60)
        print("维护计划建议")
        print("=" * 60)
        
        maintenance_schedule = []
        
        for _, row in self.simulation_results.iterrows():
            years = row['years']
            load_type = row['load_type']
            igbt_life = row['igbt_remaining_life']
            cap_life = row['capacitor_remaining_life']
            
            # 根据剩余寿命确定维护建议
            if igbt_life < 20:
                maintenance = "立即更换IGBT"
                urgency = "紧急"
            elif igbt_life < 50:
                maintenance = "计划更换IGBT"
                urgency = "高"
            elif igbt_life < 80:
                maintenance = "加强监测"
                urgency = "中"
            else:
                maintenance = "正常维护"
                urgency = "低"
            
            maintenance_schedule.append({
                'years': years,
                'load_type': load_type,
                'igbt_life': igbt_life,
                'cap_life': cap_life,
                'maintenance': maintenance,
                'urgency': urgency
            })
        
        maintenance_df = pd.DataFrame(maintenance_schedule)
        
        # 显示关键维护节点
        critical_points = maintenance_df[maintenance_df['urgency'].isin(['紧急', '高'])]
        if not critical_points.empty:
            print("关键维护节点:")
            for _, point in critical_points.iterrows():
                print(f"  {point['years']}年 {point['load_type']}负载: {point['maintenance']} ({point['urgency']}优先级)")
        
        return maintenance_df
    
    def plot_detailed_analysis(self):
        """绘制详细分析图表"""
        if self.simulation_results is None:
            print("请先运行仿真获取数据")
            return
            
        # 创建更大的图形以避免重叠
        fig = plt.figure(figsize=(20, 15), dpi=100)
        fig.suptitle('IGBT和电容器详细寿命分析', fontsize=18, fontweight='bold', y=0.98)
        
        # 创建3x3子图网格，增加间距
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        axes = []
        
        for i in range(3):
            for j in range(3):
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)
        
        # 将axes重新组织为2D数组
        axes = np.array(axes).reshape(3, 3)
        
        # 1. 寿命趋势对比
        ax1 = axes[0, 0]
        for load_type in self.simulation_results['load_type'].unique():
            data = self.simulation_results[self.simulation_results['load_type'] == load_type]
            ax1.plot(data['years'], data['igbt_remaining_life'], 'o-', 
                    label=f'IGBT-{load_type}', linewidth=2, markersize=8)
            ax1.plot(data['years'], data['capacitor_remaining_life'], 's--', 
                    label=f'电容-{load_type}', linewidth=2, markersize=8)
        format_axis_labels(ax1, '运行年数', '剩余寿命 (%)', 'IGBT和电容剩余寿命趋势')
        add_grid(ax1, alpha=0.3)
        ax1.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        set_adaptive_ylim(ax1, np.concatenate([
            self.simulation_results['igbt_remaining_life'], 
            self.simulation_results['capacitor_remaining_life']
        ]))
        
        # 2. 温度分析
        ax2 = axes[0, 1]
        for load_type in self.simulation_results['load_type'].unique():
            data = self.simulation_results[self.simulation_results['load_type'] == load_type]
            ax2.plot(data['years'], data['avg_igbt_temp'], 'o-', 
                    label=f'IGBT-{load_type}', linewidth=2, markersize=8)
        format_axis_labels(ax2, '运行年数', '平均温度 (°C)', 'IGBT平均工作温度')
        add_grid(ax2, alpha=0.3)
        ax2.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        set_adaptive_ylim(ax2, self.simulation_results['avg_igbt_temp'])
        
        # 3. 寿命消耗率
        ax3 = axes[0, 2]
        for load_type in self.simulation_results['load_type'].unique():
            data = self.simulation_results[self.simulation_results['load_type'] == load_type]
            ax3.plot(data['years'], data['igbt_life_consumption']*100, 'o-', 
                    label=f'{load_type}负载', linewidth=2, markersize=8)
        format_axis_labels(ax3, '运行年数', '寿命消耗率 (%)', 'IGBT寿命消耗率')
        add_grid(ax3, alpha=0.3)
        ax3.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        set_adaptive_ylim(ax3, self.simulation_results['igbt_life_consumption']*100)
        
        # 4. 10年预测热力图
        ax4 = axes[1, 0]
        ten_year_data = self.simulation_results[self.simulation_results['years'] == 10]
        load_types = ten_year_data['load_type'].values
        igbt_life = ten_year_data['igbt_remaining_life'].values
        cap_life = ten_year_data['capacitor_remaining_life'].values
        
        x = np.arange(len(load_types))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, igbt_life, width, label='IGBT剩余寿命', alpha=0.8)
        bars2 = ax4.bar(x + width/2, cap_life, width, label='电容剩余寿命', alpha=0.8)
        
        format_axis_labels(ax4, '负载类型', '剩余寿命 (%)', '10年运行后剩余寿命对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(load_types, rotation=45, ha='right', fontsize=10)
        ax4.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        add_grid(ax4, alpha=0.3)
        
        # 添加数值标签，避免重叠，增加标签间距
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 5. 温度分布
        ax5 = axes[1, 1]
        all_temps = []
        all_loads = []
        for _, row in self.simulation_results.iterrows():
            all_temps.extend([row['avg_igbt_temp'], row['max_igbt_temp']])
            all_loads.extend([f"{row['load_type']}-平均", f"{row['load_type']}-最高"])
        
        ax5.bar(range(len(all_temps)), all_temps, alpha=0.7)
        format_axis_labels(ax5, '工况', '温度 (°C)', 'IGBT温度分布')
        ax5.set_xticks(range(len(all_temps)))
        ax5.set_xticklabels(all_loads, rotation=45, ha='right', fontsize=9)
        add_grid(ax5, alpha=0.3)
        set_adaptive_ylim(ax5, all_temps)
        
        # 6. 寿命预测曲线
        ax6 = axes[1, 2]
        years_extended = np.arange(1, 16)  # 扩展到15年
        
        for load_type in self.simulation_results['load_type'].unique():
            data = self.simulation_results[self.simulation_results['load_type'] == load_type]
            # 简单线性外推
            if len(data) >= 2:
                slope = (data['igbt_remaining_life'].iloc[-1] - data['igbt_remaining_life'].iloc[0]) / (data['years'].iloc[-1] - data['years'].iloc[0])
                predicted_life = data['igbt_remaining_life'].iloc[0] + slope * (years_extended - data['years'].iloc[0])
                predicted_life = np.maximum(predicted_life, 0)  # 不低于0
                
                ax6.plot(years_extended, predicted_life, '--', label=f'{load_type}负载预测', linewidth=2)
                ax6.plot(data['years'], data['igbt_remaining_life'], 'o-', linewidth=2, markersize=6)
        
        format_axis_labels(ax6, '运行年数', 'IGBT剩余寿命 (%)', 'IGBT寿命预测曲线')
        add_grid(ax6, alpha=0.3)
        ax6.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        set_adaptive_ylim(ax6, [0, 100])  # 寿命百分比范围
        
        # 7. 负载影响分析
        ax7 = axes[2, 0]
        load_impact = []
        load_labels = []
        
        for load_type in self.simulation_results['load_type'].unique():
            data = self.simulation_results[self.simulation_results['load_type'] == load_type]
            ten_year_life = data[data['years'] == 10]['igbt_remaining_life'].iloc[0]
            load_impact.append(ten_year_life)
            load_labels.append(load_type)
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        wedges, texts, autotexts = ax7.pie(load_impact, labels=load_labels, autopct='%1.1f%%', colors=colors)
        ax7.set_title('10年后IGBT剩余寿命分布', fontsize=12, pad=15, fontweight='bold')
        
        # 设置饼图文字大小，避免重叠
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        for text in texts:
            text.set_fontsize(10)
        
        # 8. 维护优先级矩阵
        ax8 = axes[2, 1]
        maintenance_priority = []
        priority_labels = []
        
        for _, row in self.simulation_results.iterrows():
            if row['years'] in [5, 10]:  # 重点关注5年和10年
                life = row['igbt_remaining_life']
                if life < 50:
                    priority = '高'
                elif life < 80:
                    priority = '中'
                else:
                    priority = '低'
                
                maintenance_priority.append(priority)
                priority_labels.append(f"{row['years']}年-{row['load_type']}")
        
        priority_counts = pd.Series(maintenance_priority).value_counts()
        bars = ax8.bar(priority_counts.index, priority_counts.values, color=['red', 'orange', 'green'])
        format_axis_labels(ax8, '维护优先级', '工况数量', '维护优先级分布')
        add_grid(ax8, alpha=0.3)
        
        # 添加数值标签，增加间距避免重叠
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 9. 综合评估
        ax9 = axes[2, 2]
        # 计算综合评分（考虑寿命、温度、负载等因素）
        scores = []
        score_labels = []
        
        for _, row in self.simulation_results.iterrows():
            if row['years'] == 10:  # 只评估10年情况
                life_score = row['igbt_remaining_life'] / 100
                temp_score = max(0, 1 - (row['avg_igbt_temp'] - 100) / 200)  # 温度评分
                load_factor = {'light': 1.0, 'medium': 0.8, 'heavy': 0.6}[row['load_type']]
                
                total_score = (life_score * 0.6 + temp_score * 0.3 + load_factor * 0.1) * 100
                scores.append(total_score)
                score_labels.append(row['load_type'])
        
        bars = ax9.bar(score_labels, scores, color=['lightcoral', 'lightblue', 'lightgreen'])
        format_axis_labels(ax9, '负载类型', '综合评分', '10年运行综合评估')
        add_grid(ax9, alpha=0.3)
        
        # 添加数值标签，增加间距避免重叠
        for i, score in enumerate(scores):
            ax9.text(i, score + 2, f'{score:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 优化布局，避免重叠
        plt.tight_layout()
        
        # 显示图形
        finalize_plot(fig)
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        if self.simulation_results is None:
            print("请先运行仿真获取数据")
            return
            
        print("\n" + "=" * 80)
        print("35kV/25MW级联储能PCS长期寿命综合分析报告")
        print("=" * 80)
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 分析寿命趋势
        self.analyze_life_trends()
        
        # 计算维护计划
        maintenance_df = self.calculate_maintenance_schedule()
        
        # 关键发现
        print("\n" + "=" * 60)
        print("关键发现与建议")
        print("=" * 60)
        
        # 找出最关键的维护点
        if maintenance_df is not None:
            critical_maintenance = maintenance_df[maintenance_df['urgency'].isin(['紧急', '高'])]
            if not critical_maintenance.empty:
                print("需要重点关注的情况:")
                for _, point in critical_maintenance.iterrows():
                    print(f"  • {point['years']}年 {point['load_type']}负载: IGBT剩余寿命{point['igbt_life']:.1f}%")
        
        # 10年预测总结
        ten_year_summary = self.simulation_results[self.simulation_results['years'] == 10]
        print(f"\n10年运行预测总结:")
        for _, row in ten_year_summary.iterrows():
            status = "良好" if row['igbt_remaining_life'] > 80 else "需要关注" if row['igbt_remaining_life'] > 50 else "需要更换"
            print(f"  • {row['load_type']}负载: IGBT剩余寿命{row['igbt_remaining_life']:.1f}% ({status})")
        
        # 维护策略建议
        print(f"\n维护策略建议:")
        print("  1. 建立分级维护体系:")
        print("     • 轻负载工况: 5年检查一次")
        print("     • 中等负载工况: 3年检查一次")
        print("     • 重负载工况: 2年检查一次")
        print("  2. 实施预测性维护:")
        print("     • 实时监测IGBT结温")
        print("     • 定期分析温度循环数据")
        print("     • 建立寿命预测模型")
        print("  3. 优化运行策略:")
        print("     • 避免长期重负载运行")
        print("     • 实施负载均衡")
        print("     • 优化冷却系统")
        
        print("\n" + "=" * 80)
    
    def save_detailed_results(self):
        """保存详细分析结果"""
        if self.simulation_results is None:
            print("请先运行仿真获取数据")
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'result/详细寿命分析报告_{timestamp}.csv'
        
        # 确保result目录存在
        import os
        os.makedirs('result', exist_ok=True)
        
        # 添加分析指标
        detailed_results = self.simulation_results.copy()
        detailed_results['年化寿命消耗率'] = detailed_results.groupby('load_type')['igbt_life_consumption'].diff() / detailed_results.groupby('load_type')['years'].diff()
        detailed_results['维护优先级'] = detailed_results['igbt_remaining_life'].apply(
            lambda x: '紧急' if x < 20 else '高' if x < 50 else '中' if x < 80 else '低'
        )
        detailed_results['运行状态'] = detailed_results['igbt_remaining_life'].apply(
            lambda x: '需要更换' if x < 50 else '需要关注' if x < 80 else '良好'
        )
        
        detailed_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"详细分析结果已保存到: {filename}")
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

def run_detailed_analysis():
    """运行详细分析（兼容原有接口）"""
    print("开始详细长期寿命分析...")
    
    # 创建仿真对象
    simulator = LongTermLifeSimulation()
    
    # 运行仿真获取数据
    years_list = [1, 3, 5, 10]
    load_types = ['light', 'medium', 'heavy']
    simulator.simulate_long_term_life(years_list, load_types)
    
    # 生成综合分析报告
    simulator.generate_comprehensive_report()
    
    # 绘制详细分析图表
    print("\n生成详细分析图表...")
    simulator.plot_detailed_analysis()
    
    # 保存详细结果
    filename = simulator.save_detailed_results()
    
    print(f"\n详细分析完成！")
    print(f"分析结果已保存到: {filename}")
    
    return simulator

if __name__ == "__main__":
    # 可以选择运行基础仿真或详细分析
    print("选择运行模式:")
    print("1. 基础长期寿命仿真")
    print("2. 详细寿命分析")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        run_long_term_life_simulation()
    elif choice == "2":
        run_detailed_analysis()
    else:
        print("默认运行详细寿命分析...")
        run_detailed_analysis() 