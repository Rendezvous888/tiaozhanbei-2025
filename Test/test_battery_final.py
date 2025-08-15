"""
电池模型最终测试脚本

这是一个完整的电池测试脚本，用于测试battery_model.py中的BatteryModel类。
功能包括：
1. 测试电池在不同负载模式下的响应
2. 展示SoC、电压、电流、温度等关键参数的变化
3. 生成详细的图表和分析报告
4. 保存测试结果到本地文件

作者: AI Assistant
创建时间: 2025-01-15
版本: 1.0 (最终版)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端，确保图片保存
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from battery_model import BatteryModel, BatteryModelConfig
from load_profile import generate_load_profile_new, generate_environment_temperature

class BatteryTestSuite:
    """电池测试套件"""
    
    def __init__(self):
        """初始化测试套件"""
        self.results = {}
        self.test_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置matplotlib
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        
        print("="*60)
        print("电池模型综合测试套件")
        print("="*60)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试ID: {self.test_timestamp}")
        
    def run_daily_cycle_test(self):
        """运行日常充放电循环测试"""
        
        print("\n1. 日常充放电循环测试")
        print("-" * 40)
        
        # 创建电池实例
        battery = BatteryModel(
            initial_soc=0.5,
            initial_temperature_c=25.0
        )
        
        print(f"电池配置:")
        print(f"  容量: {battery.config.rated_capacity_ah} Ah")
        print(f"  额定电流: {battery.config.rated_current_a} A")
        print(f"  串联电芯数: {battery.config.series_cells}")
        print(f"  初始SOC: {battery.state_of_charge:.1%}")
        print(f"  初始电压: {battery.get_voltage():.1f} V")
        
        # 仿真参数
        step_seconds = 60
        total_hours = 24.0
        total_steps = int(total_hours * 3600 / step_seconds)
        
        # 生成负载曲线
        power_profile = generate_load_profile_new(
            day_type="summer-weekday",
            step_seconds=step_seconds
        )
        
        # 确保长度匹配
        if len(power_profile) < total_steps:
            cycles = int(np.ceil(total_steps / len(power_profile)))
            power_profile = np.tile(power_profile, cycles)[:total_steps]
        else:
            power_profile = power_profile[:total_steps]
        
        # 生成环境温度
        temp_profile = generate_environment_temperature(
            day_type="summer-weekday",
            size=total_steps
        )
        
        # 运行仿真
        print("开始仿真...")
        results = self._simulate_battery(
            battery, power_profile, temp_profile, step_seconds, "日常循环"
        )
        
        # 分析和保存结果
        analysis = self._analyze_results(results)
        self.results['daily_cycle'] = {
            'data': results,
            'analysis': analysis
        }
        
        # 生成图表
        self._plot_daily_cycle_results(results, analysis)
        
        return results, analysis
    
    def run_constant_power_test(self):
        """运行恒功率放电测试"""
        
        print("\n2. 恒功率放电测试")
        print("-" * 40)
        
        # 创建电池实例
        battery = BatteryModel(
            initial_soc=0.9,
            initial_temperature_c=25.0
        )
        
        # 计算测试功率（80%额定功率）
        rated_power = (battery.config.rated_current_a * 
                      battery.config.series_cells * 
                      battery.config.nominal_voltage_per_cell_v)
        test_power = 0.8 * rated_power
        
        print(f"功率设置:")
        print(f"  额定功率: {rated_power/1000:.1f} kW")
        print(f"  测试功率: {test_power/1000:.1f} kW ({test_power/rated_power:.1%})")
        print(f"  初始SOC: {battery.state_of_charge:.1%}")
        
        # 仿真参数
        step_seconds = 10
        max_duration_hours = 2.0
        max_steps = int(max_duration_hours * 3600 / step_seconds)
        
        # 生成恒功率负载
        power_profile = np.full(max_steps, test_power)
        temp_profile = np.full(max_steps, 25.0)
        
        # 运行仿真
        print("开始仿真...")
        results = self._simulate_battery(
            battery, power_profile, temp_profile, step_seconds, "恒功率放电",
            stop_at_low_soc=True
        )
        
        # 分析和保存结果
        analysis = self._analyze_results(results)
        self.results['constant_power'] = {
            'data': results,
            'analysis': analysis
        }
        
        # 生成图表
        self._plot_constant_power_results(results, analysis)
        
        return results, analysis
    
    def run_temperature_test(self):
        """运行温度响应测试"""
        
        print("\n3. 温度响应测试")
        print("-" * 40)
        
        # 创建电池实例
        battery = BatteryModel(
            initial_soc=0.7,
            initial_temperature_c=25.0
        )
        
        print(f"温度测试配置:")
        print(f"  温度范围: 10°C - 45°C")
        print(f"  初始SOC: {battery.state_of_charge:.1%}")
        
        # 仿真参数
        step_seconds = 60
        duration_hours = 4.0
        total_steps = int(duration_hours * 3600 / step_seconds)
        
        # 生成温度扫描曲线
        temp_profile = np.linspace(10.0, 45.0, total_steps)
        
        # 生成中等功率负载
        rated_power = (battery.config.rated_current_a * 
                      battery.config.series_cells * 
                      battery.config.nominal_voltage_per_cell_v)
        power_profile = np.full(total_steps, 0.5 * rated_power)
        
        # 运行仿真
        print("开始仿真...")
        results = self._simulate_battery(
            battery, power_profile, temp_profile, step_seconds, "温度扫描"
        )
        
        # 分析和保存结果
        analysis = self._analyze_results(results)
        self.results['temperature'] = {
            'data': results,
            'analysis': analysis
        }
        
        # 生成图表
        self._plot_temperature_results(results, analysis)
        
        return results, analysis
    
    def _simulate_battery(self, battery, power_profile, temp_profile, step_seconds, 
                         test_name, stop_at_low_soc=False):
        """运行电池仿真"""
        
        # 初始化记录数组
        time_points = []
        soc_points = []
        voltage_points = []
        current_points = []
        power_points = []
        temp_points = []
        ambient_temp_points = []
        c_rate_points = []
        
        start_time = datetime.now()
        
        for i, (power, ambient_temp) in enumerate(zip(power_profile, temp_profile)):
            # 计算所需电流
            # 注意：在电池模型中，current > 0 表示放电（SOC下降）
            # 因此 power > 0（系统输出功率）对应 current > 0（电池放电）
            current_voltage = battery.get_voltage()
            if current_voltage > 100:
                required_current = power / current_voltage  # 保持符号一致性
            else:
                required_current = 0.0
            
            # 限制电流
            max_current = 3.0 * battery.config.rated_current_a
            required_current = np.clip(required_current, -max_current, max_current)
            
            # 更新电池状态
            battery.update_state(required_current, step_seconds, ambient_temp)
            
            # 记录数据
            time_points.append(i * step_seconds)
            soc_points.append(battery.state_of_charge)
            voltage_points.append(battery.get_voltage())
            current_points.append(required_current)
            power_points.append(power)
            temp_points.append(battery.cell_temperature_c)
            ambient_temp_points.append(ambient_temp)
            c_rate_points.append(abs(required_current) / battery.config.rated_capacity_ah)
            
            # 检查停止条件
            if stop_at_low_soc and battery.state_of_charge <= 0.05:
                print(f"SOC降至5%以下，停止仿真（步骤 {i}）")
                break
            
            # 显示进度
            if i % max(1, len(power_profile) // 10) == 0:
                progress = (i + 1) / len(power_profile) * 100
                print(f"  进度: {progress:.1f}% - SOC: {battery.state_of_charge:.1%}, "
                      f"电压: {battery.get_voltage():.1f}V, 温度: {battery.cell_temperature_c:.1f}°C")
        
        simulation_time = (datetime.now() - start_time).total_seconds()
        print(f"仿真完成，耗时: {simulation_time:.2f}秒")
        
        return {
            'test_name': test_name,
            'time_s': np.array(time_points),
            'soc': np.array(soc_points),
            'voltage_v': np.array(voltage_points),
            'current_a': np.array(current_points),
            'power_w': np.array(power_points),
            'temperature_c': np.array(temp_points),
            'ambient_temp_c': np.array(ambient_temp_points),
            'c_rate': np.array(c_rate_points),
            'simulation_time_s': simulation_time,
            'step_seconds': step_seconds
        }
    
    def _analyze_results(self, results):
        """分析仿真结果"""
        
        data = results
        
        # 基础统计
        analysis = {
            'duration_h': data['time_s'][-1] / 3600.0,
            'soc_initial': data['soc'][0],
            'soc_final': data['soc'][-1],
            'soc_change': data['soc'][-1] - data['soc'][0],
            'soc_range': [np.min(data['soc']), np.max(data['soc'])],
            'voltage_range': [np.min(data['voltage_v']), np.max(data['voltage_v'])],
            'current_range': [np.min(data['current_a']), np.max(data['current_a'])],
            'temp_initial': data['temperature_c'][0],
            'temp_final': data['temperature_c'][-1],
            'temp_max': np.max(data['temperature_c']),
            'temp_rise': np.max(data['temperature_c']) - data['temperature_c'][0],
            'max_c_rate': np.max(data['c_rate']),
            'avg_c_rate': np.mean(data['c_rate']),
            'max_power_kw': np.max(np.abs(data['power_w'])) / 1000.0
        }
        
        # 能量统计
        dt_h = data['step_seconds'] / 3600.0
        energy_out = np.sum(data['power_w'][data['power_w'] > 0]) * dt_h  # Wh
        energy_in = -np.sum(data['power_w'][data['power_w'] < 0]) * dt_h  # Wh
        analysis['energy_out_kwh'] = energy_out / 1000.0
        analysis['energy_in_kwh'] = energy_in / 1000.0
        analysis['net_energy_kwh'] = (energy_out - energy_in) / 1000.0
        
        # 效率
        if energy_out > 0:
            analysis['efficiency_percent'] = (energy_in / energy_out) * 100.0
        else:
            analysis['efficiency_percent'] = 0.0
        
        return analysis
    
    def _plot_daily_cycle_results(self, results, analysis):
        """绘制日常循环测试结果"""
        
        data = results
        time_h = data['time_s'] / 3600.0
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('电池日常循环测试结果', fontsize=16, fontweight='bold')
        
        # SOC
        axes[0, 0].plot(time_h, data['soc'] * 100, 'b-', linewidth=2)
        axes[0, 0].axhline(y=20, color='r', linestyle='--', alpha=0.7, label='低SOC警告')
        axes[0, 0].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='高SOC警告')
        axes[0, 0].set_ylabel('SOC (%)')
        axes[0, 0].set_title('荷电状态变化')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 100)
        
        # 电压
        axes[0, 1].plot(time_h, data['voltage_v'], 'r-', linewidth=2)
        axes[0, 1].set_ylabel('电压 (V)')
        axes[0, 1].set_title('端电压变化')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 电流
        axes[0, 2].plot(time_h, data['current_a'], 'g-', linewidth=2, label='电流')
        axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0, 2].set_ylabel('电流 (A)')
        axes[0, 2].set_title('电流变化 (正值=放电，负值=充电)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # 功率
        axes[1, 0].plot(time_h, data['power_w'] / 1000, 'm-', linewidth=2, label='功率')
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 0].set_xlabel('时间 (h)')
        axes[1, 0].set_ylabel('功率 (kW)')
        axes[1, 0].set_title('功率变化 (正值=放电，负值=充电)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 温度
        axes[1, 1].plot(time_h, data['temperature_c'], 'orange', linewidth=2, label='电池温度')
        axes[1, 1].plot(time_h, data['ambient_temp_c'], 'cyan', linewidth=1, 
                       linestyle='--', label='环境温度')
        axes[1, 1].axhline(y=60, color='r', linestyle='--', alpha=0.7, label='过温警告')
        axes[1, 1].set_xlabel('时间 (h)')
        axes[1, 1].set_ylabel('温度 (°C)')
        axes[1, 1].set_title('温度变化')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # C倍率
        axes[1, 2].plot(time_h, data['c_rate'], 'purple', linewidth=2)
        axes[1, 2].axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='1C')
        axes[1, 2].axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='2C')
        axes[1, 2].set_xlabel('时间 (h)')
        axes[1, 2].set_ylabel('C倍率')
        axes[1, 2].set_title('放电倍率')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # 保存图片
        save_path = f"pic/battery_daily_cycle_{self.test_timestamp}.png"
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  图表已保存: {save_path}")
        except Exception as e:
            print(f"  保存图片失败: {e}")
        
        plt.close()  # 关闭图形以释放内存
    
    def _plot_constant_power_results(self, results, analysis):
        """绘制恒功率放电结果"""
        
        data = results
        time_min = data['time_s'] / 60.0
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('恒功率放电测试结果', fontsize=14, fontweight='bold')
        
        # SOC
        ax1.plot(time_min, data['soc'] * 100, 'b-', linewidth=2)
        ax1.set_ylabel('SOC (%)')
        ax1.set_title('SOC变化曲线')
        ax1.grid(True)
        
        # 电压
        ax2.plot(time_min, data['voltage_v'], 'r-', linewidth=2)
        ax2.set_ylabel('电压 (V)')
        ax2.set_title('端电压变化曲线')
        ax2.grid(True)
        
        # 电流
        ax3.plot(time_min, data['current_a'], 'g-', linewidth=2)
        ax3.set_xlabel('时间 (分钟)')
        ax3.set_ylabel('电流 (A)')
        ax3.set_title('放电电流曲线')
        ax3.grid(True)
        
        # 温度
        ax4.plot(time_min, data['temperature_c'], 'orange', linewidth=2)
        ax4.set_xlabel('时间 (分钟)')
        ax4.set_ylabel('温度 (°C)')
        ax4.set_title('电池温度变化')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = f"pic/battery_constant_power_{self.test_timestamp}.png"
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  图表已保存: {save_path}")
        except Exception as e:
            print(f"  保存图片失败: {e}")
        
        plt.close()
    
    def _plot_temperature_results(self, results, analysis):
        """绘制温度测试结果"""
        
        data = results
        time_h = data['time_s'] / 3600.0
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('温度响应测试结果', fontsize=14, fontweight='bold')
        
        # SOC vs 温度
        ax1.plot(data['ambient_temp_c'], data['soc'] * 100, 'b-', linewidth=2)
        ax1.set_xlabel('环境温度 (°C)')
        ax1.set_ylabel('SOC (%)')
        ax1.set_title('SOC vs 环境温度')
        ax1.grid(True)
        
        # 电压 vs 温度
        ax2.plot(data['ambient_temp_c'], data['voltage_v'], 'r-', linewidth=2)
        ax2.set_xlabel('环境温度 (°C)')
        ax2.set_ylabel('电压 (V)')
        ax2.set_title('电压 vs 环境温度')
        ax2.grid(True)
        
        # 电池温度 vs 时间
        ax3.plot(time_h, data['temperature_c'], 'orange', linewidth=2, label='电池温度')
        ax3.plot(time_h, data['ambient_temp_c'], 'cyan', linewidth=1, 
                linestyle='--', label='环境温度')
        ax3.set_xlabel('时间 (h)')
        ax3.set_ylabel('温度 (°C)')
        ax3.set_title('温度变化时程')
        ax3.grid(True)
        ax3.legend()
        
        # C倍率 vs 温度
        ax4.plot(data['ambient_temp_c'], data['c_rate'], 'purple', linewidth=2)
        ax4.set_xlabel('环境温度 (°C)')
        ax4.set_ylabel('C倍率')
        ax4.set_title('C倍率 vs 环境温度')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = f"pic/battery_temperature_{self.test_timestamp}.png"
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  图表已保存: {save_path}")
        except Exception as e:
            print(f"  保存图片失败: {e}")
        
        plt.close()
    
    def generate_summary_report(self):
        """生成测试总结报告"""
        
        print("\n" + "="*60)
        print("测试总结报告")
        print("="*60)
        
        if not self.results:
            print("没有测试结果可供总结")
            return
        
        report_lines = []
        report_lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"测试ID: {self.test_timestamp}")
        report_lines.append("")
        
        # 各测试场景总结
        for test_name, test_data in self.results.items():
            analysis = test_data['analysis']
            
            report_lines.append(f"【{test_data['data']['test_name']}】")
            report_lines.append(f"  测试时长: {analysis['duration_h']:.2f} 小时")
            report_lines.append(f"  SOC变化: {analysis['soc_initial']:.1%} → {analysis['soc_final']:.1%} "
                              f"(净变化: {analysis['soc_change']:.1%})")
            report_lines.append(f"  电压范围: {analysis['voltage_range'][0]:.1f}V - {analysis['voltage_range'][1]:.1f}V")
            report_lines.append(f"  温度变化: {analysis['temp_initial']:.1f}°C → {analysis['temp_final']:.1f}°C "
                              f"(最高: {analysis['temp_max']:.1f}°C)")
            report_lines.append(f"  最大C倍率: {analysis['max_c_rate']:.2f}C")
            report_lines.append(f"  最大功率: {analysis['max_power_kw']:.1f}kW")
            if analysis['energy_out_kwh'] > 0:
                report_lines.append(f"  能量输出: {analysis['energy_out_kwh']:.1f}kWh")
            if analysis['energy_in_kwh'] > 0:
                report_lines.append(f"  能量输入: {analysis['energy_in_kwh']:.1f}kWh")
            report_lines.append("")
        
        # 综合评估
        report_lines.append("【综合评估】")
        
        # 找出性能最优/最差的场景
        max_temp_test = max(self.results.items(), key=lambda x: x[1]['analysis']['temp_max'])
        max_c_rate_test = max(self.results.items(), key=lambda x: x[1]['analysis']['max_c_rate'])
        
        report_lines.append(f"最高温度场景: {max_temp_test[1]['data']['test_name']} "
                          f"({max_temp_test[1]['analysis']['temp_max']:.1f}°C)")
        report_lines.append(f"最高倍率场景: {max_c_rate_test[1]['data']['test_name']} "
                          f"({max_c_rate_test[1]['analysis']['max_c_rate']:.2f}C)")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_path = f"result/battery_test_report_{self.test_timestamp}.txt"
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"测试报告已保存: {report_path}")
        except Exception as e:
            print(f"保存报告失败: {e}")
        
        # 打印报告
        print(report_content)
        
        return report_content
    
    def run_all_tests(self):
        """运行所有测试"""
        
        print("开始运行电池综合测试...")
        
        # 1. 日常循环测试
        try:
            self.run_daily_cycle_test()
        except Exception as e:
            print(f"日常循环测试失败: {e}")
        
        # 2. 恒功率放电测试
        try:
            self.run_constant_power_test()
        except Exception as e:
            print(f"恒功率放电测试失败: {e}")
        
        # 3. 温度响应测试
        try:
            self.run_temperature_test()
        except Exception as e:
            print(f"温度响应测试失败: {e}")
        
        # 生成总结报告
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("所有测试已完成！")
        print(f"图表保存在: pic/battery_*_{self.test_timestamp}.png")
        print(f"报告保存在: result/battery_test_report_{self.test_timestamp}.txt")


def main():
    """主函数"""
    
    # 创建测试套件
    test_suite = BatteryTestSuite()
    
    # 运行所有测试
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
