"""
电池模型综合测试脚本

该脚本测试 battery_model.py 中的 BatteryModel 类，包括：
1. 模拟多种负载场景（日常循环、高强度、恒功率等）
2. 展示电池关键参数变化（SoC、电压、电流、温度）
3. 分析电池性能和安全状态
4. 生成详细的图表和报告

作者: AI Assistant
创建时间: 2025-01-15
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Any

# 添加项目根目录到路径，以便导入模块
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 切换到项目根目录，确保能找到ocv_data.mat文件
os.chdir(parent_dir)

from battery_model import BatteryModel, BatteryModelConfig
from load_profile import generate_load_profile_new, generate_environment_temperature
try:
    import plot_utils
except ImportError:
    print("plot_utils模块未找到，将使用基础matplotlib功能")


class BatteryTester:
    """电池测试器类，用于系统性测试电池模型性能"""
    
    def __init__(self, battery_config: BatteryModelConfig = None):
        """初始化测试器
        
        Args:
            battery_config: 电池配置参数，如不提供则使用默认配置
        """
        self.config = battery_config or BatteryModelConfig()
        self.test_results = {}
        self.test_history = []
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("=== 电池模型综合测试器 ===")
        print(f"电池容量: {self.config.rated_capacity_ah:.1f} Ah")
        print(f"额定电流: {self.config.rated_current_a:.1f} A")
        print(f"串联电芯数: {self.config.series_cells}")
        print(f"额定功率: {self.config.rated_current_a * self.config.series_cells * self.config.nominal_voltage_per_cell_v / 1000:.1f} kW")
    
    def test_scenario_daily_cycle(self, duration_hours: float = 24.0, 
                                 initial_soc: float = 0.5, 
                                 initial_temp: float = 25.0) -> Dict[str, Any]:
        """测试日常充放电循环场景"""
        
        print(f"\n--- 测试场景：日常循环 ({duration_hours:.1f}小时) ---")
        
        # 创建电池实例
        battery = BatteryModel(
            config=self.config,
            initial_soc=initial_soc,
            initial_temperature_c=initial_temp
        )
        
        # 生成负载曲线
        step_seconds = 60  # 1分钟步长
        total_steps = int(duration_hours * 3600 / step_seconds)
        
        # 使用夏季工作日的负载曲线
        power_profile = generate_load_profile_new(
            day_type="summer-weekday", 
            step_seconds=step_seconds
        )
        
        # 重复负载曲线以覆盖整个测试期间
        cycles_needed = int(np.ceil(total_steps / len(power_profile)))
        extended_power = np.tile(power_profile, cycles_needed)[:total_steps]
        
        # 生成环境温度
        temp_profile = generate_environment_temperature(
            day_type="summer-weekday", 
            size=total_steps
        )
        
        # 运行仿真
        results = self._run_simulation(
            battery, extended_power, temp_profile, step_seconds, "日常循环"
        )
        
        # 分析结果
        analysis = self._analyze_results(results, "日常循环")
        
        # 保存结果
        self.test_results['daily_cycle'] = {
            'results': results,
            'analysis': analysis
        }
        
        return self.test_results['daily_cycle']
    
    def test_scenario_high_power(self, duration_hours: float = 2.0,
                                initial_soc: float = 0.8,
                                power_ratio: float = 2.0) -> Dict[str, Any]:
        """测试高功率放电场景"""
        
        print(f"\n--- 测试场景：高功率放电 ({power_ratio:.1f}倍额定功率) ---")
        
        # 创建电池实例
        battery = BatteryModel(
            config=self.config,
            initial_soc=initial_soc,
            initial_temperature_c=25.0
        )
        
        # 生成高功率负载曲线
        step_seconds = 30  # 30秒步长，提高时间分辨率
        total_steps = int(duration_hours * 3600 / step_seconds)
        
        # 创建高功率脉冲模式
        rated_power = self.config.rated_current_a * self.config.series_cells * self.config.nominal_voltage_per_cell_v
        high_power = rated_power * power_ratio
        
        power_profile = np.zeros(total_steps)
        for i in range(total_steps):
            # 模拟脉冲负载：高功率放电 + 间歇
            cycle_position = (i * step_seconds) % 600  # 10分钟周期
            if cycle_position < 300:  # 前5分钟高功率放电
                power_profile[i] = high_power
            else:  # 后5分钟低功率或充电
                power_profile[i] = -high_power * 0.3
        
        # 环境温度（较高，模拟恶劣工况）
        temp_profile = np.full(total_steps, 35.0)
        
        # 运行仿真
        results = self._run_simulation(
            battery, power_profile, temp_profile, step_seconds, "高功率放电"
        )
        
        # 分析结果
        analysis = self._analyze_results(results, "高功率放电")
        
        # 保存结果
        self.test_results['high_power'] = {
            'results': results,
            'analysis': analysis
        }
        
        return self.test_results['high_power']
    
    def test_scenario_constant_power(self, power_w: float = None,
                                   duration_hours: float = 1.0,
                                   initial_soc: float = 0.9) -> Dict[str, Any]:
        """测试恒功率放电场景"""
        
        if power_w is None:
            # 默认使用80%额定功率
            power_w = 0.8 * self.config.rated_current_a * self.config.series_cells * self.config.nominal_voltage_per_cell_v
        
        print(f"\n--- 测试场景：恒功率放电 ({power_w/1000:.1f} kW) ---")
        
        # 创建电池实例
        battery = BatteryModel(
            config=self.config,
            initial_soc=initial_soc,
            initial_temperature_c=25.0
        )
        
        # 生成恒功率负载曲线
        step_seconds = 10  # 10秒步长，高精度
        total_steps = int(duration_hours * 3600 / step_seconds)
        
        power_profile = np.full(total_steps, power_w)
        temp_profile = np.full(total_steps, 25.0)
        
        # 运行仿真
        results = self._run_simulation(
            battery, power_profile, temp_profile, step_seconds, "恒功率放电"
        )
        
        # 分析结果
        analysis = self._analyze_results(results, "恒功率放电")
        
        # 保存结果
        self.test_results['constant_power'] = {
            'results': results,
            'analysis': analysis
        }
        
        return self.test_results['constant_power']
    
    def test_scenario_temperature_sweep(self, temp_range: Tuple[float, float] = (10.0, 45.0),
                                      duration_hours: float = 6.0) -> Dict[str, Any]:
        """测试温度扫描场景"""
        
        print(f"\n--- 测试场景：温度扫描 ({temp_range[0]:.1f}°C - {temp_range[1]:.1f}°C) ---")
        
        # 创建电池实例
        battery = BatteryModel(
            config=self.config,
            initial_soc=0.5,
            initial_temperature_c=25.0
        )
        
        # 生成温度扫描曲线
        step_seconds = 60  # 1分钟步长
        total_steps = int(duration_hours * 3600 / step_seconds)
        
        # 温度线性变化
        temp_profile = np.linspace(temp_range[0], temp_range[1], total_steps)
        
        # 中等功率负载
        rated_power = self.config.rated_current_a * self.config.series_cells * self.config.nominal_voltage_per_cell_v
        power_profile = np.full(total_steps, 0.5 * rated_power)
        
        # 运行仿真
        results = self._run_simulation(
            battery, power_profile, temp_profile, step_seconds, "温度扫描"
        )
        
        # 分析结果
        analysis = self._analyze_results(results, "温度扫描")
        
        # 保存结果
        self.test_results['temperature_sweep'] = {
            'results': results,
            'analysis': analysis
        }
        
        return self.test_results['temperature_sweep']
    
    def _run_simulation(self, battery: BatteryModel, power_profile: np.ndarray,
                       temp_profile: np.ndarray, step_seconds: float,
                       scenario_name: str) -> Dict[str, Any]:
        """运行电池仿真"""
        
        print(f"运行仿真：{scenario_name} (共 {len(power_profile)} 步)")
        
        # 初始化记录数组
        time_points = []
        soc_points = []
        voltage_points = []
        current_points = []
        power_points = []
        temp_points = []
        ambient_temp_points = []
        safety_status_points = []
        
        # 开始仿真
        start_time = datetime.now()
        
        for i, (power, ambient_temp) in enumerate(zip(power_profile, temp_profile)):
            # 计算电流（基于功率和当前电压）
            current_voltage = battery.get_voltage()
            if current_voltage > 0:
                current = power / current_voltage
            else:
                current = 0.0
            
            # 限制电流不超过3倍额定电流（安全保护）
            max_current = 3.0 * self.config.rated_current_a
            current = np.clip(current, -max_current, max_current)
            
            # 更新电池状态
            battery.update_state(current, step_seconds, ambient_temp)
            
            # 记录数据
            time_points.append(i * step_seconds)
            soc_points.append(battery.state_of_charge)
            voltage_points.append(battery.get_voltage())
            current_points.append(current)
            power_points.append(power)
            temp_points.append(battery.cell_temperature_c)
            ambient_temp_points.append(ambient_temp)
            
            # 获取安全状态
            safety_check = battery.check_safety_limits()
            safety_status_points.append(safety_check)
            
            # 进度显示
            if i % (len(power_profile) // 10) == 0:
                progress = (i + 1) / len(power_profile) * 100
                print(f"进度: {progress:.1f}% - SOC: {battery.state_of_charge:.1%}, "
                      f"电压: {battery.get_voltage():.1f}V, 温度: {battery.cell_temperature_c:.1f}°C")
        
        simulation_time = (datetime.now() - start_time).total_seconds()
        print(f"仿真完成，耗时: {simulation_time:.2f}秒")
        
        return {
            'scenario_name': scenario_name,
            'time_s': np.array(time_points),
            'soc': np.array(soc_points),
            'voltage_v': np.array(voltage_points),
            'current_a': np.array(current_points),
            'power_w': np.array(power_points),
            'temperature_c': np.array(temp_points),
            'ambient_temp_c': np.array(ambient_temp_points),
            'safety_status': safety_status_points,
            'simulation_time_s': simulation_time,
            'step_seconds': step_seconds
        }
    
    def _analyze_results(self, results: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
        """分析仿真结果"""
        
        print(f"分析结果：{scenario_name}")
        
        # 基础统计
        soc_range = [np.min(results['soc']), np.max(results['soc'])]
        voltage_range = [np.min(results['voltage_v']), np.max(results['voltage_v'])]
        current_range = [np.min(results['current_a']), np.max(results['current_a'])]
        temp_range = [np.min(results['temperature_c']), np.max(results['temperature_c'])]
        
        # 能量统计
        dt_h = results['step_seconds'] / 3600.0
        energy_consumed_wh = np.sum(results['power_w'][results['power_w'] > 0]) * dt_h
        energy_charged_wh = -np.sum(results['power_w'][results['power_w'] < 0]) * dt_h
        net_energy_wh = energy_consumed_wh - energy_charged_wh
        
        # SOC变化
        soc_change = results['soc'][-1] - results['soc'][0]
        soc_excursion = np.max(results['soc']) - np.min(results['soc'])
        
        # 温度统计
        temp_rise = np.max(results['temperature_c']) - results['temperature_c'][0]
        
        # 安全状态统计
        safety_violations = sum(1 for status in results['safety_status'] 
                              if not status['is_safe'])
        
        # C倍率统计
        c_rates = np.abs(results['current_a']) / self.config.rated_capacity_ah
        max_c_rate = np.max(c_rates)
        avg_c_rate = np.mean(c_rates)
        
        analysis = {
            'soc_range': soc_range,
            'soc_change': soc_change,
            'soc_excursion': soc_excursion,
            'voltage_range_v': voltage_range,
            'current_range_a': current_range,
            'temperature_range_c': temp_range,
            'temperature_rise_c': temp_rise,
            'energy_consumed_wh': energy_consumed_wh,
            'energy_charged_wh': energy_charged_wh,
            'net_energy_wh': net_energy_wh,
            'max_c_rate': max_c_rate,
            'avg_c_rate': avg_c_rate,
            'safety_violations': safety_violations,
            'efficiency_percent': (energy_charged_wh / energy_consumed_wh * 100) if energy_consumed_wh > 0 else 0,
            'duration_h': results['time_s'][-1] / 3600.0
        }
        
        # 打印关键指标
        print(f"  SOC变化: {soc_change:.1%} (范围: {soc_range[0]:.1%} - {soc_range[1]:.1%})")
        print(f"  电压范围: {voltage_range[0]:.1f}V - {voltage_range[1]:.1f}V")
        print(f"  温度上升: {temp_rise:.1f}°C (最高: {temp_range[1]:.1f}°C)")
        print(f"  最大C倍率: {max_c_rate:.2f}C")
        print(f"  能量效率: {analysis['efficiency_percent']:.1f}%")
        print(f"  安全违规: {safety_violations}次")
        
        return analysis
    
    def plot_scenario_results(self, scenario_name: str, save_path: str = None):
        """绘制特定场景的结果"""
        
        if scenario_name not in self.test_results:
            print(f"场景 '{scenario_name}' 的结果不存在")
            return
        
        results = self.test_results[scenario_name]['results']
        analysis = self.test_results[scenario_name]['analysis']
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 时间轴（小时）
        time_h = results['time_s'] / 3600.0
        
        # 子图1：SOC变化
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_h, results['soc'] * 100, 'b-', linewidth=2, label='SOC')
        ax1.axhline(y=20, color='r', linestyle='--', alpha=0.7, label='低SOC警告')
        ax1.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='高SOC警告')
        ax1.set_ylabel('SOC (%)')
        ax1.set_title('荷电状态变化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # 子图2：电压变化
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_h, results['voltage_v'], 'r-', linewidth=2, label='端电压')
        ax2.set_ylabel('电压 (V)')
        ax2.set_title('端电压变化')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 子图3：电流变化
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_h, results['current_a'], 'g-', linewidth=2, label='电流')
        ax3.axhline(y=self.config.rated_current_a, color='orange', linestyle='--', 
                   alpha=0.7, label='额定电流')
        ax3.axhline(y=-self.config.rated_current_a, color='orange', linestyle='--', alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('电流 (A)')
        ax3.set_title('电流变化')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 子图4：功率变化
        ax4 = fig.add_subplot(gs[1, 1])
        power_kw = results['power_w'] / 1000
        ax4.plot(time_h, power_kw, 'm-', linewidth=2, label='功率')
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_ylabel('功率 (kW)')
        ax4.set_title('功率变化')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 子图5：温度变化
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time_h, results['temperature_c'], 'orange', linewidth=2, label='电池温度')
        ax5.plot(time_h, results['ambient_temp_c'], 'cyan', linewidth=1, 
                linestyle='--', label='环境温度')
        ax5.axhline(y=60, color='r', linestyle='--', alpha=0.7, label='过温警告')
        ax5.set_xlabel('时间 (h)')
        ax5.set_ylabel('温度 (°C)')
        ax5.set_title('温度变化')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 子图6：C倍率
        ax6 = fig.add_subplot(gs[2, 1])
        c_rates = np.abs(results['current_a']) / self.config.rated_capacity_ah
        ax6.plot(time_h, c_rates, 'purple', linewidth=2, label='C倍率')
        ax6.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='1C')
        ax6.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='2C')
        ax6.set_xlabel('时间 (h)')
        ax6.set_ylabel('C倍率')
        ax6.set_title('放电倍率')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # 设置整体标题
        fig.suptitle(f'电池测试结果 - {scenario_name}', fontsize=16, fontweight='bold')
        
        # 添加统计信息文本
        stats_text = f"""测试统计:
SOC变化: {analysis['soc_change']:.1%}
温度上升: {analysis['temperature_rise_c']:.1f}°C  
最大C倍率: {analysis['max_c_rate']:.2f}C
能量效率: {analysis['efficiency_percent']:.1f}%
安全违规: {analysis['safety_violations']}次"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # 保存图片
        if save_path is None:
            save_path = f"../pic/battery_test_{scenario_name.replace(' ', '_')}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        
        plt.show()
    
    def plot_comparison_summary(self, save_path: str = None):
        """绘制所有场景的对比总结"""
        
        if not self.test_results:
            print("没有测试结果可供对比")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        scenarios = list(self.test_results.keys())
        scenario_names = [self.test_results[s]['results']['scenario_name'] for s in scenarios]
        
        # 1. SOC变化对比
        soc_changes = [self.test_results[s]['analysis']['soc_change'] * 100 for s in scenarios]
        bars1 = ax1.bar(scenario_names, soc_changes, color=['blue', 'orange', 'green', 'red'][:len(scenarios)])
        ax1.set_ylabel('SOC变化 (%)')
        ax1.set_title('SOC变化对比')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars1, soc_changes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 2. 温度上升对比
        temp_rises = [self.test_results[s]['analysis']['temperature_rise_c'] for s in scenarios]
        bars2 = ax2.bar(scenario_names, temp_rises, color=['blue', 'orange', 'green', 'red'][:len(scenarios)])
        ax2.set_ylabel('温度上升 (°C)')
        ax2.set_title('温度上升对比')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, temp_rises):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}°C', ha='center', va='bottom')
        
        # 3. 最大C倍率对比
        max_c_rates = [self.test_results[s]['analysis']['max_c_rate'] for s in scenarios]
        bars3 = ax3.bar(scenario_names, max_c_rates, color=['blue', 'orange', 'green', 'red'][:len(scenarios)])
        ax3.set_ylabel('最大C倍率')
        ax3.set_title('最大C倍率对比')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='1C')
        ax3.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='2C')
        ax3.legend()
        
        for bar, value in zip(bars3, max_c_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}C', ha='center', va='bottom')
        
        # 4. 能量效率对比
        efficiencies = [self.test_results[s]['analysis']['efficiency_percent'] for s in scenarios]
        bars4 = ax4.bar(scenario_names, efficiencies, color=['blue', 'orange', 'green', 'red'][:len(scenarios)])
        ax4.set_ylabel('能量效率 (%)')
        ax4.set_title('能量效率对比')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 100)
        
        for bar, value in zip(bars4, efficiencies):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.suptitle('电池测试场景对比总结', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = "../pic/battery_test_comparison_summary.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比总结图已保存: {save_path}")
        
        plt.show()
    
    def generate_test_report(self, save_path: str = None) -> str:
        """生成测试报告"""
        
        if not self.test_results:
            return "没有测试结果可供生成报告"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("电池模型综合测试报告")
        report_lines.append("=" * 60)
        report_lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 电池配置信息
        report_lines.append("电池配置:")
        report_lines.append(f"  容量: {self.config.rated_capacity_ah:.1f} Ah")
        report_lines.append(f"  额定电流: {self.config.rated_current_a:.1f} A")
        report_lines.append(f"  串联电芯数: {self.config.series_cells}")
        report_lines.append(f"  单体电压: {self.config.nominal_voltage_per_cell_v:.2f} V")
        report_lines.append("")
        
        # 各场景测试结果
        for scenario_key, test_data in self.test_results.items():
            results = test_data['results']
            analysis = test_data['analysis']
            
            report_lines.append(f"场景: {results['scenario_name']}")
            report_lines.append("-" * 40)
            report_lines.append(f"  测试时长: {analysis['duration_h']:.2f} 小时")
            report_lines.append(f"  SOC变化: {analysis['soc_change']:.1%}")
            report_lines.append(f"  SOC范围: {analysis['soc_range'][0]:.1%} - {analysis['soc_range'][1]:.1%}")
            report_lines.append(f"  电压范围: {analysis['voltage_range_v'][0]:.1f}V - {analysis['voltage_range_v'][1]:.1f}V")
            report_lines.append(f"  温度上升: {analysis['temperature_rise_c']:.1f}°C")
            report_lines.append(f"  最大C倍率: {analysis['max_c_rate']:.2f}C")
            report_lines.append(f"  平均C倍率: {analysis['avg_c_rate']:.2f}C")
            report_lines.append(f"  能量效率: {analysis['efficiency_percent']:.1f}%")
            report_lines.append(f"  安全违规次数: {analysis['safety_violations']}")
            report_lines.append("")
        
        # 总结和建议
        report_lines.append("测试总结:")
        report_lines.append("-" * 40)
        
        # 找出最大SOC变化的场景
        max_soc_change_scenario = max(self.test_results.items(), 
                                     key=lambda x: abs(x[1]['analysis']['soc_change']))
        report_lines.append(f"最大SOC变化场景: {max_soc_change_scenario[1]['results']['scenario_name']} "
                          f"({max_soc_change_scenario[1]['analysis']['soc_change']:.1%})")
        
        # 找出最大温升场景
        max_temp_scenario = max(self.test_results.items(), 
                               key=lambda x: x[1]['analysis']['temperature_rise_c'])
        report_lines.append(f"最大温升场景: {max_temp_scenario[1]['results']['scenario_name']} "
                          f"({max_temp_scenario[1]['analysis']['temperature_rise_c']:.1f}°C)")
        
        # 找出最高C倍率场景
        max_c_rate_scenario = max(self.test_results.items(), 
                                 key=lambda x: x[1]['analysis']['max_c_rate'])
        report_lines.append(f"最高C倍率场景: {max_c_rate_scenario[1]['results']['scenario_name']} "
                          f"({max_c_rate_scenario[1]['analysis']['max_c_rate']:.2f}C)")
        
        # 安全性评估
        total_violations = sum(test_data['analysis']['safety_violations'] 
                             for test_data in self.test_results.values())
        if total_violations == 0:
            report_lines.append("安全性评估: 所有测试场景均未发现安全违规")
        else:
            report_lines.append(f"安全性评估: 共发现 {total_violations} 次安全违规，需要关注")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if save_path is None:
            save_path = f"../result/battery_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"测试报告已保存: {save_path}")
        return report_content
    
    def run_all_tests(self):
        """运行所有测试场景"""
        
        print("开始运行所有测试场景...")
        
        # 1. 日常循环测试
        self.test_scenario_daily_cycle(duration_hours=24.0)
        
        # 2. 高功率测试
        self.test_scenario_high_power(duration_hours=2.0, power_ratio=2.0)
        
        # 3. 恒功率测试
        self.test_scenario_constant_power(duration_hours=1.0)
        
        # 4. 温度扫描测试
        self.test_scenario_temperature_sweep(duration_hours=4.0)
        
        print("\n所有测试场景完成！")
        
        # 生成图表
        for scenario in self.test_results.keys():
            self.plot_scenario_results(scenario)
        
        # 生成对比总结
        self.plot_comparison_summary()
        
        # 生成测试报告
        report = self.generate_test_report()
        print("\n" + "="*60)
        print(report)


def main():
    """主函数"""
    
    # 创建测试器
    tester = BatteryTester()
    
    # 选择要运行的测试
    print("\n请选择要运行的测试:")
    print("1. 日常循环测试")
    print("2. 高功率放电测试") 
    print("3. 恒功率放电测试")
    print("4. 温度扫描测试")
    print("5. 运行所有测试")
    
    choice = input("请输入选择 (1-5): ").strip()
    
    if choice == "1":
        result = tester.test_scenario_daily_cycle()
        tester.plot_scenario_results('daily_cycle')
    elif choice == "2":
        result = tester.test_scenario_high_power()
        tester.plot_scenario_results('high_power')
    elif choice == "3":
        result = tester.test_scenario_constant_power()
        tester.plot_scenario_results('constant_power')
    elif choice == "4":
        result = tester.test_scenario_temperature_sweep()
        tester.plot_scenario_results('temperature_sweep')
    elif choice == "5":
        tester.run_all_tests()
    else:
        print("无效选择，运行默认的日常循环测试")
        result = tester.test_scenario_daily_cycle()
        tester.plot_scenario_results('daily_cycle')


if __name__ == "__main__":
    main()
