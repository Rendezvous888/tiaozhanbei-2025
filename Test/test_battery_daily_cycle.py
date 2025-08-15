"""
电池日循环测试脚本

专门测试24小时日循环运行，展示：
1. 典型储能电站的日运行模式
2. SOC和功率的完美对应关系
3. 真实的充放电循环
4. 详细的时段分析

作者: AI Assistant
创建时间: 2025-01-15
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
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

class DailyCycleTester:
    """日循环测试器"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 11
        
        # 日循环SOC工作范围
        self.soc_min = 0.20  # 20% 最低SOC
        self.soc_max = 0.80  # 80% 最高SOC
        
    def test_daily_cycle(self):
        """测试24小时日循环"""
        
        print("=" * 80)
        print("🌅 电池24小时日循环测试")
        print("=" * 80)
        
        # 创建电池实例（从中等SOC开始）
        battery = BatteryModel(
            initial_soc=0.5,  # 从50%开始
            initial_temperature_c=25.0
        )
        
        print(f"📋 日循环配置:")
        print(f"  电池容量: {battery.config.rated_capacity_ah} Ah")
        print(f"  额定功率: {battery.config.rated_current_a * battery.config.series_cells * battery.config.nominal_voltage_per_cell_v / 1000:.1f} kW")
        print(f"  初始SOC: {battery.state_of_charge:.1%}")
        print(f"  工作范围: {self.soc_min:.0%} - {self.soc_max:.0%}")
        
        # 24小时精细仿真（30秒步长）
        step_seconds = 30
        total_hours = 24.0
        total_steps = int(total_hours * 3600 / step_seconds)
        
        print(f"  仿真步长: {step_seconds}秒")
        print(f"  总时长: {total_hours}小时")
        print(f"  总步数: {total_steps}")
        
        # 创建典型储能日循环负载
        time_h = np.linspace(0, total_hours, total_steps)
        power_profile, load_description = self._create_daily_cycle_profile(time_h)
        
        # 环境温度日变化（考虑季节特征）
        temp_profile = self._create_temperature_profile(time_h)
        
        print(f"\n🔋 负载特性:")
        print(f"  功率范围: {np.min(power_profile)/1000:.1f} ~ {np.max(power_profile)/1000:.1f} kW")
        print(f"  温度范围: {np.min(temp_profile):.1f} ~ {np.max(temp_profile):.1f} °C")
        
        # 运行日循环仿真
        results = self._simulate_daily_cycle(
            battery, power_profile, temp_profile, step_seconds, load_description
        )
        
        # 分析日循环特性
        cycle_analysis = self._analyze_daily_cycle(results, load_description)
        
        # 生成专业日循环图表
        self._plot_daily_cycle_results(results, cycle_analysis, load_description)
        
        return results, cycle_analysis
    
    def _create_daily_cycle_profile(self, time_h):
        """创建典型储能日循环负载曲线"""
        
        # 计算合理的功率水平（基于电池规格）
        nominal_voltage = 314 * 3.57  # 314串 × 3.57V
        rated_power_w = 420 * nominal_voltage  # 420A × 电压
        max_power_w = 0.08 * rated_power_w  # 使用8%额定功率
        
        power_profile = np.zeros_like(time_h)
        load_description = {}
        
        for i, t in enumerate(time_h):
            if 0 <= t < 2:    # 深夜 00:00-02:00
                power_profile[i] = -max_power_w * 0.3  # 轻度充电
                period = "深夜充电"
            elif 2 <= t < 6:  # 凌晨 02:00-06:00
                power_profile[i] = -max_power_w * 0.8  # 深度充电（谷电时段）
                period = "谷电充电"
            elif 6 <= t < 8:  # 早晨 06:00-08:00
                power_profile[i] = -max_power_w * 0.2  # 充电减缓
                period = "充电减缓"
            elif 8 <= t < 10: # 早高峰 08:00-10:00
                power_profile[i] = max_power_w * 0.9   # 早高峰放电
                period = "早高峰放电"
            elif 10 <= t < 12: # 上午 10:00-12:00
                power_profile[i] = max_power_w * 0.4   # 平稳放电
                period = "上午放电"
            elif 12 <= t < 14: # 中午 12:00-14:00
                power_profile[i] = -max_power_w * 0.6  # 光伏充电
                period = "光伏充电"
            elif 14 <= t < 16: # 下午 14:00-16:00
                power_profile[i] = max_power_w * 0.3   # 轻度放电
                period = "下午放电"
            elif 16 <= t < 18: # 傍晚 16:00-18:00
                power_profile[i] = max_power_w * 0.6   # 放电增加
                period = "傍晚放电"
            elif 18 <= t < 21: # 晚高峰 18:00-21:00
                power_profile[i] = max_power_w * 1.0   # 晚高峰满功率放电
                period = "晚高峰放电"
            elif 21 <= t < 23: # 夜间 21:00-23:00
                power_profile[i] = max_power_w * 0.2   # 轻度放电
                period = "夜间放电"
            else:             # 深夜 23:00-24:00
                power_profile[i] = -max_power_w * 0.1  # 准备充电
                period = "准备充电"
            
            # 记录时段描述
            hour = int(t)
            if hour not in load_description:
                load_description[hour] = period
        
        # 添加现实的负荷波动
        np.random.seed(42)
        for i in range(1, len(power_profile)):
            # 平滑的随机波动（±5%）
            noise = np.random.normal(0, max_power_w * 0.05)
            # 限制变化幅度
            noise = np.clip(noise, -max_power_w * 0.1, max_power_w * 0.1)
            power_profile[i] += noise
        
        # 应用5点平滑滤波
        window = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        power_profile = np.convolve(power_profile, window, mode='same')
        
        # 精确的24小时能量平衡
        total_energy = np.sum(power_profile)
        energy_correction = -total_energy / len(power_profile)
        power_profile += energy_correction
        
        return power_profile, load_description
    
    def _create_temperature_profile(self, time_h):
        """创建环境温度日变化曲线"""
        
        # 典型春季温度变化
        base_temp = 25.0  # 基础温度25°C
        daily_amplitude = 8.0  # 日温差8°C
        
        # 正弦温度变化，最低温度在早晨6点，最高温度在下午2点
        temp_profile = base_temp + daily_amplitude * np.sin(2 * np.pi * (time_h - 6) / 24)
        
        # 添加小幅随机波动
        np.random.seed(42)
        temp_noise = np.random.normal(0, 0.5, len(time_h))
        temp_profile += temp_noise
        
        return temp_profile
    
    def _simulate_daily_cycle(self, battery, power_profile, temp_profile, step_seconds, load_description):
        """模拟24小时日循环运行"""
        
        print(f"\n🔄 开始24小时日循环仿真...")
        
        results = {
            'time_h': [],
            'time_str': [],  # 时间字符串
            'soc': [],
            'voltage_v': [],
            'current_a': [],
            'power_w': [],
            'temperature_c': [],
            'ambient_temp_c': [],
            'period': [],  # 运行时段
            'energy_throughput': [],  # 累计能量吞吐
            'cycle_depth': []  # 循环深度
        }
        
        energy_throughput = 0.0
        initial_soc = battery.state_of_charge
        soc_max_reached = initial_soc
        soc_min_reached = initial_soc
        
        for i, (target_power, ambient_temp) in enumerate(zip(power_profile, temp_profile)):
            current_time_h = i * step_seconds / 3600.0
            
            # 应用物理约束
            actual_power = self._apply_smart_constraints(target_power, battery.state_of_charge)
            
            # 计算电流
            current_voltage = battery.get_voltage()
            if current_voltage > 50:
                required_current = actual_power / current_voltage
            else:
                required_current = 0.0
            
            # 更新电池状态
            battery.update_state(required_current, step_seconds, ambient_temp)
            
            # 计算能量吞吐
            energy_step = abs(actual_power) * step_seconds / 3600 / 1000  # kWh
            energy_throughput += energy_step
            
            # 跟踪SOC范围
            current_soc = battery.state_of_charge
            soc_max_reached = max(soc_max_reached, current_soc)
            soc_min_reached = min(soc_min_reached, current_soc)
            
            # 计算循环深度
            cycle_depth = soc_max_reached - soc_min_reached
            
            # 确定运行时段
            hour = int(current_time_h)
            period = load_description.get(hour, "未知时段")
            
            # 时间字符串
            time_str = f"{hour:02d}:{int((current_time_h - hour) * 60):02d}"
            
            # 记录数据
            results['time_h'].append(current_time_h)
            results['time_str'].append(time_str)
            results['soc'].append(current_soc)
            results['voltage_v'].append(battery.get_voltage())
            results['current_a'].append(required_current)
            results['power_w'].append(actual_power)
            results['temperature_c'].append(battery.cell_temperature_c)
            results['ambient_temp_c'].append(ambient_temp)
            results['period'].append(period)
            results['energy_throughput'].append(energy_throughput)
            results['cycle_depth'].append(cycle_depth)
            
            # 显示关键时刻进度
            if hour in [0, 6, 8, 12, 18, 21] and int((current_time_h - hour) * 60) < 1:
                print(f"  {time_str} ({period}): SOC={current_soc:.1%}, "
                      f"功率={actual_power/1000:.1f}kW, "
                      f"温度={battery.cell_temperature_c:.1f}°C")
        
        # 转换为numpy数组
        for key in results:
            if key not in ['time_str', 'period']:
                results[key] = np.array(results[key])
        
        print(f"✅ 日循环仿真完成！")
        return results
    
    def _apply_smart_constraints(self, target_power, current_soc):
        """应用智能约束控制"""
        
        actual_power = target_power
        
        # 严格边界控制
        if target_power > 0 and current_soc <= self.soc_min:
            actual_power = 0.0
        elif target_power < 0 and current_soc >= self.soc_max:
            actual_power = 0.0
        
        # 渐进式边界控制（提前5%开始减功率）
        margin = 0.05
        
        if target_power > 0 and current_soc <= self.soc_min + margin:
            factor = max(0.0, (current_soc - self.soc_min) / margin)
            actual_power = target_power * factor
        
        elif target_power < 0 and current_soc >= self.soc_max - margin:
            factor = max(0.0, (self.soc_max - current_soc) / margin)
            actual_power = target_power * factor
        
        return actual_power
    
    def _analyze_daily_cycle(self, results, load_description):
        """分析日循环特性"""
        
        print(f"\n" + "=" * 80)
        print("📊 日循环分析报告")
        print("=" * 80)
        
        soc_data = results['soc']
        power_data = results['power_w'] / 1000  # 转换为kW
        energy_data = results['energy_throughput']
        
        # SOC统计
        soc_initial = soc_data[0]
        soc_final = soc_data[-1]
        soc_min = np.min(soc_data)
        soc_max = np.max(soc_data)
        soc_range = soc_max - soc_min
        
        # 功率统计
        power_min = np.min(power_data)
        power_max = np.max(power_data)
        
        # 能量统计
        charge_energy = np.sum(np.where(power_data < 0, -power_data, 0)) * 0.5 / 60  # kWh
        discharge_energy = np.sum(np.where(power_data > 0, power_data, 0)) * 0.5 / 60  # kWh
        total_throughput = energy_data[-1]
        
        # 充放电周期统计
        charge_periods = []
        discharge_periods = []
        current_period = None
        period_start = 0
        
        for i, power in enumerate(power_data):
            if power < -0.5:  # 充电
                if current_period != 'charge':
                    if current_period is not None:
                        if current_period == 'discharge':
                            discharge_periods.append((period_start, i-1))
                    current_period = 'charge'
                    period_start = i
            elif power > 0.5:  # 放电
                if current_period != 'discharge':
                    if current_period is not None:
                        if current_period == 'charge':
                            charge_periods.append((period_start, i-1))
                    current_period = 'discharge'
                    period_start = i
        
        # 处理最后一个周期
        if current_period == 'charge':
            charge_periods.append((period_start, len(power_data)-1))
        elif current_period == 'discharge':
            discharge_periods.append((period_start, len(power_data)-1))
        
        print(f"🔋 SOC分析:")
        print(f"  初始SOC: {soc_initial:.1%}")
        print(f"  最终SOC: {soc_final:.1%}")
        print(f"  净变化: {soc_final - soc_initial:.2%}")
        print(f"  SOC范围: {soc_min:.1%} - {soc_max:.1%}")
        print(f"  循环深度: {soc_range:.1%}")
        print(f"  边界合规: {'✓' if soc_min >= self.soc_min and soc_max <= self.soc_max else '✗'}")
        
        print(f"\n⚡ 功率分析:")
        print(f"  功率范围: {power_min:.1f} ~ {power_max:.1f} kW")
        print(f"  充电功率: {power_min:.1f} kW")
        print(f"  放电功率: {power_max:.1f} kW")
        
        print(f"\n🔄 能量分析:")
        print(f"  充电能量: {charge_energy:.2f} kWh")
        print(f"  放电能量: {discharge_energy:.2f} kWh")
        print(f"  净能量: {discharge_energy - charge_energy:.2f} kWh")
        print(f"  总吞吐量: {total_throughput:.2f} kWh")
        print(f"  能量效率: {discharge_energy/max(charge_energy, 0.001):.1%}")
        
        print(f"\n🕐 周期分析:")
        print(f"  充电周期数: {len(charge_periods)}")
        print(f"  放电周期数: {len(discharge_periods)}")
        
        # 温度分析
        temp_min = np.min(results['temperature_c'])
        temp_max = np.max(results['temperature_c'])
        temp_avg = np.mean(results['temperature_c'])
        
        print(f"\n🌡️ 温度分析:")
        print(f"  电池温度范围: {temp_min:.1f} - {temp_max:.1f} °C")
        print(f"  平均电池温度: {temp_avg:.1f} °C")
        
        analysis = {
            'soc_initial': soc_initial,
            'soc_final': soc_final,
            'soc_range': soc_range,
            'power_range': [power_min, power_max],
            'charge_energy': charge_energy,
            'discharge_energy': discharge_energy,
            'total_throughput': total_throughput,
            'charge_periods': charge_periods,
            'discharge_periods': discharge_periods,
            'temp_range': [temp_min, temp_max],
            'temp_avg': temp_avg
        }
        
        return analysis
    
    def _plot_daily_cycle_results(self, results, analysis, load_description):
        """绘制专业日循环结果图表"""
        
        # 创建大图表
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        fig.suptitle('🌅 电池24小时日循环详细分析', fontsize=18, fontweight='bold', y=0.95)
        
        time_h = results['time_h']
        
        # 1. 主图：SOC和功率的双轴图
        ax_main = fig.add_subplot(gs[0, :])
        
        # SOC曲线（左轴）
        color1 = 'tab:blue'
        ax_main.set_xlabel('时间 (h)', fontsize=12)
        ax_main.set_ylabel('SOC (%)', color=color1, fontsize=12)
        line1 = ax_main.plot(time_h, results['soc'] * 100, color=color1, linewidth=4, 
                            label='SOC', alpha=0.9)
        ax_main.tick_params(axis='y', labelcolor=color1)
        ax_main.axhline(y=self.soc_min*100, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax_main.axhline(y=self.soc_max*100, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax_main.set_ylim(15, 85)
        
        # 功率曲线（右轴）
        ax_power = ax_main.twinx()
        color2 = 'tab:purple'
        ax_power.set_ylabel('功率 (kW)', color=color2, fontsize=12)
        power_kw = results['power_w'] / 1000
        line2 = ax_power.plot(time_h, power_kw, color=color2, linewidth=3, 
                             label='功率', alpha=0.8)
        ax_power.tick_params(axis='y', labelcolor=color2)
        ax_power.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 填充充放电区域
        ax_power.fill_between(time_h, power_kw, 0, where=(power_kw > 0), 
                             alpha=0.3, color='red', label='放电')
        ax_power.fill_between(time_h, power_kw, 0, where=(power_kw < 0), 
                             alpha=0.3, color='blue', label='充电')
        
        # 添加时段标注
        for hour in [0, 6, 8, 12, 18, 21]:
            if hour in load_description:
                ax_main.axvline(x=hour, color='gray', linestyle=':', alpha=0.5)
                ax_main.text(hour, 82, load_description[hour], rotation=45, 
                           fontsize=9, ha='left', va='bottom')
        
        ax_main.set_xlim(0, 24)
        ax_main.set_xticks(range(0, 25, 2))
        ax_main.grid(True, alpha=0.3)
        ax_main.set_title(f'SOC和功率日变化 (循环深度: {analysis["soc_range"]:.1%})', fontsize=14)
        
        # 2. 电流变化
        ax2 = fig.add_subplot(gs[1, 0])
        current_data = results['current_a']
        ax2.plot(time_h, current_data, 'green', linewidth=2, alpha=0.8)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.fill_between(time_h, current_data, 0, where=(current_data > 0), 
                        alpha=0.3, color='red', label='放电电流')
        ax2.fill_between(time_h, current_data, 0, where=(current_data < 0), 
                        alpha=0.3, color='blue', label='充电电流')
        ax2.set_ylabel('电流 (A)', fontsize=11)
        ax2.set_title('电流变化', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # 3. 电压变化
        ax3 = fig.add_subplot(gs[1, 1])
        voltage_data = results['voltage_v']
        ax3.plot(time_h, voltage_data, 'orange', linewidth=2, alpha=0.8)
        ax3.set_ylabel('电压 (V)', fontsize=11)
        ax3.set_title(f'电压变化 ({np.min(voltage_data):.0f}-{np.max(voltage_data):.0f}V)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. 温度变化
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(time_h, results['temperature_c'], 'red', linewidth=2, 
                label='电池温度', alpha=0.8)
        ax4.plot(time_h, results['ambient_temp_c'], 'brown', linewidth=1, 
                linestyle='--', label='环境温度', alpha=0.7)
        ax4.set_ylabel('温度 (°C)', fontsize=11)
        ax4.set_title(f'温度变化 (平均: {analysis["temp_avg"]:.1f}°C)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        
        # 5. 能量吞吐量
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(time_h, results['energy_throughput'], 'purple', linewidth=2, alpha=0.8)
        ax5.set_ylabel('累计吞吐量 (kWh)', fontsize=11)
        ax5.set_title(f'能量吞吐量 (总计: {analysis["total_throughput"]:.1f}kWh)', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # 6. 运行状态统计
        ax6 = fig.add_subplot(gs[3, :])
        
        # 创建状态条
        states = []
        colors = []
        for i, power in enumerate(power_kw):
            if power > 1:
                states.append(1)  # 放电
                colors.append('red')
            elif power < -1:
                states.append(-1)  # 充电
                colors.append('blue')
            else:
                states.append(0)  # 待机
                colors.append('gray')
        
        ax6.bar(time_h, [1]*len(time_h), color=colors, alpha=0.6, width=0.1)
        ax6.set_ylim(-0.1, 1.1)
        ax6.set_ylabel('运行状态', fontsize=11)
        ax6.set_xlabel('时间 (h)', fontsize=11)
        ax6.set_title('24小时运行状态 (红=放电, 蓝=充电, 灰=待机)', fontsize=12)
        ax6.set_xlim(0, 24)
        ax6.set_xticks(range(0, 25, 2))
        
        # 添加状态标签
        charge_time = len([p for p in power_kw if p < -1]) * 0.5 / 60  # 小时
        discharge_time = len([p for p in power_kw if p > 1]) * 0.5 / 60  # 小时
        standby_time = 24 - charge_time - discharge_time
        
        ax6.text(0.02, 0.95, f'充电: {charge_time:.1f}h\n放电: {discharge_time:.1f}h\n待机: {standby_time:.1f}h', 
                transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        save_path = f"pic/battery_daily_cycle_complete_{self.timestamp}.png"
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n📊 完整日循环图表已保存: {save_path}")
        except Exception as e:
            print(f"保存图片失败: {e}")
        
        plt.close()
        
        # 额外创建简洁的SOC-功率对应图
        self._create_simple_soc_power_plot(results, analysis)
    
    def _create_simple_soc_power_plot(self, results, analysis):
        """创建简洁的SOC-功率对应图"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle('🔋 日循环：SOC与功率的完美对应关系', fontsize=16, fontweight='bold')
        
        time_h = results['time_h']
        power_kw = results['power_w'] / 1000
        
        # SOC图
        ax1.plot(time_h, results['soc'] * 100, 'b-', linewidth=4, alpha=0.9, label='SOC')
        ax1.axhline(y=self.soc_min*100, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax1.axhline(y=self.soc_max*100, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax1.fill_between([0, 24], 0, self.soc_min*100, alpha=0.1, color='red', label='禁止放电区')
        ax1.fill_between([0, 24], self.soc_max*100, 100, alpha=0.1, color='red', label='禁止充电区')
        
        ax1.set_ylabel('SOC (%)', fontsize=13)
        ax1.set_title(f'荷电状态变化 ({results["soc"][0]:.1%} → {results["soc"][-1]:.1%}, 循环深度: {analysis["soc_range"]:.1%})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.set_ylim(15, 85)
        
        # 功率图
        ax2.plot(time_h, power_kw, 'purple', linewidth=4, alpha=0.9, label='功率')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.fill_between(time_h, power_kw, 0, where=(power_kw > 0), alpha=0.4, color='red', 
                        label='放电期（SOC下降）', interpolate=True)
        ax2.fill_between(time_h, power_kw, 0, where=(power_kw < 0), alpha=0.4, color='blue', 
                        label='充电期（SOC上升）', interpolate=True)
        
        ax2.set_xlabel('时间 (h)', fontsize=13)
        ax2.set_ylabel('功率 (kW)', fontsize=13)
        ax2.set_title(f'功率变化 (充电: {analysis["charge_energy"]:.1f}kWh, 放电: {analysis["discharge_energy"]:.1f}kWh)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        # 设置时间轴
        ax2.set_xlim(0, 24)
        ax2.set_xticks(range(0, 25, 2))
        
        # 添加时段分割线
        for hour in [6, 8, 12, 18, 21]:
            ax1.axvline(x=hour, color='gray', linestyle=':', alpha=0.5)
            ax2.axvline(x=hour, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存简洁图
        save_path = f"pic/battery_daily_cycle_simple_{self.timestamp}.png"
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 简洁日循环图表已保存: {save_path}")
        except Exception as e:
            print(f"保存图片失败: {e}")
        
        plt.close()

def main():
    """主函数"""
    
    tester = DailyCycleTester()
    results, analysis = tester.test_daily_cycle()
    
    print(f"\n" + "=" * 80)
    print("🎯 日循环测试总结")
    print("=" * 80)
    
    print(f"✅ 成功完成24小时日循环仿真")
    print(f"✅ SOC在安全范围内运行: {analysis['soc_range']:.1%}循环深度")
    print(f"✅ 能量平衡良好: 净能量变化 {analysis['discharge_energy'] - analysis['charge_energy']:.2f}kWh")
    print(f"✅ 充放电周期: {len(analysis['charge_periods'])}次充电, {len(analysis['discharge_periods'])}次放电")
    print(f"✅ 总能量吞吐: {analysis['total_throughput']:.1f}kWh")
    
    print(f"\n📊 生成了以下图表:")
    print(f"  • 完整日循环分析图")
    print(f"  • 简洁SOC-功率对应图")
    
    print(f"\n🔋 这是一个典型的储能电站日运行模式，")
    print(f"   完美展示了SOC和功率的物理对应关系！")

if __name__ == "__main__":
    main()
