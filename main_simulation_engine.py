"""
主仿真引擎
实现构网型级联储能PCS的24小时连续运行仿真
集成所有核心模块：PCS建模、寿命预测、健康度评价、优化控制
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# 导入中文字体配置
from matplotlib_config import configure_chinese_fonts

from core_pcs_model import CascadedPCS, SystemParameters
from life_prediction_model import IntegratedLifeModel
from optimization_control import HealthOptimizationController, OptimizationParameters

class MainSimulationEngine:
    """主仿真引擎"""
    
    def __init__(self):
        # 配置中文字体支持
        configure_chinese_fonts()
        
        # 系统参数
        self.system_params = SystemParameters()
        
        # 创建PCS系统
        self.pcs_system = CascadedPCS(self.system_params)
        
        # 创建寿命预测模型
        self.life_model = IntegratedLifeModel()
        
        # 创建优化控制器
        opt_params = OptimizationParameters()
        self.optimizer = HealthOptimizationController(opt_params)
        
        # 仿真参数
        self.simulation_duration = 24 * 3600  # 24小时，秒
        self.time_step = 60.0  # 1分钟步长（原来是1秒）
        self.control_update_interval = 3600  # 1小时更新一次控制（60步）
        
        # 数据记录 - 扩展更多参数
        self.time_history = []
        self.power_history = []
        self.health_history = []
        self.temperature_history = []
        self.life_consumption_history = []
        
        # 新增详细监控参数
        self.voltage_history = []           # 电压历史
        self.current_history = []           # 电流历史
        self.frequency_history = []         # 频率历史
        self.efficiency_history = []        # 效率历史
        self.power_factor_history = []      # 功率因数历史
        self.switching_loss_history = []    # 开关损耗历史
        self.conduction_loss_history = []   # 导通损耗历史
        self.capacitor_voltage_history = [] # 电容电压历史
        self.igbt_junction_temp_history = [] # IGBT结温历史
        self.module_health_history = {}     # 各模块健康度历史
        self.optimization_scores_history = [] # 优化得分历史
        self.control_commands_history = []  # 控制命令历史
        
        # 电网需求曲线（24小时）
        self.grid_demand_profile = self._generate_grid_demand_profile()
        
        # 环境温度变化
        self.ambient_temperature_profile = self._generate_ambient_temperature_profile()
        
        # 初始化寿命模型
        self._initialize_life_models()
        
        # 实时监控设置
        self.real_time_plotting = False     # 是否启用实时绘图（默认关闭以提升速度）
        self.plot_update_interval = 1800    # 若开启，则每30分钟更新一次图表
        self.plot_counter = 0
    
    def _generate_grid_demand_profile(self) -> List[float]:
        """生成24小时电网需求曲线"""
        demand_profile = []
        
        for hour in range(24):
            if 0 <= hour < 6:  # 夜间低谷
                demand = 5e6  # 5 MW
            elif 6 <= hour < 10:  # 早高峰
                demand = 20e6  # 20 MW
            elif 10 <= hour < 14:  # 午间
                demand = 15e6  # 15 MW
            elif 14 <= hour < 18:  # 晚高峰
                demand = 22e6  # 22 MW
            elif 18 <= hour < 22:  # 晚间
                demand = 18e6  # 18 MW
            else:  # 深夜
                demand = 8e6   # 8 MW
            
            demand_profile.append(demand)
        
        return demand_profile
    
    def _generate_ambient_temperature_profile(self) -> List[float]:
        """生成24小时环境温度变化曲线"""
        temp_profile = []
        
        for hour in range(24):
            if 0 <= hour < 6:  # 夜间
                temp = 20.0  # 20℃
            elif 6 <= hour < 12:  # 上午升温
                temp = 20.0 + (hour - 6) * 2.5  # 20-35℃
            elif 12 <= hour < 18:  # 下午高温
                temp = 35.0 - (hour - 12) * 1.67  # 35-25℃
            else:  # 晚间降温
                temp = 25.0 - (hour - 18) * 1.25  # 25-20℃
            
            temp_profile.append(temp)
        
        return temp_profile
    
    def _initialize_life_models(self):
        """初始化所有模块的寿命模型"""
        total_modules = (self.system_params.h_bridge_per_phase * 
                        self.system_params.total_phases)
        
        for i in range(total_modules):
            module_id = f"module_{i:03d}"
            self.life_model.add_module(module_id)
            # 初始化模块健康度历史
            self.module_health_history[module_id] = []
    
    def _record_detailed_simulation_data(self, current_time: float, power: float, 
                                        health_status: Dict, temp_distribution: Dict,
                                        control_result: Dict = None):
        """记录详细的仿真数据"""
        # 基础数据记录
        self.time_history.append(current_time)
        self.power_history.append(power)
        self.health_history.append(health_status['overall_health'])
        self.temperature_history.append(temp_distribution.get('max_igbt_temp', 25.0))
        
        # 获取系统寿命状态
        system_life_status = self.life_model.get_system_health_status()
        self.life_consumption_history.append(system_life_status.get('system_health', 100.0))
        
        # 记录详细电气参数
        self.voltage_history.append(temp_distribution.get('dc_voltage', self.system_params.rated_voltage))
        self.current_history.append(abs(power) / self.system_params.rated_voltage if self.system_params.rated_voltage > 0 else 0)
        self.frequency_history.append(temp_distribution.get('switching_frequency', 2000))
        
        # 计算和记录效率相关参数
        efficiency = self._estimate_system_efficiency()
        self.efficiency_history.append(efficiency)
        
        # 功率因数（简化计算）
        power_factor = 0.95 + 0.05 * (health_status['overall_health'] / 100.0)
        self.power_factor_history.append(power_factor)
        
        # 损耗估算
        switching_loss = self._estimate_switching_loss(temp_distribution)
        conduction_loss = self._estimate_conduction_loss(temp_distribution)
        self.switching_loss_history.append(switching_loss)
        self.conduction_loss_history.append(conduction_loss)
        
        # 电容电压和IGBT结温
        self.capacitor_voltage_history.append(temp_distribution.get('cap_voltage', self.system_params.rated_voltage * 0.5))
        self.igbt_junction_temp_history.append(temp_distribution.get('max_igbt_temp', 25.0))
        
        # 记录各模块健康度
        total_modules = (self.system_params.h_bridge_per_phase * self.system_params.total_phases)
        for i in range(total_modules):
            module_id = f"module_{i:03d}"
            module_health = self.life_model.get_module_health(module_id)
            self.module_health_history[module_id].append(module_health)
        
        # 记录优化控制结果
        if control_result:
            self.optimization_scores_history.append(control_result.get('health_optimization_score', 0.0))
            self.control_commands_history.append(control_result)
        
        # 实时绘图更新
        if self.real_time_plotting and self.plot_counter >= self.plot_update_interval:
            self._update_real_time_plots()
            self.plot_counter = 0
        
        self.plot_counter += self.time_step
    
    def _estimate_switching_loss(self, temp_distribution: Dict) -> float:
        """估算开关损耗"""
        temp = temp_distribution.get('max_igbt_temp', 25.0)
        freq = temp_distribution.get('switching_frequency', 2000)
        
        # 基于温度和频率的开关损耗模型
        base_loss = 0.02  # 基础损耗百分比
        temp_factor = 1.0 + 0.02 * (temp - 25.0) / 25.0
        freq_factor = 1.0 + 0.001 * (freq - 2000) / 2000
        
        return base_loss * temp_factor * freq_factor
    
    def _estimate_conduction_loss(self, temp_distribution: Dict) -> float:
        """估算导通损耗"""
        temp = temp_distribution.get('max_igbt_temp', 25.0)
        current = temp_distribution.get('current', self.system_params.rated_current)
        
        # 基于温度和电流的导通损耗模型
        base_loss = 0.015  # 基础损耗百分比
        temp_factor = 1.0 + 0.015 * (temp - 25.0) / 25.0
        current_factor = (current / self.system_params.rated_current) ** 2
        
        return base_loss * temp_factor * current_factor
    
    def _update_real_time_plots(self):
        """更新实时监控图表"""
        try:
            # 创建实时监控图表
            self._create_real_time_monitoring_dashboard()
        except Exception as e:
            print(f"实时绘图更新失败: {e}")
    
    def _create_real_time_monitoring_dashboard(self):
        """创建实时监控仪表板"""
        if len(self.time_history) < 2:
            return
        
        # 清除之前的图表
        plt.close('all')
        
        # 创建子图布局
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('构网型级联储能PCS实时监控仪表板', fontsize=18, fontweight='bold')
        
        # 时间轴（小时）
        time_hours = [t/3600 for t in self.time_history]
        
        # 1. 功率和效率监控
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(time_hours, [p/1e6 for p in self.power_history], 'b-', linewidth=2, label='功率')
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('功率 (MW)')
        ax1.set_title('系统功率曲线')
        ax1.grid(True, alpha=0.3)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_hours, self.efficiency_history, 'r--', linewidth=2, label='效率')
        ax1_twin.set_ylabel('效率', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # 2. 健康度监控
        ax2 = plt.subplot(3, 4, 2)
        ax2.plot(time_hours, self.health_history, 'g-', linewidth=2, label='整体健康度')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('健康度 (%)')
        ax2.set_title('系统健康状态')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # 3. 温度监控
        ax3 = plt.subplot(3, 4, 3)
        ax3.plot(time_hours, self.temperature_history, 'r-', linewidth=2, label='IGBT温度')
        ax3.plot(time_hours, self.igbt_junction_temp_history, 'orange', linewidth=2, label='结温')
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('温度 (℃)')
        ax3.set_title('温度监控')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 电压电流监控
        ax4 = plt.subplot(3, 4, 4)
        ax4_twin = ax4.twinx()
        ax4.plot(time_hours, [v/1e3 for v in self.voltage_history], 'b-', linewidth=2, label='电压')
        ax4_twin.plot(time_hours, [i/1e3 for i in self.current_history], 'r-', linewidth=2, label='电流')
        ax4.set_xlabel('时间 (小时)')
        ax4.set_ylabel('电压 (kV)', color='b')
        ax4_twin.set_ylabel('电流 (kA)', color='r')
        ax4.set_title('电压电流监控')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='y', labelcolor='b')
        ax4_twin.tick_params(axis='y', labelcolor='r')
        
        # 5. 损耗分析
        ax5 = plt.subplot(3, 4, 5)
        ax5.plot(time_hours, [l*100 for l in self.switching_loss_history], 'purple', linewidth=2, label='开关损耗')
        ax5.plot(time_hours, [l*100 for l in self.conduction_loss_history], 'brown', linewidth=2, label='导通损耗')
        ax5.set_xlabel('时间 (小时)')
        ax5.set_ylabel('损耗 (%)')
        ax5.set_title('损耗分析')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. 功率因数和频率
        ax6 = plt.subplot(3, 4, 6)
        ax6_twin = ax6.twinx()
        ax6.plot(time_hours, self.power_factor_history, 'g-', linewidth=2, label='功率因数')
        ax6_twin.plot(time_hours, [f/1e3 for f in self.frequency_history], 'orange', linewidth=2, label='频率')
        ax6.set_xlabel('时间 (小时)')
        ax6.set_ylabel('功率因数', color='g')
        ax6_twin.set_ylabel('频率 (kHz)', color='orange')
        ax6.set_title('功率因数和频率')
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='y', labelcolor='g')
        ax6_twin.tick_params(axis='y', labelcolor='orange')
        
        # 7. 电容电压监控
        ax7 = plt.subplot(3, 4, 7)
        ax7.plot(time_hours, [v/1e3 for v in self.capacitor_voltage_history], 'cyan', linewidth=2)
        ax7.set_xlabel('时间 (小时)')
        ax7.set_ylabel('电容电压 (kV)')
        ax7.set_title('电容电压监控')
        ax7.grid(True, alpha=0.3)
        
        # 8. 寿命消耗
        ax8 = plt.subplot(3, 4, 8)
        ax8.plot(time_hours, self.life_consumption_history, 'm-', linewidth=2)
        ax8.set_xlabel('时间 (小时)')
        ax8.set_ylabel('健康度 (%)')
        ax8.set_title('寿命模型健康度')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, 100)
        
        # 9. 优化得分
        ax9 = plt.subplot(3, 4, 9)
        if self.optimization_scores_history:
            ax9.plot(time_hours[-len(self.optimization_scores_history):], self.optimization_scores_history, 'gold', linewidth=2)
        ax9.set_xlabel('时间 (小时)')
        ax9.set_ylabel('优化得分')
        ax9.set_title('健康优化得分')
        ax9.grid(True, alpha=0.3)
        
        # 10. 模块健康度热力图
        ax10 = plt.subplot(3, 4, 10)
        if self.module_health_history:
            module_data = []
            module_names = []
            for module_id, health_data in self.module_health_history.items():
                if health_data:
                    module_data.append(health_data)
                    module_names.append(module_id.split('_')[-1])
            
            if module_data:
                # 取最近的健康度数据
                recent_health = [data[-1] if data else 100.0 for data in module_data]
                colors = ['red' if h < 80 else 'orange' if h < 90 else 'green' for h in recent_health]
                bars = ax10.bar(range(len(recent_health)), recent_health, color=colors, alpha=0.7)
                ax10.set_xlabel('模块编号')
                ax10.set_ylabel('健康度 (%)')
                ax10.set_title('模块健康度状态')
                ax10.set_xticks(range(len(module_names)))
                ax10.set_xticklabels(module_names)
                ax10.set_ylim(0, 100)
                ax10.grid(True, alpha=0.3)
        
        # 11. 系统状态统计
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        stats_text = f"""
系统状态统计:
• 当前时间: {time_hours[-1]:.1f} 小时
• 当前功率: {self.power_history[-1]/1e6:.1f} MW
• 当前健康度: {self.health_history[-1]:.1f}%
• 当前效率: {self.efficiency_history[-1]*100:.1f}%
• 当前温度: {self.temperature_history[-1]:.1f}℃
• 开关频率: {self.frequency_history[-1]:.0f} Hz
• 功率因数: {self.power_factor_history[-1]:.3f}
• 总损耗: {(self.switching_loss_history[-1] + self.conduction_loss_history[-1])*100:.2f}%
        """
        ax11.text(0.1, 0.9, stats_text, transform=ax11.transAxes, fontsize=10, 
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 12. 性能趋势
        ax12 = plt.subplot(3, 4, 12)
        if len(self.efficiency_history) > 10:
            # 计算移动平均
            window = min(10, len(self.efficiency_history))
            moving_avg = [np.mean(self.efficiency_history[max(0, i-window):i+1]) 
                         for i in range(len(self.efficiency_history))]
            ax12.plot(time_hours, [e*100 for e in self.efficiency_history], 'b-', alpha=0.5, label='实时效率')
            ax12.plot(time_hours, [e*100 for e in moving_avg], 'r-', linewidth=2, label=f'{window}点移动平均')
            ax12.set_xlabel('时间 (小时)')
            ax12.set_ylabel('效率 (%)')
            ax12.set_title('效率趋势分析')
            ax12.grid(True, alpha=0.3)
            ax12.legend()
        
        plt.tight_layout()
        
        # 保存实时监控图
        plt.savefig(f'real_time_monitoring_{int(time.time())}.png', dpi=150, bbox_inches='tight')
        
        # 显示图表（非阻塞）
        plt.show(block=False)
        plt.pause(0.1)
    
    def run_simulation(self) -> Dict:
        """运行完整仿真"""
        print("开始构网型级联储能PCS仿真...")
        print(f"仿真时长: {self.simulation_duration/3600:.1f} 小时")
        print(f"时间步长: {self.time_step} 秒")
        print(f"控制更新间隔: {self.control_update_interval/3600:.1f} 小时")
        
        start_time = time.time()
        
        # 仿真主循环
        current_time = 0.0
        control_update_counter = 0
        
        # 计算总步数用于进度条
        total_steps = int(self.simulation_duration / self.time_step)
        
        # 创建进度条
        with tqdm(total=total_steps, desc="仿真进度", unit="步") as pbar:
            
            while current_time <= self.simulation_duration:
                # 获取当前小时
                current_hour = int(current_time / 3600)
                hour_index = min(current_hour, 23)
                
                # 获取当前环境温度
                ambient_temp = self.ambient_temperature_profile[hour_index]
                
                # 获取当前电网需求
                grid_demand = self.grid_demand_profile[hour_index]
                
                # 更新PCS系统
                self.pcs_system.step_simulation(current_time, ambient_temp)
                
                # 设置功率参考
                self.pcs_system.set_power_reference(grid_demand)
                
                # 获取系统状态
                health_status = self.pcs_system.get_health_status()
                temp_distribution = self.pcs_system.get_temperature_distribution()
                
                # 更新寿命模型
                self._update_life_models(current_time, temp_distribution)
                
                # 记录数据
                self._record_detailed_simulation_data(current_time, grid_demand, health_status, temp_distribution)
                
                # 定期更新优化控制
                if control_update_counter >= self.control_update_interval:
                    optimization_result = self._update_optimization_control(current_time, grid_demand, health_status)
                    # 更新数据记录中的优化结果
                    self._record_detailed_simulation_data(current_time, grid_demand, health_status, temp_distribution, optimization_result)
                    control_update_counter = 0
                
                # 时间推进
                current_time += self.time_step
                control_update_counter += self.time_step
                
                # 更新进度条
                pbar.update(1)
                
                # 进度显示（每小时更新一次）
                if int(current_time) % 3600 == 0:
                    pbar.set_postfix({
                        '时间': f"{current_hour:02d}:00",
                        '健康度': f"{health_status['overall_health']:.1f}%",
                        '功率': f"{grid_demand/1e6:.1f}MW"
                    })
        
        simulation_time = time.time() - start_time
        print(f"仿真完成！耗时: {simulation_time:.2f} 秒")
        
        # 生成仿真报告
        simulation_report = self._generate_simulation_report()
        
        return simulation_report
    
    def _update_life_models(self, current_time: float, temp_distribution: Dict):
        """更新寿命模型"""
        total_modules = (self.system_params.h_bridge_per_phase * 
                        self.system_params.total_phases)
        
        for i in range(total_modules):
            module_id = f"module_{i:03d}"
            
            # 获取模块温度（简化：使用平均值）
            avg_igbt_temp = temp_distribution.get('max_igbt_temp', 25.0)
            avg_cap_temp = temp_distribution.get('max_cap_temp', 25.0)
            
            # 估算电流和电压
            estimated_current = self.system_params.rated_current / total_modules
            estimated_voltage = self.system_params.rated_voltage / total_modules
            estimated_ripple = estimated_current * 0.1  # 10%纹波
            
            # 更新寿命模型
            self.life_model.update_module_conditions(
                module_id, avg_igbt_temp, estimated_current,
                avg_cap_temp, estimated_voltage, estimated_ripple, current_time
            )
            
            # 计算寿命消耗
            self.life_model.calculate_module_life_consumption(module_id, self.time_step)
    
    def _update_optimization_control(self, current_time: float, 
                                   grid_demand: float, health_status: Dict):
        """更新优化控制"""
        # 获取当前条件
        current_conditions = {
            'power': grid_demand,
            'battery_soc': 0.5,  # 简化：固定SOC
            'temperature_distribution': self.pcs_system.get_temperature_distribution()
        }
        
        # 生成24小时需求预测
        demand_forecast = self.grid_demand_profile * 1  # 简化：重复当前日需求
        
        # 执行优化
        optimization_result = self.optimizer.optimize_system_operation(
            current_conditions, demand_forecast, health_status
        )
        
        # 应用优化结果（简化：仅记录）
        if current_time % 3600 == 0:  # 每小时记录一次
            print(f"优化控制更新 - 时间: {current_time/3600:.1f}h")
            print(f"  优化得分: {optimization_result['health_optimization_score']:.2f}")
            print(f"  开关频率: {optimization_result['optimized_switching_frequency']:.0f} Hz")
        
        return optimization_result
    
    def _generate_simulation_report(self) -> Dict:
        """生成仿真报告"""
        # 计算统计指标
        avg_health = np.mean(self.health_history)
        min_health = np.min(self.health_history)
        max_health = np.max(self.health_history)
        
        avg_temp = np.mean(self.temperature_history)
        max_temp = np.max(self.temperature_history)
        min_temp = np.min(self.temperature_history)
        
        # 新增详细参数统计
        avg_efficiency = np.mean(self.efficiency_history) if self.efficiency_history else 0.95
        avg_power_factor = np.mean(self.power_factor_history) if self.power_factor_history else 0.95
        avg_switching_loss = np.mean(self.switching_loss_history) if self.switching_loss_history else 0.02
        avg_conduction_loss = np.mean(self.conduction_loss_history) if self.conduction_loss_history else 0.015
        
        # 电压电流统计
        avg_voltage = np.mean(self.voltage_history) if self.voltage_history else self.system_params.rated_voltage
        avg_current = np.mean(self.current_history) if self.current_history else 0.0
        voltage_variation = np.std(self.voltage_history) if self.voltage_history else 0.0
        current_variation = np.std(self.current_history) if self.current_history else 0.0
        
        # 频率统计
        avg_frequency = np.mean(self.frequency_history) if self.frequency_history else 2000
        frequency_variation = np.std(self.frequency_history) if self.frequency_history else 0.0
        
        # 模块健康度统计
        module_health_stats = {}
        if self.module_health_history:
            for module_id, health_data in self.module_health_history.items():
                if health_data:
                    module_health_stats[module_id] = {
                        'average_health': np.mean(health_data),
                        'min_health': np.min(health_data),
                        'max_health': np.max(health_data),
                        'health_degradation': np.max(health_data) - np.min(health_data)
                    }
        
        # 寿命预测
        operating_profile = {
            'average_temperature': avg_temp,
            'average_power': np.mean(np.abs(self.power_history)) / self.system_params.rated_power,
            'duty_cycle': 0.6  # 估算
        }
        
        predicted_lifetime = self.life_model.predict_system_lifetime(operating_profile)
        
        # 生成报告
        report = {
            'simulation_summary': {
                'duration_hours': self.simulation_duration / 3600,
                'time_steps': len(self.time_history),
                'total_modules': (self.system_params.h_bridge_per_phase * 
                                self.system_params.total_phases)
            },
            'health_metrics': {
                'average_health': avg_health,
                'min_health': min_health,
                'max_health': max_health,
                'health_degradation': max_health - min_health
            },
            'temperature_metrics': {
                'average_temperature': avg_temp,
                'max_temperature': max_temp,
                'min_temperature': min_temp,
                'temperature_variation': max_temp - min_temp
            },
            'efficiency_metrics': {
                'average_efficiency': avg_efficiency,
                'efficiency_variation': np.std(self.efficiency_history) if self.efficiency_history else 0.0,
                'power_factor': avg_power_factor,
                'total_losses': avg_switching_loss + avg_conduction_loss
            },
            'electrical_metrics': {
                'average_voltage': avg_voltage,
                'voltage_variation': voltage_variation,
                'average_current': avg_current,
                'current_variation': current_variation,
                'average_frequency': avg_frequency,
                'frequency_variation': frequency_variation
            },
            'loss_analysis': {
                'switching_loss': avg_switching_loss,
                'conduction_loss': avg_conduction_loss,
                'total_loss_percentage': (avg_switching_loss + avg_conduction_loss) * 100
            },
            'module_health_analysis': module_health_stats,
            'life_prediction': {
                'predicted_lifetime_hours': predicted_lifetime,
                'predicted_lifetime_years': predicted_lifetime / 8760 if predicted_lifetime != float('inf') else float('inf')
            },
            'system_performance': {
                'power_tracking_error': self._calculate_power_tracking_error(),
                'efficiency_estimate': self._estimate_system_efficiency(),
                'stability_score': self._calculate_stability_score()
            }
        }
        
        return report
    
    def _calculate_power_tracking_error(self) -> float:
        """计算功率跟踪误差"""
        if len(self.power_history) < 2:
            return 0.0
        
        # 计算功率变化率
        power_changes = [abs(self.power_history[i] - self.power_history[i-1]) 
                        for i in range(1, len(self.power_history))]
        
        # 归一化到额定功率
        normalized_changes = [change / self.system_params.rated_power for change in power_changes]
        
        return np.mean(normalized_changes)
    
    def _estimate_system_efficiency(self) -> float:
        """估算系统效率"""
        # 基于健康度和温度估算效率
        avg_health = np.mean(self.health_history) / 100.0
        avg_temp = np.mean(self.temperature_history)
        
        # 基础效率
        base_efficiency = 0.95
        
        # 健康度影响
        health_factor = 0.8 + 0.2 * avg_health
        
        # 温度影响
        temp_factor = 1.0
        if avg_temp > 60:
            temp_factor = 0.95
        elif avg_temp > 40:
            temp_factor = 0.98
        
        estimated_efficiency = base_efficiency * health_factor * temp_factor
        
        return min(0.99, max(0.85, estimated_efficiency))
    
    def _calculate_stability_score(self) -> float:
        """计算系统稳定性评分"""
        if len(self.health_history) < 2:
            return 100.0
        
        # 健康度变化率
        health_variation = np.std(self.health_history)
        
        # 温度变化率
        temp_variation = np.std(self.temperature_history) if self.temperature_history else 0.0
        
        # 功率变化率
        power_variation = np.std([abs(self.power_history[i] - self.power_history[i-1]) 
                                 for i in range(1, len(self.power_history))]) if len(self.power_history) > 1 else 0.0
        
        # 归一化到0-100分
        health_score = max(0, 100 - health_variation * 2)
        temp_score = max(0, 100 - temp_variation * 2)
        power_score = max(0, 100 - (power_variation / self.system_params.rated_power) * 1000)
        
        # 综合稳定性评分
        stability_score = (health_score + temp_score + power_score) / 3
        
        return min(100.0, max(0.0, stability_score))
    
    def _create_comprehensive_analysis_plots(self):
        """创建综合分析图表"""
        if len(self.time_history) < 2:
            return
        
        # 创建综合分析图表
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('构网型级联储能PCS综合分析报告', fontsize=16, fontweight='bold')
        
        time_hours = [t/3600 for t in self.time_history]
        
        # 1. 功率和效率对比
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        ax1.plot(time_hours, [p/1e6 for p in self.power_history], 'b-', linewidth=2, label='功率')
        ax1_twin.plot(time_hours, [e*100 for e in self.efficiency_history], 'r--', linewidth=2, label='效率')
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('功率 (MW)', color='b')
        ax1_twin.set_ylabel('效率 (%)', color='r')
        ax1.set_title('功率与效率关系')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # 2. 健康度与温度关系
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        ax2.plot(time_hours, self.health_history, 'g-', linewidth=2, label='健康度')
        ax2_twin.plot(time_hours, self.temperature_history, 'orange', linewidth=2, label='温度')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('健康度 (%)', color='g')
        ax2_twin.set_ylabel('温度 (℃)', color='orange')
        ax2.set_title('健康度与温度关系')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='g')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        
        # 3. 损耗分析
        ax3 = axes[0, 2]
        ax3.plot(time_hours, [l*100 for l in self.switching_loss_history], 'purple', linewidth=2, label='开关损耗')
        ax3.plot(time_hours, [l*100 for l in self.conduction_loss_history], 'brown', linewidth=2, label='导通损耗')
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('损耗 (%)')
        ax3.set_title('损耗分析')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 电压电流关系
        ax4 = axes[1, 0]
        ax4_twin = ax4.twinx()
        ax4.plot(time_hours, [v/1e3 for v in self.voltage_history], 'b-', linewidth=2, label='电压')
        ax4_twin.plot(time_hours, [i/1e3 for i in self.current_history], 'r-', linewidth=2, label='电流')
        ax4.set_xlabel('时间 (小时)')
        ax4.set_ylabel('电压 (kV)', color='b')
        ax4_twin.set_ylabel('电流 (kA)', color='r')
        ax4.set_title('电压电流关系')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='y', labelcolor='b')
        ax4_twin.tick_params(axis='y', labelcolor='r')
        
        # 5. 功率因数变化
        ax5 = axes[1, 1]
        ax5.plot(time_hours, self.power_factor_history, 'g-', linewidth=2)
        ax5.set_xlabel('时间 (小时)')
        ax5.set_ylabel('功率因数')
        ax5.set_title('功率因数变化')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0.9, 1.0)
        
        # 6. 频率变化
        ax6 = axes[1, 2]
        ax6.plot(time_hours, [f/1e3 for f in self.frequency_history], 'orange', linewidth=2)
        ax6.set_xlabel('时间 (小时)')
        ax6.set_ylabel('频率 (kHz)')
        ax6.set_title('开关频率变化')
        ax6.grid(True, alpha=0.3)
        
        # 7. 模块健康度分布
        ax7 = axes[2, 0]
        if self.module_health_history:
            module_names = []
            final_health = []
            for module_id, health_data in self.module_health_history.items():
                if health_data:
                    module_names.append(module_id.split('_')[-1])
                    final_health.append(health_data[-1])
            
            if final_health:
                colors = ['red' if h < 80 else 'orange' if h < 90 else 'green' for h in final_health]
                bars = ax7.bar(range(len(final_health)), final_health, color=colors, alpha=0.7)
                ax7.set_xlabel('模块编号')
                ax7.set_ylabel('最终健康度 (%)')
                ax7.set_title('模块健康度分布')
                ax7.set_xticks(range(len(module_names)))
                ax7.set_xticklabels(module_names)
                ax7.set_ylim(0, 100)
                ax7.grid(True, alpha=0.3)
        
        # 8. 寿命消耗趋势
        ax8 = axes[2, 1]
        ax8.plot(time_hours, self.life_consumption_history, 'm-', linewidth=2)
        ax8.set_xlabel('时间 (小时)')
        ax8.set_ylabel('健康度 (%)')
        ax8.set_title('寿命消耗趋势')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, 100)
        
        # 9. 系统性能雷达图
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # 计算性能指标
        health_score = np.mean(self.health_history) / 100.0
        efficiency_score = np.mean(self.efficiency_history)
        temp_score = 1.0 - min(1.0, np.mean(self.temperature_history) / 100.0)
        stability_score = self._calculate_stability_score() / 100.0
        
        # 创建雷达图数据
        categories = ['健康度', '效率', '温度控制', '稳定性']
        values = [health_score, efficiency_score, temp_score, stability_score]
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax9 = plt.subplot(3, 3, 9, projection='polar')
        ax9.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax9.fill(angles, values, alpha=0.25, color='blue')
        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(categories)
        ax9.set_ylim(0, 1)
        ax9.set_title('系统性能雷达图')
        
        plt.tight_layout()
        return fig
    
    def plot_simulation_results(self, save_path: str = None):
        """绘制仿真结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('构网型级联储能PCS仿真结果', fontsize=16)
        
        # 时间轴（小时）
        time_hours = [t/3600 for t in self.time_history]
        
        # 1. 功率曲线
        axes[0, 0].plot(time_hours, [p/1e6 for p in self.power_history], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('时间 (小时)')
        axes[0, 0].set_ylabel('功率 (MW)')
        axes[0, 0].set_title('系统功率曲线')
        axes[0, 0].grid(True)
        
        # 2. 健康度变化
        axes[0, 1].plot(time_hours, self.health_history, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('时间 (小时)')
        axes[0, 1].set_ylabel('健康度 (%)')
        axes[0, 1].set_title('系统健康度变化')
        axes[0, 1].grid(True)
        axes[0, 1].set_ylim(0, 100)
        
        # 3. 温度变化
        axes[1, 0].plot(time_hours, self.temperature_history, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('时间 (小时)')
        axes[1, 0].set_ylabel('温度 (℃)')
        axes[1, 0].set_title('IGBT温度变化')
        axes[1, 0].grid(True)
        
        # 4. 寿命消耗
        axes[1, 1].plot(time_hours, self.life_consumption_history, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('时间 (小时)')
        axes[1, 1].set_ylabel('健康度 (%)')
        axes[1, 1].set_title('寿命模型健康度')
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"仿真结果图已保存到: {save_path}")
        
        plt.show()

def main():
    """主函数"""
    print("=" * 60)
    print("构网型级联储能PCS关键器件寿命预测及健康度分析仿真")
    print("=" * 60)
    
    # 创建仿真引擎
    engine = MainSimulationEngine()
    
    # 运行仿真
    report = engine.run_simulation()
    
    # 打印仿真报告
    print("\n" + "=" * 60)
    print("仿真报告")
    print("=" * 60)
    
    print(f"仿真时长: {report['simulation_summary']['duration_hours']:.1f} 小时")
    print(f"总模块数: {report['simulation_summary']['total_modules']}")
    print(f"时间步数: {report['simulation_summary']['time_steps']}")
    
    print(f"\n健康度指标:")
    print(f"  平均健康度: {report['health_metrics']['average_health']:.2f}%")
    print(f"  最高健康度: {report['health_metrics']['max_health']:.2f}%")
    print(f"  最低健康度: {report['health_metrics']['min_health']:.2f}%")
    print(f"  健康度下降: {report['health_metrics']['health_degradation']:.2f}%")
    
    print(f"\n温度指标:")
    print(f"  平均温度: {report['temperature_metrics']['average_temperature']:.1f}℃")
    print(f"  最高温度: {report['temperature_metrics']['max_temperature']:.1f}℃")
    print(f"  最低温度: {report['temperature_metrics']['min_temperature']:.1f}℃")
    print(f"  温度变化: {report['temperature_metrics']['temperature_variation']:.1f}℃")
    
    print(f"\n效率指标:")
    print(f"  平均效率: {report['efficiency_metrics']['average_efficiency']*100:.2f}%")
    print(f"  效率变化: {report['efficiency_metrics']['efficiency_variation']*100:.2f}%")
    print(f"  功率因数: {report['efficiency_metrics']['power_factor']:.3f}")
    print(f"  总损耗: {report['efficiency_metrics']['total_losses']*100:.2f}%")
    
    print(f"\n电气参数:")
    print(f"  平均电压: {report['electrical_metrics']['average_voltage']/1e3:.1f} kV")
    print(f"  电压变化: {report['electrical_metrics']['voltage_variation']/1e3:.2f} kV")
    print(f"  平均电流: {report['electrical_metrics']['average_current']/1e3:.1f} kA")
    print(f"  电流变化: {report['electrical_metrics']['current_variation']/1e3:.2f} kA")
    print(f"  平均频率: {report['electrical_metrics']['average_frequency']:.0f} Hz")
    print(f"  频率变化: {report['electrical_metrics']['frequency_variation']:.0f} Hz")
    
    print(f"\n损耗分析:")
    print(f"  开关损耗: {report['loss_analysis']['switching_loss']*100:.2f}%")
    print(f"  导通损耗: {report['loss_analysis']['conduction_loss']*100:.2f}%")
    print(f"  总损耗: {report['loss_analysis']['total_loss_percentage']:.2f}%")
    
    print(f"\n寿命预测:")
    if report['life_prediction']['predicted_lifetime_years'] != float('inf'):
        print(f"  预测寿命: {report['life_prediction']['predicted_lifetime_years']:.1f} 年")
    else:
        print(f"  预测寿命: 无限")
    
    print(f"\n系统性能:")
    print(f"  功率跟踪误差: {report['system_performance']['power_tracking_error']:.4f}")
    print(f"  估算效率: {report['system_performance']['efficiency_estimate']:.3f}")
    print(f"  稳定性评分: {report['system_performance']['stability_score']:.1f}/100")
    
    # 显示模块健康度分析
    if report['module_health_analysis']:
        print(f"\n模块健康度分析:")
        for module_id, health_data in report['module_health_analysis'].items():
            print(f"  {module_id}: 平均{health_data['average_health']:.1f}%, "
                  f"最高{health_data['max_health']:.1f}%, "
                  f"最低{health_data['min_health']:.1f}%")
    
    # 绘制结果
    engine.plot_simulation_results("simulation_results.png")
    
    # 创建综合分析图表
    print("\n正在生成综合分析图表...")
    comprehensive_fig = engine._create_comprehensive_analysis_plots()
    if comprehensive_fig:
        comprehensive_fig.savefig("comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        print("综合分析图表已保存为: comprehensive_analysis.png")
        plt.show()
    
    print("\n仿真完成！")

if __name__ == "__main__":
    main()
