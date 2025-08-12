"""
仿真配置文件
控制构网型级联储能PCS仿真的各种参数和设置
"""

class SimulationConfig:
    """仿真配置类"""
    
    def __init__(self):
        # 基础仿真参数
        self.simulation_duration_hours = 24.0  # 仿真时长（小时）
        self.time_step_seconds = 60.0          # 时间步长（秒）
        self.control_update_interval_hours = 1.0  # 控制更新间隔（小时）
        
        # 实时监控设置
        self.enable_real_time_monitoring = True    # 是否启用实时监控
        self.plot_update_interval_seconds = 300    # 图表更新间隔（秒）
        self.save_real_time_plots = True          # 是否保存实时监控图
        self.real_time_plot_dpi = 150             # 实时图表DPI
        
        # 数据记录设置
        self.record_detailed_metrics = True       # 是否记录详细指标
        self.record_module_health = True          # 是否记录模块健康度
        self.record_optimization_results = True   # 是否记录优化结果
        self.record_electrical_parameters = True  # 是否记录电气参数
        
        # 图表设置
        self.enable_comprehensive_analysis = True  # 是否启用综合分析
        self.comprehensive_plot_dpi = 300         # 综合分析图DPI
        self.show_plots_interactively = True      # 是否交互式显示图表
        
        # 性能优化设置
        self.enable_progress_bar = True           # 是否显示进度条
        self.progress_update_interval = 60        # 进度更新间隔（秒）
        self.enable_memory_optimization = False   # 是否启用内存优化
        
        # 输出设置
        self.save_simulation_results = True       # 是否保存仿真结果
        self.output_directory = "simulation_output"  # 输出目录
        self.save_format = "png"                 # 保存格式
        
        # 调试设置
        self.enable_debug_output = False         # 是否启用调试输出
        self.log_level = "INFO"                  # 日志级别
        self.save_debug_data = False             # 是否保存调试数据
    
    def get_simulation_duration_seconds(self) -> float:
        """获取仿真时长（秒）"""
        return self.simulation_duration_hours * 3600
    
    def get_control_update_interval_seconds(self) -> float:
        """获取控制更新间隔（秒）"""
        return self.control_update_interval_hours * 3600
    
    def get_total_steps(self) -> int:
        """获取总仿真步数"""
        return int(self.get_simulation_duration_seconds() / self.time_step_seconds)
    
    def validate_config(self) -> bool:
        """验证配置参数的有效性"""
        errors = []
        
        if self.simulation_duration_hours <= 0:
            errors.append("仿真时长必须大于0")
        
        if self.time_step_seconds <= 0:
            errors.append("时间步长必须大于0")
        
        if self.control_update_interval_hours <= 0:
            errors.append("控制更新间隔必须大于0")
        
        if self.plot_update_interval_seconds <= 0:
            errors.append("图表更新间隔必须大于0")
        
        if self.simulation_duration_hours < self.control_update_interval_hours:
            errors.append("仿真时长必须大于控制更新间隔")
        
        if self.time_step_seconds > self.get_control_update_interval_seconds():
            errors.append("时间步长不能大于控制更新间隔")
        
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("=" * 60)
        print("仿真配置摘要")
        print("=" * 60)
        print(f"仿真时长: {self.simulation_duration_hours:.1f} 小时 ({self.get_simulation_duration_seconds():.0f} 秒)")
        print(f"时间步长: {self.time_step_seconds:.1f} 秒")
        print(f"总仿真步数: {self.get_total_steps():,}")
        print(f"控制更新间隔: {self.control_update_interval_hours:.1f} 小时")
        print(f"实时监控: {'启用' if self.enable_real_time_monitoring else '禁用'}")
        print(f"图表更新间隔: {self.plot_update_interval_seconds:.0f} 秒")
        print(f"详细指标记录: {'启用' if self.record_detailed_metrics else '禁用'}")
        print(f"综合分析: {'启用' if self.enable_comprehensive_analysis else '禁用'}")
        print(f"输出目录: {self.output_directory}")
        print(f"保存格式: {self.save_format}")
        print("=" * 60)

# 预定义配置方案
class PresetConfigs:
    """预定义配置方案"""
    
    @staticmethod
    def quick_test():
        """快速测试配置"""
        config = SimulationConfig()
        config.simulation_duration_hours = 1.0
        config.time_step_seconds = 10.0
        config.control_update_interval_hours = 0.1
        config.plot_update_interval_seconds = 60
        config.enable_real_time_monitoring = False
        return config
    
    @staticmethod
    def standard_simulation():
        """标准仿真配置"""
        config = SimulationConfig()
        config.simulation_duration_hours = 24.0
        config.time_step_seconds = 60.0
        config.control_update_interval_hours = 1.0
        config.plot_update_interval_seconds = 300
        return config
    
    @staticmethod
    def detailed_analysis():
        """详细分析配置"""
        config = SimulationConfig()
        config.simulation_duration_hours = 24.0
        config.time_step_seconds = 30.0
        config.control_update_interval_hours = 0.5
        config.plot_update_interval_seconds = 180
        config.enable_real_time_monitoring = True
        config.record_detailed_metrics = True
        config.enable_comprehensive_analysis = True
        return config
    
    @staticmethod
    def long_term_simulation():
        """长期仿真配置"""
        config = SimulationConfig()
        config.simulation_duration_hours = 168.0  # 1周
        config.time_step_seconds = 300.0         # 5分钟
        config.control_update_interval_hours = 6.0
        config.plot_update_interval_seconds = 3600
        config.enable_real_time_monitoring = False
        config.enable_memory_optimization = True
        return config

if __name__ == "__main__":
    # 测试配置
    print("测试预定义配置方案...")
    
    configs = [
        ("快速测试", PresetConfigs.quick_test()),
        ("标准仿真", PresetConfigs.standard_simulation()),
        ("详细分析", PresetConfigs.detailed_analysis()),
        ("长期仿真", PresetConfigs.long_term_simulation())
    ]
    
    for name, config in configs:
        print(f"\n{name}配置:")
        config.print_config_summary()
        if config.validate_config():
            print("✓ 配置验证通过")
        else:
            print("✗ 配置验证失败")
