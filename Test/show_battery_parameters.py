"""
电池参数展示脚本

显示项目中电池模型的详细参数配置，包括：
1. 基础电池参数
2. 热管理参数
3. 安全限制参数
4. 性能参数
5. 系统级配置

作者: AI Assistant
创建时间: 2025-01-15
"""

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from battery_model import BatteryModelConfig
from enhanced_battery_model import PyBaMMConfig
from energy_storage_battery_model import EnergyStorageConfig
from device_parameters import SystemParameters, IGBTParameters, CapacitorParameters, ThermalParameters

def show_basic_battery_parameters():
    """显示基础电池参数"""
    
    print("=" * 80)
    print("基础电池模型参数 (BatteryModelConfig)")
    print("=" * 80)
    
    config = BatteryModelConfig()
    
    print(f"📋 电池基本规格:")
    print(f"  串联电芯数量: {config.series_cells} 个")
    print(f"  电池容量: {config.rated_capacity_ah} Ah")
    print(f"  额定电流: {config.rated_current_a} A")
    print(f"  单体额定电压: {config.nominal_voltage_per_cell_v} V")
    
    # 计算系统级参数
    total_voltage = config.series_cells * config.nominal_voltage_per_cell_v
    total_power = config.rated_current_a * total_voltage
    energy_capacity = config.rated_capacity_ah * total_voltage / 1000  # kWh
    
    print(f"\n🔋 计算得出的系统参数:")
    print(f"  模块总电压: {total_voltage:.1f} V ({total_voltage/1000:.2f} kV)")
    print(f"  额定功率: {total_power/1000:.1f} kW ({total_power/1000000:.2f} MW)")
    print(f"  能量容量: {energy_capacity:.1f} kWh")
    print(f"  C倍率 (1C): {config.rated_capacity_ah} A")
    
    print(f"\n⚡ 电气参数:")
    print(f"  基础内阻 (25°C): {config.base_string_resistance_ohm_25c:.6f} Ω")
    print(f"  单体内阻 (25°C): {config.base_string_resistance_ohm_25c/config.series_cells:.8f} Ω")
    
    print(f"\n🌡️ 热管理参数:")
    print(f"  热阻: {config.thermal_resistance_k_per_w:.2e} K/W")
    print(f"  热容: {config.thermal_capacity_j_per_k:.2e} J/K")
    
    print(f"\n🔄 温度影响参数:")
    print(f"  低温容量损失系数: {config.low_temp_capacity_loss_per_k_c} /K")
    print(f"  高温容量增益系数: {config.high_temp_capacity_gain_per_k_c} /K")

def show_energy_storage_parameters():
    """显示储能系统参数"""
    
    print("\n" + "=" * 80)
    print("储能系统电池参数 (EnergyStorageConfig)")
    print("=" * 80)
    
    config = EnergyStorageConfig()
    
    print(f"🏭 系统规格:")
    print(f"  电池类型: {config.battery_type} (磷酸铁锂/三元锂)")
    print(f"  系统容量: {config.battery_capacity_ah} Ah")
    print(f"  串联数: {config.series_cells} 个")
    print(f"  额定功率: {config.rated_power_mw} MW")
    
    print(f"\n⚡ 电压范围:")
    print(f"  最小模块电压: {config.module_voltage_min_kv} kV")
    print(f"  最大模块电压: {config.module_voltage_max_kv} kV")
    print(f"  额定模块电压: {(config.module_voltage_min_kv + config.module_voltage_max_kv)/2:.1f} kV")
    
    # 计算电流范围
    max_current_at_min_voltage = config.rated_power_mw * 1000 / config.module_voltage_min_kv
    max_current_at_max_voltage = config.rated_power_mw * 1000 / config.module_voltage_max_kv
    
    print(f"\n🔌 电流范围:")
    print(f"  最小电压时电流: {max_current_at_min_voltage:.1f} A")
    print(f"  最大电压时电流: {max_current_at_max_voltage:.1f} A")
    print(f"  最大过载倍数: {config.max_overload_ratio}x")
    
    print(f"\n🌡️ 温度限制:")
    print(f"  最低工作温度: {config.min_temperature_c}°C")
    print(f"  最高工作温度: {config.max_temperature_c}°C")
    print(f"  工作温度范围: {config.max_temperature_c - config.min_temperature_c}°C")
    
    print(f"\n⏰ 寿命参数:")
    print(f"  目标寿命: {config.target_life_years} 年")
    print(f"  年日历衰减率: {config.calendar_fade_per_year*100:.1f}% /年")
    print(f"  循环衰减率: {config.cycle_fade_per_cycle*1000:.1f} ‰ /循环")
    
    print(f"\n🔬 仿真模型:")
    print(f"  PyBaMM模型类型: {config.pybamm_model_type}")
    print(f"  热模型: {config.thermal_model}")
    print(f"  老化模型: {'启用' if config.ageing_model else '禁用'}")

def show_system_parameters():
    """显示系统级参数"""
    
    print("\n" + "=" * 80)
    print("系统级参数 (SystemParameters)")
    print("=" * 80)
    
    sys_params = SystemParameters()
    
    print(f"🏗️ 系统配置:")
    print(f"  系统电压范围: {sys_params.system_voltage_kV[0]} - {sys_params.system_voltage_kV[1]} kV")
    print(f"  系统频率范围: {sys_params.system_frequency_Hz[0]} - {sys_params.system_frequency_Hz[1]} Hz")
    print(f"  时间步长: {sys_params.time_step_seconds} 秒")
    
    print(f"\n🔗 级联配置:")
    print(f"  每相级联模块数: {sys_params.cascaded_power_modules}")
    print(f"  模块开关频率: {sys_params.module_switching_frequency_Hz} Hz")
    print(f"  额定电流: {sys_params.rated_current_A} A")
    print(f"  过载能力: {sys_params.overload_capability_pu}")
    
    print(f"\n🔋 电池系统:")
    print(f"  电池容量: {sys_params.battery_capacity_Ah} Ah")
    print(f"  串联电池数: {sys_params.battery_series_cells}")
    print(f"  能量时长: {sys_params.energy_hours} 小时")
    print(f"  电池供应商: {sys_params.battery_note}")
    
    print(f"\n📊 损耗配置:")
    print(f"  杂项损耗比例: {sys_params.misc_loss_fraction*100:.1f}%")
    print(f"  固定辅助损耗: {sys_params.aux_loss_w/1000:.1f} kW")
    
    print(f"\n🌡️ 环境条件:")
    print(f"  环境温度范围: {sys_params.ambient_temperature_C[0]} - {sys_params.ambient_temperature_C[1]}°C")
    print(f"  水冷入口温度: {sys_params.water_cooling_inlet_temperature_C}°C")
    print(f"  功率器件: {sys_params.power_device}")

def show_performance_metrics():
    """显示性能指标"""
    
    print("\n" + "=" * 80)
    print("电池性能指标分析")
    print("=" * 80)
    
    # 基于基础配置计算性能指标
    config = BatteryModelConfig()
    storage_config = EnergyStorageConfig()
    
    # 功率密度计算
    module_voltage = config.series_cells * config.nominal_voltage_per_cell_v
    module_power = config.rated_current_a * module_voltage
    power_density_w_per_wh = module_power / (config.rated_capacity_ah * module_voltage)
    
    print(f"⚡ 功率性能:")
    print(f"  模块功率: {module_power/1000:.1f} kW")
    print(f"  功率密度: {power_density_w_per_wh:.2f} W/Wh")
    print(f"  1C放电功率: {config.rated_capacity_ah * module_voltage / 1000:.1f} kW")
    print(f"  最大放电倍率: 3C (理论最大 {3 * config.rated_capacity_ah} A)")
    
    # 能量密度
    energy_wh = config.rated_capacity_ah * module_voltage
    print(f"\n🔋 能量性能:")
    print(f"  模块能量: {energy_wh/1000:.1f} kWh")
    print(f"  比能量: {energy_wh/config.rated_capacity_ah:.1f} Wh/Ah")
    print(f"  能量效率: ~95% (典型值)")
    
    # 循环性能
    daily_cycles_per_year = 365
    cycles_15_years = daily_cycles_per_year * storage_config.target_life_years
    total_fade_15_years = (storage_config.calendar_fade_per_year * storage_config.target_life_years + 
                          storage_config.cycle_fade_per_cycle * cycles_15_years)
    
    print(f"\n🔄 循环性能:")
    print(f"  目标寿命: {storage_config.target_life_years} 年")
    print(f"  预计总循环: {cycles_15_years} 次")
    print(f"  15年后预计容量保持率: {(1-total_fade_15_years)*100:.1f}%")
    
    # 温度性能
    print(f"\n🌡️ 温度性能:")
    print(f"  工作温度范围: {storage_config.min_temperature_c}°C 到 {storage_config.max_temperature_c}°C")
    print(f"  最佳工作温度: 20-30°C")
    print(f"  热管理方式: 液冷/风冷")
    
    # 安全性能
    print(f"\n🛡️ 安全性能:")
    print(f"  过载保护: {storage_config.max_overload_ratio}倍额定电流")
    print(f"  过温保护: {storage_config.max_temperature_c}°C")
    print(f"  过压保护: {storage_config.module_voltage_max_kv} kV")
    print(f"  SOC工作范围: 5% - 95% (推荐)")

def show_comparison_table():
    """显示不同电池配置的对比表"""
    
    print("\n" + "=" * 80)
    print("电池配置对比表")
    print("=" * 80)
    
    # 配置对比
    basic_config = BatteryModelConfig()
    storage_config = EnergyStorageConfig()
    
    print(f"{'参数':<20} {'基础模型':<20} {'储能模型':<20}")
    print("-" * 65)
    print(f"{'电池容量 (Ah)':<20} {basic_config.rated_capacity_ah:<20} {storage_config.battery_capacity_ah:<20}")
    print(f"{'串联数':<20} {basic_config.series_cells:<20} {storage_config.series_cells:<20}")
    print(f"{'额定电流 (A)':<20} {basic_config.rated_current_a:<20} {'变动':<20}")
    print(f"{'系统规模':<20} {'模块级':<20} {'25MW级':<20}")
    print(f"{'电池类型':<20} {'通用':<20} {storage_config.battery_type:<20}")
    print(f"{'目标寿命 (年)':<20} {'N/A':<20} {storage_config.target_life_years:<20}")
    print(f"{'仿真精度':<20} {'工程级':<20} {'PyBaMM高精度':<20}")
    
    # 性能对比
    basic_voltage = basic_config.series_cells * basic_config.nominal_voltage_per_cell_v
    basic_power = basic_config.rated_current_a * basic_voltage
    storage_voltage_avg = (storage_config.module_voltage_min_kv + storage_config.module_voltage_max_kv) / 2 * 1000
    
    print(f"\n{'性能指标':<20} {'基础模型':<20} {'储能模型':<20}")
    print("-" * 65)
    print(f"{'模块电压 (V)':<20} {basic_voltage:<20.0f} {storage_voltage_avg:<20.0f}")
    print(f"{'模块功率 (kW)':<20} {basic_power/1000:<20.1f} {storage_config.rated_power_mw*1000:<20.0f}")
    print(f"{'能量 (kWh)':<20} {basic_config.rated_capacity_ah*basic_voltage/1000:<20.1f} {storage_config.battery_capacity_ah*storage_voltage_avg/1000:<20.1f}")
    print(f"{'功率密度 (W/Wh)':<20} {basic_power/(basic_config.rated_capacity_ah*basic_voltage):<20.2f} {(storage_config.rated_power_mw*1e6)/(storage_config.battery_capacity_ah*storage_voltage_avg):<20.2f}")

def main():
    """主函数"""
    
    print("🔋 35kV/25MW级联储能PCS电池参数详细说明")
    print("项目: 构网型级联储能PCS")
    print("时间:", "2025-01-15")
    
    # 显示各类参数
    show_basic_battery_parameters()
    show_energy_storage_parameters()
    show_system_parameters()
    show_performance_metrics()
    show_comparison_table()
    
    print("\n" + "=" * 80)
    print("📝 参数说明总结")
    print("=" * 80)
    print("""
🎯 关键特点:
  • 大容量储能: 314Ah × 312串 = 约350kWh模块
  • 高压系统: 30-40.5kV电压范围
  • 大功率: 25MW级联系统额定功率
  • 长寿命: 15年设计寿命，>5000循环
  • 多重保护: 过载、过温、过压全方位保护

⚙️ 技术特色:
  • PyBaMM高精度电化学建模
  • 热-电化学耦合仿真
  • 先进的老化模型
  • 多场景测试验证
  • 实时安全监控

🔧 应用场景:
  • 电网调频调压
  • 削峰填谷
  • 新能源消纳
  • 紧急备用电源
  • 微电网储能
    """)
    
    print("\n📊 了解更多信息，请运行:")
    print("  python Test/test_battery_final.py    # 运行电池性能测试")
    print("  python battery_model.py             # 查看电池模型示例")
    print("  python energy_storage_battery_model.py  # 查看储能系统示例")

if __name__ == "__main__":
    main()
