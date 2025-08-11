"""
PCS与储能电池集成接口

该模块提供PCS系统与储能电池模型的集成接口，实现：
1. 电池状态监控和管理
2. 充放电控制策略
3. 安全保护机制
4. 系统级优化控制
5. 实时数据交换

集成特点：
- 支持多种电池化学体系（LFP/NCM）
- 实时状态监控和预测
- 智能充放电策略
- 安全保护机制
- 与PCS控制系统的无缝集成
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging

from energy_storage_battery_model import EnergyStorageBatteryModel, EnergyStorageConfig


@dataclass
class PCSControlParameters:
    """PCS控制参数"""
    
    # 基本控制参数
    control_frequency_hz: float = 1000.0  # 控制频率
    voltage_control_band_kv: float = 0.5  # 电压控制带宽
    current_control_band_a: float = 10.0  # 电流控制带宽
    
    # 充放电策略参数
    max_charge_power_mw: float = 25.0  # 最大充电功率
    max_discharge_power_mw: float = 25.0  # 最大放电功率
    min_soc_threshold: float = 0.1  # 最小SOC阈值
    max_soc_threshold: float = 0.9  # 最大SOC阈值
    
    # 安全保护参数
    over_voltage_threshold_kv: float = 41.0  # 过压阈值
    under_voltage_threshold_kv: float = 29.0  # 欠压阈值
    over_current_threshold_a: float = 1000.0  # 过流阈值
    over_temperature_threshold_c: float = 55.0  # 过温阈值
    
    # 优化控制参数
    power_ramp_rate_mw_per_min: float = 5.0  # 功率爬坡率
    frequency_response_enabled: bool = True  # 频率响应使能
    voltage_regulation_enabled: bool = True  # 电压调节使能


@dataclass
class GridRequirements:
    """电网需求参数"""
    
    # 功率需求
    target_power_mw: float = 0.0  # 目标功率
    power_direction: Literal["charge", "discharge", "idle"] = "idle"  # 功率方向
    
    # 频率调节需求
    frequency_deviation_hz: float = 0.0  # 频率偏差
    frequency_response_power_mw: float = 0.0  # 频率响应功率
    
    # 电压调节需求
    voltage_deviation_percent: float = 0.0  # 电压偏差
    voltage_regulation_power_mw: float = 0.0  # 电压调节功率
    
    # 时间要求
    response_time_ms: float = 100.0  # 响应时间要求


class PCSBatteryIntegration:
    """PCS与储能电池集成系统"""
    
    def __init__(
        self,
        battery_config: Optional[EnergyStorageConfig] = None,
        pcs_config: Optional[PCSControlParameters] = None,
        use_pybamm: bool = True
    ):
        # 配置参数
        self.battery_config = battery_config or EnergyStorageConfig()
        self.pcs_config = pcs_config or PCSControlParameters()
        
        # 创建电池模型
        self.battery = EnergyStorageBatteryModel(
            config=self.battery_config,
            use_pybamm=use_pybamm
        )
        
        # 系统状态
        self.system_status = "initializing"  # initializing, ready, running, fault, maintenance
        self.operation_mode = "standby"  # standby, charging, discharging, frequency_response, voltage_regulation
        
        # 控制状态
        self.control_active = False
        self.last_control_time = time.time()
        self.control_history = []
        
        # 性能指标
        self.performance_metrics = {
            'total_energy_throughput_mwh': 0.0,
            'total_cycles': 0.0,
            'availability_percent': 100.0,
            'response_time_ms': 0.0,
            'efficiency_percent': 95.0
        }
        
        # 日志设置
        self._setup_logging()
        
        # 初始化完成
        self.system_status = "ready"
        self.logger.info("PCS-电池集成系统初始化完成")
    
    def _setup_logging(self):
        """设置日志系统"""
        self.logger = logging.getLogger('PCSBatteryIntegration')
        self.logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler('pcs_battery_integration.log')
        file_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def start_control(self):
        """启动控制系统"""
        if self.system_status != "ready":
            self.logger.error(f"系统状态不正确: {self.system_status}")
            return False
        
        self.control_active = True
        self.system_status = "running"
        self.logger.info("控制系统已启动")
        return True
    
    def stop_control(self):
        """停止控制系统"""
        self.control_active = False
        self.system_status = "ready"
        self.logger.info("控制系统已停止")
    
    def update_grid_requirements(self, grid_req: GridRequirements) -> Dict[str, Any]:
        """更新电网需求并执行控制"""
        if not self.control_active:
            return {"status": "error", "message": "控制系统未启动"}
        
        try:
            # 检查电池状态
            battery_status = self.battery.get_status()
            
            # 验证需求可行性
            feasibility = self._check_requirement_feasibility(grid_req, battery_status)
            if not feasibility['feasible']:
                return {
                    "status": "warning",
                    "message": "需求不可行",
                    "details": feasibility['reasons']
                }
            
            # 计算控制指令
            control_command = self._calculate_control_command(grid_req, battery_status)
            
            # 执行控制
            execution_result = self._execute_control_command(control_command)
            
            # 更新性能指标
            self._update_performance_metrics(grid_req, execution_result)
            
            # 记录控制历史
            self._record_control_history(grid_req, control_command, execution_result)
            
            return {
                "status": "success",
                "control_command": control_command,
                "execution_result": execution_result,
                "battery_status": battery_status
            }
            
        except Exception as e:
            self.logger.error(f"控制更新失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def _check_requirement_feasibility(
        self, 
        grid_req: GridRequirements, 
        battery_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检查需求可行性"""
        reasons = []
        feasible = True
        
        # 检查功率需求
        if grid_req.target_power_mw > self.pcs_config.max_discharge_power_mw:
            reasons.append(f"功率需求 {grid_req.target_power_mw} MW 超过最大放电功率 {self.pcs_config.max_discharge_power_mw} MW")
            feasible = False
        
        # 检查SOC限制
        if grid_req.power_direction == "discharge" and battery_status['soc'] < self.pcs_config.min_soc_threshold:
            reasons.append(f"SOC {battery_status['soc']:.3f} 低于最小阈值 {self.pcs_config.min_soc_threshold}")
            feasible = False
        
        if grid_req.power_direction == "charge" and battery_status['soc'] > self.pcs_config.max_soc_threshold:
            reasons.append(f"SOC {battery_status['soc']:.3f} 高于最大阈值 {self.pcs_config.max_soc_threshold}")
            feasible = False
        
        # 检查安全状态
        if any(battery_status['safety_status'].values()):
            reasons.append("电池存在安全告警")
            feasible = False
        
        # 检查响应时间
        if grid_req.response_time_ms < 100:  # 假设最小响应时间100ms
            reasons.append(f"响应时间要求 {grid_req.response_time_ms} ms 过于严格")
            feasible = False
        
        return {
            "feasible": feasible,
            "reasons": reasons
        }
    
    def _calculate_control_command(
        self, 
        grid_req: GridRequirements, 
        battery_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算控制指令"""
        command = {
            'timestamp': time.time(),
            'power_setpoint_mw': 0.0,
            'voltage_setpoint_kv': 0.0,
            'current_setpoint_a': 0.0,
            'operation_mode': 'idle'
        }
        
        # 基本功率控制
        if grid_req.power_direction == "charge":
            command['power_setpoint_mw'] = -min(
                abs(grid_req.target_power_mw),
                self.pcs_config.max_charge_power_mw
            )
            command['operation_mode'] = 'charging'
        elif grid_req.power_direction == "discharge":
            command['power_setpoint_mw'] = min(
                grid_req.target_power_mw,
                self.pcs_config.max_discharge_power_mw
            )
            command['operation_mode'] = 'discharging'
        
        # 频率响应控制
        if grid_req.frequency_response_enabled and abs(grid_req.frequency_deviation_hz) > 0.01:
            freq_response_power = self._calculate_frequency_response_power(grid_req.frequency_deviation_hz)
            command['power_setpoint_mw'] += freq_response_power
            command['operation_mode'] = 'frequency_response'
        
        # 电压调节控制
        if grid_req.voltage_regulation_enabled and abs(grid_req.voltage_deviation_percent) > 1.0:
            voltage_reg_power = self._calculate_voltage_regulation_power(grid_req.voltage_deviation_percent)
            command['power_setpoint_mw'] += voltage_reg_power
            command['operation_mode'] = 'voltage_regulation'
        
        # 计算电压和电流设定值
        if battery_status['voltage_v'] > 0:
            command['voltage_setpoint_kv'] = battery_status['voltage_kv']
            command['current_setpoint_a'] = command['power_setpoint_mw'] * 1e6 / (battery_status['voltage_v'])
        
        # 功率爬坡限制
        command = self._apply_power_ramp_limiting(command)
        
        return command
    
    def _calculate_frequency_response_power(self, frequency_deviation_hz: float) -> float:
        """计算频率响应功率"""
        # 简化的频率响应算法
        # 频率偏差每0.1Hz对应1MW功率响应
        response_power = frequency_deviation_hz * 10.0  # MW/Hz
        
        # 限制响应功率范围
        max_response = min(self.pcs_config.max_discharge_power_mw, 
                          self.pcs_config.max_charge_power_mw)
        response_power = max(-max_response, min(max_response, response_power))
        
        return response_power
    
    def _calculate_voltage_regulation_power(self, voltage_deviation_percent: float) -> float:
        """计算电压调节功率"""
        # 简化的电压调节算法
        # 电压偏差每1%对应0.5MW功率调节
        regulation_power = voltage_deviation_percent * 0.5  # MW/%
        
        # 限制调节功率范围
        max_regulation = min(self.pcs_config.max_discharge_power_mw, 
                            self.pcs_config.max_charge_power_mw)
        regulation_power = max(-max_regulation, min(max_regulation, regulation_power))
        
        return regulation_power
    
    def _apply_power_ramp_limiting(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """应用功率爬坡限制"""
        if not self.control_history:
            return command
        
        # 获取上次功率设定值
        last_power = self.control_history[-1].get('power_setpoint_mw', 0.0)
        current_power = command['power_setpoint_mw']
        
        # 计算功率变化
        power_change = current_power - last_power
        max_change = self.pcs_config.power_ramp_rate_mw_per_min / 60.0  # 转换为每秒
        
        # 限制功率变化
        if abs(power_change) > max_change:
            if power_change > 0:
                command['power_setpoint_mw'] = last_power + max_change
            else:
                command['power_setpoint_mw'] = last_power - max_change
        
        return command
    
    def _execute_control_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """执行控制指令"""
        start_time = time.time()
        
        try:
            # 更新电池状态（模拟控制执行）
            dt = 1.0 / self.pcs_config.control_frequency_hz
            
            # 计算电流
            current_a = command['current_setpoint_a']
            
            # 更新电池模型
            battery_status = self.battery.update_state(
                current_a=current_a,
                dt_s=dt,
                ambient_temp_c=25.0,  # 假设环境温度
                operation_mode=command['operation_mode']
            )
            
            # 计算执行结果
            execution_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            result = {
                'execution_time_ms': execution_time,
                'actual_power_mw': battery_status['power_mw'],
                'actual_voltage_kv': battery_status['voltage_kv'],
                'actual_current_a': battery_status['current_a'],
                'battery_status': battery_status,
                'success': True
            }
            
            # 更新操作模式
            self.operation_mode = command['operation_mode']
            
            return result
            
        except Exception as e:
            self.logger.error(f"控制指令执行失败: {e}")
            return {
                'execution_time_ms': (time.time() - start_time) * 1000,
                'success': False,
                'error': str(e)
            }
    
    def _update_performance_metrics(
        self, 
        grid_req: GridRequirements, 
        execution_result: Dict[str, Any]
    ):
        """更新性能指标"""
        if execution_result['success']:
            # 更新能量吞吐量
            power_mw = abs(execution_result['actual_power_mw'])
            dt_hours = 1.0 / self.pcs_config.control_frequency_hz / 3600.0
            self.performance_metrics['total_energy_throughput_mwh'] += power_mw * dt_hours
            
            # 更新响应时间
            self.performance_metrics['response_time_ms'] = execution_result['execution_time_ms']
            
            # 更新效率（简化计算）
            if grid_req.target_power_mw != 0:
                efficiency = abs(execution_result['actual_power_mw'] / grid_req.target_power_mw)
                self.performance_metrics['efficiency_percent'] = efficiency * 100.0
    
    def _record_control_history(
        self, 
        grid_req: GridRequirements, 
        control_command: Dict[str, Any], 
        execution_result: Dict[str, Any]
    ):
        """记录控制历史"""
        history_entry = {
            'timestamp': datetime.now(),
            'grid_requirements': {
                'target_power_mw': grid_req.target_power_mw,
                'power_direction': grid_req.power_direction,
                'frequency_deviation_hz': grid_req.frequency_deviation_hz,
                'voltage_deviation_percent': grid_req.voltage_deviation_percent
            },
            'control_command': control_command,
            'execution_result': execution_result,
            'battery_status': self.battery.get_status()
        }
        
        self.control_history.append(history_entry)
        
        # 限制历史记录数量
        if len(self.control_history) > 10000:
            self.control_history = self.control_history[-5000:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'system_status': self.system_status,
            'operation_mode': self.operation_mode,
            'control_active': self.control_active,
            'battery_status': self.battery.get_status(),
            'performance_metrics': self.performance_metrics.copy(),
            'safety_status': self.battery.safety_status.copy(),
            'last_control_time': self.last_control_time
        }
    
    def get_control_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取控制历史"""
        return self.control_history[-limit:]
    
    def export_control_data(self, filepath: str):
        """导出控制数据"""
        try:
            export_data = {
                'system_config': {
                    'battery_config': self.battery_config.__dict__,
                    'pcs_config': self.pcs_config.__dict__
                },
                'current_status': self.get_system_status(),
                'control_history': self.control_history[-1000:],  # 最近1000条记录
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"控制数据已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"数据导出失败: {e}")
    
    def run_demo_scenario(self, duration_minutes: float = 60.0):
        """运行演示场景"""
        self.logger.info(f"开始运行演示场景，持续时间: {duration_minutes} 分钟")
        
        # 启动控制
        if not self.start_control():
            return
        
        # 场景时间步长
        dt_minutes = 1.0
        total_steps = int(duration_minutes / dt_minutes)
        
        # 模拟电网需求变化
        for step in range(total_steps):
            time_minutes = step * dt_minutes
            
            # 创建电网需求
            grid_req = self._create_demo_grid_requirements(time_minutes, duration_minutes)
            
            # 更新控制
            result = self.update_grid_requirements(grid_req)
            
            if result['status'] != 'success':
                self.logger.warning(f"步骤 {step} 控制失败: {result}")
            
            # 等待时间步长
            time.sleep(dt_minutes * 60.0)
        
        # 停止控制
        self.stop_control()
        self.logger.info("演示场景运行完成")
    
    def _create_demo_grid_requirements(self, time_minutes: float, total_duration: float) -> GridRequirements:
        """创建演示电网需求"""
        # 模拟日负荷变化
        hour = (time_minutes / 60.0) % 24.0
        
        if 6 <= hour <= 9:  # 早高峰
            power_direction = "discharge"
            target_power = 20.0 + 5.0 * np.sin((hour - 7.5) * np.pi / 1.5)
        elif 18 <= hour <= 21:  # 晚高峰
            power_direction = "discharge"
            target_power = 22.0 + 3.0 * np.sin((hour - 19.5) * np.pi / 1.5)
        elif 23 <= hour or hour <= 6:  # 夜间低谷
            power_direction = "charge"
            target_power = 15.0 + 5.0 * np.random.random()
        else:  # 平时
            if np.random.random() > 0.5:
                power_direction = "discharge"
                target_power = 10.0 + 5.0 * np.random.random()
            else:
                power_direction = "charge"
                target_power = 8.0 + 3.0 * np.random.random()
        
        # 添加频率和电压调节需求
        frequency_deviation = 0.05 * np.sin(time_minutes * 0.1) + 0.02 * np.random.random()
        voltage_deviation = 2.0 * np.sin(time_minutes * 0.05) + 1.0 * np.random.random()
        
        return GridRequirements(
            target_power_mw=target_power,
            power_direction=power_direction,
            frequency_deviation_hz=frequency_deviation,
            voltage_deviation_percent=voltage_deviation,
            response_time_ms=200.0
        )


def create_pcs_battery_integration_example():
    """创建PCS-电池集成示例"""
    # 电池配置
    battery_config = EnergyStorageConfig(
        battery_type="LFP",
        pybamm_model_type="SPM",
        thermal_model="lumped",
        ageing_model=True
    )
    
    # PCS控制配置
    pcs_config = PCSControlParameters(
        control_frequency_hz=1000.0,
        max_charge_power_mw=25.0,
        max_discharge_power_mw=25.0,
        power_ramp_rate_mw_per_min=5.0
    )
    
    # 创建集成系统
    integration = PCSBatteryIntegration(
        battery_config=battery_config,
        pcs_config=pcs_config,
        use_pybamm=True
    )
    
    print("PCS-电池集成系统创建成功！")
    print(f"电池类型: {battery_config.battery_type}")
    print(f"电池容量: {battery_config.battery_capacity_ah} Ah")
    print(f"串联数: {battery_config.series_cells}")
    print(f"额定功率: {battery_config.rated_power_mw} MW")
    print(f"控制频率: {pcs_config.control_frequency_hz} Hz")
    
    return integration


if __name__ == "__main__":
    # 创建集成系统
    integration = create_pcs_battery_integration_example()
    
    # 运行演示场景
    print("\n开始运行演示场景...")
    integration.run_demo_scenario(duration_minutes=10.0)  # 10分钟演示
    
    # 获取系统状态
    status = integration.get_system_status()
    print(f"\n系统状态: {status['system_status']}")
    print(f"操作模式: {status['operation_mode']}")
    print(f"电池SOC: {status['battery_status']['soc']:.3f}")
    print(f"电池电压: {status['battery_status']['voltage_kv']:.1f} kV")
    
    # 导出数据
    integration.export_control_data('pcs_battery_integration_demo.json')
    print("\n演示完成，数据已导出！")
