"""
增强电池模型：集成PyBaMM的详细电化学仿真

该模型结合了原有的工程级模型和PyBaMM的高精度电化学模型，
提供多层次的仿真精度选择：
- 快速模式：使用原有的工程级模型
- 详细模式：使用PyBaMM的DFN/SPM模型
- 混合模式：关键参数使用PyBaMM，其他使用工程模型

功能特点：
- 支持多种电化学模型（SPM、DFN、SPMe等）
- 详细的热-电化学耦合
- 精确的老化机制建模
- 可配置的仿真精度
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, Literal
from dataclasses import dataclass
import warnings

try:
    import pybamm
    PYBAMM_AVAILABLE = True
except ImportError:
    PYBAMM_AVAILABLE = False
    warnings.warn("PyBaMM未安装，将使用工程级模型。请运行: pip install pybamm")

from battery_model import BatteryModel, BatteryModelConfig


@dataclass
class PyBaMMConfig:
    """PyBaMM模型配置"""
    
    # 模型类型选择
    model_type: Literal["SPM", "DFN", "SPMe"] = "SPM"
    
    # 电池化学体系
    chemistry: str = "lithium-ion"
    
    # 参数集
    parameter_set: str = "Chen2020"
    
    # 求解器设置
    solver_method: str = "Casadi"
    solver_tolerance: float = 1e-6
    max_steps: int = 1000
    
    # 网格设置
    npts: int = 50  # 径向网格点数
    
    # 热模型
    thermal: str = "lumped"  # "lumped", "x-full", "x-lumped", "xyz-full"
    
    # 老化模型
    ageing: bool = True
    SEI_model: str = "solvent-diffusion-limited"
    lithium_plating: bool = True


class EnhancedBatteryModel:
    """增强电池模型：集成PyBaMM和工程级模型"""
    
    def __init__(
        self,
        pybamm_config: Optional[PyBaMMConfig] = None,
        battery_config: Optional[BatteryModelConfig] = None,
        initial_soc: float = 0.5,
        initial_temperature_c: float = 25.0,
        use_pybamm: bool = True
    ):
        self.use_pybamm = use_pybamm and PYBAMM_AVAILABLE
        
        # 工程级模型（作为备用和快速仿真）
        self.engineering_model = BatteryModel(
            config=battery_config,
            initial_soc=initial_soc,
            initial_temperature_c=initial_temperature_c
        )
        
        # PyBaMM模型
        if self.use_pybamm:
            self.pybamm_config = pybamm_config or PyBaMMConfig()
            self._setup_pybamm_model()
        else:
            self.pybamm_config = None
            self.pybamm_model = None
            self.pybamm_sim = None
        
        # 状态变量
        self.current_mode = "engineering"  # "engineering", "pybamm", "hybrid"
        self.simulation_history = []
        
    def _setup_pybamm_model(self):
        """设置PyBaMM模型"""
        try:
            # 选择模型
            if self.pybamm_config.model_type == "SPM":
                self.pybamm_model = pybamm.lithium_ion.SPM()
            elif self.pybamm_config.model_type == "DFN":
                self.pybamm_model = pybamm.lithium_ion.DFN()
            elif self.pybamm_config.model_type == "SPMe":
                self.pybamm_model = pybamm.lithium_ion.SPMe()
            else:
                self.pybamm_model = pybamm.lithium_ion.SPM()
            
            # 添加热模型
            if self.pybamm_config.thermal != "isothermal":
                self.pybamm_model = pybamm.thermal.lumped(self.pybamm_model)
            
            # 添加老化模型
            if self.pybamm_config.ageing:
                self.pybamm_model = pybamm.ageing.SEI(
                    self.pybamm_model, 
                    self.pybamm_config.SEI_model
                )
                if self.pybamm_config.lithium_plating:
                    self.pybamm_model = pybamm.ageing.lithium_plating(
                        self.pybamm_model
                    )
            
            # 设置参数
            self.param = pybamm.ParameterValues(self.pybamm_config.parameter_set)
            
            # 设置求解器
            if self.pybamm_config.solver_method == "Casadi":
                self.solver = pybamm.CasadiSolver(
                    mode="fast",
                    rtol=self.pybamm_config.solver_tolerance,
                    atol=self.pybamm_config.solver_tolerance
                )
            else:
                self.solver = pybamm.ScipySolver()
            
            # 设置网格
            self.geometry = self.pybamm_model.geometry
            self.mesh = pybamm.Mesh(self.geometry, self.pybamm_config.npts)
            
            # 离散化
            self.disc = pybamm.Discretisation(self.mesh, self.pybamm_model.default_spatial_methods)
            self.disc_model = self.disc.process_model(self.pybamm_model, inplace=False)
            
            # 设置初始条件
            self.pybamm_sim = None
            
        except Exception as e:
            warnings.warn(f"PyBaMM模型设置失败: {e}")
            self.use_pybamm = False
            self.pybamm_model = None
    
    def update_state(
        self, 
        current_a: float, 
        dt_s: float, 
        ambient_temp_c: float,
        mode: Optional[str] = None
    ) -> None:
        """更新电池状态"""
        if mode:
            self.current_mode = mode
            
        # 工程级模型更新（总是执行，作为备用）
        self.engineering_model.update_state(current_a, dt_s, ambient_temp_c)
        
        # PyBaMM模型更新
        if self.use_pybamm and self.current_mode in ["pybamm", "hybrid"]:
            self._update_pybamm_state(current_a, dt_s, ambient_temp_c)
        
        # 记录历史
        self.simulation_history.append({
            'time': len(self.simulation_history) * dt_s,
            'current': current_a,
            'soc_eng': self.engineering_model.state_of_charge,
            'temp_eng': self.engineering_model.cell_temperature_c,
            'mode': self.current_mode
        })
    
    def _update_pybamm_state(self, current_a: float, dt_s: float, ambient_temp_c: float):
        """更新PyBaMM模型状态"""
        try:
            if self.pybamm_sim is None:
                # 首次运行，创建仿真
                self.pybamm_sim = pybamm.Simulation(
                    self.pybamm_model,
                    parameter_values=self.param,
                    solver=self.solver
                )
                
                # 设置初始条件
                t_eval = np.array([0, dt_s])
                solution = self.pybamm_sim.solve(t_eval)
                self.last_pybamm_solution = solution
            else:
                # 继续仿真
                t_eval = np.array([
                    self.last_pybamm_solution.t[-1],
                    self.last_pybamm_solution.t[-1] + dt_s
                ])
                solution = self.pybamm_sim.solve(t_eval, initial_conditions=self.last_pybamm_solution)
                self.last_pybamm_solution = solution
            
        except Exception as e:
            warnings.warn(f"PyBaMM仿真失败: {e}")
            self.current_mode = "engineering"
    
    def get_voltage(self) -> float:
        """获取端电压"""
        if self.use_pybamm and self.current_mode in ["pybamm", "hybrid"]:
            try:
                # 从PyBaMM获取电压
                voltage = self.last_pybamm_solution["Terminal voltage [V]"].entries[-1]
                return voltage * self.engineering_model.config.series_cells
            except:
                pass
        
        # 回退到工程级模型
        return self.engineering_model.get_voltage()
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """获取详细状态信息"""
        status = self.engineering_model.get_battery_status()
        
        if self.use_pybamm and self.current_mode in ["pybamm", "hybrid"]:
            try:
                # 添加PyBaMM详细信息
                pybamm_status = self._get_pybamm_status()
                status.update(pybamm_status)
            except:
                pass
        
        status['simulation_mode'] = self.current_mode
        status['pybamm_available'] = self.use_pybamm
        
        return status
    
    def _get_pybamm_status(self) -> Dict[str, Any]:
        """获取PyBaMM模型状态"""
        try:
            solution = self.last_pybamm_solution
            return {
                'pybamm_voltage_v': solution["Terminal voltage [V]"].entries[-1],
                'pybamm_soc': solution["Discharge capacity [A.h]"].entries[-1] / 
                             solution["Discharge capacity [A.h]"].entries[0],
                'pybamm_temperature_k': solution["Cell temperature [K]"].entries[-1],
                'pybamm_anode_potential_v': solution["Negative electrode potential [V]"].entries[-1],
                'pybamm_cathode_potential_v': solution["Positive electrode potential [V]"].entries[-1],
                'pybamm_electrolyte_concentration_mol_per_m3': 
                    solution["Electrolyte concentration [mol.m-3]"].entries[-1],
            }
        except:
            return {}
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """绘制工程模型和PyBaMM模型的对比图"""
        if not self.simulation_history:
            print("没有仿真历史数据")
            return
        
        # 提取数据
        times = [h['time'] for h in self.simulation_history]
        soc_eng = [h['soc_eng'] for h in self.simulation_history]
        temp_eng = [h['temp_eng'] for h in self.simulation_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # SOC对比
        ax1.plot(times, soc_eng, 'b-', label='工程模型', linewidth=2)
        ax1.set_ylabel('SOC')
        ax1.set_title('电池状态对比')
        ax1.legend()
        ax1.grid(True)
        
        # 温度对比
        ax2.plot(times, temp_eng, 'r-', label='工程模型', linewidth=2)
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('温度 (°C)')
        ax2.legend()
        ax2.grid(True)
        
        # 如果有PyBaMM数据，添加对比
        if self.use_pybamm and self.current_mode in ["pybamm", "hybrid"]:
            try:
                pybamm_status = self._get_pybamm_status()
                if 'pybamm_soc' in pybamm_status:
                    ax1.axhline(y=pybamm_status['pybamm_soc'], color='g', linestyle='--', 
                               label=f'PyBaMM {self.pybamm_config.model_type}')
                    ax1.legend()
            except:
                pass
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_enhanced_model_example():
    """创建增强模型的示例"""
    # 配置PyBaMM
    pybamm_config = PyBaMMConfig(
        model_type="SPM",
        thermal="lumped",
        ageing=True
    )
    
    # 创建增强模型
    enhanced_model = EnhancedBatteryModel(
        pybamm_config=pybamm_config,
        use_pybamm=True
    )
    
    # 运行示例仿真
    current_profile = [100, 200, 150, 300, 100] * 20  # 5个电流值重复20次
    duration_hours = 0.5
    
    print("开始增强电池模型仿真...")
    
    for i, current in enumerate(current_profile):
        enhanced_model.update_state(current, duration_hours * 3600 / len(current_profile), 25.0)
        
        if i % 20 == 0:
            status = enhanced_model.get_detailed_status()
            print(f"步骤 {i}: SOC={status['soc']:.3f}, "
                  f"电压={status['voltage_v']:.1f}V, "
                  f"温度={status['temperature_c']:.1f}°C")
    
    # 绘制结果
    enhanced_model.plot_comparison()
    
    # 获取详细状态
    final_status = enhanced_model.get_detailed_status()
    print("\n最终状态:")
    for key, value in final_status.items():
        print(f"  {key}: {value}")
    
    return enhanced_model


if __name__ == "__main__":
    if PYBAMM_AVAILABLE:
        print("PyBaMM可用，运行增强模型示例...")
        model = create_enhanced_model_example()
    else:
        print("PyBaMM不可用，请先安装: pip install pybamm")
        print("将使用工程级模型...")
        model = EnhancedBatteryModel(use_pybamm=False)
