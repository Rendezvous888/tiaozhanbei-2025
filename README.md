# 🔋 35 kV/25 MW级联储能PCS仿真系统

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com/your-username/tiaozhanbei-2025)
[![Version](https://img.shields.io/badge/Version-2.0-orange.svg)](https://github.com/your-username/tiaozhanbei-2025/releases)

> 🚀 **专业级储能变流器仿真平台** - 集成级联H桥拓扑、IGBT寿命预测、热分析、控制优化等先进功能

## 📖 目录

- [项目概述](#项目概述)
- [✨ 核心特性](#-核心特性)
- [🏗️ 系统架构](#️-系统架构)
- [📁 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
- [📊 功能模块](#-功能模块)
- [🔧 安装配置](#-安装配置)
- [💻 使用示例](#-使用示例)
- [📈 仿真结果](#-仿真结果)
- [🎯 应用场景](#-应用场景)
- [🔬 技术特点](#-技术特点)
- [📚 文档资源](#-文档资源)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)

## 🎯 项目概述

本项目是一个**完整的35 kV/25 MW级联储能PCS仿真系统**，专为电力电子研究、储能系统设计和工程应用而开发。系统采用级联H桥拓扑结构，集成了先进的IGBT器件建模、热分析、寿命预测和控制优化算法。

### 🌟 主要应用领域
- **储能电站设计与优化**
- **电力电子器件寿命评估**
- **控制策略研究与开发**
- **系统性能分析与优化**
- **教学与科研支持**

## ✨ 核心特性

| 特性 | 描述 | 优势 |
|------|------|------|
| 🔌 **级联H桥拓扑** | 每相40个H桥单元串联 | 81电平输出，高电压分辨率 |
| 🔥 **IGBT热建模** | 动态结温和壳温仿真 | 精确的温度分布分析 |
| ⏰ **寿命预测** | 基于雨流计数法的疲劳分析 | 15年寿命预测能力 |
| 🎛️ **控制优化** | PI、MPC、自适应控制 | 多种控制策略对比 |
| 📊 **长期仿真** | 1-10年运行仿真 | 真实工况模拟 |
| 🎨 **中文支持** | 完整的中文界面 | 适合国内用户使用 |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    35 kV/25 MW PCS系统                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  控制层     │  │  功率层     │  │  保护层     │        │
│  │ (PI/MPC)   │  │ (H桥拓扑)   │  │ (温度/寿命) │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  仿真引擎   │  │  数据分析   │  │  可视化     │        │
│  │ (Simulink)  │  │ (Pandas)    │  │ (Matplotlib)│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
tiaozhanbei-2025/
├── 📁 核心模块/
│   ├── main_simulation.py              # 🚀 主仿真程序
│   ├── pcs_simulation_model.py         # 🔌 PCS系统模型
│   ├── h_bridge_model.py              # 🌉 H桥单元建模
│   ├── control_optimization.py         # 🎛️ 控制与优化
│   └── enhanced_battery_model.py      # 🔋 电池模型
├── 📁 高级功能/
│   ├── long_term_life_simulation.py    # ⏰ 长期寿命仿真
│   ├── detailed_life_analysis.py       # 📊 详细寿命分析
│   ├── enhanced_igbt_life_model.py     # 🔥 IGBT寿命模型
│   └── advanced_device_modeling.py     # ⚡ 先进器件建模
├── 📁 分析工具/
│   ├── analyze_modulation_quality.py   # 📈 调制质量分析
│   ├── compare_pwm_strategies.py       # 🔄 PWM策略对比
│   └── plot_utils.py                   # 🎨 绘图工具
├── 📁 测试与调试/
│   ├── Test/                           # 🧪 测试文件
│   ├── Debug/                          # 🐛 调试工具
│   └── demo_simulation.py              # 🎬 演示版本
├── 📁 文档资源/
│   ├── 项目总结.md                     # 📋 项目总结
│   ├── 先进PWM策略总结.md              # 📚 PWM策略
│   └── 长期寿命仿真总结报告.md         # 📊 寿命分析
├── 📁 结果输出/
│   └── result/                         # 📈 仿真结果图表
├── requirements.txt                     # 📦 依赖包列表
└── README.md                           # 📖 项目说明
```

## 🚀 快速开始

### 1️⃣ 环境要求
- **Python**: 3.7+ (推荐3.8+)
- **操作系统**: Windows 10/11, Linux, macOS
- **内存**: 建议8GB+
- **存储**: 至少2GB可用空间

### 2️⃣ 安装步骤

```bash
# 克隆项目
git clone https://github.com/your-username/tiaozhanbei-2025.git
cd tiaozhanbei-2025

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3️⃣ 快速运行

```bash
# 运行主仿真程序
python main_simulation.py

# 运行演示版本
python demo_simulation.py

# 运行特定模块
python h_bridge_model.py
```

## 📊 功能模块

### 🔌 PCS系统建模 (`pcs_simulation_model.py`)

**核心类**:
- `PCSParameters`: 系统参数配置
- `CascadedHBridge`: 级联H桥拓扑建模
- `ThermalModel`: IGBT热模型
- `LifePrediction`: 寿命预测模型
- `BatteryManagement`: 电池管理系统

**主要功能**:
- 35 kV/25 MW系统参数配置
- 级联H桥拓扑建模（每相40个单元）
- IGBT结温和壳温动态仿真
- 基于雨流计数法的寿命评估
- 24小时储能电站运行仿真

### 🌉 H桥单元建模 (`h_bridge_model.py`)

**核心类**:
- `HBridgeUnit`: 单个H桥单元建模
- `CascadedHBridgeSystem`: 级联H桥系统

**主要功能**:
- PWM调制信号生成（相移PWM）
- 开关损耗和导通损耗计算
- 谐波频谱分析（FFT）
- 81电平电压输出
- 多电平调制策略

### 🎛️ 控制与优化 (`control_optimization.py`)

**核心类**:
- `PCSController`: 基本控制器（PI控制）
- `AdvancedControlStrategies`: 高级控制策略
- `PCSOptimizer`: 参数优化器
- `PerformanceEvaluator`: 性能评估器

**控制策略**:
- 电压环和电流环PI控制
- 模型预测控制（MPC）
- 自适应控制
- 模糊控制
- 遗传算法参数优化

### ⏰ 长期寿命仿真 (`long_term_life_simulation.py`)

**主要功能**:
- 1年、3年、5年、10年寿命预测
- 轻负载、中等负载、重负载多工况分析
- IGBT Coffin-Manson疲劳寿命模型
- 电容Arrhenius温度加速模型
- 分级维护体系建议

## 🔧 安装配置

### 依赖包说明

| 包名 | 版本 | 用途 |
|------|------|------|
| `numpy` | ≥1.21.0 | 数值计算基础 |
| `matplotlib` | ≥3.5.0 | 图形绘制和可视化 |
| `scipy` | ≥1.7.0 | 科学计算和优化 |
| `pandas` | ≥1.3.0 | 数据处理和分析 |
| `rainflow` | ≥3.0.0 | 雨流计数法分析 |
| `deap` | ≥1.3.0 | 遗传算法优化 |
| `pybamm` | ≥23.9.0 | 电池建模支持 |

### 中文字体配置

```python
# 在代码开头添加以下配置
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

## 💻 使用示例

### 基础仿真示例

```python
from pcs_simulation_model import PCSSimulation
from device_parameters import DeviceParameters

# 创建仿真实例
sim = PCSSimulation()

# 设置仿真参数
sim.set_simulation_time(24)  # 24小时仿真
sim.set_load_profile('daily')  # 日常负载模式

# 运行仿真
results = sim.run_simulation()

# 生成报告
sim.generate_report()
sim.plot_results()
```

### 寿命分析示例

```python
from long_term_life_simulation import LongTermLifeSimulation

# 创建长期寿命仿真
life_sim = LongTermLifeSimulation()

# 设置仿真年限
life_sim.set_simulation_years(10)

# 运行多工况分析
results = life_sim.run_multi_scenario_analysis()

# 生成寿命报告
life_sim.generate_life_report()
```

### 控制优化示例

```python
from control_optimization import PCSOptimizer

# 创建优化器
optimizer = PCSOptimizer()

# 设置优化目标
optimizer.set_objectives(['efficiency', 'lifetime', 'response_time'])

# 运行优化
best_params = optimizer.optimize_parameters()

# 应用优化结果
optimizer.apply_parameters(best_params)
```

## 📈 仿真结果

### 系统性能指标

| 指标 | 数值 | 状态 |
|------|------|------|
| **平均效率** | 99.98% | ✅ 优秀 |
| **最大结温** | 150°C | ✅ 安全 |
| **IGBT寿命** | 100% | ✅ 良好 |
| **电容寿命** | 100% | ✅ 良好 |
| **THD** | <2% | ✅ 优秀 |

### 长期寿命分析结果

#### IGBT剩余寿命（10年后）
- **轻负载**: 96.9% ✅ 良好
- **中等负载**: 76.9% ⚠️ 需要关注
- **重负载**: 41.3% ❌ 需要更换

### 输出文件

- `PCS_仿真报告_YYYYMMDD_HHMMSS.csv`: 详细仿真结果
- `长期寿命分析报告.pdf`: 寿命预测报告
- `系统性能分析图表.png`: 可视化结果

## 🎯 应用场景

### 🏭 储能电站
- **电网调峰调频**: 支持电网稳定性
- **新能源并网**: 平滑功率波动
- **电能质量改善**: 降低谐波含量

### 🏢 工业应用
- **大功率电机驱动**: 变频器系统
- **电力电子设备**: 测试和验证
- **系统集成**: 整体解决方案

### 🎓 研究开发
- **拓扑结构优化**: 性能提升研究
- **控制策略研究**: 算法对比分析
- **器件选型分析**: 成本效益评估
- **系统性能评估**: 可靠性分析

## 🔬 技术特点

### 🌉 级联H桥拓扑
- **模块数**: 每相40个H桥单元
- **电压等级**: 81电平（-40到+40）
- **电压分辨率**: 505 V
- **总输出电压**: 35 kV
- **功率容量**: 25 MW

### 🔥 IGBT器件建模
- **开关频率**: 1000 Hz
- **损耗模型**: 开关损耗 + 导通损耗
- **热网络**: 结到壳、壳到环境热阻热容
- **温度限制**: 最大结温150°C

### ⏰ 寿命预测技术
- **IGBT寿命模型**: Coffin-Manson疲劳模型
- **电容寿命模型**: Arrhenius温度加速模型
- **雨流计数**: 温度循环分析
- **多工况分析**: 不同负载条件下的寿命表现

### 🎛️ 先进控制策略
- **基本控制**: PI电压环、电流环控制
- **高级控制**: 模型预测控制、自适应控制
- **优化算法**: 遗传算法参数优化
- **性能指标**: 效率、响应速度、稳定性

## 📚 文档资源

### 📋 项目文档
- [项目总结](./项目总结.md) - 完整的项目概述和功能说明
- [先进PWM策略总结](./先进PWM策略总结.md) - PWM调制技术详解
- [长期寿命仿真总结报告](./长期寿命仿真总结报告.md) - 寿命分析结果
- [IGBT寿命计算方法对比分析](./IGBT寿命计算方法对比分析.md) - 寿命模型对比

### 🎬 演示和测试
- [演示版本](./demo_simulation.py) - 快速体验系统功能
- [测试文件](./Test/) - 完整的测试套件
- [调试工具](./Debug/) - 问题诊断和调试

### 📊 结果示例
- [仿真结果图表](./result/) - 可视化输出示例
- [性能分析报告](./enhanced_simulation_report.py) - 自动报告生成

