#!/usr/bin/env python3
"""
PyBaMM集成测试脚本

该脚本测试增强电池模型的基本功能，包括：
1. PyBaMM模型创建和配置
2. 基本仿真运行
3. 结果对比和可视化
4. 错误处理和回退机制
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_pybamm_availability():
    """测试PyBaMM是否可用"""
    print("=" * 50)
    print("测试PyBaMM可用性")
    print("=" * 50)
    
    try:
        import pybamm
        print(f"✅ PyBaMM版本: {pybamm.__version__}")
        print(f"✅ PyBaMM路径: {pybamm.__file__}")
        return True
    except ImportError as e:
        print(f"❌ PyBaMM导入失败: {e}")
        print("请运行: pip install pybamm")
        return False

def test_enhanced_model_creation():
    """测试增强模型创建"""
    print("\n" + "=" * 50)
    print("测试增强模型创建")
    print("=" * 50)
    
    try:
        from enhanced_battery_model import EnhancedBatteryModel, PyBaMMConfig
        
        # 测试配置创建
        config = PyBaMMConfig(
            model_type="SPM",
            thermal="lumped",
            ageing=True
        )
        print(f"✅ PyBaMMConfig创建成功: {config}")
        
        # 测试模型创建
        model = EnhancedBatteryModel(
            pybamm_config=config,
            use_pybamm=True
        )
        print(f"✅ EnhancedBatteryModel创建成功")
        print(f"   - PyBaMM可用: {model.use_pybamm}")
        print(f"   - 当前模式: {model.current_mode}")
        
        return model
        
    except Exception as e:
        print(f"❌ 增强模型创建失败: {e}")
        return None

def test_basic_simulation(model):
    """测试基本仿真功能"""
    if model is None:
        return
    
    print("\n" + "=" * 50)
    print("测试基本仿真功能")
    print("=" * 50)
    
    try:
        # 简单的电流曲线
        current_profile = [50, 100, 150, 200, 150, 100, 50]
        dt = 1.0  # 1秒时间步
        
        print("开始仿真...")
        start_time = time.time()
        
        for i, current in enumerate(current_profile):
            model.update_state(current, dt, 25.0)
            
            if i % 2 == 0:
                status = model.get_detailed_status()
                print(f"  步骤 {i}: 电流={current}A, SOC={status['soc']:.3f}, "
                      f"电压={status['voltage_v']:.1f}V")
        
        simulation_time = time.time() - start_time
        print(f"✅ 仿真完成，耗时: {simulation_time:.3f}秒")
        
        # 获取最终状态
        final_status = model.get_detailed_status()
        print(f"✅ 最终状态:")
        for key, value in final_status.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {type(value).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本仿真失败: {e}")
        return False

def test_mode_switching(model):
    """测试模式切换"""
    if model is None:
        return
    
    print("\n" + "=" * 50)
    print("测试模式切换")
    print("=" * 50)
    
    try:
        # 测试工程模式
        print("切换到工程模式...")
        model.update_state(100, 1.0, 25.0, mode="engineering")
        status = model.get_detailed_status()
        print(f"  工程模式: {status['simulation_mode']}")
        
        # 测试PyBaMM模式（如果可用）
        if model.use_pybamm:
            print("切换到PyBaMM模式...")
            model.update_state(100, 1.0, 25.0, mode="pybamm")
            status = model.get_detailed_status()
            print(f"  PyBaMM模式: {status['simulation_mode']}")
        
        # 测试混合模式
        print("切换到混合模式...")
        model.update_state(100, 1.0, 25.0, mode="hybrid")
        status = model.get_detailed_status()
        print(f"  混合模式: {status['simulation_mode']}")
        
        print("✅ 模式切换测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 模式切换测试失败: {e}")
        return False

def test_visualization(model):
    """测试可视化功能"""
    if model is None:
        return
    
    print("\n" + "=" * 50)
    print("测试可视化功能")
    print("=" * 50)
    
    try:
        # 生成更多数据用于绘图
        print("生成绘图数据...")
        for i in range(20):
            current = 100 + 50 * np.sin(i * 0.5)
            model.update_state(current, 0.5, 25.0)
        
        # 测试绘图
        print("生成对比图...")
        model.plot_comparison()
        print("✅ 可视化测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 50)
    print("测试错误处理")
    print("=" * 50)
    
    try:
        # 测试PyBaMM不可用的情况
        print("测试PyBaMM不可用的情况...")
        from enhanced_battery_model import EnhancedBatteryModel
        
        model = EnhancedBatteryModel(use_pybamm=False)
        print(f"✅ 无PyBaMM模式创建成功: {model.use_pybamm}")
        
        # 测试基本功能
        model.update_state(100, 1.0, 25.0)
        status = model.get_detailed_status()
        print(f"✅ 无PyBaMM模式仿真成功: SOC={status['soc']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n" + "=" * 50)
    print("性能基准测试")
    print("=" * 50)
    
    try:
        from enhanced_battery_model import EnhancedBatteryModel, PyBaMMConfig
        
        # 工程级模型性能测试
        print("测试工程级模型性能...")
        eng_model = EnhancedBatteryModel(use_pybamm=False)
        
        start_time = time.time()
        for i in range(1000):
            eng_model.update_state(100, 0.1, 25.0)
        eng_time = time.time() - start_time
        
        print(f"  工程级模型: 1000步，耗时 {eng_time:.3f}秒")
        print(f"  平均每步: {eng_time/1000*1000:.2f}ms")
        
        # PyBaMM模型性能测试（如果可用）
        try:
            pybamm_config = PyBaMMConfig(model_type="SPM")
            pybamm_model = EnhancedBatteryModel(
                pybamm_config=pybamm_config,
                use_pybamm=True
            )
            
            print("测试PyBaMM SPM模型性能...")
            start_time = time.time()
            for i in range(100):  # 减少步数，因为PyBaMM较慢
                pybamm_model.update_state(100, 0.1, 25.0)
            pybamm_time = time.time() - start_time
            
            print(f"  PyBaMM SPM: 100步，耗时 {pybamm_time:.3f}秒")
            print(f"  平均每步: {pybamm_time/100*1000:.2f}ms")
            
        except Exception as e:
            print(f"  PyBaMM性能测试跳过: {e}")
        
        print("✅ 性能基准测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")
        return False

def main():
    """主函数"""
    print("PyBaMM集成测试开始")
    print("=" * 60)
    
    # 测试PyBaMM可用性
    pybamm_available = test_pybamm_availability()
    
    # 测试增强模型创建
    model = test_enhanced_model_creation()
    
    # 测试基本仿真
    if model:
        test_basic_simulation(model)
        test_mode_switching(model)
        test_visualization(model)
    
    # 测试错误处理
    test_error_handling()
    
    # 性能基准测试
    run_performance_benchmark()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    
    if pybamm_available:
        print("✅ PyBaMM集成成功，可以使用增强功能")
    else:
        print("⚠️  PyBaMM不可用，将使用工程级模型")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
