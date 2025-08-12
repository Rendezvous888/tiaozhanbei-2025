#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断仿真只运行一步的问题
"""

import numpy as np
import traceback
from load_profile import generate_profiles
from pcs_simulation_model import PCSSimulation

def debug_simulation_steps():
    """诊断仿真步骤问题"""
    print("=== 仿真步骤诊断 ===")
    
    try:
        # 1. 检查负载曲线
        print("\n1. 检查负载曲线...")
        P_profile, T_amb = generate_profiles('summer-weekday', 60)
        t = np.arange(len(P_profile)) * (60 / 3600.0)  # 小时
        
        print(f"  时间向量长度: {len(t)}")
        print(f"  功率向量长度: {len(P_profile)}")
        print(f"  温度向量长度: {len(T_amb)}")
        print(f"  时间范围: {t[0]:.3f} ~ {t[-1]:.3f} 小时")
        print(f"  功率范围: {P_profile.min()/1e6:.2f} ~ {P_profile.max()/1e6:.2f} MW")
        
        # 2. 创建仿真系统
        print("\n2. 创建仿真系统...")
        pcs_sim = PCSSimulation()
        print(f"  时间步长: {pcs_sim.time_step_seconds} 秒")
        print(f"  电池模块: {pcs_sim.battery_module is not None}")
        print(f"  级联系统: {pcs_sim.cascaded_system is not None}")
        
        # 3. 运行仿真并监控
        print("\n3. 运行仿真...")
        print("  开始仿真循环...")
        
        # 手动运行仿真循环来诊断
        results = manual_simulation_loop(pcs_sim, t, P_profile, T_amb)
        
        print(f"\n4. 仿真结果分析...")
        print(f"  实际运行步数: {len(results['SOC_history'])}")
        print(f"  预期步数: {len(P_profile)}")
        print(f"  时间历史长度: {len(results['time'])}")
        
        if len(results['SOC_history']) == 1:
            print("  ⚠️  问题确认：仿真只运行了1步！")
            print("  可能原因：")
            print("    - 电池SOC越界导致提前退出")
            print("    - 安全状态检查失败")
            print("    - 异常处理导致循环中断")
        else:
            print(f"  ✅ 仿真正常运行，步数: {len(results['SOC_history'])}")
            
    except Exception as e:
        print(f"❌ 诊断过程中出现错误: {e}")
        traceback.print_exc()

def manual_simulation_loop(pcs_sim, t, P_profile, T_amb):
    """手动运行仿真循环来诊断问题"""
    default_dt_h = pcs_sim.time_step_seconds / 3600.0
    dt = t[1] - t[0] if len(t) > 1 else default_dt_h
    dt_seconds = float(dt) * 3600.0
    
    # 初始化结果数组
    Tj_history = []
    Tc_history = []
    P_loss_history = []
    SOC_history = []
    efficiency_history = []
    I_rms_history = []
    Tamb_history = []
    time_history = []
    
    print(f"    时间步长: {dt:.6f} 小时 ({dt_seconds:.1f} 秒)")
    print(f"    总步数: {len(P_profile)}")
    
    for i, P_cmd in enumerate(P_profile):
        try:
            if i < 5 or i % 100 == 0:  # 打印前5步和每100步的信息
                print(f"    步骤 {i+1}: P_cmd={P_cmd/1e6:.2f} MW")
            
            # 检查电池状态
            if hasattr(pcs_sim, 'battery_module') and pcs_sim.battery_module is not None:
                soc_now = pcs_sim.battery_module.state_of_charge
                print(f"      当前SOC: {soc_now:.4f}")
            else:
                soc_now = getattr(pcs_sim.battery, 'SOC', 0.5)
                print(f"      当前SOC: {soc_now:.4f}")
            
            # 基于SOC的功率裁剪
            P_out = float(P_cmd)
            margin = 0.02
            if soc_now <= (pcs_sim.params.SOC_min + margin) and P_out > 0:
                scale = max(0.0, (soc_now - pcs_sim.params.SOC_min) / margin)
                P_out *= scale
                print(f"      SOC过低，功率缩放: {scale:.4f}")
            if soc_now >= (pcs_sim.params.SOC_max - margin) and P_out < 0:
                scale = max(0.0, (pcs_sim.params.SOC_max - soc_now) / margin)
                P_out *= scale
                print(f"      SOC过高，功率缩放: {scale:.4f}")
            
            # 计算电流
            I_rms = abs(P_out) / (np.sqrt(3) * pcs_sim.params.V_grid) if pcs_sim.params.V_grid > 0 else 0.0
            
            # 计算损耗
            if pcs_sim.cascaded_system is not None:
                losses = pcs_sim.cascaded_system.calculate_total_losses(I_rms)
                P_loss_conv = float(losses['total_loss'])
            else:
                if P_out > 0:
                    P_loss_conv, _, _, _ = pcs_sim.hbridge.calculate_total_losses(P_out, 'discharge')
                else:
                    P_loss_conv, _, _, _ = pcs_sim.hbridge.calculate_total_losses(abs(P_out), 'charge')
            
            # 系统损耗
            P_loss_misc = abs(P_out) * float(pcs_sim.params.misc_loss_fraction) + float(pcs_sim.params.aux_loss_w)
            P_loss = P_loss_conv + P_loss_misc
            
            # 更新温度
            pcs_sim.params.T_amb = float(T_amb[i])
            Tj, Tc = pcs_sim.thermal.update_temperature(P_loss, dt)
            
            # 更新电池
            # 计算电池电流
            if bool(getattr(pcs_sim.params, 'soc_from_grid_power', False)):
                P_batt_abs = abs(P_out)
            else:
                P_batt_abs = abs(P_out) + P_loss
            
            if pcs_sim.battery_module is not None:
                if P_out >= 0:
                    signed_current = + P_batt_abs / max(1e-6, pcs_sim.params.V_battery)
                else:
                    signed_current = - P_batt_abs / max(1e-6, pcs_sim.params.V_battery)
                
                print(f"      电池电流: {signed_current:.2f} A")
                pcs_sim.battery_module.update_state(float(signed_current), dt_seconds, float(T_amb[i]))
                SOC = float(pcs_sim.battery_module.state_of_charge)
            else:
                P_charge = (+P_batt_abs if P_out < 0 else -P_batt_abs)
                SOC = pcs_sim.battery.update_soc(P_charge, dt)
            
            # 计算效率
            if abs(P_out) > 1e-3:
                efficiency = abs(P_out) / max(1e-3, P_batt_abs)
            else:
                efficiency = np.nan
            
            # 记录历史
            Tj_history.append(Tj)
            Tc_history.append(Tc)
            P_loss_history.append(P_loss)
            SOC_history.append(SOC)
            efficiency_history.append(efficiency)
            I_rms_history.append(I_rms)
            Tamb_history.append(pcs_sim.params.T_amb)
            time_history.append(t[i])
            
            # 检查安全状态
            if hasattr(pcs_sim.battery_module, 'safety_status') and pcs_sim.battery_module is not None:
                if any(pcs_sim.battery_module.safety_status.values()):
                    print(f"      ⚠️  安全状态触发，停止仿真")
                    break
            
            if i >= 10:  # 只运行前10步来诊断
                print(f"      完成前10步诊断，停止仿真")
                break
                
        except Exception as e:
            print(f"      ❌ 步骤 {i+1} 出现错误: {e}")
            traceback.print_exc()
            break
    
    return {
        'time': time_history,
        'power': P_profile[:len(SOC_history)],
        'SOC_history': SOC_history,
        'Tj_history': Tj_history,
        'Tc_history': Tc_history,
        'P_loss_history': P_loss_history,
        'efficiency_history': efficiency_history,
        'I_rms_history': I_rms_history,
        'Tamb_history': Tamb_history
    }

if __name__ == "__main__":
    debug_simulation_steps()
