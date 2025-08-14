#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按35kV系统参数调试级联H桥（PS-PWM, fsw=1000Hz）
"""

import numpy as np
import matplotlib.pyplot as plt
from h_bridge_model import CascadedHBridgeSystem
from scipy import signal


def thd_time_domain(V, t, f_grid):
	fs = 1.0 / (t[1] - t[0])
	b, a = signal.butter(4, (2*f_grid) / (fs/2), btype='low')
	Vf = signal.filtfilt(b, a, V)
	Vh = V - Vf
	Vr = np.sqrt(np.mean(V**2))
	Vfr = np.sqrt(np.mean(Vf**2))
	Vhr = np.sqrt(np.mean(Vh**2))
	thd = (Vhr / Vfr) * 100 if Vfr > 0 else float('inf')
	return thd, Vr, Vfr, Vhr


def run_debug(N_modules=35, Vdc_per_module=1000, fsw=1000, f_grid=50, m=0.9):
	print("=== 35kV 系统调试（PS-PWM）===")
	print(f"N={N_modules}, Vdc/module={Vdc_per_module}V, fsw={fsw}Hz, f_grid={f_grid}Hz, m={m}")
	system = CascadedHBridgeSystem(N_modules, Vdc_per_module, fsw, f_grid, modulation_strategy="PS-PWM")
	# 仿真1个工频周期
	t = np.linspace(0, 1.0/f_grid, 10000)
	V_total, V_modules = system.generate_phase_shifted_pwm(t, m)
	# 统计
	Vmin, Vmax = float(np.min(V_total)), float(np.max(V_total))
	Vr = float(np.sqrt(np.mean(V_total**2)))
	thd, Vrms, Vf_rms, Vh_rms = thd_time_domain(V_total, t, f_grid)
	print(f"输出范围: [{Vmin:.0f}, {Vmax:.0f}] V  Vrms={Vr:.0f} V  THD≈{thd:.2f}%")
	# 期望峰值（近似）：N*Vdc*m
	Vpk_theory = N_modules * Vdc_per_module * m
	print(f"理论峰值≈{Vpk_theory:.0f} V ({'-' if Vmin<-0.9*Vpk_theory else ''}symmetry check)")
	# 绘图
	fig, axes = plt.subplots(3, 1, figsize=(14, 10))
	axes[0].plot(t*1000, V_total/1000, 'b-', lw=1)
	axes[0].set_title('Cascaded Output Voltage (kV)')
	axes[0].grid(True)
	# 局部放大
	idx0 = int(0.10*len(t)); idx1 = int(0.12*len(t))
	axes[1].plot(t[idx0:idx1]*1000, V_total[idx0:idx1]/1000, 'b-', lw=1)
	axes[1].set_title('Zoomed Output (kV)')
	axes[1].grid(True)
	# 基波/谐波分量
	b, a = signal.butter(4, (2*f_grid)/( (1.0/(t[1]-t[0]))/2 ), btype='low')
	Vfund = signal.filtfilt(b, a, V_total)
	axes[2].plot(t*1000, V_total/1000, alpha=0.5, label='Total')
	axes[2].plot(t*1000, Vfund/1000, label='Fundamental', lw=2)
	axes[2].legend(); axes[2].grid(True)
	plt.tight_layout(); plt.show()
	return {
		"Vmin": Vmin, "Vmax": Vmax, "Vrms": Vr, "THD": thd,
		"Vf_rms": float(Vf_rms), "Vh_rms": float(Vh_rms)
	}


if __name__ == "__main__":
	run_debug()
