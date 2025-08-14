#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两种PWM调制策略（PS-PWM 与 NLM）在相同参数下的输出与THD
"""

import numpy as np
from h_bridge_model import CascadedHBridgeSystem


def compute_thd_time_domain(V, t, f_grid):
	"""使用低通法估计THD（简便近似）"""
	from scipy import signal
	fs = 1.0 / (t[1] - t[0])
	cutoff = 2 * f_grid
	nyq = fs / 2
	b, a = signal.butter(4, cutoff / nyq, btype='low')
	V_fund = signal.filtfilt(b, a, V)
	V_h = V - V_fund
	Vr = np.sqrt(np.mean(V**2))
	Vf = np.sqrt(np.mean(V_fund**2))
	Vh = np.sqrt(np.mean(V_h**2))
	thd = (Vh / Vf) * 100 if Vf > 0 else float('inf')
	return thd, Vr, Vf, Vh


def compute_thd_fft(V, t, f_grid):
	"""FFT方法计算THD（更稳健）"""
	fs = 1.0 / (t[1] - t[0])
	if len(V) % 2 == 1:
		V = V[:-1]
		t = t[:-1]
	window = np.blackman(len(V))
	Vw = V * window
	n = 2 ** int(np.log2(len(Vw)) + 2)
	Y = np.fft.fft(Vw, n)
	f = np.fft.fftfreq(n, 1 / fs)
	pos = f >= 0
	f = f[pos]
	Y = np.abs(Y[pos])
	# 窗补偿
	Y = Y * (2.0 / np.sum(window))
	# 基波索引
	k1 = np.argmin(np.abs(f - f_grid))
	A1 = Y[k1]
	power_total = np.sum(Y**2)
	power_fund = A1**2
	power_harm = max(0.0, power_total - power_fund)
	thd = np.sqrt(power_harm) / A1 * 100 if A1 > 0 else float('inf')
	return thd, f, Y


def compare():
	N = 5
	Vdc = 1000
	fsw = 1000
	f_grid = 50
	t = np.linspace(0, 0.02, 4000)
	m = 0.8

	# PS-PWM
	sys_ps = CascadedHBridgeSystem(N, Vdc, fsw, f_grid, modulation_strategy="PS-PWM")
	V_ps, _ = sys_ps.generate_phase_shifted_pwm(t, m)
	thd_ps_fft, _, _ = compute_thd_fft(V_ps, t, f_grid)
	thd_ps_td, Vr_ps, Vf_ps, Vh_ps = compute_thd_time_domain(V_ps, t, f_grid)

	# NLM
	sys_nlm = CascadedHBridgeSystem(N, Vdc, fsw, f_grid, modulation_strategy="NLM")
	V_nlm, _ = sys_nlm.generate_phase_shifted_pwm(t, m)
	thd_nlm_fft, _, _ = compute_thd_fft(V_nlm, t, f_grid)
	thd_nlm_td, Vr_nlm, Vf_nlm, Vh_nlm = compute_thd_time_domain(V_nlm, t, f_grid)

	print("=== PWM策略对比 @ fsw=1000 Hz, m=0.8 ===")
	print(f"PS-PWM:  THD(FFT)={thd_ps_fft:.2f}%  THD(LP)={thd_ps_td:.2f}%  Vrms={Vr_ps:.1f}V  Vf_rms={Vf_ps:.1f}V")
	print(f"NLM:     THD(FFT)={thd_nlm_fft:.2f}%  THD(LP)={thd_nlm_td:.2f}%  Vrms={Vr_nlm:.1f}V  Vf_rms={Vf_nlm:.1f}V")

	# 返回便于上层使用
	return {
		"PS-PWM": {"THD_FFT": thd_ps_fft, "THD_LP": thd_ps_td, "Vrms": Vr_ps, "Vf_rms": Vf_ps},
		"NLM": {"THD_FFT": thd_nlm_fft, "THD_LP": thd_nlm_td, "Vrms": Vr_nlm, "Vf_rms": Vf_nlm},
	}


if __name__ == "__main__":
	compare()

