import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== H桥单元详细建模 =====================
class HBridgeUnit:
    def __init__(self, Vdc=1000, fsw=1000, f_grid=50, device_params=None):
        """
        H桥单元建模
        Args:
            Vdc: 直流电压 (V)
            fsw: 开关频率 (Hz)
            f_grid: 电网频率 (Hz)
            device_params: 设备参数对象
        """
        self.Vdc = Vdc
        self.fsw = fsw
        self.f_grid = f_grid
        self.Ts = 1.0 / fsw  # 开关周期
        
        # 导入设备参数
        if device_params is None:
            from device_parameters import get_optimized_parameters
            device_params = get_optimized_parameters()
        
        # IGBT参数 - 使用device_parameters.py中的参数
        self.Vce_sat = device_params['igbt'].Vce_sat_V["25C"][0]  # 饱和压降 (V) - 25°C典型值
        self.Vf = device_params['igbt'].diode_Vf_V[0]  # 二极管正向压降 (V)
        self.t_on = device_params['igbt'].switching_times_us['td_on'][0] * 1e-6  # 开通时间 (s)
        self.t_off = device_params['igbt'].switching_times_us['td_off'][0] * 1e-6  # 关断时间 (s)
        self.t_rr = device_params['igbt'].diode_Qr_uC[0] / device_params['igbt'].diode_Irm_A[0] * 1e-6  # 反向恢复时间 (s)
        
        # 损耗参数 - 基于device_parameters.py中的开关损耗
        self.E_on = device_params['igbt'].switching_energy_mJ['Eon'][0] * 1e-3  # 开通损耗 (J)
        self.E_off = device_params['igbt'].switching_energy_mJ['Eoff'][0] * 1e-3  # 关断损耗 (J)
        self.E_rr = device_params['igbt'].diode_Erec_mJ[0] * 1e-3  # 反向恢复损耗 (J)
    
    def generate_carrier_wave(self, t):
        """生成载波信号（三角波）"""
        # 修正：生成对称三角波，范围[-1, 1]
        frac = t * self.fsw - np.floor(t * self.fsw)
        carrier = 4.0 * np.abs(frac - 0.5) - 1.0
        return carrier
    
    def generate_carrier_wave_with_shift(self, t, carrier_shift):
        """生成带相移的载波信号（三角波）"""
        # 修正：载波相移按载波周期比例偏移，生成对称三角波
        # carrier_shift ∈ [0, 1)
        time_offset = float(carrier_shift) / float(self.fsw)
        shifted_t = t + time_offset
        frac = shifted_t * self.fsw - np.floor(shifted_t * self.fsw)
        carrier = 4.0 * np.abs(frac - 0.5) - 1.0
        return carrier
    
    def generate_reference_wave(self, t, modulation_index, phase_shift=0):
        """生成参考信号（正弦波）"""
        omega = 2 * np.pi * self.f_grid
        reference = modulation_index * np.sin(omega * t + phase_shift)
        return reference
    
    def pwm_comparison(self, t, modulation_index, phase_shift=0):
        """PWM比较，生成开关信号"""
        carrier = self.generate_carrier_wave(t)
        reference = self.generate_reference_wave(t, modulation_index, phase_shift)
        
        # 生成PWM信号
        pwm_positive = reference > carrier
        pwm_negative = reference < -carrier
        
        return pwm_positive, pwm_negative
    
    def calculate_output_voltage(self, t, modulation_index, phase_shift=0):
        """计算H桥输出电压"""
        pwm_pos, pwm_neg = self.pwm_comparison(t, modulation_index, phase_shift)
        
        # 输出电压计算
        V_out = np.zeros_like(t)
        V_out[pwm_pos] = self.Vdc
        V_out[pwm_neg] = -self.Vdc
        
        return V_out
    
    def calculate_output_voltage_with_carrier_shift(self, t, modulation_index, carrier_shift):
        """计算带载波相移的H桥输出电压"""
        # 修正：使用载波相移，参考波不改变
        carrier = self.generate_carrier_wave_with_shift(t, carrier_shift)
        reference = self.generate_reference_wave(t, modulation_index, 0)  # 参考波无相移
        
        # 生成PWM信号
        pwm_positive = reference > carrier
        pwm_negative = reference < -carrier
        
        # 输出电压计算
        V_out = np.zeros_like(t)
        V_out[pwm_positive] = self.Vdc
        V_out[pwm_negative] = -self.Vdc
        
        return V_out
    
    def calculate_switching_losses(self, I_rms, duty_cycle):
        """计算开关损耗（按参考条件做线性归一化）"""
        try:
            from device_parameters import get_optimized_parameters
            igbt_params = get_optimized_parameters()['igbt']
            Vref = float(getattr(igbt_params, 'Vces_V', 1700))
            Iref = float(getattr(igbt_params, 'Ic_dc_A', 1500))
        except Exception:
            Vref, Iref = 1700.0, 1500.0

        # 修正：使用更合理的缩放因子，避免过度缩小
        # 对于电流，使用平方根关系（更符合实际）
        scale_I = float(max(0.3, min(2.0, np.sqrt(I_rms / max(1e-6, Iref)))))
        # 对于电压，使用线性关系但限制最小缩放
        scale_V = float(max(0.7, min(1.5, self.Vdc / max(1e-6, Vref))))

        E_on_eff = self.E_on * scale_I * scale_V
        E_off_eff = self.E_off * scale_I * scale_V
        E_rr_eff = self.E_rr * scale_I * scale_V

        # 修正：使用更合理的开关损耗模型
        # 对于级联H桥，每个模块在每个开关周期有2个开关事件（IGBT开关）
        # 考虑二极管反向恢复损耗
        
        # 每个开关周期的开关损耗
        P_sw = 2 * (E_on_eff + E_off_eff + E_rr_eff) * self.fsw
        
        return float(P_sw)
    
    def calculate_conduction_losses(self, I_rms, duty_cycle):
        """计算导通损耗"""
        # 修正：使用更合理的导通损耗模型
        # 对于级联H桥，导通损耗应该考虑实际的电流分布
        
        # 每个模块有2个IGBT和2个二极管
        # IGBT导通损耗（考虑占空比）
        P_cond_igbt = 2 * self.Vce_sat * I_rms * duty_cycle
        
        # 二极管导通损耗（互补占空比）
        P_cond_diode = 2 * self.Vf * I_rms * (1 - duty_cycle)
        
        # 总导通损耗
        total_conduction_loss = P_cond_igbt + P_cond_diode
        
        return float(total_conduction_loss)
    
# ===================== 级联H桥建模 =====================
class CascadedHBridgeSystem:
    def __init__(self, N_modules=40, Vdc_per_module=1000, fsw=1000, f_grid=50, modulation_strategy="NLM"):
        """
        级联H桥系统建模
        Args:
            N_modules: 每相模块数
            Vdc_per_module: 每模块直流电压
            fsw: 开关频率
            f_grid: 电网频率
            modulation_strategy: 调制策略（仅支持"NLM" 最近电平调制）
        """
        self.N_modules = N_modules
        self.Vdc_per_module = Vdc_per_module
        self.fsw = fsw
        self.f_grid = f_grid
        self.modulation_strategy = "NLM"  # 强制使用NLM
        
        # 创建H桥单元
        self.hbridge_units = [HBridgeUnit(Vdc_per_module, fsw, f_grid) for _ in range(N_modules)]
        
        # 系统参数
        self.V_total = N_modules * Vdc_per_module  # 总输出电压
        self.fundamental_freq = f_grid

    def generate_phase_shifted_pwm(self, t, modulation_index, phase_shift=0):
        """生成输出电压（仅使用NLM调制策略）"""
        return self._generate_output_nlm(t, modulation_index)

    def _generate_output_nlm(self, t, modulation_index):
        """最近电平调制（Nearest Level Modulation, NLM）
        思路：目标电压 v_ref = m*sin(wt)。最近电平 L = round(N * m * sin(wt)) ∈ [-N, N]。
        每个时刻选取 |L| 个模块输出 sign(L)*Vdc，其余为0。
        """
        omega = 2 * np.pi * self.f_grid
        v_ref_norm = modulation_index * np.sin(omega * t)  # 归一化到[-m, m]
        L = np.round(self.N_modules * v_ref_norm).astype(int)  # 目标电平（模块数）
        L = np.clip(L, -self.N_modules, self.N_modules)
        
        # 为每个模块生成输出
        V_modules = [np.zeros_like(t) for _ in range(self.N_modules)]
        for idx_time in range(len(t)):
            level = L[idx_time]
            if level > 0:
                # 使前 level 个模块输出 +Vdc
                for k in range(level):
                    V_modules[k][idx_time] = +self.Vdc_per_module
            elif level < 0:
                # 使前 |level| 个模块输出 -Vdc
                for k in range(-level):
                    V_modules[k][idx_time] = -self.Vdc_per_module
            # 其余模块保持为0
        
        V_total = np.sum(V_modules, axis=0)
        return V_total, V_modules

    def calculate_harmonic_spectrum(self, V_output, t):
        """计算谐波频谱（仅用于可视化）。不要用它直接算THD。"""
        fs = 1.0 / (t[1] - t[0])
        if len(V_output) % 2 != 0:
            V_output = V_output[:-1]
            t = t[:-1]
            fs = 1.0 / (t[1] - t[0])
        window = np.blackman(len(V_output))
        V_windowed = V_output * window
        fft_length = 2**int(np.log2(len(V_windowed)) + 2)
        fft_result = np.fft.fft(V_windowed, fft_length)
        freqs = np.fft.fftfreq(fft_length, 1/fs)
        magnitude_spectrum = np.abs(fft_result)
        positive_freqs = freqs >= 0
        freqs_positive = freqs[positive_freqs]
        magnitude_positive = magnitude_spectrum[positive_freqs]
        return freqs_positive, magnitude_positive

    def calculate_thd_time_domain(self, V_output, t):
        """基于时域正交投影的精确THD计算。
        THD = sqrt(V_rms^2 - V1_rms^2) / V1_rms
        其中 V1_rms 通过与基频正交基的内积估算得到。
        """
        omega = 2.0 * np.pi * self.f_grid
        # 保证使用整数个工频周期
        T = 1.0 / self.f_grid
        t_span = t[-1] - t[0]
        cycles = max(1, int(round(t_span / T)))
        T_used = cycles * T
        mask = t - t[0] <= T_used + 1e-12
        v = V_output[mask]
        t_used = t[mask]
        # 计算总RMS
        V_rms = np.sqrt(np.mean(v**2))
        # 正交基
        sin_wt = np.sin(omega * t_used)
        cos_wt = np.cos(omega * t_used)
        # 估算基波峰值分量（A、B为正交系数）
        A = 2.0 * np.mean(v * sin_wt)
        B = 2.0 * np.mean(v * cos_wt)
        V1_peak = np.sqrt(A*A + B*B)
        V1_rms = V1_peak / np.sqrt(2.0)
        # 数值误差保护
        rest_power = max(0.0, V_rms*V_rms - V1_rms*V1_rms)
        THD = np.sqrt(rest_power) / max(1e-9, V1_rms)
        return THD
    
    def calculate_total_losses(self, I_rms, duty_cycle=0.5):
        """计算系统总损耗"""
        total_switching_loss = 0
        total_conduction_loss = 0
        
        for hbridge in self.hbridge_units:
            # 修正：每个模块承载整个系统的电流，不是电流被模块数除
            I_module = I_rms
            
            # 开关损耗
            P_sw = hbridge.calculate_switching_losses(I_module, duty_cycle)
            total_switching_loss += P_sw
            
            # 导通损耗
            P_cond = hbridge.calculate_conduction_losses(I_module, duty_cycle)
            total_conduction_loss += P_cond
        
        total_loss = total_switching_loss + total_conduction_loss
        
        return {
            'total_loss': total_loss,
            'switching_loss': total_switching_loss,
            'conduction_loss': total_conduction_loss
        }

# ===================== 仿真和可视化 =====================
def simulate_hbridge_system():
    """运行H桥系统仿真"""
    print("=== H桥系统仿真 ===")
    
    # 导入设备参数
    from device_parameters import get_optimized_parameters
    device_params = get_optimized_parameters()
    
    # 系统参数（统一与NLM测试脚本）
    N_modules = 40
    Vdc_per_module = 875  # V - 35kV / 40
    fsw = device_params['system'].module_switching_frequency_Hz  # Hz（保持1kHz）
    f_grid = 50  # Hz
    
    # 创建级联H桥系统
    cascaded_system = CascadedHBridgeSystem(N_modules, Vdc_per_module, fsw, f_grid)
    
    # 仿真时间（使用多周期以提高THD稳定性）
    t = np.linspace(0, 0.1, 5000)  # 5个工频周期，与测试脚本一致
    
    # 调制比
    modulation_index = 0.8
    
    print(f"系统参数:")
    print(f"- 模块数: {N_modules}")
    print(f"- 每模块直流电压: {Vdc_per_module} V")
    print(f"- 总输出电压: {cascaded_system.V_total} V")
    print(f"- 开关频率: {fsw} Hz")
    print(f"- 调制比: {modulation_index}")
    
    # 生成输出电压
    V_total, V_modules = cascaded_system.generate_phase_shifted_pwm(t, modulation_index)
    
    # 计算谐波频谱（仅用于可视化）
    freqs, magnitude = cascaded_system.calculate_harmonic_spectrum(V_total, t)
    
    # 计算损耗
    I_rms = 100  # 假设RMS电流
    losses = cascaded_system.calculate_total_losses(I_rms)
    
    print(f"\n损耗分析:")
    print(f"- 总损耗: {losses['total_loss']:.2f} W")
    print(f"- 开关损耗: {losses['switching_loss']:.2f} W")
    print(f"- 导通损耗: {losses['conduction_loss']:.2f} W")
    
    # 绘制结果
    plot_hbridge_results(t, V_total, V_modules, freqs, magnitude, cascaded_system)
    
    return cascaded_system, V_total, losses

def plot_hbridge_results(t, V_total, V_modules, freqs, magnitude, system):
    """绘制H桥仿真结果（自适应显示）"""
    # 导入自适应绘图工具
    from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot
    
    # 创建自适应图形 - 增加到3行4列以容纳更多图表
    fig, axes = create_adaptive_figure(3, 4, title='Cascaded H-Bridge System Comprehensive Analysis')
    
    # 第一行：基本波形分析
    # 输出电压波形
    axes[0, 0].plot(t * 1000, V_total / 1000, 'b-', linewidth=2)
    format_axis_labels(axes[0, 0], 'Time (ms)', 'Output Voltage (kV)', 'Cascaded H-Bridge Output Voltage')
    add_grid(axes[0, 0])
    set_adaptive_ylim(axes[0, 0], V_total / 1000)
    
    # 单个模块电压波形（显示前5个模块）
    for i in range(min(5, len(V_modules))):
        axes[0, 1].plot(t * 1000, V_modules[i], alpha=0.7, label=f'Module {i+1}')
    format_axis_labels(axes[0, 1], 'Time (ms)', 'Voltage (V)', 'Individual H-Bridge Module Output')
    axes[0, 1].legend()
    add_grid(axes[0, 1])
    set_adaptive_ylim(axes[0, 1], np.array(V_modules[:5]).flatten())
    
    # 谐波频谱
    axes[0, 2].plot(freqs, magnitude, 'r-', linewidth=2)
    format_axis_labels(axes[0, 2], 'Frequency (Hz)', 'Magnitude (V)', 'Output Voltage Harmonic Spectrum')
    axes[0, 2].set_xlim(0, 5000)  # 显示到5kHz
    add_grid(axes[0, 2])
    set_adaptive_ylim(axes[0, 2], magnitude)
    
    # 谐波含量分析（柱状图）
    harmonic_orders = [1, 3, 5, 7, 9, 11, 13, 15]
    harmonic_magnitudes = []
    for order in harmonic_orders:
        freq_target = order * system.f_grid
        idx = np.argmin(np.abs(freqs - freq_target))
        harmonic_magnitudes.append(magnitude[idx])
    
    axes[0, 3].bar(harmonic_orders, harmonic_magnitudes, color='orange', alpha=0.7)
    format_axis_labels(axes[0, 3], 'Harmonic Order', 'Magnitude (V)', 'Harmonic Content Analysis')
    add_grid(axes[0, 3])
    
    # 使用时域THD（与测试脚本一致）
    thd = system.calculate_thd_time_domain(V_total, t) * 100.0
    
    # 在图表上显示THD
    axes[0, 3].text(0.02, 0.98, f'THD: {thd:.2f}%', 
                     transform=axes[0, 3].transAxes, 
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # 第二行：损耗和效率分析
    # 不同电流下的损耗分析
    current_range = np.linspace(10, 200, 50)
    switching_losses = []
    conduction_losses = []
    total_losses = []
    
    for I in current_range:
        losses = system.calculate_total_losses(I)
        switching_losses.append(losses['switching_loss'])
        conduction_losses.append(losses['conduction_loss'])
        total_losses.append(losses['total_loss'])
    
    axes[1, 0].plot(current_range, switching_losses, 'r-', label='Switching Loss', linewidth=2)
    axes[1, 0].plot(current_range, conduction_losses, 'g-', label='Conduction Loss', linewidth=2)
    axes[1, 0].plot(current_range, total_losses, 'b-', label='Total Loss', linewidth=2)
    format_axis_labels(axes[1, 0], 'RMS Current (A)', 'Power Loss (W)', 'Power Loss vs Current')
    axes[1, 0].legend()
    add_grid(axes[1, 0])
    
    # 效率曲线
    power_output = current_range * system.V_total * 0.8  # 假设功率因数0.8
    efficiency = [(po / (po + tl)) * 100 for po, tl in zip(power_output, total_losses)]
    
    # 修正：功率计算应该考虑实际的RMS电压
    # 对于调制比为0.8的正弦波，RMS电压约为峰值的0.8/√2
    rms_voltage_factor = 0.8 / np.sqrt(2)  # 调制比和正弦波RMS因子的组合
    power_output = current_range * (system.V_total * rms_voltage_factor) * 0.8  # 功率因数0.8
    efficiency = [(po / (po + tl)) * 100 for po, tl in zip(power_output, total_losses)]
    
    axes[1, 1].plot(current_range, efficiency, 'purple', linewidth=2)
    format_axis_labels(axes[1, 1], 'RMS Current (A)', 'Efficiency (%)', 'System Efficiency vs Current')
    add_grid(axes[1, 1])
    set_adaptive_ylim(axes[1, 1], efficiency)
    
    # 损耗分布饼图
    losses_100A = system.calculate_total_losses(100)
    loss_labels = ['Switching Loss', 'Conduction Loss']
    loss_values = [losses_100A['switching_loss'], losses_100A['conduction_loss']]
    colors = ['red', 'green']
    
    axes[1, 2].pie(loss_values, labels=loss_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Loss Distribution at 100A')
    
    # 温度分析（基于损耗估算）
    # 假设热阻和热容
    thermal_resistance = 0.1  # K/W
    ambient_temp = 25  # °C
    
    temp_rise = [tl * thermal_resistance for tl in total_losses]
    junction_temp = [ambient_temp + tr for tr in temp_rise]
    
    axes[1, 3].plot(current_range, junction_temp, 'orange', linewidth=2)
    format_axis_labels(axes[1, 3], 'RMS Current (A)', 'Junction Temperature (°C)', 'Temperature Rise vs Current')
    add_grid(axes[1, 3])
    
    # 第三行：高级分析
    # 不同调制比下的THD分析（时域法）
    modulation_range = np.linspace(0.1, 1.0, 20)
    thd_values = []
    
    for mi in modulation_range:
        V_test, _ = system.generate_phase_shifted_pwm(t, mi)
        thd = system.calculate_thd_time_domain(V_test, t) * 100.0
        thd_values.append(thd)
    
    axes[2, 0].plot(modulation_range, thd_values, 'brown', linewidth=2)
    format_axis_labels(axes[2, 0], 'Modulation Index', 'THD (%)', 'THD vs Modulation Index')
    add_grid(axes[2, 0])
    
    # 开关频率对损耗的影响
    freq_range = np.linspace(500, 2000, 20)
    freq_losses = []
    
    for f in freq_range:
        # 创建临时系统计算损耗
        temp_system = CascadedHBridgeSystem(system.N_modules, system.Vdc_per_module, f, system.f_grid)
        losses = temp_system.calculate_total_losses(100)
        freq_losses.append(losses['total_loss'])
    
    axes[2, 1].plot(freq_range, freq_losses, 'teal', linewidth=2)
    format_axis_labels(axes[2, 1], 'Switching Frequency (Hz)', 'Total Loss (W)', 'Loss vs Switching Frequency')
    add_grid(axes[2, 1])
    
    # 模块数量对输出电压质量的影响
    module_range = [10, 20, 30, 40, 50, 60]
    quality_metrics = []
    
    for n in module_range:
        temp_system = CascadedHBridgeSystem(n, system.Vdc_per_module, system.fsw, system.f_grid)
        V_test, _ = temp_system.generate_phase_shifted_pwm(t, 0.8)
        thd = temp_system.calculate_thd_time_domain(V_test, t) * 100.0
        quality_metrics.append(100 - thd)  # 质量指标（越高越好）
    
    axes[2, 2].plot(module_range, quality_metrics, 'darkblue', linewidth=2, marker='o')
    format_axis_labels(axes[2, 2], 'Number of Modules', 'Quality Index (%)', 'Output Quality vs Module Count')
    add_grid(axes[2, 2])
    
    # 系统信息总结
    axes[2, 3].axis('off')
    
    # 设置中文字体支持
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    info_text = f"""System Configuration:
- Number of Modules: {system.N_modules}
- Voltage per Module: {system.Vdc_per_module} V
- Total Output Voltage: {system.V_total/1000:.1f} kV
- Switching Frequency: {system.fsw} Hz
- Grid Frequency: {system.f_grid} Hz

Performance Metrics:
- Number of Levels: {system.N_modules * 2 + 1}
- Voltage Resolution: {system.Vdc_per_module} V
- Max Modulation Index: 1.0
- Estimated Efficiency: {efficiency[len(efficiency)//2]:.1f}% @ {current_range[len(current_range)//2]:.0f}A"""
    
    axes[2, 3].text(0.05, 0.95, info_text, transform=axes[2, 3].transAxes, 
                   fontsize=8, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # 使用自适应工具优化布局
    optimize_layout(fig)
    
    # 完成绘图
    finalize_plot(fig)

# ===================== PWM调制策略分析 =====================
def analyze_pwm_strategies():
    """分析不同PWM调制策略"""
    print("\n=== PWM调制策略分析 ===")
    
    # 导入设备参数
    from device_parameters import get_optimized_parameters
    device_params = get_optimized_parameters()
    
    # 系统参数 - 使用device_parameters.py中的参数
    N_modules = device_params['system'].cascaded_power_modules
    Vdc_per_module = 1000
    fsw = device_params['system'].module_switching_frequency_Hz
    f_grid = 50
    
    cascaded_system = CascadedHBridgeSystem(N_modules, Vdc_per_module, fsw, f_grid)
    
    # 仿真时间
    t = np.linspace(0, 0.02, 10000)
    
    # 不同调制比
    modulation_indices = [0.5, 0.7, 0.9, 1.0]
    
    # 创建自适应图形
    from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot
    
    fig, axes = create_adaptive_figure(2, 2, title='Output Characteristics at Different Modulation Indices')
    
    for i, mi in enumerate(modulation_indices):
        row = i // 2
        col = i % 2
        
        # 生成输出电压
        V_total, _ = cascaded_system.generate_phase_shifted_pwm(t, mi)
        
        # 计算谐波频谱
        freqs, magnitude = cascaded_system.calculate_harmonic_spectrum(V_total, t)
        
        # 绘制时域波形
        axes[row, col].plot(t * 1000, V_total / 1000, 'b-', linewidth=2)
        format_axis_labels(axes[row, col], 'Time (ms)', 'Voltage (kV)', f'Modulation Index = {mi}')
        add_grid(axes[row, col])
        set_adaptive_ylim(axes[row, col], V_total / 1000)
        
        # 计算THD（使用时域方法，与测试脚本一致）
        thd = cascaded_system.calculate_thd_time_domain(V_total, t) * 100.0
        
        axes[row, col].text(0.02, 0.98, f'THD: {thd:.2f}%', 
                           transform=axes[row, col].transAxes, 
                           verticalalignment='top', fontsize=10)
    
    # 使用自适应工具优化布局
    optimize_layout(fig)
    
    # 完成绘图
    finalize_plot(fig)
    
    print("PWM调制策略分析完成")

# ===================== 3D和高级分析图表 =====================
def plot_advanced_analysis(system, t):
    """生成高级分析图表，包括3D图表和动态分析"""
    from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot
    
    # 创建3D图表
    from mpl_toolkits.mplot3d import Axes3D
    
    # 创建自适应图形 - 2行3列
    fig, axes = create_adaptive_figure(2, 3, title='Advanced H-Bridge Analysis - 3D and Dynamic Charts')
    
    # 第一行：3D分析
    # 3D损耗表面图
    current_range = np.linspace(10, 200, 20)
    freq_range = np.linspace(500, 2000, 20)
    X, Y = np.meshgrid(current_range, freq_range)
    Z = np.zeros_like(X)
    
    for i, freq in enumerate(freq_range):
        for j, current in enumerate(current_range):
            temp_system = CascadedHBridgeSystem(system.N_modules, system.Vdc_per_module, freq, system.f_grid)
            losses = temp_system.calculate_total_losses(current)
            Z[i, j] = losses['total_loss']
    
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Current (A)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_zlabel('Total Loss (W)')
    ax1.set_title('3D Loss Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 3D效率表面图
    Z_efficiency = np.zeros_like(X)
    for i, freq in enumerate(freq_range):
        for j, current in enumerate(current_range):
            power_output = current * system.V_total * 0.8
            if power_output > 0:
                Z_efficiency[i, j] = (power_output / (power_output + Z[i, j])) * 100
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_efficiency, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Current (A)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_zlabel('Efficiency (%)')
    ax2.set_title('3D Efficiency Surface')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 调制比vs频率vsTHD的3D图
    mod_range = np.linspace(0.1, 1.0, 15)
    X_mod, Y_mod = np.meshgrid(mod_range, freq_range)
    Z_thd = np.zeros_like(X_mod)
    
    for i, freq in enumerate(freq_range):
        for j, mod in enumerate(mod_range):
            temp_system = CascadedHBridgeSystem(system.N_modules, system.Vdc_per_module, freq, system.f_grid)
            V_test, _ = temp_system.generate_phase_shifted_pwm(t, mod)
            freqs_test, magnitude_test = temp_system.calculate_harmonic_spectrum(V_test, t)
            # 使用时域THD计算（与测试脚本一致）
            thd = temp_system.calculate_thd_time_domain(V_test, t) * 100.0
            Z_thd[i, j] = thd
    
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X_mod, Y_mod, Z_thd, cmap='coolwarm', alpha=0.8)
    ax3.set_xlabel('Modulation Index')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_zlabel('THD (%)')
    ax3.set_title('3D THD Analysis')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 第二行：动态分析和对比
    # 不同模块数量的性能对比
    module_counts = [10, 20, 30, 40, 50, 60]
    efficiency_comparison = []
    thd_comparison = []
    
    for n in module_counts:
        temp_system = CascadedHBridgeSystem(n, system.Vdc_per_module, system.fsw, system.f_grid)
        losses = temp_system.calculate_total_losses(100)
        
        # 修正：功率计算应该考虑实际的RMS电压
        # 对于调制比为0.8的正弦波，RMS电压约为峰值的0.8/√2
        rms_voltage_factor = 0.8 / np.sqrt(2)  # 调制比和正弦波RMS因子的组合
        total_power = 100 * (temp_system.V_total * rms_voltage_factor) * 0.8
        efficiency = (total_power / (total_power + losses['total_loss'])) * 100
        efficiency_comparison.append(efficiency)
        
        # 计算THD（使用时域方法）
        V_test, _ = temp_system.generate_phase_shifted_pwm(t, 0.8)
        thd = temp_system.calculate_thd_time_domain(V_test, t) * 100.0
        thd_comparison.append(thd)
    
    # 双Y轴图表：效率vsTHD
    ax4 = axes[1, 0]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(module_counts, efficiency_comparison, 'b-', linewidth=2, label='Efficiency')
    line2 = ax4_twin.plot(module_counts, thd_comparison, 'r-', linewidth=2, label='THD')
    
    ax4.set_xlabel('Number of Modules')
    ax4.set_ylabel('Efficiency (%)', color='b')
    ax4_twin.set_ylabel('THD (%)', color='r')
    ax4.set_title('Efficiency vs THD vs Module Count')
    ax4.grid(True)
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    # 成本效益分析
    cost_estimate = [n * 1000 for n in module_counts]  # 假设每个模块1000元
    axes[1, 1].plot(module_counts, cost_estimate, 'g-', linewidth=2, marker='o')
    format_axis_labels(axes[1, 1], 'Number of Modules', 'Estimated Cost (¥)', 'Cost vs Module Count')
    add_grid(axes[1, 1])
    
    # 性能指标雷达图
    # 计算综合性能指标
    performance_metrics = {
        'Efficiency': np.mean(efficiency_comparison) / 100,
        'THD Quality': (100 - np.mean(thd_comparison)) / 100,
        'Cost Efficiency': 1 / (np.mean(cost_estimate) / 10000),  # 归一化成本
        'Voltage Levels': min(1.0, system.N_modules / 60),  # 归一化电压等级
        'Switching Performance': min(1.0, system.fsw / 2000)  # 归一化开关性能
    }
    
    # 创建雷达图
    categories = list(performance_metrics.keys())
    values = list(performance_metrics.values())
    
    # 闭合雷达图
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax6 = fig.add_subplot(2, 3, 6, projection='polar')
    ax6.plot(angles, values, 'o-', linewidth=2, color='purple')
    ax6.fill(angles, values, alpha=0.25, color='purple')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 1)
    ax6.set_title('System Performance Radar Chart')
    
    # 使用自适应工具优化布局
    optimize_layout(fig)
    
    # 完成绘图
    finalize_plot(fig)
    
    return fig

# ===================== 实时监控仪表板 =====================
def create_monitoring_dashboard(system, t):
    """创建实时监控仪表板"""
    from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot
    
    # 创建仪表板布局
    fig, axes = create_adaptive_figure(2, 4, title='H-Bridge System Real-Time Monitoring Dashboard')
    
    # 第一行：实时状态指标
    # 当前运行状态
    axes[0, 0].axis('off')
    status_text = f"""System Status: RUNNING
Operating Time: {t[-1]:.3f} s
Grid Frequency: {system.f_grid} Hz
Switching Frequency: {system.fsw} Hz
DC Voltage: {system.Vdc_per_module} V
Module Count: {system.N_modules}"""
    
    axes[0, 0].text(0.05, 0.95, status_text, transform=axes[0, 0].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # 实时损耗监控
    current_monitoring = np.linspace(50, 150, 100)
    real_time_losses = []
    for I in current_monitoring:
        losses = system.calculate_total_losses(I)
        real_time_losses.append(losses['total_loss'])
    
    axes[0, 1].plot(current_monitoring, real_time_losses, 'r-', linewidth=2)
    format_axis_labels(axes[0, 1], 'Current (A)', 'Loss (W)', 'Real-Time Loss Monitoring')
    add_grid(axes[0, 1])
    
    # 温度监控
    thermal_resistance = 0.1
    ambient_temp = 25
    temp_rise = [tl * thermal_resistance for tl in real_time_losses]
    junction_temp = [ambient_temp + tr for tr in temp_rise]
    
    axes[0, 2].plot(current_monitoring, junction_temp, 'orange', linewidth=2)
    format_axis_labels(axes[0, 2], 'Current (A)', 'Temperature (°C)', 'Temperature Monitoring')
    add_grid(axes[0, 2])
    
    # 效率监控
    power_output = current_monitoring * system.V_total * 0.8
    efficiency = [(po / (po + tl)) * 100 for po, tl in zip(power_output, real_time_losses)]
    
    # 修正：功率计算应该考虑实际的RMS电压
    rms_voltage_factor = 0.8 / np.sqrt(2)  # 调制比和正弦波RMS因子的组合
    power_output = current_monitoring * (system.V_total * rms_voltage_factor) * 0.8
    efficiency = [(po / (po + tl)) * 100 for po, tl in zip(power_output, real_time_losses)]
    
    axes[0, 3].plot(current_monitoring, efficiency, 'purple', linewidth=2)
    format_axis_labels(axes[0, 3], 'Current (A)', 'Efficiency (%)', 'Efficiency Monitoring')
    add_grid(axes[0, 3])
    
    # 第二行：告警和趋势
    # 告警状态
    axes[1, 0].axis('off')
    
    # 检查告警条件
    warnings = []
    if max(junction_temp) > 100:
        warnings.append("⚠️ High Temperature")
    if min(efficiency) < 90:
        warnings.append("⚠️ Low Efficiency")
    if max(real_time_losses) > 5000:
        warnings.append("⚠️ High Power Loss")
    
    if not warnings:
        warnings.append("✅ All Systems Normal")
    
    warning_text = "System Alerts:\n" + "\n".join(warnings)
    color = "red" if "⚠️" in warning_text else "green"
    
    axes[1, 0].text(0.05, 0.95, warning_text, transform=axes[1, 0].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    
    # 趋势分析
    # 计算趋势（使用简单的线性回归）
    x_norm = (current_monitoring - np.mean(current_monitoring)) / np.std(current_monitoring)
    z_efficiency = np.polyfit(x_norm, efficiency, 1)
    trend_slope = z_efficiency[0]
    
    trend_color = 'green' if trend_slope > 0 else 'red'
    trend_symbol = '↗️' if trend_slope > 0 else '↘️'
    
    axes[1, 1].plot(current_monitoring, efficiency, 'purple', linewidth=2)
    axes[1, 1].plot(current_monitoring, np.polyval(z_efficiency, x_norm), '--', color=trend_color, alpha=0.7)
    format_axis_labels(axes[1, 1], 'Current (A)', 'Efficiency (%)', f'Efficiency Trend {trend_symbol}')
    add_grid(axes[1, 1])
    
    # 性能分布直方图
    axes[1, 2].hist(efficiency, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    format_axis_labels(axes[1, 2], 'Efficiency (%)', 'Frequency', 'Efficiency Distribution')
    add_grid(axes[1, 2])
    
    # 系统健康度仪表
    axes[1, 3].axis('off')
    
    # 计算综合健康度
    health_score = 0
    if max(junction_temp) <= 100:
        health_score += 25
    if min(efficiency) >= 90:
        health_score += 25
    if max(real_time_losses) <= 5000:
        health_score += 25
    if trend_slope >= 0:
        health_score += 25
    
    health_color = 'green' if health_score >= 75 else 'orange' if health_score >= 50 else 'red'
    
    health_text = f"""System Health Score:
{health_score}/100

Health Status: {'Excellent' if health_score >= 75 else 'Good' if health_score >= 50 else 'Warning'}"""
    
    axes[1, 3].text(0.05, 0.95, health_text, transform=axes[1, 3].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=health_color, alpha=0.8))
    
    # 使用自适应工具优化布局
    optimize_layout(fig)
    
    # 完成绘图
    finalize_plot(fig)
    
    return fig

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 运行H桥系统仿真
    system, V_output, losses = simulate_hbridge_system()
    
    # 分析PWM策略
    analyze_pwm_strategies()
    
    # 生成高级分析图表
    print("\n正在生成高级分析图表...")
    t_analysis = np.linspace(0, 0.02, 10000)
    advanced_fig = plot_advanced_analysis(system, t_analysis)
    
    # 创建实时监控仪表板
    print("正在创建实时监控仪表板...")
    monitoring_fig = create_monitoring_dashboard(system, t_analysis)
    
    print("\n仿真完成！")
    print("已生成以下图表：")
    print("1. 基础仿真结果（12个子图）")
    print("2. PWM策略分析（4个子图）")
    print("3. 高级分析图表（6个子图，包含3D图表）")
    print("4. 实时监控仪表板（8个子图）")
    print("总计：30个分析图表") 