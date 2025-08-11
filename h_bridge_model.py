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
        # 三角波载波
        carrier = 2 * (t * self.fsw - np.floor(t * self.fsw + 0.5))
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
    
    def calculate_switching_losses(self, I_rms, duty_cycle):
        """计算开关损耗（按参考条件做线性归一化）"""
        try:
            from device_parameters import get_optimized_parameters
            igbt_params = get_optimized_parameters()['igbt']
            Vref = float(getattr(igbt_params, 'Vces_V', 1700))
            Iref = float(getattr(igbt_params, 'Ic_dc_A', 1500))
        except Exception:
            Vref, Iref = 1700.0, 1500.0

        # 参考能量（J/次），按电流、电压线性缩放
        scale_I = float(max(0.1, min(5.0, I_rms / max(1e-6, Iref))))
        scale_V = float(max(0.5, min(1.2, self.Vdc / max(1e-6, Vref))))

        E_on_eff = self.E_on * scale_I * scale_V
        E_off_eff = self.E_off * scale_I * scale_V
        E_rr_eff = self.E_rr * scale_I * scale_V

        # 总开关损耗功率（W）
        P_sw = (E_on_eff + E_off_eff + E_rr_eff) * self.fsw
        return float(P_sw)
    
    def calculate_conduction_losses(self, I_rms, duty_cycle):
        """计算导通损耗"""
        # IGBT导通损耗
        P_cond_igbt = self.Vce_sat * I_rms * duty_cycle
        
        # 二极管导通损耗
        P_cond_diode = self.Vf * I_rms * (1 - duty_cycle)
        
        return P_cond_igbt + P_cond_diode
    
# ===================== 级联H桥建模 =====================
class CascadedHBridgeSystem:
    def __init__(self, N_modules=40, Vdc_per_module=1000, fsw=1000, f_grid=50):
        """
        级联H桥系统建模
        Args:
            N_modules: 每相模块数
            Vdc_per_module: 每模块直流电压
            fsw: 开关频率
            f_grid: 电网频率
        """
        self.N_modules = N_modules
        self.Vdc_per_module = Vdc_per_module
        self.fsw = fsw
        self.f_grid = f_grid
        
        # 创建H桥单元
        self.hbridge_units = [HBridgeUnit(Vdc_per_module, fsw, f_grid) for _ in range(N_modules)]
        
        # 系统参数
        self.V_total = N_modules * Vdc_per_module  # 总输出电压
        self.fundamental_freq = f_grid
        
    def generate_phase_shifted_pwm(self, t, modulation_index, phase_shift=0):
        """生成相移PWM信号"""
        # 计算每个模块的相移
        phase_shifts = np.linspace(0, 2*np.pi, self.N_modules, endpoint=False)
        
        # 生成每个模块的输出电压
        V_modules = []
        for i, hbridge in enumerate(self.hbridge_units):
            module_phase_shift = phase_shift + phase_shifts[i]
            V_module = hbridge.calculate_output_voltage(t, modulation_index, module_phase_shift)
            V_modules.append(V_module)
        
        # 级联叠加
        V_total = np.sum(V_modules, axis=0)
        
        return V_total, V_modules
    
    def calculate_harmonic_spectrum(self, V_output, t):
        """计算谐波频谱"""
        # 采样频率
        fs = 1.0 / (t[1] - t[0])
        
        # FFT分析
        fft_result = np.fft.fft(V_output)
        freqs = np.fft.fftfreq(len(V_output), 1/fs)
        
        # 计算幅值谱
        magnitude_spectrum = np.abs(fft_result) / len(V_output)
        
        # 只取正频率部分
        positive_freqs = freqs >= 0
        freqs_positive = freqs[positive_freqs]
        magnitude_positive = magnitude_spectrum[positive_freqs]
        
        return freqs_positive, magnitude_positive
    
    def calculate_total_losses(self, I_rms, duty_cycle=0.5):
        """计算系统总损耗"""
        total_switching_loss = 0
        total_conduction_loss = 0
        
        for hbridge in self.hbridge_units:
            # 每个模块的电流
            I_module = I_rms / self.N_modules
            
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
    
    # 系统参数 - 使用device_parameters.py中的参数
    N_modules = device_params['system'].cascaded_power_modules
    Vdc_per_module = 1000  # V - 根据系统电压计算
    fsw = device_params['system'].module_switching_frequency_Hz  # Hz
    f_grid = 50  # Hz
    
    # 创建级联H桥系统
    cascaded_system = CascadedHBridgeSystem(N_modules, Vdc_per_module, fsw, f_grid)
    
    # 仿真时间
    t = np.linspace(0, 0.02, 10000)  # 一个工频周期
    
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
    
    # 计算谐波频谱
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
    
    # 创建自适应图形
    fig, axes = create_adaptive_figure(2, 2, title='Cascaded H-Bridge System Simulation Results')
    
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
    axes[1, 0].plot(freqs, magnitude, 'r-', linewidth=2)
    format_axis_labels(axes[1, 0], 'Frequency (Hz)', 'Magnitude (V)', 'Output Voltage Harmonic Spectrum')
    axes[1, 0].set_xlim(0, 5000)  # 显示到5kHz
    add_grid(axes[1, 0])
    set_adaptive_ylim(axes[1, 0], magnitude)
    
    # 系统信息
    axes[1, 1].axis('off')
    
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

Output Characteristics:
- Number of Levels: {system.N_modules * 2 + 1}
- Voltage Resolution: {system.Vdc_per_module} V
- Max Modulation Index: 1.0"""
    
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
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
        
        # 计算THD
        fundamental_idx = np.argmin(np.abs(freqs - f_grid))
        fundamental_magnitude = magnitude[fundamental_idx]
        harmonic_power = np.sum(magnitude**2) - fundamental_magnitude**2
        thd = np.sqrt(harmonic_power) / fundamental_magnitude * 100
        
        axes[row, col].text(0.02, 0.98, f'THD: {thd:.2f}%', 
                           transform=axes[row, col].transAxes, 
                           verticalalignment='top', fontsize=10)
    
    # 使用自适应工具优化布局
    optimize_layout(fig)
    
    # 完成绘图
    finalize_plot(fig)
    
    print("PWM调制策略分析完成")

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 运行H桥系统仿真
    system, V_output, losses = simulate_hbridge_system()
    
    # 分析PWM策略
    analyze_pwm_strategies()
    
    print("\n仿真完成！") 