/**
 * 构网型级联储能PCS嵌入式控制器
 * 实现关键算法在DSP/ARM平台上的运行
 * 包含寿命预测、健康度评价、优化控制等核心功能
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// 系统配置参数
#define MAX_MODULES_PER_PHASE 40
#define TOTAL_PHASES 3
#define MAX_TOTAL_MODULES (MAX_MODULES_PER_PHASE * TOTAL_PHASES)
#define MAX_TEMPERATURE_HISTORY 1000
#define MAX_TIME_HISTORY 1000

// 数据类型定义
typedef struct {
    float temperature;
    float current;
    float voltage;
    float ripple_current;
    float time;
} ModuleConditions;

typedef struct {
    float remaining_life;
    float total_consumption;
    int thermal_cycles;
    float max_cycle_amplitude;
    float avg_cycle_amplitude;
} IGBTLifeStatus;

typedef struct {
    float remaining_life;
    float total_consumption;
    float max_temperature;
    float max_voltage;
    float max_ripple_current;
} CapacitorLifeStatus;

typedef struct {
    float overall_health;
    float igbt_health;
    float capacitor_health;
    int module_count;
    int critical_modules[MAX_TOTAL_MODULES];
    int critical_count;
} SystemHealthStatus;

typedef struct {
    float power_profile[24];
    float switching_frequency;
    float modulation_index;
    float switching_reduction;
    float optimization_score;
} OptimizationResult;

// 全局变量
static ModuleConditions module_conditions[MAX_TOTAL_MODULES];
static IGBTLifeStatus igbt_life_status[MAX_TOTAL_MODULES];
static CapacitorLifeStatus cap_life_status[MAX_TOTAL_MODULES];
static SystemHealthStatus system_health;
static OptimizationResult optimization_result;

// 系统参数
static const float SYSTEM_RATED_POWER = 25.0e6f;  // 25 MW
static const float SYSTEM_RATED_VOLTAGE = 35.0e3f;  // 35 kV
static const float SYSTEM_RATED_CURRENT = 420.0f;  // 420 A
static const float IGBT_BASE_LIFETIME = 100000.0f;  // 小时
static const float CAP_BASE_LIFETIME = 100000.0f;  // 小时

// 函数声明
void initialize_system(void);
void update_module_conditions(int module_id, float temp, float current, 
                            float voltage, float ripple, float time);
float calculate_igbt_life_consumption(int module_id, float time_step);
float calculate_capacitor_life_consumption(int module_id, float time_step);
void update_system_health(void);
void identify_critical_modules(void);
OptimizationResult optimize_system_operation(float current_power, 
                                           float battery_soc, 
                                           SystemHealthStatus health);
float calculate_health_score(int module_id);
void apply_optimization_control(OptimizationResult opt_result);

/**
 * 系统初始化
 */
void initialize_system(void) {
    int i;
    
    // 初始化模块条件
    for (i = 0; i < MAX_TOTAL_MODULES; i++) {
        module_conditions[i].temperature = 25.0f;
        module_conditions[i].current = 0.0f;
        module_conditions[i].voltage = 0.0f;
        module_conditions[i].ripple_current = 0.0f;
        module_conditions[i].time = 0.0f;
        
        // 初始化寿命状态
        igbt_life_status[i].remaining_life = 1.0f;
        igbt_life_status[i].total_consumption = 0.0f;
        igbt_life_status[i].thermal_cycles = 0;
        igbt_life_status[i].max_cycle_amplitude = 0.0f;
        igbt_life_status[i].avg_cycle_amplitude = 0.0f;
        
        cap_life_status[i].remaining_life = 1.0f;
        cap_life_status[i].total_consumption = 0.0f;
        cap_life_status[i].max_temperature = 25.0f;
        cap_life_status[i].max_voltage = 0.0f;
        cap_life_status[i].max_ripple_current = 0.0f;
    }
    
    // 初始化系统健康状态
    system_health.overall_health = 100.0f;
    system_health.igbt_health = 100.0f;
    system_health.capacitor_health = 100.0f;
    system_health.module_count = MAX_TOTAL_MODULES;
    system_health.critical_count = 0;
    
    // 初始化优化结果
    memset(&optimization_result, 0, sizeof(OptimizationResult));
    optimization_result.switching_frequency = 1000.0f;
    optimization_result.modulation_index = 0.8f;
    
    printf("系统初始化完成\n");
}

/**
 * 更新模块运行条件
 */
void update_module_conditions(int module_id, float temp, float current, 
                            float voltage, float ripple, float time) {
    if (module_id < 0 || module_id >= MAX_TOTAL_MODULES) {
        return;
    }
    
    module_conditions[module_id].temperature = temp;
    module_conditions[module_id].current = current;
    module_conditions[module_id].voltage = voltage;
    module_conditions[module_id].ripple_current = ripple;
    module_conditions[module_id].time = time;
    
    // 更新电容最大值记录
    if (temp > cap_life_status[module_id].max_temperature) {
        cap_life_status[module_id].max_temperature = temp;
    }
    if (voltage > cap_life_status[module_id].max_voltage) {
        cap_life_status[module_id].max_voltage = voltage;
    }
    if (ripple > cap_life_status[module_id].max_ripple_current) {
        cap_life_status[module_id].max_ripple_current = ripple;
    }
}

/**
 * 计算IGBT寿命消耗
 */
float calculate_igbt_life_consumption(int module_id, float time_step) {
    if (module_id < 0 || module_id >= MAX_TOTAL_MODULES) {
        return 0.0f;
    }
    
    ModuleConditions *cond = &module_conditions[module_id];
    IGBTLifeStatus *status = &igbt_life_status[module_id];
    
    // 基于温度的寿命消耗
    float temp_factor = 1.0f;
    if (cond->temperature > 125.0f) {
        temp_factor = powf(2.0f, (cond->temperature - 125.0f) / 10.0f);
    } else if (cond->temperature > 100.0f) {
        temp_factor = powf(1.5f, (cond->temperature - 100.0f) / 25.0f);
    }
    
    // 基于电流应力的寿命消耗
    float current_factor = 1.0f;
    float current_stress = fabsf(cond->current) / 1500.0f;
    if (current_stress > 1.0f) {
        current_factor = 1.0f + 0.2f * (current_stress - 1.0f);
    }
    
    // 计算寿命消耗
    float life_consumption = (time_step / IGBT_BASE_LIFETIME) * temp_factor * current_factor;
    
    // 更新状态
    status->total_consumption += life_consumption;
    status->remaining_life = fmaxf(0.0f, 1.0f - status->total_consumption);
    
    return life_consumption;
}

/**
 * 计算电容寿命消耗
 */
float calculate_capacitor_life_consumption(int module_id, float time_step) {
    if (module_id < 0 || module_id >= MAX_TOTAL_MODULES) {
        return 0.0f;
    }
    
    ModuleConditions *cond = &module_conditions[module_id];
    CapacitorLifeStatus *status = &cap_life_status[module_id];
    
    // 基于温度的寿命消耗
    float temp_factor = 1.0f;
    if (cond->temperature > 70.0f) {
        temp_factor = powf(2.0f, (cond->temperature - 70.0f) / 10.0f);
    } else if (cond->temperature > 50.0f) {
        temp_factor = powf(1.5f, (cond->temperature - 50.0f) / 20.0f);
    }
    
    // 基于电压应力的寿命消耗
    float voltage_factor = 1.0f;
    float voltage_stress = cond->voltage / 1200.0f;
    if (voltage_stress > 0.8f) {
        voltage_factor = 1.0f + 0.5f * (voltage_stress - 0.8f) / 0.2f;
    }
    
    // 基于纹波电流的寿命消耗
    float ripple_factor = 1.0f;
    float ripple_stress = cond->ripple_current / 80.0f;
    if (ripple_stress > 0.8f) {
        ripple_factor = 1.0f + 0.3f * (ripple_stress - 0.8f) / 0.2f;
    }
    
    // 计算寿命消耗
    float life_consumption = (time_step / CAP_BASE_LIFETIME) * temp_factor * voltage_factor * ripple_factor;
    
    // 更新状态
    status->total_consumption += life_consumption;
    status->remaining_life = fmaxf(0.0f, 1.0f - status->total_consumption);
    
    return life_consumption;
}

/**
 * 计算模块健康度得分
 */
float calculate_health_score(int module_id) {
    if (module_id < 0 || module_id >= MAX_TOTAL_MODULES) {
        return 0.0f;
    }
    
    // 取IGBT和电容健康度的最小值
    float igbt_health = igbt_life_status[module_id].remaining_life * 100.0f;
    float cap_health = cap_life_status[module_id].remaining_life * 100.0f;
    
    return fminf(igbt_health, cap_health);
}

/**
 * 更新系统健康状态
 */
void update_system_health(void) {
    int i;
    float total_igbt_health = 0.0f;
    float total_cap_health = 0.0f;
    
    for (i = 0; i < MAX_TOTAL_MODULES; i++) {
        total_igbt_health += igbt_life_status[i].remaining_life;
        total_cap_health += cap_life_status[i].remaining_life;
    }
    
    // 计算平均健康度
    system_health.igbt_health = (total_igbt_health / MAX_TOTAL_MODULES) * 100.0f;
    system_health.capacitor_health = (total_cap_health / MAX_TOTAL_MODULES) * 100.0f;
    
    // 系统整体健康度取最小值
    system_health.overall_health = fminf(system_health.igbt_health, 
                                        system_health.capacitor_health);
    
    // 识别关键模块
    identify_critical_modules();
}

/**
 * 识别关键模块（寿命消耗最严重的）
 */
void identify_critical_modules(void) {
    int i;
    system_health.critical_count = 0;
    
    for (i = 0; i < MAX_TOTAL_MODULES; i++) {
        float module_health = calculate_health_score(i);
        
        // 如果健康度低于20%，标记为关键模块
        if (module_health < 20.0f) {
            if (system_health.critical_count < MAX_TOTAL_MODULES) {
                system_health.critical_modules[system_health.critical_count] = i;
                system_health.critical_count++;
            }
        }
    }
}

/**
 * 优化系统运行
 */
OptimizationResult optimize_system_operation(float current_power, 
                                           float battery_soc, 
                                           SystemHealthStatus health) {
    OptimizationResult result;
    memset(&result, 0, sizeof(OptimizationResult));
    
    // 基于健康状态调整优化策略
    float health_factor = health.overall_health / 100.0f;
    
    // 功率曲线优化
    if (health_factor < 0.5f) {
        // 保守策略：优先保护设备
        for (int i = 0; i < 24; i++) {
            if (i < 6) {  // 夜间充电
                result.power_profile[i] = fminf(5.0e6f, 10.0e6f * (1.0f - battery_soc));
            } else if (i < 18) {  // 日间放电
                result.power_profile[i] = -fminf(8.0e6f, fabsf(current_power));
            } else {  // 晚间充电
                result.power_profile[i] = fminf(6.0e6f, 10.0e6f * (1.0f - battery_soc));
            }
        }
    } else if (health_factor < 0.8f) {
        // 平衡策略
        for (int i = 0; i < 24; i++) {
            if (i < 6) {
                result.power_profile[i] = fminf(8.0e6f, 15.0e6f * (1.0f - battery_soc));
            } else if (i < 18) {
                result.power_profile[i] = -fminf(12.0e6f, fabsf(current_power));
            } else {
                result.power_profile[i] = fminf(10.0e6f, 15.0e6f * (1.0f - battery_soc));
            }
        }
    } else {
        // 激进策略
        for (int i = 0; i < 24; i++) {
            if (i < 6) {
                result.power_profile[i] = fminf(12.0e6f, 20.0e6f * (1.0f - battery_soc));
            } else if (i < 18) {
                result.power_profile[i] = -fminf(18.0e6f, fabsf(current_power));
            } else {
                result.power_profile[i] = fminf(15.0e6f, 20.0e6f * (1.0f - battery_soc));
            }
        }
    }
    
    // 开关频率优化
    if (health_factor < 0.6f) {
        result.switching_frequency = 700.0f;  // 降低30%
    } else if (health_factor < 0.8f) {
        result.switching_frequency = 850.0f;  // 降低15%
    } else {
        result.switching_frequency = 1000.0f;  // 标准频率
    }
    
    // 调制策略优化
    if (health_factor < 0.6f) {
        result.modulation_index = 0.6f;
        result.switching_reduction = 0.3f;
    } else if (health_factor < 0.8f) {
        result.modulation_index = 0.75f;
        result.switching_reduction = 0.15f;
    } else {
        result.modulation_index = 0.9f;
        result.switching_reduction = 0.0f;
    }
    
    // 计算优化得分
    result.optimization_score = health_factor * 100.0f;
    
    return result;
}

/**
 * 应用优化控制
 */
void apply_optimization_control(OptimizationResult opt_result) {
    // 更新全局优化结果
    memcpy(&optimization_result, &opt_result, sizeof(OptimizationResult));
    
    // 这里可以添加实际的硬件控制代码
    // 例如：设置PWM频率、调制比等
    
    printf("优化控制已应用:\n");
    printf("  开关频率: %.0f Hz\n", opt_result.switching_frequency);
    printf("  调制指数: %.2f\n", opt_result.modulation_index);
    printf("  开关减少: %.1f%%\n", opt_result.switching_reduction * 100.0f);
    printf("  优化得分: %.1f\n", opt_result.optimization_score);
}

/**
 * 主控制循环
 */
void main_control_loop(void) {
    int module_id;
    float time_step = 1.0f;  // 1秒控制周期
    float current_time = 0.0f;
    
    printf("开始主控制循环...\n");
    
    while (1) {
        // 模拟更新所有模块的运行条件
        for (module_id = 0; module_id < MAX_TOTAL_MODULES; module_id++) {
            // 这里应该从实际传感器读取数据
            float temp = 25.0f + 10.0f * sinf(current_time / 3600.0f);  // 模拟温度变化
            float current = SYSTEM_RATED_CURRENT / MAX_TOTAL_MODULES;
            float voltage = SYSTEM_RATED_VOLTAGE / MAX_TOTAL_MODULES;
            float ripple = current * 0.1f;
            
            // 更新模块条件
            update_module_conditions(module_id, temp, current, voltage, ripple, current_time);
            
            // 计算寿命消耗
            calculate_igbt_life_consumption(module_id, time_step);
            calculate_capacitor_life_consumption(module_id, time_step);
        }
        
        // 更新系统健康状态
        update_system_health();
        
        // 每小时执行一次优化
        if (fmodf(current_time, 3600.0f) < time_step) {
            // 执行优化
            OptimizationResult opt_result = optimize_system_operation(
                15.0e6f,  // 当前功率
                0.5f,     // 电池SOC
                system_health
            );
            
            // 应用优化结果
            apply_optimization_control(opt_result);
            
            // 打印状态
            printf("时间: %.1f小时, 系统健康度: %.1f%%, 关键模块数: %d\n",
                   current_time / 3600.0f, system_health.overall_health, 
                   system_health.critical_count);
        }
        
        // 时间推进
        current_time += time_step;
        
        // 控制延时（实际应用中应该使用定时器）
        // delay_ms(1000);
    }
}

/**
 * 主函数
 */
int main(void) {
    printf("构网型级联储能PCS嵌入式控制器启动\n");
    printf("系统规格: %d MW, %d kV, %d A\n", 
           (int)(SYSTEM_RATED_POWER/1e6), 
           (int)(SYSTEM_RATED_VOLTAGE/1e3), 
           (int)SYSTEM_RATED_CURRENT);
    printf("模块配置: %d相 x %d模块 = %d总模块\n", 
           TOTAL_PHASES, MAX_MODULES_PER_PHASE, MAX_TOTAL_MODULES);
    
    // 初始化系统
    initialize_system();
    
    // 启动主控制循环
    main_control_loop();
    
    return 0;
}
