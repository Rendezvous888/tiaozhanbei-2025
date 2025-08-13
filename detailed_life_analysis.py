#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细寿命分析模块
提供IGBT和电容器的详细寿命分析功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from plot_utils import create_adaptive_figure, optimize_layout, set_adaptive_ylim, format_axis_labels, add_grid, finalize_plot

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class DetailedLifeAnalysis:
    """详细寿命分析类"""
    
    def __init__(self):
        # 从之前的仿真结果加载数据
        self.results_file = 'result/长期寿命分析结果_20250801_113525.csv'
        self.results = pd.read_csv(self.results_file)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def analyze_life_trends(self):
        """分析寿命趋势"""
        print("=" * 60)
        print("详细寿命趋势分析")
        print("=" * 60)
        
        # 按负载类型分组分析
        load_types = self.results['load_type'].unique()
        
        for load_type in load_types:
            data = self.results[self.results['load_type'] == load_type]
            print(f"\n{load_type}负载工况分析:")
            print(f"  1年运行: IGBT剩余寿命 {data[data['years']==1]['igbt_remaining_life'].iloc[0]:.1f}%")
            print(f"  5年运行: IGBT剩余寿命 {data[data['years']==5]['igbt_remaining_life'].iloc[0]:.1f}%")
            print(f"  10年运行: IGBT剩余寿命 {data[data['years']==10]['igbt_remaining_life'].iloc[0]:.1f}%")
            
            # 计算年化寿命消耗率
            if len(data) > 1:
                first_year = data[data['years']==1]['igbt_life_consumption'].iloc[0]
                ten_year = data[data['years']==10]['igbt_life_consumption'].iloc[0]
                annual_rate = (ten_year - first_year) / 9  # 9年间的平均年化率
                print(f"  年化寿命消耗率: {annual_rate*100:.2f}%/年")
    
    def calculate_maintenance_schedule(self):
        """计算维护计划"""
        print("\n" + "=" * 60)
        print("维护计划建议")
        print("=" * 60)
        
        maintenance_schedule = []
        
        for _, row in self.results.iterrows():
            years = row['years']
            load_type = row['load_type']
            igbt_life = row['igbt_remaining_life']
            cap_life = row['capacitor_remaining_life']
            
            # 根据剩余寿命确定维护建议
            if igbt_life < 20:
                maintenance = "立即更换IGBT"
                urgency = "紧急"
            elif igbt_life < 50:
                maintenance = "计划更换IGBT"
                urgency = "高"
            elif igbt_life < 80:
                maintenance = "加强监测"
                urgency = "中"
            else:
                maintenance = "正常维护"
                urgency = "低"
            
            maintenance_schedule.append({
                'years': years,
                'load_type': load_type,
                'igbt_life': igbt_life,
                'cap_life': cap_life,
                'maintenance': maintenance,
                'urgency': urgency
            })
        
        maintenance_df = pd.DataFrame(maintenance_schedule)
        
        # 显示关键维护节点
        critical_points = maintenance_df[maintenance_df['urgency'].isin(['紧急', '高'])]
        if not critical_points.empty:
            print("关键维护节点:")
            for _, point in critical_points.iterrows():
                print(f"  {point['years']}年 {point['load_type']}负载: {point['maintenance']} ({point['urgency']}优先级)")
        
        return maintenance_df
    
    def plot_detailed_analysis(self):
        """绘制详细分析图表"""
        # 使用自适应绘图工具创建图形
        fig, axes = create_adaptive_figure(3, 3, title='IGBT和电容器详细寿命分析', title_size=16)
        
        # 1. 寿命趋势对比
        ax1 = axes[0, 0]
        for load_type in self.results['load_type'].unique():
            data = self.results[self.results['load_type'] == load_type]
            ax1.plot(data['years'], data['igbt_remaining_life'], 'o-', 
                    label=f'IGBT-{load_type}', linewidth=2, markersize=6)
            ax1.plot(data['years'], data['capacitor_remaining_life'], 's--', 
                    label=f'电容-{load_type}', linewidth=2, markersize=6)
        format_axis_labels(ax1, '运行年数', '剩余寿命 (%)', 'IGBT和电容剩余寿命趋势')
        add_grid(ax1, alpha=0.3)
        ax1.legend(fontsize=8, loc='best')
        set_adaptive_ylim(ax1, np.concatenate([
            self.results['igbt_remaining_life'], 
            self.results['capacitor_remaining_life']
        ]))
        
        # 2. 温度分析
        ax2 = axes[0, 1]
        for load_type in self.results['load_type'].unique():
            data = self.results[self.results['load_type'] == load_type]
            ax2.plot(data['years'], data['avg_igbt_temp'], 'o-', 
                    label=f'IGBT-{load_type}', linewidth=2, markersize=6)
        format_axis_labels(ax2, '运行年数', '平均温度 (°C)', 'IGBT平均工作温度')
        add_grid(ax2, alpha=0.3)
        ax2.legend(fontsize=8, loc='best')
        set_adaptive_ylim(ax2, self.results['avg_igbt_temp'])
        
        # 3. 寿命消耗率
        ax3 = axes[0, 2]
        for load_type in self.results['load_type'].unique():
            data = self.results[self.results['load_type'] == load_type]
            ax3.plot(data['years'], data['igbt_life_consumption']*100, 'o-', 
                    label=f'{load_type}负载', linewidth=2, markersize=6)
        format_axis_labels(ax3, '运行年数', '寿命消耗率 (%)', 'IGBT寿命消耗率')
        add_grid(ax3, alpha=0.3)
        ax3.legend(fontsize=8, loc='best')
        set_adaptive_ylim(ax3, self.results['igbt_life_consumption']*100)
        
        # 4. 10年预测热力图
        ax4 = axes[1, 0]
        ten_year_data = self.results[self.results['years'] == 10]
        load_types = ten_year_data['load_type'].values
        igbt_life = ten_year_data['igbt_remaining_life'].values
        cap_life = ten_year_data['capacitor_remaining_life'].values
        
        x = np.arange(len(load_types))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, igbt_life, width, label='IGBT剩余寿命', alpha=0.8)
        bars2 = ax4.bar(x + width/2, cap_life, width, label='电容剩余寿命', alpha=0.8)
        
        format_axis_labels(ax4, '负载类型', '剩余寿命 (%)', '10年运行后剩余寿命对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(load_types, rotation=45, ha='right')
        ax4.legend(fontsize=8, loc='best')
        add_grid(ax4, alpha=0.3)
        
        # 添加数值标签，避免重叠
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 5. 温度分布
        ax5 = axes[1, 1]
        all_temps = []
        all_loads = []
        for _, row in self.results.iterrows():
            all_temps.extend([row['avg_igbt_temp'], row['max_igbt_temp']])
            all_loads.extend([f"{row['load_type']}-平均", f"{row['load_type']}-最高"])
        
        ax5.bar(range(len(all_temps)), all_temps, alpha=0.7)
        format_axis_labels(ax5, '工况', '温度 (°C)', 'IGBT温度分布')
        ax5.set_xticks(range(len(all_temps)))
        ax5.set_xticklabels(all_loads, rotation=45, ha='right', fontsize=8)
        add_grid(ax5, alpha=0.3)
        set_adaptive_ylim(ax5, all_temps)
        
        # 6. 寿命预测曲线
        ax6 = axes[1, 2]
        years_extended = np.arange(1, 16)  # 扩展到15年
        
        for load_type in self.results['load_type'].unique():
            data = self.results[self.results['load_type'] == load_type]
            # 简单线性外推
            if len(data) >= 2:
                slope = (data['igbt_remaining_life'].iloc[-1] - data['igbt_remaining_life'].iloc[0]) / (data['years'].iloc[-1] - data['years'].iloc[0])
                predicted_life = data['igbt_remaining_life'].iloc[0] + slope * (years_extended - data['years'].iloc[0])
                predicted_life = np.maximum(predicted_life, 0)  # 不低于0
                
                ax6.plot(years_extended, predicted_life, '--', label=f'{load_type}负载预测', linewidth=2)
                ax6.plot(data['years'], data['igbt_remaining_life'], 'o-', linewidth=2, markersize=6)
        
        format_axis_labels(ax6, '运行年数', 'IGBT剩余寿命 (%)', 'IGBT寿命预测曲线')
        add_grid(ax6, alpha=0.3)
        ax6.legend(fontsize=8, loc='best')
        set_adaptive_ylim(ax6, [0, 100])  # 寿命百分比范围
        
        # 7. 负载影响分析
        ax7 = axes[2, 0]
        load_impact = []
        load_labels = []
        
        for load_type in self.results['load_type'].unique():
            data = self.results[self.results['load_type'] == load_type]
            ten_year_life = data[data['years'] == 10]['igbt_remaining_life'].iloc[0]
            load_impact.append(ten_year_life)
            load_labels.append(load_type)
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        wedges, texts, autotexts = ax7.pie(load_impact, labels=load_labels, autopct='%1.1f%%', colors=colors)
        ax7.set_title('10年后IGBT剩余寿命分布', fontsize=10, pad=10)
        
        # 设置饼图文字大小，避免重叠
        for autotext in autotexts:
            autotext.set_fontsize(8)
        
        # 8. 维护优先级矩阵
        ax8 = axes[2, 1]
        maintenance_priority = []
        priority_labels = []
        
        for _, row in self.results.iterrows():
            if row['years'] in [5, 10]:  # 重点关注5年和10年
                life = row['igbt_remaining_life']
                if life < 50:
                    priority = '高'
                elif life < 80:
                    priority = '中'
                else:
                    priority = '低'
                
                maintenance_priority.append(priority)
                priority_labels.append(f"{row['years']}年-{row['load_type']}")
        
        priority_counts = pd.Series(maintenance_priority).value_counts()
        bars = ax8.bar(priority_counts.index, priority_counts.values, color=['red', 'orange', 'green'])
        format_axis_labels(ax8, '维护优先级', '工况数量', '维护优先级分布')
        add_grid(ax8, alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 9. 综合评估
        ax9 = axes[2, 2]
        # 计算综合评分（考虑寿命、温度、负载等因素）
        scores = []
        score_labels = []
        
        for _, row in self.results.iterrows():
            if row['years'] == 10:  # 只评估10年情况
                life_score = row['igbt_remaining_life'] / 100
                temp_score = max(0, 1 - (row['avg_igbt_temp'] - 100) / 200)  # 温度评分
                load_factor = {'light': 1.0, 'medium': 0.8, 'heavy': 0.6}[row['load_type']]
                
                total_score = (life_score * 0.6 + temp_score * 0.3 + load_factor * 0.1) * 100
                scores.append(total_score)
                score_labels.append(row['load_type'])
        
        bars = ax9.bar(score_labels, scores, color=['lightcoral', 'lightblue', 'lightgreen'])
        format_axis_labels(ax9, '负载类型', '综合评分', '10年运行综合评估')
        add_grid(ax9, alpha=0.3)
        
        # 添加数值标签
        for i, score in enumerate(scores):
            ax9.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 优化布局，避免重叠
        optimize_layout(fig, tight_layout=True, h_pad=2.0, w_pad=2.0)
        
        # 显示图形
        finalize_plot(fig)
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n" + "=" * 80)
        print("35kV/25MW级联储能PCS长期寿命综合分析报告")
        print("=" * 80)
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 分析寿命趋势
        self.analyze_life_trends()
        
        # 计算维护计划
        maintenance_df = self.calculate_maintenance_schedule()
        
        # 关键发现
        print("\n" + "=" * 60)
        print("关键发现与建议")
        print("=" * 60)
        
        # 找出最关键的维护点
        critical_maintenance = maintenance_df[maintenance_df['urgency'].isin(['紧急', '高'])]
        if not critical_maintenance.empty:
            print("需要重点关注的情况:")
            for _, point in critical_maintenance.iterrows():
                print(f"  • {point['years']}年 {point['load_type']}负载: IGBT剩余寿命{point['igbt_life']:.1f}%")
        
        # 10年预测总结
        ten_year_summary = self.results[self.results['years'] == 10]
        print(f"\n10年运行预测总结:")
        for _, row in ten_year_summary.iterrows():
            status = "良好" if row['igbt_remaining_life'] > 80 else "需要关注" if row['igbt_remaining_life'] > 50 else "需要更换"
            print(f"  • {row['load_type']}负载: IGBT剩余寿命{row['igbt_remaining_life']:.1f}% ({status})")
        
        # 维护策略建议
        print(f"\n维护策略建议:")
        print("  1. 建立分级维护体系:")
        print("     • 轻负载工况: 5年检查一次")
        print("     • 中等负载工况: 3年检查一次")
        print("     • 重负载工况: 2年检查一次")
        print("  2. 实施预测性维护:")
        print("     • 实时监测IGBT结温")
        print("     • 定期分析温度循环数据")
        print("     • 建立寿命预测模型")
        print("  3. 优化运行策略:")
        print("     • 避免长期重负载运行")
        print("     • 实施负载均衡")
        print("     • 优化冷却系统")
        
        print("\n" + "=" * 80)
    
    def save_detailed_results(self):
        """保存详细分析结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'result/详细寿命分析报告_{timestamp}.csv'
        
        # 确保result目录存在
        import os
        os.makedirs('result', exist_ok=True)
        
        # 添加分析指标
        detailed_results = self.results.copy()
        detailed_results['年化寿命消耗率'] = detailed_results.groupby('load_type')['igbt_life_consumption'].diff() / detailed_results.groupby('load_type')['years'].diff()
        detailed_results['维护优先级'] = detailed_results['igbt_remaining_life'].apply(
            lambda x: '紧急' if x < 20 else '高' if x < 50 else '中' if x < 80 else '低'
        )
        detailed_results['运行状态'] = detailed_results['igbt_remaining_life'].apply(
            lambda x: '需要更换' if x < 50 else '需要关注' if x < 80 else '良好'
        )
        
        detailed_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"详细分析结果已保存到: {filename}")
        return filename

def run_detailed_analysis():
    """运行详细分析"""
    print("开始详细长期寿命分析...")
    
    # 创建分析对象
    analyzer = DetailedLifeAnalysis()
    
    # 生成综合分析报告
    analyzer.generate_comprehensive_report()
    
    # 绘制详细分析图表
    print("\n生成详细分析图表...")
    analyzer.plot_detailed_analysis()
    
    # 保存详细结果
    filename = analyzer.save_detailed_results()
    
    print(f"\n详细分析完成！")
    print(f"分析结果已保存到: {filename}")
    
    return analyzer

if __name__ == "__main__":
    run_detailed_analysis() 