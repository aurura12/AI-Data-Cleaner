import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适配无显示环境
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import warnings
import re
import matplotlib.font_manager as fm
import sys
import io
# --- 0. 环境配置：字体与绘图风格 (保持不变) ---
warnings.filterwarnings('ignore')
sns.set(style="whitegrid", palette="deep")
plt.rcParams['axes.unicode_minus'] = False 

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import (
    FONT_FILE,
    CLEANED_DATA_FILE,
    EDA_REPORT_DIR,
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_app"))
from schema_analyzer import DataSchema

sys_name = platform.system()
font_path = FONT_FILE
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print(f"成功加载自定义字体: {font_path}")
elif sys_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
elif sys_name == "Darwin":  # Mac
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

class ChipAnalyzer:
    def __init__(self, file_path, output_dir, schema=None):
        """
        初始化分析器
        ##### 【修改】: 简化了初始化逻辑，不再重复清洗，直接读取已清洗好的数据 #####
        
        参数:
            file_path: 清洗后的 CSV 文件路径
            output_dir: 输出目录
            schema: DataSchema 对象（可选）；提供后用于语义列名查找
        """
        self.output_dir = output_dir
        self.schema = schema
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print(f"--- 1. 读取清洗后的数据 ---")
        try:
            self.df = pd.read_csv(file_path)
            print(f"文件源: {os.path.basename(file_path)}")
            print(f"总行数: {len(self.df)} | 总列数: {len(self.df.columns)}")
        except Exception as e:
            print(f"读取文件失败: {e}")
            self.df = pd.DataFrame()
            return

        # schema 就绪提示
        if self.schema:
            print(f"Schema 已加载: {len(self.schema.columns)} 列")

        # ##### 【修改】: 直接使用清洗脚本生成的 Process_Date，不再解析中文日期 #####
        date_col = self._resolve_col('Process_Date', 'date', '日期', 'process')
        if date_col:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            # 如果找到的日期列不是 Process_Date，显式重命名
            if date_col != 'Process_Date':
                self.df['Process_Date'] = self.df[date_col]
            self.df = self.df.sort_values('Process_Date').reset_index(drop=True)

        # 兼容 Label_Pass
        pass_col = self._resolve_col('Is_Pass', 'pass', '良率', 'label')
        if pass_col and pass_col not in self.df.columns:
            if 'Label_Pass' in self.df.columns:
                self.df['Is_Pass'] = self.df['Label_Pass']

        # 简单检查必要字段
        is_pass_col = 'Is_Pass' if 'Is_Pass' in self.df.columns else self._resolve_col(None, 'pass', '良率', 'label')
        height_col = self._resolve_col('Total_Indium_Height', 'height', '高度', 'indium')
        required_cols = [c for c in [is_pass_col, height_col] if c]
        missing = [c for c in required_cols if c and c not in self.df.columns]
        if missing:
            print(f"警告: 输入数据缺少关键列 {missing}，部分图表可能无法生成。")
        else:
            yield_col = 'Is_Pass' if 'Is_Pass' in self.df.columns else is_pass_col
            if yield_col and yield_col in self.df.columns:
                print(f"数据加载成功，当前良率: {self.df[yield_col].mean():.2%}")

    def _resolve_col(self, default_name, *keywords):
        """
        语义列名解析: 优先通过 schema 按关键词查找，回退到 default_name。
        default_name 为 None 时不回退（仅用 schema 查找）。
        """
        if self.schema:
            found = self.schema.find_column(*keywords)
            if found and found in self.df.columns:
                return found
        if default_name and default_name in self.df.columns:
            return default_name
        return default_name

    def save_processed_data(self, file_name='chip_data_enriched.csv'):
        save_path = os.path.join(self.output_dir, file_name)
        self.df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n[保存成功] 分析中间数据已保存至: {save_path}")

    # ===============================================================
    #  绘图逻辑 
    # ===============================================================

    def analyze_status_distribution(self):
        """
        0. 宏观分布 
        ##### 【修改】: 优先展示压连情况四分类 #####
        """
        print(">>> 生成良率分布图...")
        plt.figure(figsize=(8, 6))

        target_col = self._resolve_col(None, '压连') or next((c for c in self.df.columns if '压连' in c), None)
        if target_col:
            statuses = pd.to_numeric(self.df[target_col], errors='coerce')
            # 顺序：良好、轻微压连、严重压连、虚焊
            status_order = [0, 1, 2, -1]
            status_labels = {
                0: '良好 (0)',
                1: '轻微压连 (1)',
                2: '严重压连 (2)',
                -1: '虚焊 (-1)'
            }
            value_counts = statuses.value_counts(dropna=True)
            counts = [int(value_counts.get(v, 0)) for v in status_order]
            labels = [status_labels[v] for v in status_order]
            # 使用指定的颜色方案
            colors = []
            for label in labels:
                if '(-1)' in label: 
                    colors.append('#F39C12') # 橙色 (虚焊)
                elif '(2)' in label: 
                    colors.append('#C0392B') # 红色 (严重)
                elif '(0)' in label: 
                    colors.append('#2ECC71') # 绿色 (正常)
                elif '(1)' in label: 
                    colors.append('#27AE60') # 深绿 (轻微)
                else: 
                    colors.append('#95A5A6') # 灰色 (未知)
        elif 'Is_Pass' in self.df.columns:
            pass_col = 'Is_Pass'
        else:
            pass_col = self._resolve_col(None, 'pass', '良率', 'label')
            if not pass_col or pass_col not in self.df.columns:
                return

        if pass_col == 'Is_Pass' or pass_col is None:
            # 已有 Is_Pass
            counts_series = self.df['Is_Pass' if 'Is_Pass' in self.df.columns else pass_col].value_counts().sort_index()
            labels = ['不良 (Fail)' if idx == 0 else '良品 (Pass)' for idx in counts_series.index]
            colors = ['#E74C3C' if idx == 0 else '#2ECC71' for idx in counts_series.index]
            counts = counts_series.values.tolist()
        else:
            return

        ax = sns.barplot(x=labels, y=counts, palette=colors)
        max_count = max(counts) if counts else 0
        offset = max_count * 0.02 if max_count else 0.1
        for i, v in enumerate(counts):
            ax.text(i, v + offset, str(v), ha='center', fontweight='bold', fontsize=12)

        plt.title("产线最终良率分布", fontsize=14)
        plt.ylabel("芯片数量 (Count)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '0_生产状态分布统计.png'), dpi=300)
        plt.close()

    def analyze_global_correlations(self):
        """
        1. 关键参数相关性
        ##### 【修改】: 加入 Equipment_Temp, Vacuum_Level，并使用新列名 #####
        """
        # 使用新列名列表
        # 优先用 schema 的数值特征，回退到硬编码列表
        if self.schema:
            numeric_features = self.schema.get_numeric_features()
            pass_col = self._resolve_col(None, 'pass', '良率', 'label') or 'Is_Pass'
            target_cols = numeric_features + ([pass_col] if pass_col in self.df.columns else [])
        else:
            target_cols = ['Total_Indium_Height', 'Force_kg', 'Equipment_Temp', 
                           'Vacuum_Level', 'Indium_Taper_Zscore', 'Calc_Circuit_Range', 'Is_Pass']
        
        valid_cols = [c for c in target_cols if c in self.df.columns]
        if len(valid_cols) < 2: return

        plt.figure(figsize=(11, 9))
        corr = self.df[valid_cols].corr()
        
        # 中文映射表更新
        name_map = {
            'Total_Indium_Height': '总高度',
            'Force_kg': '压力',
            'Equipment_Temp': '设备温度',
            'Vacuum_Level': '真空度',
            'Indium_Taper_Zscore': '锥度异常',
            'Calc_Circuit_Range': '平整度',
            'Is_Pass': '良率'
        }
        
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", 
                    xticklabels=[name_map.get(x,x) for x in corr.columns],
                    yticklabels=[name_map.get(x,x) for x in corr.index],
                    vmin=-1, vmax=1)
        
        plt.title("工艺与环境参数相关性矩阵", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '1_参数相关性分析.png'), dpi=300)
        plt.close()

    def analyze_pass_fail_dist(self):
        """
        2. 良次品特征对比
        ##### 【修改】: 布局改为 2x3，加入温度和真空度分析 #####
        """
        print("--- 正在生成物理特征分布对比图 (含环境参数) ---")

        # 锁定6大核心特征 (包含环境)
        if self.schema:
            numeric_features = self.schema.get_numeric_features()
            core_features = numeric_features[:6]
            feature_map = {c: c for c in core_features}
        else:
            feature_map = {
                'Total_Indium_Height': '铟柱总高度 (μm)',
                'Force_kg': '倒焊压力 (kg)',
                'Calc_Circuit_Range': '电路端平整度 (极差)',
                'Indium_Taper_Zscore': '形状异常度 (Z-Score)',
                'Equipment_Temp': '设备温度 (℃)',   # 新增
                'Vacuum_Level': '真空度读数'        # 新增
            }
            core_features = list(feature_map.keys())

        pass_col = 'Is_Pass' if 'Is_Pass' in self.df.columns else self._resolve_col(None, 'pass', '良率', 'label')
        if not pass_col or pass_col not in self.df.columns:
            return

        plot_df = self.df.copy()
        pass_label_map = {1: '良品', 0: '不良'}
        if self.schema:
            pass_label_map = {1: self.schema.pass_label, 0: self.schema.fail_label}
        plot_df['良率状态'] = plot_df[pass_col].map(pass_label_map)
        
        # 改为 2行3列
        fig, axes = plt.subplots(2, 3, figsize=(16, 11)) 
        axes = axes.flatten()
        
        my_palette = {'良品': '#2ECC71', '不良': '#E74C3C'}
        
        for i, col in enumerate(core_features):
            ax = axes[i]
            chinese_name = feature_map[col]
            
            if col in plot_df.columns and plot_df[col].notna().sum() > 0:
                sub_df = plot_df.dropna(subset=[col, '良率状态'])
                
                # A. 箱线图
                sns.boxplot(x='良率状态', y=col, data=sub_df, 
                            palette=my_palette, width=0.5, showfliers=False, ax=ax)
                
                # B. 散点图
                sns.stripplot(x='良率状态', y=col, data=sub_df, 
                            color='#2C3E50', alpha=0.3, size=3, jitter=True, ax=ax)
                
                # C. 辅助线 (良品中位数)
                pass_median = sub_df[sub_df['良率状态']=='良品'][col].median()
                if not np.isnan(pass_median):
                    ax.axhline(pass_median, color='#27AE60', linestyle='--', alpha=0.8, linewidth=1.5, 
                            label=f'良品中位数: {pass_median:.1f}')
                
                ax.set_title(f"{chinese_name} 分布", fontsize=12, fontweight='bold')
                ax.set_ylabel(chinese_name, fontsize=11)
                ax.set_xlabel("") # 省略X轴标签
                ax.legend(loc='upper right', frameon=True, fontsize='x-small')
                
            else:
                ax.text(0.5, 0.5, f"数据缺失:\n{chinese_name}", 
                        horizontalalignment='center', verticalalignment='center',
                        color='gray')
                ax.set_axis_off()

        plt.suptitle(f"关键参数 vs 良率分布 (样本数 N={len(plot_df)})", fontsize=15, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, '2_核心特征分布_2x3_中文.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[图表生成] 已保存: {save_path}")

    # 下面的 Time_Seq_Day, Position_Code 等逻辑基本通用，
    # 只要确保输入文件包含这些列即可直接运行，无需大幅修改。

    

    def analyze_yield_trend_weekly(self):
        """
        3. 周度趋势分析 (带产量标注版)
        """
        # 1. 检查列名
        date_col = self._resolve_col('Process_Date', 'date', '日期', 'process')
        if not date_col or date_col not in self.df.columns: 
            print(">>> 跳过周度分析: 缺少日期列")
            return

        print(">>> 3. 生成周度趋势图 (含产量标注)...")
        
        plt.figure(figsize=(14, 8))
        
        # --- 数据预处理 ---
        df_ts = self.df.copy()
        # 强制转为时间格式
        if date_col != 'Process_Date':
            df_ts['Process_Date'] = pd.to_datetime(df_ts[date_col], errors='coerce')
        else:
            df_ts['Process_Date'] = pd.to_datetime(df_ts['Process_Date'], errors='coerce')
        df_ts = df_ts.dropna(subset=['Process_Date'])
        df_ts = df_ts.set_index('Process_Date').sort_index()

        if df_ts.empty: return

        # 确定使用哪一列作为良率
        pass_col = self._resolve_col(None, 'pass', '良率', 'label')
        target_col = 'Label_Pass' if 'Label_Pass' in df_ts.columns else (pass_col or 'Is_Pass')
        
        # 按周重采样: W-MON = 每周一
        weekly_stats = df_ts[target_col].resample('W-MON').agg(['mean', 'count'])
        weekly_stats = weekly_stats[weekly_stats['count'] > 0] # 去掉无产出的周
        
        if weekly_stats.empty: return

        x_dates = weekly_stats.index.strftime('%Y-%m-%d')
        x_axis = range(len(x_dates))

        # --- 第一轴: 产量 (柱状图) ---
        ax1 = plt.gca()
        bars = ax1.bar(x_axis, weekly_stats['count'], color='#AED6F1', alpha=0.8, label='周产量(颗)')
        ax1.set_ylabel('产量 (Count)', color='#2E86C1', fontsize=12)

        # 5.0 参考线（标注单位为产量）
        ax1.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='参考线 (产量=5.0)')
        ax1.text(0.98, 0.95, '产量 5.0', transform=ax1.transAxes, color='red',
                 fontsize=10, ha='right', va='top')
        
        # 【新增】给柱状图添加数字标注
        for i, count in enumerate(weekly_stats['count']):
            ax1.text(i, count, str(int(count)), 
                     ha='center', va='bottom',  # 底部对齐是指文字底部对齐数据点(即显示在柱子上方)
                     fontsize=10, 
                     fontweight='bold', 
                     color='#2874A6') # 深蓝色字

        # --- 第二轴: 良率 (折线图) ---
        ax2 = ax1.twinx()
        ax2.plot(x_axis, weekly_stats['mean'], color='#E74C3C', marker='o', linewidth=2.5, label='周良率')
        
        '''
        # 给折线图添加百分比标注
        for i, rate in enumerate(weekly_stats['mean']):
            ax2.annotate(f"{rate:.0%}", (i, rate), 
                         xytext=(0, 10), textcoords='offset points', 
                         ha='center', fontsize=9, color='#C0392B')'''
            
        ax2.set_ylabel('良率 (Yield)', color='#E74C3C', fontsize=12)
        ax2.set_ylim(-0.05, 1.15) # 稍微调高一点，防止文字被切掉
        
        # 设置X轴
        ax1.set_xticks(x_axis)
        ax1.set_xticklabels(x_dates, rotation=45, ha='right')
        
        # 图例与标题
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.title("周度生产监控: 产量与良率趋势 (Weekly Report)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '3_周度趋势分析.png'), dpi=300)
        plt.close()

    def analyze_height_drift(self):
        """
        4. 高度漂移分析 (修复版)
        适配 Total_Indium_Height 并自动计算天数序列
        """
        # 1. 检查核心数据列 (适配新列名)
        height_col = self._resolve_col('Total_Indium_Height', 'height', '高度', 'indium')
        if not height_col or height_col not in self.df.columns: 
            print(">>> 跳过高度漂移图: 缺少高度列")
            return
            
        print(">>> 4. 生成高度漂移图...")
        
        # 2. 准备绘图数据 (不修改原数据)
        plot_df = self.df.copy()

        # 3. 动态计算 Time_Seq_Day (如果不存在)
        time_col = self._resolve_col('Time_Seq_Day', 'time_seq', 'time', '生产天数')
        date_col = self._resolve_col('Process_Date', 'date', '日期', 'process')
        if not time_col or time_col not in plot_df.columns:
            if date_col and date_col in plot_df.columns:
                plot_df[date_col] = pd.to_datetime(plot_df[date_col])
                start_date = plot_df[date_col].min()
                # 计算每一行距离第一天的天数
                plot_df['Time_Seq_Day'] = (plot_df[date_col] - start_date).dt.days
                time_col = 'Time_Seq_Day'
            else:
                print(">>> 跳过高度漂移图: 缺少日期列用于计算时间序列")
                return
        else:
            time_col = time_col

        plt.figure(figsize=(12, 6))
        
        # 4. 生成用于图例的文字标签
        pass_col = self._resolve_col(None, 'pass', '良率', 'label')
        target_label = 'Label_Pass' if 'Label_Pass' in plot_df.columns else (pass_col or 'Is_Pass')
        
        if target_label in plot_df.columns:
            plot_df['Status_Text'] = plot_df[target_label].map({1: '良品 (Pass)', 0: '不良 (Fail)'})
            hue_col = 'Status_Text'
            palette_dict = {'不良 (Fail)': '#E74C3C', '良品 (Pass)': '#2ECC71'} # 红/绿
        else:
            hue_col = None
            palette_dict = None

        # 5. 绘制散点图
        sns.scatterplot(x=time_col, y=height_col, hue=hue_col, 
                       data=plot_df, palette=palette_dict, 
                       s=60, alpha=0.7)
        
        # 6. 绘制趋势线 (拟合线)
        sns.regplot(x=time_col, y=height_col, data=plot_df, scatter=False, 
                    line_kws={'color': '#3498DB', 'linestyle': '--', 'alpha': 0.8}, 
                    label='整体趋势 (Overall Trend)')

        plt.title("总高度随生产天数漂移 (Process Drift)", fontsize=14)
        plt.xlabel("距离生产首日的天数 (Days from Start)")
        plt.ylabel("铟柱总高度 (Total Indium Height μm)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '4_高度长期漂移.png'), dpi=300)
        plt.close()
        
    

    def analyze_position_effect(self):
        """5. 晶圆位置分析"""
        pos_col = self._resolve_col('Position_Code', 'position', 'code', '位置')
        if not pos_col or pos_col not in self.df.columns: return
        
        pass_col = 'Is_Pass' if 'Is_Pass' in self.df.columns else self._resolve_col(None, 'pass', '良率', 'label')
        if not pass_col or pass_col not in self.df.columns: return
        
        pos_stats = self.df.groupby(pos_col)[pass_col].agg(['mean', 'count']).reset_index()
        pos_stats = pos_stats[pos_stats['count'] >= 1].sort_values('mean', ascending=True)
        
        if pos_stats.empty: return

        plt.figure(figsize=(12, 6))
        sns.barplot(x=pos_col, y='mean', data=pos_stats, palette='magma')
        plt.axhline(self.df[pass_col].mean(), color='red', linestyle='--', label='平均良率')
        
        plt.title("各位置良率排行 (Position Check)", fontsize=14)
        plt.ylabel("良率")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '5_位置良率分析.png'), dpi=300)
        plt.close()


    def run_full_analysis(self):
        """主控函数：执行所有分析"""
        if self.df.empty:
            print("数据为空，无法分析。")
            return
            
        print("\n>>> 开始生成统计图表...")
        self.analyze_status_distribution()
        self.analyze_global_correlations()
        self.analyze_pass_fail_dist()
        self.analyze_yield_trend_weekly()
        self.analyze_height_drift()
        self.analyze_position_effect()
        #self.analyze_advanced_spatial()
        print(f">>> 分析完成。结果已保存在: {self.output_dir}")

if __name__ == "__main__":
    # 使用相对路径
    input_csv = CLEANED_DATA_FILE
    output_dir = EDA_REPORT_DIR

    analyzer = ChipAnalyzer(input_csv, output_dir)
    analyzer.save_processed_data('chip_data_for_viz.csv')
    analyzer.run_full_analysis()
   