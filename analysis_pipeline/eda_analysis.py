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
from domain_adapter import load_or_build_schema

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

class GenericAnalyzer:
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

        # 通用日期列解析：优先 schema，其次常见命名
        date_col = self._get_date_column()
        if date_col:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            if date_col != 'Process_Date':
                self.df['Process_Date'] = self.df[date_col]
            self.df = self.df.sort_values('Process_Date').reset_index(drop=True)

        # 通用目标列解析
        target_col = self._get_target_column()
        if target_col:
            print(f"目标列: {target_col}")

    def _resolve_col(self, default_name, *keywords):
        """
        语义列名解析: 优先通过 schema 按关键词查找，回退到 default_name。
        如果两者都找不到，返回 None。
        """
        if self.schema:
            found = self.schema.find_column(*keywords)
            if found and found in self.df.columns:
                return found
        if default_name and default_name in self.df.columns:
            return default_name
        return None

    def _get_target_column(self):
        """通用目标列解析：优先 schema，其次常见命名回退。"""
        if self.schema and self.schema.target_column:
            if self.schema.target_column in self.df.columns:
                return self.schema.target_column
        for name in ['Is_Pass', 'Label_Pass', 'target', 'label', 'y', 'class', '结果', '状态', '良率']:
            if name in self.df.columns:
                return name
        return None

    def _get_date_column(self):
        """通用日期列解析：优先 schema，其次常见命名回退。"""
        if self.schema:
            found = self.schema.find_column('date', '时间', '日期', 'process', 'time')
            if found and found in self.df.columns:
                return found
        for name in ['Process_Date', '日期', 'date', 'time', '时间']:
            if name in self.df.columns:
                return name
        for c in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[c]):
                return c
        return None

    def _get_display_name(self, col):
        """从 schema 获取列的显示名，回退到列名本身。"""
        if self.schema:
            for c in self.schema.columns:
                if c.raw_name == col:
                    return c.display_name or c.semantic_name or col
        return col

    def _get_unit_suffix(self, col):
        """从 schema 获取列的单位后缀字符串。"""
        if self.schema:
            for c in self.schema.columns:
                if c.raw_name == col and c.physical_unit:
                    return f" ({c.physical_unit})"
        return ""

    def _get_numeric_features(self):
        """获取数值特征列（优先 schema，其次全量数值列）。"""
        if self.schema:
            return self.schema.get_numeric_features()
        return [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]

    def save_processed_data(self, file_name='data_enriched.csv'):
        save_path = os.path.join(self.output_dir, file_name)
        self.df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n[保存成功] 分析中间数据已保存至: {save_path}")

    # ===============================================================
    #  绘图逻辑 
    # ===============================================================

    def analyze_target_distribution(self):
        """
        0. 目标分布图 (通用版)
        支持二分类/多分类的自适应展示，兼容数值和字符串目标列
        """
        print(">>> 生成目标分布图...")
        target_col = self._get_target_column()
        if not target_col:
            print(">>> 跳过目标分布图: 未识别到目标列")
            return

        plt.figure(figsize=(8, 6))

        # 尝试数值转换，失败则使用原始字符串值
        raw_values = self.df[target_col]
        numeric_values = pd.to_numeric(raw_values, errors='coerce')
        is_numeric = numeric_values.notna().sum() > 0

        if is_numeric:
            value_counts = numeric_values.value_counts(dropna=True).sort_index()
        else:
            value_counts = raw_values.value_counts(dropna=True)

        if value_counts.empty:
            print(">>> 目标列无有效取值，跳过")
            plt.close()
            return

        n_unique = len(value_counts)
        target_type = self.schema.target_type if self.schema and self.schema.target_type else (
            'binary' if n_unique <= 2 else 'multiclass'
        )

        if n_unique <= 2 or target_type == 'binary':
            pass_label = self.schema.pass_label if self.schema and self.schema.pass_label else 'Pass'
            fail_label = self.schema.fail_label if self.schema and self.schema.fail_label else 'Fail'
            labels = []
            colors = []

            for v in value_counts.index:
                v_str = str(v)
                is_pass = False
                if self.schema and self.schema.pass_values and v_str in self.schema.pass_values:
                    is_pass = True
                elif self.schema and self.schema.fail_values and v_str in self.schema.fail_values:
                    is_pass = False
                elif is_numeric and v > 0:
                    is_pass = True
                elif not is_numeric and v_str.lower() in ('pass', 'true', 'yes', '合格', '良品', '好', '1', 'y'):
                    is_pass = True
                else:
                    is_pass = False

                labels.append(pass_label if is_pass else fail_label)
                colors.append('#2ECC71' if is_pass else '#E74C3C')

            ax = sns.barplot(x=labels, y=value_counts.values, palette=colors)
        else:
            labels = []
            for v in value_counts.index:
                v_str = str(v)
                if self.schema and self.schema.target_mapping and v_str in self.schema.target_mapping:
                    labels.append(self.schema.target_mapping[v_str])
                else:
                    labels.append(v_str)
            colors = sns.color_palette("husl", n_unique)
            ax = sns.barplot(x=labels, y=value_counts.values, palette=colors)

        max_val = max(value_counts.values)
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + max_val * 0.02, str(v), ha='center', fontweight='bold', fontsize=12)

        target_display = self._get_display_name(target_col)
        plt.title(f"{target_display} 分布", fontsize=14)
        plt.ylabel("样本数量 (Count)")
        plt.xticks(rotation=45 if n_unique > 4 else 0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '0_目标分布统计.png'), dpi=300)
        plt.close()

    def analyze_status_distribution(self):
        """[保留别名] 委托给通用版本"""
        self.analyze_target_distribution()

    def analyze_global_correlations(self):
        """
        1. 关键参数相关性 (通用版)
        """
        # 通用数值特征列表
        numeric_features = self._get_numeric_features()
        target_col = self._get_target_column()
        target_cols = list(numeric_features)
        if target_col and target_col not in target_cols and target_col in self.df.columns:
            target_cols.append(target_col)

        valid_cols = [c for c in target_cols if c in self.df.columns]
        if len(valid_cols) < 2:
            return

        plt.figure(figsize=(11, 9))
        corr = self.df[valid_cols].corr()

        # 动态名称映射
        name_map = {col: self._get_display_name(col) for col in valid_cols}

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
                    xticklabels=[name_map.get(x, x) for x in corr.columns],
                    yticklabels=[name_map.get(x, x) for x in corr.index],
                    vmin=-1, vmax=1)

        plt.title("参数相关性矩阵 (Correlation Matrix)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '1_参数相关性分析.png'), dpi=300)
        plt.close()

    def analyze_pass_fail_dist(self):
        """
        2. 合格/不合格特征对比 (通用版)
        """
        print("--- 正在生成特征分布对比图 ---")

        numeric_features = self._get_numeric_features()
        core_features = numeric_features[:min(6, len(numeric_features))]
        if not core_features:
            print(">>> 跳过特征对比图: 无数值特征")
            return

        feature_map = {c: self._get_display_name(c) + self._get_unit_suffix(c) for c in core_features}

        target_col = self._get_target_column()
        if not target_col or target_col not in self.df.columns:
            print(">>> 跳过特征对比图: 未识别到目标列")
            return

        pass_label = self.schema.pass_label if self.schema and self.schema.pass_label else 'Pass'
        fail_label = self.schema.fail_label if self.schema and self.schema.fail_label else 'Fail'
        plot_df = self.df.copy()

        # 根据目标列值映射 pass/fail
        # 优先 schema.pass_values/fail_values，其次 0/1 启发式
        pass_set, fail_set = set(), set()
        if self.schema and self.schema.pass_values:
            pass_set = set(self.schema.pass_values)
            fail_set = set(self.schema.fail_values) if self.schema.fail_values else set()
        elif self.schema and self.schema.fail_values:
            fail_set = set(self.schema.fail_values)
        else:
            # 无 schema 或无 pass/fail 映射时，使用 0/1 启发式
            pass_set = {'1', '1.0', 1, 1.0}
            fail_set = {'0', '0.0', 0, 0.0}

        def map_status(val):
            if str(val) in [str(v) for v in pass_set]:
                return pass_label
            elif str(val) in [str(v) for v in fail_set]:
                return fail_label
            return None

        plot_df['目标状态'] = plot_df[target_col].apply(map_status)
        plot_df = plot_df.dropna(subset=['目标状态'])

        if plot_df.empty:
            print(">>> 跳过特征对比图: 无有效的 pass/fail 状态")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        axes = axes.flatten()

        my_palette = {pass_label: '#2ECC71', fail_label: '#E74C3C'}

        for i, col in enumerate(core_features):
            ax = axes[i]
            display_name = feature_map[col]

            if col in plot_df.columns and plot_df[col].notna().sum() > 0:
                sub_df = plot_df.dropna(subset=[col, '目标状态'])

                sns.boxplot(x='目标状态', y=col, data=sub_df,
                            palette=my_palette, width=0.5, showfliers=False, ax=ax)

                sns.stripplot(x='目标状态', y=col, data=sub_df,
                              color='#2C3E50', alpha=0.3, size=3, jitter=True, ax=ax)

                pass_median = sub_df[sub_df['目标状态'] == pass_label][col].median()
                if not np.isnan(pass_median):
                    ax.axhline(pass_median, color='#27AE60', linestyle='--', alpha=0.8, linewidth=1.5,
                               label=f'{pass_label}中位数: {pass_median:.1f}')

                ax.set_title(f"{display_name} 分布", fontsize=12, fontweight='bold')
                ax.set_ylabel(display_name, fontsize=11)
                ax.set_xlabel("")
                ax.legend(loc='upper right', frameon=True, fontsize='x-small')
            else:
                ax.text(0.5, 0.5, f"数据缺失:\n{display_name}",
                        horizontalalignment='center', verticalalignment='center', color='gray')
                ax.set_axis_off()

        plt.suptitle(f"关键参数 vs {pass_label}/{fail_label} 分布 (样本数 N={len(plot_df)})", fontsize=15, y=1.02)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, '2_核心特征分布_2x3.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[图表生成] 已保存: {save_path}")

    # 下面的 Time_Seq_Day, Position_Code 等逻辑基本通用，
    # 只要确保输入文件包含这些列即可直接运行，无需大幅修改。

    

    def analyze_yield_trend_weekly(self):
        """
        3. 周度趋势分析 (通用版)
        """
        # 1. 检查列名
        date_col = self._get_date_column()
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

        # 确定使用哪一列作为目标值
        target_col = self._get_target_column()
        if not target_col or target_col not in df_ts.columns:
            print(">>> 跳过周度分析: 未识别到目标列")
            return
        
        # 按周重采样: W-MON = 每周一
        weekly_stats = df_ts[target_col].resample('W-MON').agg(['mean', 'count'])
        weekly_stats = weekly_stats[weekly_stats['count'] > 0] # 去掉无产出的周
        
        if weekly_stats.empty: return

        x_dates = weekly_stats.index.strftime('%Y-%m-%d')
        x_axis = range(len(x_dates))

        # --- 第一轴: 数量 (柱状图) ---
        ax1 = plt.gca()
        bars = ax1.bar(x_axis, weekly_stats['count'], color='#AED6F1', alpha=0.8, label='周样本数')
        ax1.set_ylabel('样本数 (Count)', color='#2E86C1', fontsize=12)

        # 5.0 参考线
        ax1.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='参考线 (Count=5.0)')
        ax1.text(0.98, 0.95, 'Count 5.0', transform=ax1.transAxes, color='red',
                 fontsize=10, ha='right', va='top')
        
        # 柱状图数字标注
        for i, count in enumerate(weekly_stats['count']):
            ax1.text(i, count, str(int(count)), 
                     ha='center', va='bottom',
                     fontsize=10, 
                     fontweight='bold', 
                     color='#2874A6')

        # --- 第二轴: 目标均值 (折线图) ---
        ax2 = ax1.twinx()
        ax2.plot(x_axis, weekly_stats['mean'], color='#E74C3C', marker='o', linewidth=2.5, label='周目标均值')
            
        target_display = self._get_display_name(target_col) if target_col else '目标'
        ax2.set_ylabel(f'{target_display} 均值', color='#E74C3C', fontsize=12)
        # 仅二分类目标使用 [0,1] 区间，否则自动缩放
        if self.schema and self.schema.target_type == 'binary':
            ax2.set_ylim(-0.05, 1.15)
        
        # 设置X轴
        ax1.set_xticks(x_axis)
        ax1.set_xticklabels(x_dates, rotation=45, ha='right')
        
        # 图例与标题
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.title("周度趋势: 样本数与目标均值 (Weekly Trend)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '3_周度趋势分析.png'), dpi=300)
        plt.close()

    def analyze_height_drift(self):
        """
        4. 数值特征漂移分析 (通用版)
        取首个数值特征，按日期绘制漂移
        """
        # 1. 获取数值特征列
        numeric_features = self._get_numeric_features()
        if not numeric_features:
            print(">>> 跳过漂移图: 无数值特征")
            return
        feat_col = numeric_features[0]

        if feat_col not in self.df.columns:
            print(">>> 跳过漂移图: 第一数值特征列不存在")
            return

        print(f">>> 4. 生成特征漂移图 ({feat_col})...")

        # 2. 准备绘图数据
        plot_df = self.df.copy()

        # 3. 日期列处理
        time_col = None
        date_col = self._get_date_column()
        if date_col and date_col in plot_df.columns:
            plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors='coerce')
            start_date = plot_df[date_col].min()
            plot_df['Time_Seq_Day'] = (plot_df[date_col] - start_date).dt.days
            time_col = 'Time_Seq_Day'
        else:
            # 尝试使用现有时间序列列
            for tc in ['Time_Seq_Day', 'time_seq', 'seq']:
                if tc in plot_df.columns:
                    time_col = tc
                    break

        if not time_col:
            print(">>> 跳过漂移图: 无日期列用于计算时间序列")
            return

        plt.figure(figsize=(12, 6))

        # 4. 分类着色（根据目标列）
        target_col = self._get_target_column()
        if target_col and target_col in plot_df.columns:
            pass_label = self.schema.pass_label if self.schema and self.schema.pass_label else 'Pass'
            fail_label = self.schema.fail_label if self.schema and self.schema.fail_label else 'Fail'
            pass_set = set(self.schema.pass_values) if self.schema and self.schema.pass_values else {'1', '1.0', 1, 1.0}
            fail_set = set(self.schema.fail_values) if self.schema and self.schema.fail_values else {'0', '0.0', 0, 0.0}

            def map_status(val):
                if str(val) in [str(v) for v in pass_set]:
                    return f'{pass_label} (Pass)'
                elif str(val) in [str(v) for v in fail_set]:
                    return f'{fail_label} (Fail)'
                return 'Unknown'

            plot_df['Status_Text'] = plot_df[target_col].apply(map_status)
            hue_col = 'Status_Text'
            palette_dict = {f'{fail_label} (Fail)': '#E74C3C', f'{pass_label} (Pass)': '#2ECC71'}
        else:
            hue_col = None
            palette_dict = None

        # 5. 散点图
        sns.scatterplot(x=time_col, y=feat_col, hue=hue_col,
                       data=plot_df, palette=palette_dict,
                       s=60, alpha=0.7)

        # 6. 趋势线
        sns.regplot(x=time_col, y=feat_col, data=plot_df, scatter=False,
                    line_kws={'color': '#3498DB', 'linestyle': '--', 'alpha': 0.8},
                    label='整体趋势 (Overall Trend)')

        feat_display = self._get_display_name(feat_col)
        feat_unit = self._get_unit_suffix(feat_col)
        plt.title(f"{feat_display} 随时间漂移 (Process Drift)", fontsize=14)
        plt.xlabel("距离首日天数 (Days from Start)")
        plt.ylabel(f"{feat_display}{feat_unit}")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '4_特征漂移分析.png'), dpi=300)
        plt.close()
        
    

    def analyze_position_effect(self):
        """5. 位置效应分析 (通用版)"""
        pos_col = None
        if self.schema:
            pos_col = self.schema.find_column('position', '位置', 'code')
        if not pos_col:
            for name in ['Position_Code', 'position', '位置', 'code', 'pos']:
                if name in self.df.columns:
                    pos_col = name
                    break
        if not pos_col or pos_col not in self.df.columns:
            print(">>> 跳过位置分析: 无位置列")
            return

        target_col = self._get_target_column()
        if not target_col or target_col not in self.df.columns:
            print(">>> 跳过位置分析: 未识别到目标列")
            return

        pos_stats = self.df.groupby(pos_col)[target_col].agg(['mean', 'count']).reset_index()
        pos_stats = pos_stats[pos_stats['count'] >= 1].sort_values('mean', ascending=True)
        if pos_stats.empty:
            return

        plt.figure(figsize=(12, 6))
        sns.barplot(x=pos_col, y='mean', data=pos_stats, palette='magma')
        plt.axhline(self.df[target_col].mean(), color='red', linestyle='--', label='全局均值')
        pos_display = self._get_display_name(pos_col)
        plt.title(f"各{pos_display}目标均值排行 ({pos_display} Check)", fontsize=14)
        plt.ylabel("目标均值")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '5_位置效应分析.png'), dpi=300)
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

    # 加载贯穿整条流水线的 DataSchema（领域无关的关键）
    schema = load_or_build_schema()
    analyzer = GenericAnalyzer(input_csv, output_dir, schema=schema)
    analyzer.save_processed_data('data_for_viz.csv')
    analyzer.run_full_analysis()
   