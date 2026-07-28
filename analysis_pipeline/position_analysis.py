import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import re
import matplotlib.font_manager as fm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import FONT_FILE, CLEANED_DATA_FILE, POSITION_REPORT_DIR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_app"))
from schema_analyzer import DataSchema
from domain_adapter import load_or_build_schema


# ==========================================
# 领域无关辅助函数
# ==========================================

def _display_name(schema, col):
    """从 schema 获取列显示名"""
    if schema:
        for c in schema.columns:
            if c.raw_name == col:
                return c.display_name or c.semantic_name or col
    return col


def _unit_suffix(schema, col):
    """从 schema 获取列的单位后缀"""
    if schema:
        for c in schema.columns:
            if c.raw_name == col and c.physical_unit:
                return f" ({c.physical_unit})"
    return ""


def _numeric_features(schema, df, exclude_cols=None):
    """获取数值特征列（优先 schema，其次全量数值列）"""
    exclude_cols = set(exclude_cols or [])
    if schema:
        return [c for c in schema.get_numeric_features() if c not in exclude_cols]
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]


# ==========================================
# 0. 环境配置：字体与绘图风格（改为函数，import时不执行）
# ==========================================

_plot_configured = False

def _configure_plotting():
    """配置 matplotlib 和 seaborn 绘图风格（仅执行一次）"""
    global _plot_configured
    if _plot_configured:
        return
    _plot_configured = True
    
    sns.set(style="whitegrid", palette="deep")
    plt.rcParams['axes.unicode_minus'] = False
    
    sys_name = platform.system()
    font_path = FONT_FILE
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print(f"成功加载自定义字体: {font_path}")
    elif sys_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
    elif sys_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']


def run_position_analysis(input_path=None, output_dir=None, schema=None):
    """
    晶圆位置效应分析

    参数:
        input_path: 清洗后数据CSV路径
        output_dir: 输出目录
        schema: DataSchema对象（可选）
    """
    if input_path is None:
        input_path = CLEANED_DATA_FILE
    if output_dir is None:
        output_dir = POSITION_REPORT_DIR

    os.makedirs(output_dir, exist_ok=True)
    _configure_plotting()

    print(f"--- 读取数据: {os.path.basename(input_path)} ---")
    try:
        df = pd.read_csv(input_path)
        print(f"数据行数: {len(df)}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_path}")
        return []

    # --- 语义列名解析 ---
    pos_col = None
    pass_col = None

    if schema:
        pos_col = schema.find_column('position', 'code', '位置')
        pass_col = schema.find_column('pass', '良率', 'label')
        if not pass_col:
            # 用 schema 的目标列作为良率列
            tc = schema.get_target_column()
            if tc and tc in df.columns:
                pass_col = tc
                df['_Pass_Target'] = df[tc]
    else:
        # 无 schema 时启发式查找位置列
        for c in df.columns:
            if any(k in c.lower() for k in ['position', '位置', 'code', '工位', '区域', 'zone']):
                pos_col = c
                break

    # 尝试找良率列：优先 schema 目标列，其次 Is_Pass/Label_Pass，最后 0/1 列
    if not pass_col:
        for name in ['Is_Pass', 'Label_Pass', 'pass', 'Pass', 'label', 'Label', '良率', '合格']:
            if name in df.columns:
                pass_col = name
                break
    if not pass_col:
        for c in df.columns:
            uniq = set(str(u) for u in df[c].dropna().unique())
            if uniq <= {"0", "1"}:
                pass_col = c
                break

    if not pass_col:
        print("错误: 数据中未找到良率/目标列")
        return []

    if pos_col not in df.columns:
        print("信息: 无位置编码列，跳过位置分析")
        return []

    # 确保没有空值
    df_pos = df.dropna(subset=[pos_col, pass_col]).copy()

    def extract_pos_num(val):
        match = re.search(r'(\d+)', str(val))
        return int(match.group(1)) if match else 999

    unique_positions = sorted(df_pos[pos_col].unique(), key=extract_pos_num)
    df_pos = df_pos[df_pos[pos_col].isin(unique_positions)]
    print(f"分析位置范围: {unique_positions}")

    # ==========================================
    # 2. 聚合统计
    # ==========================================
    agg_dict = {pass_col: 'mean'}
    
    # 通用：取前 3 个数值特征作为辅助分析列
    for feat in _numeric_features(schema, df_pos, exclude_cols=[pos_col, pass_col])[:3]:
        if feat in df_pos.columns:
            agg_dict[feat] = 'mean'

    rename_map = {pass_col: 'Yield_Rate'}
    summary_table = df_pos.groupby(pos_col).agg(agg_dict).rename(columns=rename_map)

    print("\n=== 各位置统计摘要 ===")
    print(summary_table.reindex(unique_positions))

    # ==========================================
    # 3. 可视化
    # ==========================================
    plt.rcParams['figure.dpi'] = 120

    results = []

    # --- 图表 1: 目标均值排行 ---
    pos_display = _display_name(schema, pos_col)
    label_display = _display_name(schema, pass_col)
    print(f"\n绘制图表 1: 各{pos_display}排行...")
    plt.figure(figsize=(12, 6))
    bar = sns.barplot(x=pos_col, y=pass_col, data=df_pos,
                      order=unique_positions, ci=None, palette="viridis")
    avg_yield = df_pos[pass_col].mean()
    plt.axhline(y=avg_yield, color='r', linestyle='--', label=f'全局均值 {avg_yield:.1%}')
    plt.title(f'各{pos_display}排名 ({pos_display} Ranking)', fontsize=14)
    plt.ylabel(f'{label_display} (Rate)')
    plt.xlabel(f'{pos_display} (Position)')
    # 仅二分类目标使用 [0,1] 区间，否则自动缩放
    if schema and schema.target_type == 'binary':
        plt.ylim(0, 1.15)
    for p in bar.patches:
        height = p.get_height()
        if height > 0:
            bar.annotate(f'{height:.1%}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=10)
    plt.legend(loc='lower right')
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, '1_Position_Yield_Rate.png')
    plt.savefig(chart1_path)
    plt.close()
    results.append({"chart_name": f"{pos_display}排行", "image_path": chart1_path})

    # --- 图表 2: 堆叠图 ---
    pass_label = schema.pass_label if schema and schema.pass_label else 'Pass'
    fail_label = schema.fail_label if schema and schema.fail_label else 'Fail'
    print(f"绘制图表 2: {pass_label}/{fail_label}分布堆叠图...")
    plt.figure(figsize=(14, 7))
    df_pos['Status_Str'] = df_pos[pass_col].map({1: f'Pass ({pass_label})', 0: f'Fail ({fail_label})'})
    ct = pd.crosstab(df_pos[pos_col], df_pos['Status_Str'], normalize='index')
    ct = ct.reindex(unique_positions)
    colors_map = {f'Fail ({fail_label})': '#E74C3C', f'Pass ({pass_label})': '#2ECC71'}
    valid_cols = [c for c in ct.columns if c in colors_map]
    color_list = [colors_map[c] for c in valid_cols]
    ct[valid_cols].plot(kind='bar', stacked=True, color=color_list,
                        figsize=(14, 7), edgecolor='white', width=0.8)
    plt.title(f'各{pos_display}{pass_label}/{fail_label}占比 (Stacked)', fontsize=14)
    plt.ylabel('占比 (Ratio)')
    plt.xlabel(pos_display)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, '2_Position_Pass_Fail_Ratio.png')
    plt.savefig(chart2_path)
    plt.close()
    results.append({"chart_name": f"{pass_label}{fail_label}堆叠", "image_path": chart2_path})

    # --- 图表 3: 数值特征分布箱线图 ---
    print("绘制图表 3: 特征分布...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    numeric_feats = _numeric_features(schema, df_pos, exclude_cols=[pos_col, pass_col])
    feat1 = numeric_feats[0] if len(numeric_feats) > 0 else None
    feat2 = numeric_feats[1] if len(numeric_feats) > 1 else None

    if feat1 and feat1 in df_pos.columns:
        sns.boxplot(x=pos_col, y=feat1, data=df_pos,
                    order=unique_positions, ax=axes[0], palette="Blues", showfliers=False)
        axes[0].set_title(f'{_display_name(schema, feat1)} 分布', fontsize=12)
        axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
        axes[0].set_xlabel("")
    else:
        axes[0].text(0.5, 0.5, "缺少特征数据", ha='center')

    if feat2 and feat2 in df_pos.columns:
        sns.boxplot(x=pos_col, y=feat2, data=df_pos,
                    order=unique_positions, ax=axes[1], palette="Greens", showfliers=False)
        global_median = df_pos[feat2].median()
        axes[1].axhline(global_median, color='red', linestyle='--', alpha=0.6,
                        label=f'全局中位: {global_median:.2f}')
        axes[1].set_title(f'{_display_name(schema, feat2)} 分布', fontsize=12)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "缺少特征数据", ha='center')

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, '3_Position_Physical_Features.png')
    plt.savefig(chart3_path)
    plt.close()
    results.append({"chart_name": "特征分布", "image_path": chart3_path})

    print(f"\n[Done] 位置分析图表已保存至: {output_dir}")
    return results


if __name__ == "__main__":
    # 子进程入口: 使用默认路径，并加载贯穿整条流水线的 DataSchema
    schema = load_or_build_schema()
    run_position_analysis(schema=schema)
