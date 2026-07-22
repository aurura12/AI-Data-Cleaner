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

# ==========================================
# 0. 环境配置：字体与绘图风格
# ==========================================

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

    print(f"--- 读取数据: {os.path.basename(input_path)} ---")
    try:
        df = pd.read_csv(input_path)
        print(f"数据行数: {len(df)}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_path}")
        return []

    # --- 语义列名解析 ---
    if schema:
        pos_col = schema.find_column('position', 'code', '位置') or 'Position_Code'
        pass_col = schema.find_column('pass', '良率', 'label') or 'Is_Pass'
    else:
        pos_col = 'Position_Code'
        pass_col = 'Is_Pass'

    if pass_col not in df.columns and 'Label_Pass' in df.columns:
        df['Is_Pass'] = df['Label_Pass']
        pass_col = 'Is_Pass'

    if pass_col not in df.columns:
        print("错误: 数据中未找到良率列")
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
    height_col = 'Total_Indium_Height'
    if schema:
        height_col = schema.find_column('height', 'indium', '高度') or height_col

    if height_col in df_pos.columns:
        agg_dict[height_col] = 'mean'

    if 'Force_kg' in df_pos.columns:
        agg_dict['Force_kg'] = 'mean'
    elif '倒焊压力' in df_pos.columns:
        agg_dict['倒焊压力'] = 'mean'

    if 'Indium_Taper_Zscore' in df_pos.columns:
        agg_dict['Indium_Taper_Zscore'] = 'mean'

    rename_map = {pass_col: 'Yield_Rate'}
    summary_table = df_pos.groupby(pos_col).agg(agg_dict).rename(columns=rename_map)

    print("\n=== 各位置统计摘要 ===")
    print(summary_table.reindex(unique_positions))

    # ==========================================
    # 3. 可视化
    # ==========================================
    plt.rcParams['figure.dpi'] = 120

    results = []

    # --- 图表 1: 良率排行 ---
    print("\n绘制图表 1: 各位置良率排行...")
    plt.figure(figsize=(12, 6))
    bar = sns.barplot(x=pos_col, y=pass_col, data=df_pos,
                      order=unique_positions, ci=None, palette="viridis")
    avg_yield = df_pos[pass_col].mean()
    plt.axhline(y=avg_yield, color='r', linestyle='--', label=f'全局平均良率 {avg_yield:.1%}')
    plt.title('各位置良率排行 (Position Yield)', fontsize=14)
    plt.ylabel('良率 (Pass Rate)')
    plt.xlabel('位置编码 (Position)')
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
    results.append({"chart_name": "位置良率排行", "image_path": chart1_path})

    # --- 图表 2: 堆叠图 ---
    print("绘制图表 2: 良品/不良品分布堆叠图...")
    plt.figure(figsize=(14, 7))
    pass_label = schema.pass_label if schema and schema.pass_label else '良品'
    fail_label = schema.fail_label if schema and schema.fail_label else '不良'
    df_pos['Status_Str'] = df_pos[pass_col].map({1: f'Pass ({pass_label})', 0: f'Fail ({fail_label})'})
    ct = pd.crosstab(df_pos[pos_col], df_pos['Status_Str'], normalize='index')
    ct = ct.reindex(unique_positions)
    colors_map = {f'Fail ({fail_label})': '#E74C3C', f'Pass ({pass_label})': '#2ECC71'}
    valid_cols = [c for c in ct.columns if c in colors_map]
    color_list = [colors_map[c] for c in valid_cols]
    ct[valid_cols].plot(kind='bar', stacked=True, color=color_list,
                        figsize=(14, 7), edgecolor='white', width=0.8)
    plt.title('各位置良品/不良品占比 (Stacked)', fontsize=14)
    plt.ylabel('占比 (Ratio)')
    plt.xlabel('位置编码')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, '2_Position_Pass_Fail_Ratio.png')
    plt.savefig(chart2_path)
    plt.close()
    results.append({"chart_name": "良品不良品堆叠", "image_path": chart2_path})

    # --- 图表 3: 物理参数箱线图 ---
    print("绘制图表 3: 物理特征分布...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    taper_col = 'Indium_Taper_Zscore'
    if 'Indium_Taper_Zscore' in df_pos.columns:
        sns.boxplot(x=pos_col, y='Indium_Taper_Zscore', data=df_pos,
                    order=unique_positions, ax=axes[0], palette="Blues", showfliers=False)
        axes[0].set_title('铟柱锥度 Z-Score 分布', fontsize=12)
        axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
        axes[0].set_xlabel("")
    else:
        axes[0].text(0.5, 0.5, "缺少锥度数据", ha='center')

    if height_col in df_pos.columns:
        sns.boxplot(x=pos_col, y=height_col, data=df_pos,
                    order=unique_positions, ax=axes[1], palette="Greens", showfliers=False)
        global_median = df_pos[height_col].median()
        axes[1].axhline(global_median, color='red', linestyle='--', alpha=0.6,
                        label=f'全局中位: {global_median:.2f}')
        axes[1].set_title(f'{height_col} 分布', fontsize=12)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "缺少高度数据", ha='center')

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, '3_Position_Physical_Features.png')
    plt.savefig(chart3_path)
    plt.close()
    results.append({"chart_name": "物理特征分布", "image_path": chart3_path})

    print(f"\n[Done] 位置分析图表已保存至: {output_dir}")
    return results


if __name__ == "__main__":
    # 子进程入口: 使用默认路径，无 schema
    run_position_analysis()
