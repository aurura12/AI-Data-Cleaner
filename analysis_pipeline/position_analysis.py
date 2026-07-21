import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适配无显示环境
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import re
import matplotlib.font_manager as fm

# ==========================================
# 0. 环境配置：字体与绘图风格
# ==========================================
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import FONT_FILE, CLEANED_DATA_FILE, POSITION_REPORT_DIR

input_path = CLEANED_DATA_FILE
output_dir = POSITION_REPORT_DIR

os.makedirs(output_dir, exist_ok=True)

sns.set(style="whitegrid", palette="deep")
plt.rcParams['axes.unicode_minus'] = False 

# 字体设置
sys_name = platform.system()
font_path = FONT_FILE
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print(f"成功加载自定义字体: {font_path}")
elif sys_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
elif sys_name == "Darwin":  # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

# ==========================================
# 1. 数据加载与预处理
# ==========================================
print(f"--- 读取数据: {os.path.basename(input_path)} ---")
try:
    df = pd.read_csv(input_path)
    print(f"数据行数: {len(df)}")
except FileNotFoundError:
    print(f"错误: 找不到文件 {input_path}")
    exit()

# --- 核心列名映射 ---
pos_col = 'Position_Code'

# 检查良率列 (优先使用 Is_Pass)
if 'Is_Pass' in df.columns:
    yield_col = 'Is_Pass'
elif 'Label_Pass' in df.columns:
    df['Is_Pass'] = df['Label_Pass']
    yield_col = 'Is_Pass'
else:
    print("错误: 数据中未找到良率列 (Is_Pass 或 Label_Pass)")
    yield_col = None

if yield_col:
    # 确保没有空值
    df_pos = df.dropna(subset=[pos_col, yield_col]).copy()

    # 【优化】智能自然排序 (M1 -> M2 -> ... -> M10)
    # 提取数字进行排序
    def extract_pos_num(val):
        match = re.search(r'(\d+)', str(val))
        return int(match.group(1)) if match else 999

    unique_positions = sorted(df_pos[pos_col].unique(), key=extract_pos_num)

    # 过滤无效位置
    df_pos = df_pos[df_pos[pos_col].isin(unique_positions)]

    print(f"分析位置范围: {unique_positions}")
else:
    exit()

# ==========================================
# 2. 统计分析
# ==========================================
# 聚合统计
agg_dict = {
    yield_col: 'mean',
    'Total_Indium_Height': 'mean', 
}

# ##### 【修改】: 适配 Force_kg #####
if 'Force_kg' in df_pos.columns: 
    agg_dict['Force_kg'] = 'mean'
elif '倒焊压力' in df_pos.columns: # 兼容旧列名
    agg_dict['倒焊压力'] = 'mean'

# 添加 Indium_Taper_Zscore 统计
if 'Indium_Taper_Zscore' in df_pos.columns:
    agg_dict['Indium_Taper_Zscore'] = 'mean'

summary_table = df_pos.groupby(pos_col).agg(agg_dict).rename(columns={yield_col: 'Yield_Rate'})

print("\n=== 各位置统计摘要 (Position Summary) ===")
print(summary_table.reindex(unique_positions))

# ==========================================
# 3. 可视化分析
# ==========================================
plt.rcParams['figure.dpi'] = 120

# --- 图表 1: 各位置良率对比 (M1 - M12...) ---
print("\n绘制图表 1: 各位置良率排行...")
plt.figure(figsize=(12, 6))

bar = sns.barplot(x=pos_col, y=yield_col, data=df_pos, 
                  order=unique_positions, ci=None, palette="viridis")

# ##### 【修改】: 动态计算全局平均良率，不再写死 #####
avg_yield = df_pos[yield_col].mean()
plt.axhline(y=avg_yield, color='r', linestyle='--', label=f'全局平均良率 {avg_yield:.1%}')

plt.title('各位置良率排行 (Position Yield)', fontsize=14)
plt.ylabel('良率 (Pass Rate)')
plt.xlabel('位置编码 (Position)')
plt.ylim(0, 1.15) 

# 标数值
for p in bar.patches:
    height = p.get_height()
    if height > 0:
        bar.annotate(f'{height:.1%}', 
                     (p.get_x() + p.get_width() / 2., height), 
                     ha='center', va='bottom', fontsize=10)

plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_Position_Yield_Rate.png'))
plt.close()

# --- 图表 2: 良率状态堆叠图 (Pass vs Fail) ---
# ##### 【修改】: 使用 Is_Pass 生成堆叠图，更加通用 #####
print("绘制图表 2: 良品/不良品分布堆叠图...")
plt.figure(figsize=(14, 7))
    
# 映射标签名
df_pos['Status_Str'] = df_pos[yield_col].map({1: 'Pass (良品)', 0: 'Fail (不良)'})

# 交叉表
ct = pd.crosstab(df_pos[pos_col], df_pos['Status_Str'], normalize='index')
ct = ct.reindex(unique_positions)
    
# 颜色: 不良=红, 良品=绿
colors = {'Fail (不良)': '#E74C3C', 'Pass (良品)': '#2ECC71'}
valid_cols = [c for c in ct.columns if c in colors]
color_list = [colors[c] for c in valid_cols]

ct[valid_cols].plot(kind='bar', stacked=True, color=color_list,
            figsize=(14, 7), edgecolor='white', width=0.8)

plt.title('各位置良品/不良品占比 (Stacked)', fontsize=14)
plt.ylabel('占比 (Ratio)')
plt.xlabel('位置编码')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_Position_Pass_Fail_Ratio.png'))
plt.close()

# --- 图表 3: 关键物理参数分布箱线图 ---
# ##### 【修改】: 确保列名正确存在才画图 #####
print("绘制图表 3: 物理特征分布...")
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# 子图1: 锥度
if 'Indium_Taper_Zscore' in df_pos.columns:
    sns.boxplot(x=pos_col, y='Indium_Taper_Zscore', data=df_pos, 
                order=unique_positions, ax=axes[0], palette="Blues", showfliers=False)
    axes[0].set_title('铟柱锥度 Z-Score 分布 (值越大形状越异常)', fontsize=12)
    axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_xlabel("")
else:
    axes[0].text(0.5, 0.5, "缺少 Indium_Taper_Zscore 数据", ha='center')

# 子图2: 总高度 (Total Height)
if 'Total_Indium_Height' in df_pos.columns:
    sns.boxplot(x=pos_col, y='Total_Indium_Height', data=df_pos, 
                order=unique_positions, ax=axes[1], palette="Greens", showfliers=False)
    
    global_median = df_pos['Total_Indium_Height'].median()
    axes[1].axhline(global_median, color='red', linestyle='--', alpha=0.6, label=f'全局中位: {global_median:.2f}')
    
    axes[1].set_title('总铟柱高度分布 (Total Height)', fontsize=12)
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, "缺少 Total_Indium_Height 数据", ha='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_Position_Physical_Features.png'))
plt.close()

print(f"\n[Done] 位置分析图表已保存至: {output_dir}")