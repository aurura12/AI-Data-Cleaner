import os
import json
import re
import sys
import pandas as pd
import requests
from typing import Dict, Optional
from openai import OpenAI
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import (
    ROOT_DIR,
    OUTPUT_DIR,
    CLEANED_DATA_FILE,
    ANALYSIS_SUMMARY_FILE,
    AI_TEXT_RESULTS_FILE,
)

# ==========================================
# 1. 文本分析结果收集
# ==========================================

def collect_analysis_text_data(base_dir: str) -> Dict:
    """
    收集所有文本分析结果（CSV、统计数据等）
    优先读取analysis_summary.json，如果不存在则从CSV文件计算
    返回: 包含所有分析结果的字典
    """
    analysis_data = {
        'kpi_stats': {},
        'position_stats': {},
        'eda_summary': {}
    }
    
    output_dir = OUTPUT_DIR
    
    # 优先尝试读取摘要JSON文件
    summary_file = os.path.join(output_dir, 'analysis_summary.json')
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # 只收集 EDA 和 Position 数据，移除 ML 相关
            if 'eda_analysis' in summary:
                eda = summary['eda_analysis']
                
                # KPI统计
                if 'yield_stats' in eda:
                    yield_stats = eda['yield_stats']
                    analysis_data['kpi_stats'] = {
                        'total': yield_stats.get('total', 0),
                        'pass_rate': yield_stats.get('pass_rate', 0) * 100,
                        'pass_count': yield_stats.get('pass_count', 0),
                        'fail_count': yield_stats.get('fail_count', 0)
                    }
                
                # 高度统计
                if 'height_stats' in eda:
                    analysis_data['kpi_stats']['height_stats'] = eda['height_stats']
                
                # 位置统计
                if 'position_stats' in eda:
                    pos = eda['position_stats']
                    analysis_data['position_stats'] = {
                        'worst_position': pos.get('worst', {}),
                        'best_position': pos.get('best', {})
                    }
                
                # 时间趋势
                if 'time_trend' in eda:
                    analysis_data['eda_summary'] = {
                        'height_drift': {
                            'early_mean': eda['time_trend'].get('early_mean_height', 0),
                            'late_mean': eda['time_trend'].get('late_mean_height', 0),
                            'drift_amount': abs(eda['time_trend'].get('drift', 0)),
                            'drift_trend': 'negative' if eda['time_trend'].get('drift', 0) < 0 else 'positive'
                        }
                    }
            
            print("从analysis_summary.json加载数据摘要（KPI将从CSV重新计算）")
        except Exception as e:
            print(f"读取摘要JSON失败: {e}，将使用CSV文件计算")
    
    # 1. 读取清洗后的数据，计算KPI（总是重新计算，确保使用最新数据）
    data_file = os.path.join(output_dir, 'cleaned_chip_data_final.csv')
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        
        # 计算KPI（优先使用标准标签列，避免偏差）
        total = len(df)
        pass_count = 0
        fail_count = 0
        status_counts = {}

        if 'Label_Pass' in df.columns:
            pass_count = int((df['Label_Pass'] == 1).sum())
            fail_count = int((df['Label_Pass'] == 0).sum())
        elif 'Is_Pass' in df.columns:
            pass_count = int((df['Is_Pass'] == 1).sum())
            fail_count = int((df['Is_Pass'] == 0).sum())

        target_col = next((c for c in df.columns if '压连' in c), None)
        if target_col:
            for val in df[target_col].dropna():
                try:
                    v = int(float(val))
                    status_counts[v] = status_counts.get(v, 0) + 1
                except:
                    pass

        if pass_count == 0 and fail_count == 0 and status_counts:
            pass_count = status_counts.get(0, 0) + status_counts.get(1, 0)
            fail_count = status_counts.get(-1, 0) + status_counts.get(2, 0)

        analysis_data['kpi_stats'] = {
            'total': total,
            'pass_count': pass_count,
            'fail_count': fail_count,
            'pass_rate': (pass_count / total * 100) if total > 0 else 0,
            'open_count': status_counts.get(-1, 0),
            'severe_count': status_counts.get(2, 0),
            'open_rate': (status_counts.get(-1, 0) / total * 100) if total > 0 else 0,
            'severe_rate': (status_counts.get(2, 0) / total * 100) if total > 0 else 0,
            'status_distribution': status_counts
        }

        # 关键特征统计
        if 'Total_Indium_Height' in df.columns:
            analysis_data['kpi_stats']['height_stats'] = {
                'mean': float(df['Total_Indium_Height'].mean()),
                'median': float(df['Total_Indium_Height'].median()),
                'std': float(df['Total_Indium_Height'].std()),
                'min': float(df['Total_Indium_Height'].min()),
                'max': float(df['Total_Indium_Height'].max())
            }
            
        # 压力统计 (新增)
        pressure_col = next((c for c in df.columns if '压力' in c), None)
        if pressure_col:
            analysis_data['kpi_stats']['pressure_stats'] = {
                'mean': float(df[pressure_col].mean()),
                'min': float(df[pressure_col].min()),
                'max': float(df[pressure_col].max())
            }
    
    # 2. 读取位置分析结果（从CSV或重新计算）
    pos_data_file = os.path.join(output_dir, 'cleaned_chip_data_final.csv')
    if os.path.exists(pos_data_file):
        df_pos = pd.read_csv(pos_data_file)
        if 'Position_Code' in df_pos.columns and 'Is_Pass' in df_pos.columns:
            pos_stats = df_pos.groupby('Position_Code')['Is_Pass'].agg(['mean', 'count']).reset_index()
            pos_stats.columns = ['Position_Code', 'Yield_Rate', 'Count']
            analysis_data['position_stats'] = {
                'position_yield': pos_stats.to_dict('records'),
                'worst_position': pos_stats.loc[pos_stats['Yield_Rate'].idxmin()].to_dict() if len(pos_stats) > 0 else {},
                'best_position': pos_stats.loc[pos_stats['Yield_Rate'].idxmax()].to_dict() if len(pos_stats) > 0 else {}
            }
    
    # 3. EDA摘要（从数据计算）
    if os.path.exists(data_file):
        df_eda = pd.read_csv(data_file)
        if 'Time_Seq_Day' in df_eda.columns and 'Total_Indium_Height' in df_eda.columns:
            # 计算时间趋势
            median_day = df_eda['Time_Seq_Day'].quantile(0.5)
            early_mean = float(df_eda[df_eda['Time_Seq_Day'] < median_day]['Total_Indium_Height'].mean())
            late_mean = float(df_eda[df_eda['Time_Seq_Day'] >= median_day]['Total_Indium_Height'].mean())
            
            analysis_data['eda_summary'] = {
                'height_drift': {
                    'early_mean': early_mean,
                    'late_mean': late_mean,
                    'drift_trend': 'negative' if late_mean < early_mean else 'positive',
                    'drift_amount': abs(late_mean - early_mean)
                }
            }
    
    return analysis_data

# ==========================================
# 2. 使用文本LLM生成分析报告
# ==========================================

def generate_text_based_report(
    analysis_data: Dict,
    model_name: str = "qwen-plus",
    chart_data_text: str = None
) -> str:
    """
    基于文本分析结果和图表数据，使用LLM生成综合报告
    """
    # 获取KPI数据用于填充模板
    kpi = analysis_data.get('kpi_stats', {})
    total = kpi.get('total', 0)
    pass_rate = kpi.get('pass_rate', 0)
    pass_count = kpi.get('pass_count', 0)
    fail_count = kpi.get('fail_count', 0)
    open_rate = kpi.get('open_rate', 0)
    open_count = kpi.get('open_count', 0)
    severe_rate = kpi.get('severe_rate', 0)
    severe_count = kpi.get('severe_count', 0)
    status_dist = kpi.get('status_distribution', {})
    
    replacements = {
        "Total_Indium_Height": "总铟柱高度",
        "Calc_Circuit_Range": "电路端平整度",
        "Indium_Taper_Zscore": "铟柱形状异常度",
        "Force_kg": "倒焊压力",
        "Equipment_Temp": "设备温度",
        "Vacuum_Level": "真空度",
        "Time_Seq_Day": "生产天数",
        "Wafer_Index": "晶圆次序",
        "Position_Code": "位置编码",
        "Label_Pass": "良品标签",
        "Is_Pass": "是否良品",
        "Pass": "良品",
        "Fail": "不良",
    }

    analysis_summary = f"""
【数据分析结果摘要 - 请务必严格引用以下数值】

1. 核心KPI指标：
   - 总样本数：{total}
   - 整体良品率：{pass_rate:.2f}% (良品共{pass_count}颗)
   - 整体不良率：{(100 - pass_rate):.2f}% (不良品共{fail_count}颗)
   - 严重压连(2)：占比{severe_rate:.2f}%，共{severe_count}颗 [这是主要的失效原因]
   - 虚焊(-1)：占比{open_rate:.2f}%，共{open_count}颗
   - 良好(0)：{status_dist.get(0, 0)}颗
   - 轻微压连(1)：{status_dist.get(1, 0)}颗
   - [重要逻辑] 良品(Pass) = 良好(0) + 轻微压连(1)；不良(Fail) = 严重压连(2) + 虚焊(-1)。

2. 关键特征统计：
   - 总铟柱高度：良品中位数约12.10μm，不良品中位数约11.41μm，差异约0.69μm。
   - 倒焊压力：均值 {analysis_data['kpi_stats'].get('pressure_stats', {}).get('mean', 0):.2f} kg，范围 {analysis_data['kpi_stats'].get('pressure_stats', {}).get('min', 0):.2f} - {analysis_data['kpi_stats'].get('pressure_stats', {}).get('max', 0):.2f} kg。
   - [警告] 严禁使用 10-12kg，必须使用 18-24kg 这一真实区间。

3. 位置良率分析：
   - 最差位置：{analysis_data['position_stats'].get('worst_position', {}).get('Position_Code', 'N/A')}，良率为 {analysis_data['position_stats'].get('worst_position', {}).get('Yield_Rate', 0)*100:.1f}%。
   - 最佳位置：{analysis_data['position_stats'].get('best_position', {}).get('Position_Code', 'N/A')}，良率为 {analysis_data['position_stats'].get('best_position', {}).get('Yield_Rate', 0)*100:.1f}%。
"""

    for k, v in replacements.items():
        analysis_summary = analysis_summary.replace(k, v)
    
    # 如果提供了图表数据，添加到摘要中
    if chart_data_text:
        for k, v in replacements.items():
            chart_data_text = chart_data_text.replace(k, v)
        analysis_summary += f"\n\n【图表数据摘要（从绘图代码提取的文字数据）】\n{chart_data_text}"
    
    # 构建与图像识别完全相同的HTML模板提示词
    prompt = f"""【角色设定】
你现在是一位半导体良率优化专家（Yield Optimization Expert）和机器学习工程师。
我们已经完成了前期的 EDA（探索性数据分析）和位置分析阶段，现在进入给出建议总结分析阶段。

【任务目标】
根据所有图表分析结果和提供的业务背景，生成一份综合评估报告。报告需识别可能与良率下降相关的物理参数特征和数据模式，** 为产线工程师提供决策参考和排查线索**（非绝对执行指令）。分析应基于实际数据，使用客观、统计学的口吻，严格确保引用的数据、占比、百分比、物理量范围与【分析结果摘要】完全一致，严禁虚构或凭空猜测数据区间。

【语言与表述要求（必须遵守）】
1. 全文只用中文表达，禁止出现类似 Total_Indium_Height、Indium_Taper_Zscore、Equipment_Temp 这类英文/下划线字段名。
2. 禁止出现 Pass/Fail 这类英文类别名，统一使用“良品/不良”。
3. 字段必须使用中文名：总铟柱高度、电路端平整度、铟柱形状异常度、倒焊压力、设备温度、真空度、生产天数、晶圆次序、位置编码。
4. 允许出现位置编码 M1~M10（例如 M5），除此之外尽量避免英文缩写。

【数据概况与业务逻辑】
数据集：cleaned_chip_data_final.csv
预测目标（Label）：
原始列：压连情况（-1=虚焊, 0=正常, 1=轻微压连, 2=严重压连）。
建模目标：二分类（Binary Classification）。
逻辑：{{-1, 2}} = 不良 (0) ；{{0, 1}} = 良品 (1) 。（注意：这是一个非平衡数据集，不良样本较少但成本极高）。
关键特征：
* 总铟柱高度：核心特征，上下高度和。
* 电路端平整度：注意存在缺失值，仅激光调平工艺有此值。
* 铟柱形状异常度：反映形状偏离的程度。
* 倒焊压力：工艺设定压力。
* 生产天数：用来捕捉设备老化/漂移。
* 晶圆次序：反映批次效应。
* 位置编码：空间位置（M1~M10）。
补充说明：铟柱高度与倒焊压力不存在因果关系，仅可做相关性或共现分析，避免直接推断因果。

【已完成的图表分析（基于文字数据）】
{analysis_summary}

【参考报告结构（必须严格遵循）】
请严格按照以下HTML结构生成报告内容，保持与模板完全一致的格式和风格：

<div class="section-card">
  <h2>2. 分布现状</h2>
  
  <h3>2.1 压连结果分布</h3>
  <div class="chart-wrapper">
    <img src="output/analysis_report/0_生产状态分布统计.png" alt="图表: 生产状态分布">
  </div>
  <p>（分析四类生产状态的分布情况，指出严重压连是主要失效原因，并说明各状态的具体数量和占比。每个段落至少3-5句话，避免占位符文字。**务必将对该图的分析与建议紧跟在此图下方呈现，一图一段**）</p>

  <h3>2.2 关键参数特征差异</h3>
  <div class="chart-wrapper">
    <img src="output/analysis_report/2_核心特征分布_2x3_中文.png" alt="图表: 核心特征分布">
  </div>
  <p>（对比Pass/Fail样本在关键参数上的分布差异，重点分析总铟柱高度、倒焊压力、铟柱形状异常度等。指出关键阈值和风险区间。每个段落至少3-5句话，避免占位符文字。**务必将分析与建议紧跟在此图下方，一图一段**）</p>

  <h3>2.3 参数间关联性与"实验设计复盘"</h3>
  <div class="chart-wrapper">
    <img src="output/analysis_report/1_参数相关性分析.png" alt="图表: 相关性分析">
  </div>
  <div class="analysis-box">
    <p>（分析相关性热力图，揭示人为操作模式。每个段落至少3-5句话，避免占位符文字。）</p>
    <ul>
      <li><strong>良率的物质基础：</strong> 良率与总铟柱高度的正相关关系，解释其物理机制。</li>
      <li><strong>揭示不良的操作模式：</strong> 压力与高度的负相关关系，说明对低高度物料施加了高压力。</li>
      <li><strong>叠加风险：</strong> 失效样本在"低高度+差形状+高压力"三重恶劣条件下产生，解释其影响。</li>
    </ul>
  </div>
</div>

<div class="section-card">
  <h2>3. 工艺与设备维度深潜</h2>

  <h3>3.1 位置编码异质性分析：以 M5 为基准的偏差诊断</h3>
  <div class="chart-wrapper">
    <img src="output/position_analysis_v2/1_Position_Yield_Rate.png" alt="图表: 空间效应良率">
  </div>
  <div class="chart-wrapper">
    <img src="output/position_analysis_v2/2_Position_Failure_Detail.png" alt="图表: 空间效应缺陷详情">
  </div>
  <div class="analysis-box">
    <p>（分析各位置的良率差异，锁定M7、M8、M2、M4等异常位置，并解释其具体问题。每个段落至少3-5句话，避免占位符文字。**分析与建议紧跟该图下方，一图一段**）</p>
  </div>
  <div class="chart-wrapper">
    <img src="output/position_analysis_v2/3_Position_Physical_Features.png" alt="图表: 物理一致性特征">
  </div>
  <div class="analysis-box">
    <p>（以M5为稳定性标杆，分析其他位置的物理偏差，如系统性高度偏移、一致性失控、形状异常等。每个段落至少3-5句话，避免占位符文字。**分析与建议紧跟该图下方，一图一段**）</p>
  </div>
</div>

【输出要求】
1. 必须严格按照上述HTML结构生成，包括所有div与class属性
2. 使用图表路径时，必须使用相对路径（如 output/analysis_report/0_生产状态分布统计.png）
3. 分析内容要求：
   - 基于提供的图表分析结果（文字数据），使用客观、统计学的口吻进行深入、专业的分析
   - 严禁虚构数据，严禁引用错误的压力或高度数值区间，必须与上方提供的统计结果对齐
   - 每个段落至少3-5句话，避免占位符文字
   - 每个图表对应的段落必须包含至少一句建议（以“建议”开头或包含“建议”）
   - **避免使用绝对化语言**（如"必须设定"、"禁止超过"、"一定导致"等）
   - **采用建议性话术**（如"建议关注"、"风险可能显著增加"、"建议维持在"、"需关注区间"、"值得排查"等）
   - 指出潜在的高风险参数区间或需要重点监控的异常点，但以建议和参考的形式呈现
   - 数据应根据实际情况变化，基于实际统计结果进行分析
4. 保持与模板完全一致的视觉风格和结构
5. **重要：直接输出HTML代码，不要添加任何markdown代码块标记（如```html或```），不要添加任何解释文字，只输出纯HTML内容**
6. **严禁在 HTML 标签闭合处添加额外的引号或非标签文字（如 </div>" 或 alt="..."），确保 HTML 语法完全正确。**

请开始生成综合报告："""

    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_API_BASE") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = ""
        if resp and resp.choices:
            content = resp.choices[0].message.content
        if content:
            content = content.strip()
            lines = content.split('\n')
            if lines and (lines[0].startswith('```html') or lines[0].startswith('```')):
                lines = lines[1:]
                content = '\n'.join(lines).strip()
            if lines and (lines[-1].strip() == '```' or lines[-1].strip().startswith('```')):
                lines = lines[:-1]
                content = '\n'.join(lines).strip()
            content = re.sub(r'^```html?\s*\n?', '', content, flags=re.MULTILINE)
            content = re.sub(r'^```\s*\n?', '', content, flags=re.MULTILINE)
            content = re.sub(r'\n?```\s*$', '', content, flags=re.MULTILINE)
            content = content.strip()
            if not content.startswith('<div'):
                div_start = content.find('<div')
                if div_start > 0:
                    content = content[div_start:].strip()
            if not content.rstrip().endswith('</div>'):
                div_end = content.rfind('</div>')
                if div_end > 0:
                    content = content[:div_end + 6].strip()
            content = re.sub(r'src="([^"]+?\.png)\s+alt=', r'src="\1" alt=', content)
            content = re.sub(r'</(div|h\d|p|li)>"\s*alt="[^"]*"[^>]*>', r'</\1>', content)
            content = re.sub(r'</(div|h\d|p|li)>\s*alt="[^"]*"[^>]*>', r'</\1>', content)
            content = re.sub(r'<div class="\s*alt="[^"]+">', r'<div class="chart-wrapper">', content)
            content = re.sub(r'<h3>2\.3\s*参数间关联性与.*?</h3>', r'<h3>2.3 参数间关联性与“实验设计复盘”</h3>', content)
            content = re.sub(r'alt="[^"]*占位符[^"]*"', 'alt="图表"', content, flags=re.IGNORECASE)
            content = re.sub(r'(?<!<img)(?<!<div)(?<!<span)(?<!<p)\s+alt="[^"]+"', r'', content)
            content = re.sub(r'alt="([^"]+)"\s*"([^>]*>)', r'alt="\1"\2', content)
            path_fixes = {
                'output/position_analysis_v2/3_Position_Physical_Stats.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
                'output/position_analysis_v2/3_Position_Physical_Diff.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
                'output/position_analysis_v2/3_Position_Feature_Distribution.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
                'output/position_analysis_v2/3_Position_Physical_Characteristics.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
                'output/position_analysis_v2/3_Process_Parameter.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
                'output/position_analysis_v2/3_Position_Parameter_Drift.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
                'output/position_analysis_v2/3_Position_High_Risk.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
                'output/position_analysis_v2/3_Physical_Consistency.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
                'output/position_analysis_v2/4_Position_Parameter_Correlation.png': 'output/analysis_report/1_参数相关性分析.png',
                'output/position_analysis_v2/3_Position_Height_Distribution.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
                'output/position_analysis_v2/3_Position_Physical_Deviation.png': 'output/position_analysis_v2/3_Position_Physical_Features.png',
            }
            for wrong_path, correct_path in path_fixes.items():
                content = content.replace(wrong_path, correct_path)
            def _fix_img_path(match):
                src = match.group(1)
                for wrong_path, correct_path in path_fixes.items():
                    if wrong_path in src:
                        return match.group(0).replace(wrong_path, correct_path)
                if any(k in src for k in ['Physical_Stats', 'Physical_Diff', 'Feature_Distribution', 'Physical_Characteristics', 'Physical_Deviation', 'Height_Distribution', 'Process_Parameter', 'High_Risk', 'Physical_Consistency', 'Parameter_Drift']):
                    return match.group(0).replace(src, 'output/position_analysis_v2/3_Position_Physical_Features.png')
                if 'Parameter_Correlation' in src and 'position_analysis' in src:
                    return match.group(0).replace(src, 'output/analysis_report/1_参数相关性分析.png')
                return match.group(0)
            content = re.sub(r'<img src="([^"]+)"', _fix_img_path, content)
            content = re.sub(r'(<div class="chart-wrapper">.*?</div>)\s*<div class="chart-wrapper">', r'\1', content, flags=re.S)
            content = re.sub(r'<h2>3\. 时间空间维度</h2>', '<h2>3. 工艺与设备维度深潜</h2>', content)
            content = re.sub(r'<h2>3\. 时间、空间维度</h2>', '<h2>3. 工艺与设备维度深潜</h2>', content)
            content = re.sub(r'<h3>3\.1\s*时间稳定性与参数漂移.*?(?=<h3>|</div>)', '', content, flags=re.S)
            content = re.sub(r'<div class="chart-wrapper">.*?3_周度趋势分析.*?</div>\s*<p>.*?</p>', '', content, flags=re.S)
            content = re.sub(r'<div class="chart-wrapper">.*?4_高度长期漂移.*?</div>\s*<p>.*?</p>', '', content, flags=re.S)
            content = re.sub(r'<h3>3\.\d+\s*晶圆批次效应分析.*?(?=<h3>|</div>)', '', content, flags=re.S)
            content = re.sub(r'<div class="chart-wrapper">.*?6_晶圆次序效应分析.*?</div>\s*<p>.*?</p>', '', content, flags=re.S)
            content = re.sub(r'<h3>3\.2\s*位置编码异质性分析', '<h3>3.1 位置编码异质性分析', content)
            content = re.sub(r'<div class="section-card">\s*<h2>1\. 核心指标</h2>.*?</div>\s*(?=<div class="section-card">)', '', content, flags=re.S)
            def _ensure_suggestions(match):
                prefix, paragraph, suffix = match.group(1), match.group(2).strip(), match.group(3)
                if not paragraph or "建议" in paragraph:
                    return match.group(0)
                add_sentence = "建议结合该图所示趋势进行工艺复核与持续监控。"
                if paragraph.endswith("。"):
                    paragraph = paragraph + add_sentence
                else:
                    paragraph = paragraph + "。" + add_sentence
                return f"{prefix}{paragraph}{suffix}"
            content = re.sub(r'(<div class="chart-wrapper">.*?</div>\s*<p>)(.*?)(</p>)', _ensure_suggestions, content, flags=re.S)
            def _mark_advice(match):
                prefix, paragraph, suffix = match.group(1), match.group(2), match.group(3)
                if "建议" not in paragraph or "◆ 建议" in paragraph:
                    return match.group(0)
                paragraph = paragraph.replace("建议", "◆ 建议", 1)
                paragraph = paragraph.replace("◆ 建议", "<br>◆ 建议", 1)
                return f"{prefix}{paragraph}{suffix}"
            content = re.sub(r'(<p>)(.*?)(</p>)', _mark_advice, content, flags=re.S)
            return content
        return ""
    except Exception as e:
        print(f"报告生成异常: {e}")
        return ""

# ==========================================
# 3. 主函数
# ==========================================

def main():
    """主函数：基于文本分析结果生成AI报告"""
    base_dir = ROOT_DIR
    
    print("=" * 60)
    print("AI文本分析报告生成模块")
    print("=" * 60)
    
    # 1. 收集分析结果
    print("\n[步骤 1/3] 收集分析结果...")
    analysis_data = collect_analysis_text_data(base_dir)
    print(f"已收集KPI、位置分析、EDA数据")
    
    # 2. 提取图表数据（从绘图代码中提取）
    print("\n[步骤 2/4] 提取图表数据（从EDA/Position绘图代码）...")
    try:
        from chart_data_extractor import extract_all_chart_data
        chart_data_text, chart_data_dict = extract_all_chart_data(base_dir)
        print(f"已提取图表数据（长度: {len(chart_data_text)} 字符）")
        print(f"\n图表数据预览:\n{chart_data_text[:300]}...")
    except Exception as e:
        print(f"图表数据提取失败: {e}，将仅使用统计摘要")
        chart_data_text = None
    
    # 3. 生成AI报告
    print("\n[步骤 3/4] 使用LLM生成综合报告...")
    report_content = generate_text_based_report(
        analysis_data,
        model_name=os.getenv("DASHSCOPE_TEXT_MODEL") or "qwen-plus",
        chart_data_text=chart_data_text
    )
    
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'ai_text_analysis_results.json')
    
    if not report_content:
        print("AI报告生成失败")
        # 写入失败文件，供前端检测
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'error': True,
                'message': 'AI报告生成失败',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        return
    
    # 4. 保存结果
    print("\n[步骤 4/4] 保存分析结果...")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_data': analysis_data,
            'comprehensive_report': report_content,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, ensure_ascii=False, indent=2)
    
    print(f"分析结果已保存至: {results_file}")
    print(f"综合报告长度: {len(report_content)} 字符")
    
    return analysis_data, report_content

if __name__ == "__main__":
    main()
