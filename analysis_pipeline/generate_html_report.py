import os
import json
import base64
import sys
import re
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_paths import (
    ROOT_DIR,
    OUTPUT_DIR,
    CLEANED_DATA_FILE,
    AI_CHART_RESULTS_FILE,
    AI_TEXT_RESULTS_FILE,
    HTML_REPORT_FILE,
)

# 领域无关适配层
from domain_adapter import (
    load_or_build_schema,
    compute_kpi,
    collect_generic_analysis,
    build_static_report_html,
    discover_charts,
)

# ==========================================
# 1. 核心工具函数
# ==========================================
def get_base64_image(image_path):
    """读取本地图片并转换为 Base64 字符串；不存在则返回原路径。"""
    if not os.path.exists(image_path):
        return image_path
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

def load_ai_analysis_results(results_file: str) -> dict:
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def build_kpi_cards(kpi: dict) -> str:
    """基于通用 KPI 生成卡片 HTML（不依赖任何行业字段名）。"""
    total = kpi.get("total", 0)
    cards = []
    cards.append(f"""
        <div class="kpi-card">
          <div class="kpi-title">样本总数</div>
          <div class="kpi-value">{total}</div>
          <div class="kpi-sub">进入分析的全量记录</div>
        </div>""")

    if "pass_rate" in kpi:
        cards.append(f"""
        <div class="kpi-card">
          <div class="kpi-title">{kpi.get('target_label', '合格率')}</div>
          <div class="kpi-value">{kpi['pass_rate']:.2f}%</div>
          <div class="kpi-sub">合格 {kpi.get('pass_count', 0)} 个</div>
        </div>""")
        cards.append(f"""
        <div class="kpi-card">
          <div class="kpi-title">不合格率</div>
          <div class="kpi-value" style="color: #e74c3c;">{100 - kpi['pass_rate']:.2f}%</div>
          <div class="kpi-sub">不合格 {kpi.get('fail_count', 0)} 个</div>
        </div>""")
    else:
        # 无 pass/fail 映射：展示目标列取值分布
        dist = kpi.get("value_distribution", {})
        top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        for k, v in top:
            cards.append(f"""
        <div class="kpi-card">
          <div class="kpi-title">目标值 {k}</div>
          <div class="kpi-value">{v}</div>
          <div class="kpi-sub">占比 {v/total*100:.1f}%</div>
        </div>""")
    return "\n".join(cards)

# ==========================================
# 2. 报告主体
# ==========================================

# 复用原模板的 CSS 样式（与行业无关，仅视觉风格）
CSS_BLOCK = r"""
  <style>
    :root {
      --primary-blue: #3498db;
      --dark-text: #2c3e50;
      --light-text: #7f8c8d;
      --bg-color: #eff2f7;
      --card-bg: #ffffff;
      --green-bg: #dff0d8;
      --green-border: #d6e9c6;
      --green-text: #3c763d;
      --accent-green: #e1f3d8;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg-color);
      color: var(--dark-text);
      margin: 0;
      padding: 40px 20px;
      line-height: 1.6;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    .report-header { text-align: center; margin-bottom: 50px; }
    .report-header h1 { font-size: 36px; color: var(--dark-text); margin-bottom: 10px; font-weight: 700; letter-spacing: 1px; }
    .report-header p { color: var(--light-text); font-size: 16px; }
    .section-card { background: var(--card-bg); border-radius: 8px; padding: 30px; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    h2 { font-size: 24px; color: var(--primary-blue); margin-top: 0; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 2px solid var(--primary-blue); position: relative; }
    h3 { font-size: 18px; color: #444; margin-top: 30px; margin-bottom: 15px; font-weight: 600; border-left: 4px solid var(--primary-blue); padding-left: 10px; }
    .kpi-container { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 40px; }
    .kpi-card { flex: 1; background: #fff; border-radius: 12px; padding: 30px 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.05); transition: transform 0.2s; }
    .kpi-card:hover { transform: translateY(-5px); }
    .kpi-title { font-size: 16px; color: var(--dark-text); font-weight: 600; margin-bottom: 15px; }
    .kpi-value { font-size: 42px; font-weight: bold; color: var(--primary-blue); margin-bottom: 10px; }
    .kpi-sub { font-size: 13px; color: var(--light-text); }
    p { color: #555; font-size: 15px; margin-bottom: 15px; }
    .chart-wrapper { background: #fcfcfc; border: 1px solid #eee; border-radius: 8px; padding: 10px; margin: 20px 0; text-align: center; }
    img { max-width: 100%; height: auto; border-radius: 4px; }
    .advice-list { display: flex; flex-direction: column; gap: 15px; }
    .advice-item { background-color: #dcedc8; border-left: 5px solid #7cb342; color: #33691e; padding: 15px 20px; border-radius: 6px; display: flex; align-items: flex-start; font-size: 15px; line-height: 1.5; }
    .advice-icon { font-size: 20px; margin-right: 15px; margin-top: -2px; min-width: 24px; }
    .advice-content strong { display: block; margin-bottom: 4px; color: #2e5c18; font-size: 16px; }
    .tech-pill { display: inline-block; background: #e3f2fd; color: #1976d2; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-right: 5px; margin-bottom: 5px; }
    .analysis-box { background-color: #fdf6ec; border-left: 5px solid #e6a23c; padding: 20px; border-radius: 4px; color: #606266; margin-top: 20px; }
    @media (max-width: 768px) { .kpi-container { flex-direction: column; } .report-header h1 { font-size: 28px; } }
  </style>
"""

print("🔄 正在生成完整 HTML 报告...")

base_dir = ROOT_DIR

# 1. 加载 DataSchema（贯穿理解）
schema = load_or_build_schema()
df = pd.read_csv(CLEANED_DATA_FILE) if os.path.exists(CLEANED_DATA_FILE) else pd.DataFrame()

# 2. 通用 KPI
kpi_data = compute_kpi(df, schema) if not df.empty else {}
if not kpi_data or kpi_data.get("total", 0) == 0:
    kpi_data = {"total": len(df)}

# 3. 通用分析数据 + 可用图表
analysis = collect_generic_analysis(schema=schema)
chart_paths = discover_charts(ROOT_DIR)

# 4. 报告正文：优先 AI 生成（领域无关），否则通用静态兜底
ai_analysis = load_ai_analysis_results(AI_TEXT_RESULTS_FILE)
if not ai_analysis or not ai_analysis.get('comprehensive_report'):
    ai_analysis = load_ai_analysis_results(AI_CHART_RESULTS_FILE)

report_body = ""
if ai_analysis and ai_analysis.get('comprehensive_report'):
    report_body = ai_analysis['comprehensive_report']
    print("✅ 使用 AI 生成的领域无关报告正文")
else:
    report_body = build_static_report_html(analysis, schema)
    print("⚠️ 未检测到 AI 分析结果，使用通用静态报告正文")

# 5. 标题与导语（通用，基于 schema 而非硬编码行业）
target_label = schema.target_column or "目标"
if schema and schema.target_column:
    target_desc = f"目标列「{schema.target_column}」（{schema.target_type}）"
else:
    target_desc = "已识别的数据目标"
intro_text = (
    f"本报告围绕{target_desc}展开多维度数据分析，涵盖分布现状、关键特征差异、"
    f"参数关联性与机器学习归因等环节，并在此基础上给出优化建议。"
    f"全部分析结论均由数据自动统计与模型推导得出。"
)

current_date = datetime.now().strftime('%Y-%m-%d')
part_1 = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>数据分析与质量洞察报告</title>
{CSS_BLOCK}
</head>
<body>
  <div class="container">
    <header class="report-header">
      <h1>数据分析与质量洞察报告</h1>
      <p>基于全量数据的多维度参数关联性与归因洞察 | 报告生成日期：{current_date}</p>
    </header>

    <div class="section-card">
      <h2>总体介绍</h2>
      <p>{intro_text}</p>
    </div>

    <div class="section-card" style="background: transparent; padding: 0; box-shadow: none;">
      <h2 style="border-bottom: none; margin-bottom: 10px; padding-left: 10px;">核心指标概览</h2>
      <p style="padding-left: 10px; margin-bottom: 25px; font-size: 14px; color: #666;">
          基于全量样本 (N={kpi_data.get('total', 0)}) 的核心质量概况统计。
      </p>
      <div class="kpi-container">
        {build_kpi_cards(kpi_data)}
      </div>
    </div>
"""

# 6. 拼接并动态嵌入图片（仅嵌入实际存在的图表）
full_html = part_1 + "\n" + report_body + "\n</div>\n</body>\n</html>"

def _embed(m):
    rel = m.group(1)
    full = os.path.join(ROOT_DIR, rel)
    if os.path.exists(full):
        return f'src="{get_base64_image(full)}"'
    return m.group(0)
full_html = re.sub(r'src="(output/[^"]+?\.png)"', _embed, full_html)

# 7. 保存
output_filename = HTML_REPORT_FILE
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"\n✅ 成功生成独立报告文件：{output_filename}")
print("您可以直接双击打开，所有样式和图片都已完美嵌入。")
