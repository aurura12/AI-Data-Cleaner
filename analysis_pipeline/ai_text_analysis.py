import os
import json
import re
import sys
import pandas as pd
import requests
from typing import Dict, Optional
from openai import OpenAI
from datetime import datetime

# 确保子进程也能加载 .env 中的 API key
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_paths import (
    ROOT_DIR,
    OUTPUT_DIR,
    CLEANED_DATA_FILE,
    ANALYSIS_SUMMARY_FILE,
    AI_TEXT_RESULTS_FILE,
)

# 领域无关适配层：用 DataSchema 驱动 KPI / 统计 / 报告生成
from domain_adapter import (
    load_or_build_schema,
    collect_generic_analysis,
    build_report_prompt,
    discover_charts,
)

# ==========================================
# 1. 兼容旧接口：通用分析数据收集
# ==========================================

def collect_analysis_text_data(base_dir: str) -> Dict:
    """
    收集所有分析数据（领域无关）。
    优先读取 DataSchema（由 run_pipeline 或 load_or_build_schema 生成），
    再基于 schema 计算一套通用统计量。
    """
    schema = load_or_build_schema()
    return collect_generic_analysis(schema=schema)


# ==========================================
# 2. 使用文本LLM生成分析报告（基于 DataSchema，无行业硬编码）
# ==========================================

def _count_expected_sections(analysis_data: Dict) -> int:
    """根据 analysis_data 计算预期的 <div class="section-card"> 数量。"""
    count = 2  # 一、总体介绍 + 二、核心分布现状（始终存在）
    if analysis_data.get("feature_stats"):   count += 1  # 三
    if analysis_data.get("correlations"):    count += 1  # 四
    if analysis_data.get("drift"):           count += 1  # 五
    if analysis_data.get("position_stats"):  count += 1  # 六
    if analysis_data.get("ml_importance"):   count += 1  # 七
    count += 1  # 八、总结与优化建议（始终存在）
    return count


def generate_text_based_report(
    analysis_data: Dict,
    schema=None,
    model_name: str = "qwen-plus",
    chart_data_text: str = None,
    chart_paths: list = None
) -> str:
    """
    基于通用统计结果与 DataSchema，使用LLM生成综合报告。
    业务背景与目标含义全部来自 schema，不再写死任何行业知识。
    """
    expected_sections = _count_expected_sections(analysis_data)
    prompt = build_report_prompt(analysis_data, schema, chart_data_text, chart_paths)

    for attempt in range(2):  # 最多重试 1 次
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
            if not content:
                return ""

            # --- 后处理开始 ---
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

            # 验证章节数量：统计 <div class="section-card"> 出现次数
            actual_sections = len(re.findall(r'<div\s+class="section-card"', content))
            if attempt == 0 and actual_sections != expected_sections:
                print(f"[重试] 预期 {expected_sections} 个章节，实际只有 {actual_sections} 个，将重新生成...")
                prompt += "\n\n【重要警告】你上次输出的章节不完整！本次必须生成所有列出的章节，一个都不能少！"
                continue  # 重试

            content = re.sub(r'src="([^"]+?\.png)\s+alt=', r'src="\1" alt=', content)
            content = re.sub(r'</(div|h\d|p|li)>"\s*alt="[^"]*"[^>]*>', r'</\1>', content)
            content = re.sub(r'</(div|h\d|p|li)>\s*alt="[^"]*"[^>]*>', r'</\1>', content)
            content = re.sub(r'<div class="\s*alt="[^"]+">', r'<div class="chart-wrapper">', content)
            content = re.sub(r'<h3>2\.3\s*参数间关联性与.*?</h3>', r'<h3>2.3 参数间关联性与"实验设计复盘"</h3>', content)
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
            content = re.sub(r'<h2>3\. 时间空间维度</h2>', '<h2>3. 时间与空间效应分析</h2>', content)
            content = re.sub(r'<h2>3\. 时间、空间维度</h2>', '<h2>3. 时间与空间效应分析</h2>', content)
            content = re.sub(r'<h3>3\.1\s*时间稳定性与参数漂移.*?(?=<h3>|</div>)', '', content, flags=re.S)
            content = re.sub(r'<div class="chart-wrapper">.*?3_周度趋势分析.*?</div>\s*<p>.*?</p>', '', content, flags=re.S)
            content = re.sub(r'<div class="chart-wrapper">.*?4_特征漂移分析.*?</div>\s*<p>.*?</p>', '', content, flags=re.S)
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
                # 只在"建议"是句首或前面是标点/空格时插入 ◆，避免拆散"优化建议""改进建议"等复合词
                paragraph = re.sub(
                    r'(^|[\s，。；：、])建议',
                    r'\1◆ 建议',
                    paragraph, count=1
                )
                paragraph = paragraph.replace("◆ 建议", "<br>◆ 建议", 1)
                return f"{prefix}{paragraph}{suffix}"
            content = re.sub(r'(<p>)(.*?)(</p>)', _mark_advice, content, flags=re.S)

            # 后处理：将每个 section-card 内的 chart-wrapper 移到章节末尾（文字在前，图片在后）
            def _move_charts_to_end(html_text):
                """把每个 <div class="section-card"> 内部的 <div class="chart-wrapper"> 搬到 </div> 前。"""
                def _reorder_section(m):
                    section = m.group(0)
                    # 提取所有 chart-wrapper 块
                    charts = re.findall(
                        r'<div class="chart-wrapper">.*?</div>\s*', section, flags=re.S
                    )
                    if not charts:
                        return section
                    # 移除原有的 chart-wrapper 块
                    section_no_charts = re.sub(
                        r'<div class="chart-wrapper">.*?</div>\s*', '', section, flags=re.S
                    )
                    # 在 </div> 前插入所有 chart-wrapper
                    insert_pos = section_no_charts.rfind('</div>')
                    if insert_pos == -1:
                        return section
                    charts_block = ''.join(charts)
                    return (section_no_charts[:insert_pos] + charts_block
                            + section_no_charts[insert_pos:])
                return re.sub(
                    r'<div class="section-card">.*?</div>',
                    _reorder_section, html_text, flags=re.S
                )

            content = _move_charts_to_end(content)
            return content
        except Exception as e:
            print(f"报告生成异常: {e}")
            return ""
    return ""


# ==========================================
# 3. 主函数
# ==========================================

def main():
    """主函数：基于通用分析结果生成AI报告"""
    base_dir = ROOT_DIR

    print("=" * 60)
    print("AI文本分析报告生成模块（领域无关）")
    print("=" * 60)

    # 0. 加载/构建 DataSchema（贯穿整条流水线）
    schema = load_or_build_schema()
    print(f"[Schema] 已加载，共 {len(schema.columns)} 列"
          + (f"，目标列={schema.target_column}" if schema.target_column else ""))

    # 1. 收集通用分析结果
    print("\n[步骤 1/3] 收集通用分析结果（基于 DataSchema）...")
    analysis_data = collect_analysis_text_data(base_dir)
    print("已收集 KPI、特征对比、相关性、位置效应等通用统计")

    # 2. 提取图表数据（从绘图代码中提取）
    print("\n[步骤 2/4] 提取图表数据（从EDA/Position绘图代码）...")
    try:
        from chart_data_extractor import extract_all_chart_data
        chart_data_text, chart_data_dict = extract_all_chart_data(base_dir, schema=schema)
        print(f"已提取图表数据（长度: {len(chart_data_text)} 字符）")
        print(f"\n图表数据预览:\n{chart_data_text[:300]}...")
    except Exception as e:
        print(f"图表数据提取失败: {e}，将仅使用统计摘要")
        chart_data_text = None

    # 2.5 发现实际生成的图表，供 LLM 真实引用
    chart_paths = discover_charts(ROOT_DIR)
    print(f"发现 {len(chart_paths)} 张可用图表")

    # 3. 生成AI报告
    print("\n[步骤 3/4] 使用LLM生成综合报告...")
    report_content = generate_text_based_report(
        analysis_data,
        schema=schema,
        model_name=os.getenv("DASHSCOPE_TEXT_MODEL") or "qwen-plus",
        chart_data_text=chart_data_text,
        chart_paths=chart_paths
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
