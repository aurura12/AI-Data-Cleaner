"""
静态报告生成器（领域无关兜底）

直接复用 domain_adapter.build_static_report_html：
基于 DataSchema 计算通用统计量并生成 HTML，不再写死任何行业知识。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_adapter import load_or_build_schema, collect_generic_analysis, build_static_report_html
from project_paths import ROOT_DIR


def generate_static_analysis_content(base_dir: str) -> str:
    """
    基于数据分析结果生成静态 HTML 报告内容（领域无关）。
    """
    schema = load_or_build_schema()
    analysis = collect_generic_analysis(schema=schema)
    return build_static_report_html(analysis, schema)


if __name__ == "__main__":
    base_dir = ROOT_DIR
    content = generate_static_analysis_content(base_dir)
    print(f"生成的静态报告内容长度: {len(content)} 字符")
