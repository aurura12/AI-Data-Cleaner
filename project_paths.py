"""
项目路径统一配置
所有脚本通过此模块获取路径，避免硬编码。
"""
import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 各模块目录
WEB_APP_DIR = os.path.join(ROOT_DIR, "web_app")
PIPELINE_DIR = os.path.join(ROOT_DIR, "analysis_pipeline")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")

# 数据目录
DATA_INPUT_DIR = os.path.join(ROOT_DIR, "data", "input")
RAW_DATA_FILE = os.path.join(DATA_INPUT_DIR, "梳理版.csv")

# 输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
CLEANED_DATA_FILE = os.path.join(OUTPUT_DIR, "cleaned_chip_data_final.csv")
EDA_REPORT_DIR = os.path.join(OUTPUT_DIR, "analysis_report")
ML_REPORT_DIR = os.path.join(OUTPUT_DIR, "ml_report")
POSITION_REPORT_DIR = os.path.join(OUTPUT_DIR, "position_analysis_v2")
HTML_REPORT_FILE = os.path.join(OUTPUT_DIR, "report.html")

# AI 分析结果
AI_TEXT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "ai_text_analysis_results.json")
AI_CHART_RESULTS_FILE = os.path.join(OUTPUT_DIR, "ai_chart_analysis_results.json")
AI_CHART_INTERMEDIATE_FILE = os.path.join(OUTPUT_DIR, "ai_chart_analysis_intermediate.json")
ANALYSIS_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "analysis_summary.json")

# 资源文件
FONT_FILE = os.path.join(ASSETS_DIR, "fonts", "SimHei.ttf")

# 兼容旧路径别名（供迁移过渡期使用）
LEGACY_BASE_FUNCTION_OUTPUT = OUTPUT_DIR
