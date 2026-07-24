"""
一键运行完整分析流水线

执行顺序：
  1. 数据清洗
  2. EDA 探索性数据分析
  3. 机器学习归因分析
  4. 位置效应分析
  5. AI 文本解读
  6. 生成 HTML 报告

支持 --cleaning legacy|generic 参数切换清洗模式。
"""
import os
import sys
import argparse

# 将项目根目录加入路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "web_app"))

from project_paths import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    OUTPUT_DIR,
    PIPELINE_DIR,
)

# 尝试初始化 LLM client（供 generic 清洗和 schema 构建使用）
try:
    from domain_adapter import make_ai_client
    _CLIENT = make_ai_client()
    _TEXT_MODEL = os.getenv("DASHSCOPE_TEXT_MODEL") or os.getenv("TEXT_MODEL") or "qwen-plus"
    _CODER_MODEL = os.getenv("DASHSCOPE_CODER_MODEL") or os.getenv("CODER_MODEL") or "qwen2.5-coder-7b-instruct"
except Exception:
    _CLIENT = None
    _TEXT_MODEL = "qwen-plus"
    _CODER_MODEL = "qwen2.5-coder-7b-instruct"

# 解析命令行参数
parser = argparse.ArgumentParser(description='运行完整数据分析流水线')
parser.add_argument('--cleaning', choices=['legacy', 'generic'], default='generic',
                    help='数据清洗模式: legacy=传统制程清洗(仅半导体), generic=通用智能清洗(默认)')
parser.add_argument('--input', type=str, default=None,
                    help='输入数据文件路径（覆盖默认路径）')
parser.add_argument('--api-key', type=str, default=None,
                    help='OpenAI 兼容 API Key（如未设置 DASHSCOPE_API_KEY / API_KEY）')
parser.add_argument('--api-base', type=str, default=None,
                    help='OpenAI 兼容 API Base URL（如未设置 DASHSCOPE_API_BASE / BASE_URL）')
parser.add_argument('--text-model', type=str, default=None,
                    help='文本模型名（默认 qwen-plus）')
parser.add_argument('--coder-model', type=str, default=None,
                    help='代码模型名（默认 qwen2.5-coder-7b-instruct）')
args = parser.parse_args()

CLEANING_MODE = args.cleaning
INPUT_FILE = args.input or RAW_DATA_FILE

# 若命令行指定了 API 参数，覆盖环境变量
if args.api_key:
    os.environ.setdefault("DASHSCOPE_API_KEY", args.api_key)
    os.environ.setdefault("API_KEY", args.api_key)
if args.api_base:
    os.environ.setdefault("DASHSCOPE_API_BASE", args.api_base)
    os.environ.setdefault("BASE_URL", args.api_base)
if args.text_model:
    _TEXT_MODEL = args.text_model
if args.coder_model:
    _CODER_MODEL = args.coder_model

print("=" * 60)
print(f"开始执行完整分析流程 (清洗模式: {CLEANING_MODE})")
print("=" * 60)

# 1. 数据清洗
print(f"\n[步骤 1/6] 数据清洗 (模式: {CLEANING_MODE})...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if CLEANING_MODE == "generic":
    # 通用智能清洗（Schema驱动 + LLM增强）
    try:
        import pandas as pd
        from cleaning_code_generator import run_cleaning_pipeline
        from domain_adapter import load_or_build_schema

        print(f"使用通用清洗模式，读取数据: {INPUT_FILE}")
        df_raw = pd.read_csv(INPUT_FILE, encoding='utf-8-sig', low_memory=False)
        print(f"读取数据: {len(df_raw)} 行, {len(df_raw.columns)} 列")

        schema = load_or_build_schema(cleaned_csv=INPUT_FILE, force=True,
                                         client=_CLIENT, model=_TEXT_MODEL)
        cleaned_df, stats = run_cleaning_pipeline(
            df_raw, schema,
            enable_llm_enhanced=True,
            client=_CLIENT, model=_TEXT_MODEL, coder_model=_CODER_MODEL,
        )

        cleaned_df.to_csv(CLEANED_DATA_FILE, index=False, encoding='utf-8-sig')
        print(f"通用清洗完成: {len(cleaned_df)} 行, {len(cleaned_df.columns)} 列")
    except Exception as e:
        print(f"通用清洗失败 ({e})，回退到传统清洗模式...")
        CLEANING_MODE = "legacy"

if CLEANING_MODE == "legacy":
    # 传统半导体制程清洗
    try:
        from data_cleaning import clean_data_from_file

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if os.path.exists(INPUT_FILE):
            result = clean_data_from_file(INPUT_FILE, CLEANED_DATA_FILE)
            if result is None:
                print("数据清洗失败，但继续执行后续步骤...")
        else:
            print(f"找不到输入文件 {INPUT_FILE}，跳过数据清洗步骤...")
    except Exception as e:
        print(f"数据清洗异常: {e}，但继续执行后续步骤...")

# 1.5 构建并持久化 DataSchema（贯穿下游分析/报告，领域无关的关键）
try:
    from domain_adapter import load_or_build_schema
    load_or_build_schema(client=_CLIENT, model=_TEXT_MODEL)
    print("DataSchema 已构建并持久化（output/schema.json）")
except Exception as e:
    print(f"DataSchema 构建失败（不影响后续步骤）: {e}")

# 2. EDA 分析
print("\n[步骤 2/6] EDA 探索性数据分析...")
import subprocess
subprocess.run([sys.executable, os.path.join(PIPELINE_DIR, "eda_analysis.py")])

# 3. 机器学习分析
print("\n[步骤 3/6] 机器学习分析...")
subprocess.run([sys.executable, os.path.join(PIPELINE_DIR, "ml_analysis.py")])

# 4. 位置效应分析
print("\n[步骤 4/6] 位置效应分析...")
subprocess.run([sys.executable, os.path.join(PIPELINE_DIR, "position_analysis.py")])

# 5. AI 文本分析
print("\n[步骤 5/6] AI 文本分析...")
try:
    subprocess.run(
        [sys.executable, os.path.join(PIPELINE_DIR, "ai_text_analysis.py")],
        check=False,
    )
    print("文本分析完成")
except Exception as e:
    print(f"文本分析失败: {e}")
    print("   将使用静态报告内容")

# 6. 生成 HTML 报告
print("\n[步骤 6/6] 生成 HTML 报告...")
subprocess.run(
    [sys.executable, os.path.join(PIPELINE_DIR, "generate_html_report.py")],
    check=False,
)
print("HTML 报告已生成")

print("\n" + "=" * 60)
print("所有数据处理流程已完成！")
print(f"报告位置: {os.path.join(OUTPUT_DIR, 'report.html')}")
print("=" * 60)
