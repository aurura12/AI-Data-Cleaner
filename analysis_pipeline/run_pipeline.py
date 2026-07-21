"""
一键运行完整分析流水线

执行顺序：
  1. 数据清洗
  2. EDA 探索性数据分析
  3. 机器学习归因分析
  4. 位置效应分析
  5. AI 文本解读
  6. 生成 HTML 报告
"""
import os
import subprocess
import sys

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

print("=" * 60)
print("🚀 开始执行完整分析流程")
print("=" * 60)

# 1. 数据清洗
print("\n[步骤 1/6] 数据清洗...")
try:
    from data_cleaning import clean_data_from_file

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(RAW_DATA_FILE):
        result = clean_data_from_file(RAW_DATA_FILE, CLEANED_DATA_FILE)
        if result is None:
            print("⚠️ 数据清洗失败，但继续执行后续步骤...")
    else:
        print(f"⚠️ 找不到输入文件 {RAW_DATA_FILE}，跳过数据清洗步骤...")
except Exception as e:
    print(f"⚠️ 数据清洗异常: {e}，但继续执行后续步骤...")

# 2. EDA 分析
print("\n[步骤 2/6] EDA 探索性数据分析...")
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
    print("✅ 文本分析完成")
except Exception as e:
    print(f"⚠️ 文本分析失败: {e}")
    print("   将使用静态报告内容")

# 6. 生成 HTML 报告
print("\n[步骤 6/6] 生成 HTML 报告...")
subprocess.run(
    [sys.executable, os.path.join(PIPELINE_DIR, "generate_html_report.py")],
    check=False,
)
print("✅ HTML 报告已生成")

print("\n" + "=" * 60)
print("🎉 所有数据处理流程已完成！")
print(f"📄 报告位置: {os.path.join(OUTPUT_DIR, 'report.html')}")
print("=" * 60)
