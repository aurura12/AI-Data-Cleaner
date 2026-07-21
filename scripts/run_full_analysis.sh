#!/bin/bash
# 一键运行完整分析流水线
cd "$(dirname "$0")/.."
echo "正在运行完整分析流水线..."
python analysis_pipeline/run_pipeline.py
echo ""
echo "分析完成！请查看 output/report.html"
