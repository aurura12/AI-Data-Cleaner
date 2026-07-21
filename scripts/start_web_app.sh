#!/bin/bash
# 启动 Web 交互界面
cd "$(dirname "$0")/.."
echo "正在启动 AI 半导体生产数据分析 Web 界面..."
echo "访问地址: http://localhost:8502"
streamlit run web_app/app.py
