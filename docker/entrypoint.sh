#!/bin/bash
set -e

cd /app
mkdir -p data/input output assets/fonts

# 确保中文字体存在
if [ ! -f assets/fonts/SimHei.ttf ]; then
  echo "[entrypoint] 下载中文字体 SimHei.ttf ..."
  wget -q -O assets/fonts/SimHei.ttf \
    https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf || true
fi

MODE="${1:-web}"

case "$MODE" in
  web)
    echo "=========================================="
    echo " AI 半导体生产数据分析平台 - Web 模式"
    echo " 访问地址: http://localhost:${STREAMLIT_SERVER_PORT:-8502}"
    echo " 代码路径: /app"
    echo " 数据输入: /app/data/input/"
    echo " 分析输出: /app/output/"
    echo "=========================================="
    exec streamlit run web_app/app.py \
      --server.port="${STREAMLIT_SERVER_PORT:-8502}" \
      --server.address="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
      --server.headless=true \
      --browser.gatherUsageStats=false
    ;;
  pipeline)
    echo "=========================================="
    echo " AI 半导体生产数据分析平台 - 批处理模式"
    echo " 输入文件: /app/data/input/梳理版.csv"
    echo " 输出目录: /app/output/"
    echo "=========================================="
    exec python analysis_pipeline/run_pipeline.py
    ;;
  bash|shell)
    exec /bin/bash
    ;;
  *)
    echo "未知模式: $MODE"
    echo "可用模式: web | pipeline | bash"
    exit 1
    ;;
esac
