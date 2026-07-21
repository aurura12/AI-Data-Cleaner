# AI 半导体生产数据分析平台
FROM python:3.11-slim-bookworm

LABEL maintainer="ai-semiconductor-production"
LABEL description="AI Semiconductor Production Data Analysis Platform"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR ${APP_HOME}

# 系统依赖：matplotlib / weasyprint / 中文字体
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    fontconfig \
    fonts-dejavu-core \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 项目代码
COPY project_paths.py .
COPY web_app/ ./web_app/
COPY analysis_pipeline/ ./analysis_pipeline/
COPY scripts/ ./scripts/
COPY .streamlit/ ./.streamlit/

# 数据与输出目录（运行时通过 volume 挂载持久化）
RUN mkdir -p data/input output assets/fonts

# 中文字体（容器内 matplotlib / PDF 导出使用）
RUN wget -q -O assets/fonts/SimHei.ttf \
    https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf \
    && fc-cache -f

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && chmod +x scripts/start_web_app.sh scripts/run_full_analysis.sh \
    && chmod +x scripts/docker-start.sh scripts/docker-run-pipeline.sh 2>/dev/null || true

EXPOSE 8502

VOLUME ["/app/data", "/app/output"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["web"]
