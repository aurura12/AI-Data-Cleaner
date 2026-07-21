#!/bin/bash
# Docker 一键启动 Web 界面
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .env ]; then
  echo "未找到 .env 文件，正在从 .env.example 复制..."
  cp .env.example .env
  echo "请编辑 .env 填写 DASHSCOPE_API_KEY 后重新运行本脚本。"
  exit 1
fi

mkdir -p data/input output

echo "正在构建并启动 Docker 容器..."
docker compose up -d --build web

echo ""
echo "=========================================="
WEB_PORT=$(grep -E '^WEB_PORT=' .env 2>/dev/null | cut -d= -f2)
WEB_PORT=${WEB_PORT:-8502}
echo " Web 界面已启动"
echo " 访问地址: http://localhost:${WEB_PORT}"
echo " 代码路径: ${ROOT_DIR}  ->  容器内 /app"
echo " 数据目录: ${ROOT_DIR}/data/input/"
echo " 输出目录: ${ROOT_DIR}/output/"
echo "=========================================="
echo ""
echo "查看日志: docker compose logs -f web"
echo "停止服务: docker compose down"
