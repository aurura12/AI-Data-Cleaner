#!/bin/bash
# Docker 一键运行完整分析流水线
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .env ]; then
  echo "未找到 .env 文件，正在从 .env.example 复制..."
  cp .env.example .env
  echo "请编辑 .env 填写 DASHSCOPE_API_KEY 后重新运行本脚本。"
  exit 1
fi

INPUT_FILE="${ROOT_DIR}/data/input/梳理版.csv"
if [ ! -f "$INPUT_FILE" ]; then
  echo "错误: 未找到输入数据文件"
  echo "请将 CSV 数据放入: ${INPUT_FILE}"
  exit 1
fi

mkdir -p output

echo "正在 Docker 中运行完整分析流水线..."
docker compose run --rm --build pipeline

echo ""
echo "分析完成！请查看: ${ROOT_DIR}/output/report.html"
