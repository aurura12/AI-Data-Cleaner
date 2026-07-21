# Docker 部署说明

本文档说明如何使用 Docker 部署 **AI 半导体生产数据分析平台**，包括代码路径、启动脚本与环境配置。

---

## 一、前置要求

| 项目 | 要求 |
|------|------|
| 操作系统 | macOS / Linux / Windows（需 WSL2） |
| Docker | Docker Engine 20.10+ 或 Docker Desktop |
| Docker Compose | v2+（`docker compose` 命令） |
| 网络 | 构建镜像时需访问 PyPI；AI 功能需访问 DashScope API |
| API Key | 阿里云 DashScope API Key（AI 对话、报告解读等功能必填） |

验证环境：

```bash
docker --version
docker compose version
```

---

## 二、项目目录与代码路径

### 2.1 宿主机目录结构

```
ai_semiconductor_production/          # 项目根目录
├── DOCKER_DEPLOY.md                  # 本文档
├── Dockerfile                        # 镜像构建文件
├── docker-compose.yml                # 服务编排
├── .env.example                      # 环境变量模板
├── .env                              # 实际配置（需自行创建，勿提交 Git）
├── project_paths.py                  # 统一路径配置（代码内引用）
│
├── docker/
│   └── entrypoint.sh                 # 容器入口脚本
│
├── scripts/
│   ├── docker-start.sh               # ★ Docker 一键启动 Web
│   ├── docker-run-pipeline.sh        # ★ Docker 一键批处理分析
│   ├── start_web_app.sh              # 本地启动 Web（非 Docker）
│   └── run_full_analysis.sh          # 本地批处理（非 Docker）
│
├── web_app/                          # Streamlit Web 应用
│   ├── app.py                        # Web 入口
│   └── config.py                     # 界面与 API 配置
│
├── analysis_pipeline/                # 批处理分析流水线
│   └── run_pipeline.py               # 流水线入口
│
├── data/
│   └── input/
│       └── 梳理版.csv                # 默认输入数据
│
├── output/                           # 分析结果输出（持久化挂载）
└── assets/
    └── fonts/
        └── SimHei.ttf                # 中文字体（镜像内自动下载）
```

### 2.2 宿主机 ↔ 容器路径映射

容器内工作目录固定为 **`/app`**。

| 宿主机路径 | 容器内路径 | 说明 |
|------------|------------|------|
| `./` | `/app` | 项目根目录 |
| `./web_app/` | `/app/web_app/` | Web 交互界面 |
| `./analysis_pipeline/` | `/app/analysis_pipeline/` | 批处理流水线 |
| `./project_paths.py` | `/app/project_paths.py` | 路径常量定义 |
| `./data/` | `/app/data/` | 输入数据目录 |
| `./data/input/梳理版.csv` | `/app/data/input/梳理版.csv` | 默认批处理输入文件 |
| `./output/` | `/app/output/` | 分析结果输出 |
| `./assets/fonts/` | `/app/assets/fonts/` | 中文字体资源 |

### 2.3 容器内关键输出路径

| 路径 | 内容 |
|------|------|
| `/app/output/cleaned_chip_data_final.csv` | 清洗后数据 |
| `/app/output/analysis_report/` | EDA 探索性分析图表 |
| `/app/output/ml_report/` | 机器学习归因图表 |
| `/app/output/position_analysis_v2/` | 位置效应分析图表 |
| `/app/output/report.html` | 完整 HTML 分析报告 |
| `/app/output/*.json` | AI 分析中间结果 |

---

## 三、环境配置

### 3.1 创建配置文件

```bash
cd ai_semiconductor_production
cp .env.example .env
```

编辑 `.env`，至少填写 `DASHSCOPE_API_KEY`。

### 3.2 配置项说明

```ini
# --- Web 服务端口（宿主机映射端口）---
WEB_PORT=8502

# --- 阿里云 DashScope API（必填）---
# 控制台: https://dashscope.console.aliyun.com/
DASHSCOPE_API_KEY=sk-your-api-key-here
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# --- 模型选择 ---
DASHSCOPE_TEXT_MODEL=qwen-plus                          # 文本分析、报告解读
DASHSCOPE_CODER_MODEL=qwen2.5-coder-7b-instruct         # AI 个性化分析（代码生成）

# --- 可选：本地 Ollama（图表 AI 解读）---
OLLAMA_URL=http://host.docker.internal:11434
```

### 3.3 配置生效方式

环境变量通过 `docker-compose.yml` 注入容器，代码中读取优先级为：

1. 容器环境变量（`DASHSCOPE_API_KEY` 等）
2. `web_app/config.py` 中的默认值

| 环境变量 | 用途 | 是否必填 |
|----------|------|----------|
| `DASHSCOPE_API_KEY` | AI 对话、文本解读、报告生成 | **是** |
| `DASHSCOPE_API_BASE` | OpenAI 兼容 API 地址 | 否（有默认值） |
| `DASHSCOPE_TEXT_MODEL` | 文本分析模型 | 否 |
| `DASHSCOPE_CODER_MODEL` | 代码生成模型 | 否 |
| `WEB_PORT` | 宿主机访问端口 | 否（默认 8502） |
| `OLLAMA_URL` | 本地 Ollama 服务地址 | 否 |

> **注意**：`.env` 含敏感信息，请勿提交到版本库。

---

## 四、部署步骤

### 4.1 方式一：一键脚本（推荐）

**启动 Web 界面：**

```bash
bash scripts/docker-start.sh
```

脚本会自动：
- 检查并创建 `.env`（从 `.env.example` 复制）
- 创建 `data/input/` 和 `output/` 目录
- 构建镜像并后台启动 `web` 服务

**运行批处理分析：**

1. 将 CSV 数据放入 `data/input/梳理版.csv`
2. 执行：

```bash
bash scripts/docker-run-pipeline.sh
```

### 4.2 方式二：Docker Compose 命令

**首次部署 / 启动 Web：**

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填写 API Key

# 2. 创建数据目录
mkdir -p data/input output

# 3. 构建并启动
docker compose up -d --build web
```

**运行批处理流水线：**

```bash
docker compose run --rm pipeline
```

### 4.3 访问服务

启动成功后，浏览器访问：

```
http://localhost:8502
```

若修改了 `WEB_PORT`，则访问 `http://localhost:<WEB_PORT>`。

---

## 五、启动脚本说明

### 5.1 Docker 启动脚本

| 脚本 | 路径 | 功能 |
|------|------|------|
| `docker-start.sh` | `scripts/docker-start.sh` | 构建镜像并启动 Web 服务（后台运行） |
| `docker-run-pipeline.sh` | `scripts/docker-run-pipeline.sh` | 在容器中执行完整分析流水线（一次性任务） |

### 5.2 容器入口脚本

| 脚本 | 路径 | 功能 |
|------|------|------|
| `entrypoint.sh` | `docker/entrypoint.sh` | 容器启动入口，支持三种模式 |

入口模式：

| 模式 | 命令示例 | 说明 |
|------|----------|------|
| `web` | `docker compose up web` | 启动 Streamlit Web（默认） |
| `pipeline` | `docker compose run --rm pipeline` | 运行批处理分析 |
| `bash` | `docker compose run --rm web bash` | 进入容器 Shell 调试 |

### 5.3 本地启动脚本（非 Docker）

| 脚本 | 路径 | 功能 |
|------|------|------|
| `start_web_app.sh` | `scripts/start_web_app.sh` | 本地 Python 环境启动 Web |
| `run_full_analysis.sh` | `scripts/run_full_analysis.sh` | 本地 Python 环境运行流水线 |

---

## 六、Docker 服务说明

`docker-compose.yml` 定义了两个服务：

### 6.1 web 服务

| 属性 | 值 |
|------|-----|
| 容器名 | `ai-semiconductor-web` |
| 镜像 | `ai-semiconductor-production:latest` |
| 端口 | `8502`（可通过 `WEB_PORT` 修改宿主机映射） |
| 重启策略 | `unless-stopped` |
| 数据卷 | `./data` → `/app/data`，`./output` → `/app/output` |

### 6.2 pipeline 服务

| 属性 | 值 |
|------|-----|
| 容器名 | `ai-semiconductor-pipeline` |
| 运行方式 | 一次性任务（`docker compose run --rm`） |
| Profile | `pipeline`（不会随 `docker compose up` 自动启动） |
| 输入 | `/app/data/input/梳理版.csv` |
| 输出 | `/app/output/report.html` 等 |

---

## 七、常用运维命令

```bash
# 查看 Web 服务日志
docker compose logs -f web

# 查看容器状态
docker compose ps

# 停止所有服务
docker compose down

# 重新构建镜像（代码更新后）
docker compose build --no-cache web

# 进入容器调试
docker compose run --rm web bash

# 在容器内手动运行流水线
docker compose run --rm web pipeline
```

---

## 八、开发模式（代码热更新）

默认情况下，代码打包在镜像内。开发时可将代码目录挂载到容器，修改后重启即生效。

编辑 `docker-compose.yml`，取消 `web` 服务中以下注释：

```yaml
volumes:
  - ./data:/app/data
  - ./output:/app/output
  - ./web_app:/app/web_app
  - ./analysis_pipeline:/app/analysis_pipeline
```

然后重启：

```bash
docker compose up -d web
```

---

## 九、生产部署建议

1. **API Key 安全**：使用 `.env` 或密钥管理服务，不要硬编码在代码中。
2. **数据持久化**：确保 `data/` 和 `output/` 已挂载到宿主机或网络存储。
3. **反向代理**：生产环境建议在 Docker 前加 Nginx，配置 HTTPS。
4. **资源限制**：可在 `docker-compose.yml` 中为 `web` 服务添加 `deploy.resources` 限制 CPU/内存。
5. **防火墙**：仅开放必要端口（默认 `8502`）。

---

## 十、常见问题

**Q: 启动后无法访问页面？**

```bash
docker compose ps          # 确认容器在运行
docker compose logs web    # 查看错误日志
```

**Q: AI 功能报错？**

- 检查 `.env` 中 `DASHSCOPE_API_KEY` 是否正确
- 确认服务器能访问 `DASHSCOPE_API_BASE` 地址

**Q: 图表中文乱码？**

- 镜像构建时会自动下载 `SimHei.ttf` 到 `/app/assets/fonts/`
- 若字体缺失，容器启动时 `entrypoint.sh` 会尝试重新下载

**Q: 批处理提示找不到输入文件？**

- 确认宿主机存在 `data/input/梳理版.csv`
- 确认 `data` 目录已正确挂载：`docker compose run --rm web ls -la /app/data/input/`

**Q: 如何使用宿主机上的 Ollama？**

- 在宿主机启动 Ollama（默认端口 `11434`）
- `.env` 中设置 `OLLAMA_URL=http://host.docker.internal:11434`
- `docker-compose.yml` 已配置 `extra_hosts` 以支持 Linux/macOS 访问宿主机

---

## 十一、快速参考

```bash
# 完整部署流程（复制即用）
cd ai_semiconductor_production
cp .env.example .env          # 编辑填写 API Key
mkdir -p data/input output
bash scripts/docker-start.sh  # 启动 Web → http://localhost:8502

# 批处理分析
cp your_data.csv data/input/梳理版.csv
bash scripts/docker-run-pipeline.sh
open output/report.html       # macOS 查看报告
```
