# AI 半导体生产数据分析平台

基于 Python、Pandas、机器学习和大模型的生产数据分析平台，提供从数据上传/清洗、探索性分析、机器学习归因、晶圆位置效应分析，到 AI 解读和 HTML/PDF 报告导出的完整流程。

项目支持两种使用方式：

- Streamlit Web 界面：适合交互式上传数据、选择分析模块和下载结果。
- Python 批处理流水线：适合固定输入文件、定时运行或 Docker 部署。

**当前状态：** 可在本地或 Docker 环境运行，未配置固定的公共线上访问地址。AI 分析需要自行配置 OpenAI 兼容接口的 API Key。

---

## 1. 项目概览

| 项目 | 内容 |
|------|------|
| 项目名称 | AI 半导体生产数据分析平台 |
| 项目负责人 | 当前仓库未记录固定负责人，请由部署方补充 |
| 目标用户 | 工艺工程师、质量分析人员、生产管理人员、数据分析人员 |
| 输入数据 | CSV、Excel、TSV 或 TXT 表格数据 |
| Web 入口 | Streamlit，默认端口 `8502` |
| 批处理入口 | `analysis_pipeline/run_pipeline.py` |

## 2. 项目背景与原始需求

生产数据通常分散在表格中，人工完成数据清洗、指标统计、特征归因和报告整理成本较高，也容易受到字段命名、缺失值和行业数据格式差异的影响。

本项目把上述工作串成可复用流程：先识别和清洗数据，再生成统计和可视化结果，随后使用机器学习分析重要特征，最后由大模型辅助解释结果并生成可交付报告。当前实现已从半导体专用字段处理逐步扩展为支持多行业表格数据的通用分析模式，同时保留半导体生产数据的位置效应分析能力。

## 3. 项目用途

一句话概括：**将生产类表格数据自动转换为可解释的统计分析、特征归因和 AI 辅助分析报告。**

## 4. 技术架构与项目结构

### 技术栈

| 层次 | 技术 |
|------|------|
| 运行时 | Python 3.8+；Docker 镜像使用 Python 3.11 |
| Web 界面 | Streamlit |
| 数据处理 | Pandas、NumPy、OpenPyXL |
| 可视化 | Matplotlib、Seaborn |
| 机器学习 | Scikit-learn、XGBoost、SHAP |
| AI 接口 | OpenAI 兼容 API；可选 Ollama |
| 报告导出 | HTML、ReportLab、WeasyPrint |
| 容器化 | Docker、Docker Compose |

### 目录结构

```text
ai_semiconductor_production/
├── README.md                         # 项目说明与运行手册
├── AGENTS.md                         # Agent/维护人员操作规约与故障记录
├── DOCKER_DEPLOY.md                  # Docker 部署细节
├── requirements.txt                  # Python 依赖
├── .env.example                      # 环境变量模板
├── Dockerfile                        # Web 与流水线共用镜像
├── docker-compose.yml                # web/pipeline 服务编排
├── project_paths.py                  # 统一路径常量
├── docker/entrypoint.sh              # 容器入口
├── scripts/                          # 本地和 Docker 快捷脚本
├── data/input/                       # 默认输入数据目录
├── output/                           # 分析结果目录
├── assets/fonts/SimHei.ttf           # 中文字体资源
├── web_app/                          # Streamlit 交互应用
│   ├── app.py                        # Web 主入口
│   ├── config.py                     # AI、字体和界面配置
│   ├── data_cleaning.py              # 数据清洗
│   ├── descriptive_stats.py          # 统计分析与展示
│   ├── ai_chat.py                    # AI 对话/个性化分析
│   └── report_export.py               # 报告导出
└── analysis_pipeline/                # 批处理分析流水线
    ├── run_pipeline.py               # 六步流程入口
    ├── eda_analysis.py               # EDA
    ├── ml_analysis.py                # 机器学习归因
    ├── position_analysis.py          # 位置效应
    ├── ai_text_analysis.py           # AI 文本解读
    └── generate_html_report.py       # HTML 报告生成
```

批处理流水线顺序为：`数据清洗 → DataSchema → EDA → 机器学习归因 → 位置效应 → AI 文本解读 → HTML 报告`。

## 5. 环境配置

### 本地环境

- Python 3.8 或更高版本；建议 Python 3.11。
- Windows、macOS 或 Linux 均可运行。
- PDF 导出依赖 `WeasyPrint`，部分操作系统还需要额外系统库；只生成 HTML 时可不处理该依赖。
- AI 功能需要能够访问配置的 OpenAI 兼容 API。

```powershell
python -m venv venv
venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS 激活方式：

```bash
source venv/bin/activate
```

### 环境变量

```powershell
Copy-Item .env.example .env
# 编辑 .env，至少填写 DASHSCOPE_API_KEY
```

| 变量 | 用途 | 默认值/要求 |
|------|------|-------------|
| `WEB_PORT` | 宿主机访问端口 | `8502` |
| `DASHSCOPE_API_KEY` | AI 文本分析、对话、报告解读 | AI 功能必填 |
| `DASHSCOPE_API_BASE` | OpenAI 兼容接口地址 | DashScope 默认地址 |
| `DASHSCOPE_TEXT_MODEL` | 文本分析模型 | `qwen-plus` |
| `DASHSCOPE_CODER_MODEL` | 代码生成/个性化分析模型 | `qwen2.5-coder-7b-instruct` |
| `OLLAMA_URL` | 可选的本地 Ollama 地址 | `http://host.docker.internal:11434` |

不要把 `.env`、真实 API Key 或包含敏感业务数据的输入文件提交到 Git。

## 6. 运行方式与访问入口

### Web 交互模式（推荐）

```powershell
streamlit run web_app/app.py
```

浏览器访问 `http://localhost:8502`。侧边栏支持上传 CSV、Excel、TSV 或 TXT 文件，也可以使用项目内默认数据；页面支持中文/英文切换、浅色/深色主题和数据合并/替换选择。

也可以使用：

```bash
bash scripts/start_web_app.sh
```

### 批处理模式

将输入文件放到 `data/input/梳理版.csv`，然后运行：

```powershell
python analysis_pipeline/run_pipeline.py
```

常用参数：

```powershell
python analysis_pipeline/run_pipeline.py --input path/to/data.csv --cleaning generic
python analysis_pipeline/run_pipeline.py --cleaning legacy
```

结果位于 `output/`，其中 `output/report.html` 是完整 HTML 报告。也可以运行 `bash scripts/run_full_analysis.sh`。

### Docker 模式

```bash
cp .env.example .env
# 编辑 .env 填写 API Key
docker compose up -d --build web
```

访问 `http://localhost:8502`。批处理使用：

```bash
docker compose run --rm --build pipeline
```

Windows 用户可使用 Docker Desktop，并在 PowerShell 中执行等价的 `docker compose` 命令；`DOCKER_DEPLOY.md` 记录了完整容器路径、挂载和运维命令。

### 权限与可用性检查

当前项目没有账号体系、测试账号或权限申请流程。可通过以下命令检查服务：

```bash
docker compose ps
docker compose logs -f web
```

Docker 健康检查地址为 `http://localhost:8502/_stcore/health`。

## 7. 部署路径与输出

项目没有固定生产服务器或统一部署目录，部署路径由使用方决定。Docker 容器内工作目录固定为 `/app`：

| 宿主机 | 容器 | 用途 |
|--------|------|------|
| `./data` | `/app/data` | 输入数据和运行时数据 |
| `./output` | `/app/output` | 分析结果持久化 |
| `./web_app` | `/app/web_app` | Web 代码 |
| `./analysis_pipeline` | `/app/analysis_pipeline` | 流水线代码 |

关键输出：

| 路径 | 内容 |
|------|------|
| `output/cleaned_chip_data_final.csv` | 清洗后数据 |
| `output/analysis_report/` | EDA 图表 |
| `output/ml_report/` | 机器学习归因图表 |
| `output/position_analysis_v2/` | 位置效应图表 |
| `output/report.html` | 完整 HTML 报告 |
| `output/*.json` | Schema、AI 分析及汇总中间结果 |

## 8. 最新进度

### 已完成

- Streamlit 交互式数据上传、数据合并/替换和多语言界面。
- 通用智能数据清洗与传统半导体清洗模式。
- 描述性统计、EDA、相关性和趋势分析。
- XGBoost/SHAP 特征重要性和决策树规则分析。
- 晶圆/位置效应分析。
- AI 文本解读、智能问答和个性化分析。
- HTML/PDF 报告导出及中文字体支持。
- Docker Web 服务与一次性批处理服务。
- 分析流程对多行业数据的通用化适配。

### 当前维护重点

- 不同数据集的字段语义和目标列识别仍依赖规则与 LLM 推断，需要用真实数据回归验证。
- PDF 导出在不同操作系统上的系统库兼容性需要按部署环境确认。
- 生产环境部署、鉴权、任务调度和持久化存储尚未在本仓库形成统一方案。

## 9. 关键注意事项与已知问题

- AI 分析并非离线功能；未配置 API Key 时，基础数据处理仍可运行，但 Schema、AI 解读、智能问答等能力可能不可用或回退到静态逻辑。
- 批处理默认读取 `data/input/梳理版.csv`；使用其他文件名时必须传 `--input`。
- 新数据上传后，应用会根据数据指纹清理旧分析结果，避免不同数据集相互污染；请先保存需要保留的旧输出。
- 中文图表依赖 `assets/fonts/SimHei.ttf`。Docker 构建和容器启动时会尝试下载字体，离线环境需要提前准备。
- `output/` 中的 CSV、HTML、JSON 和图像会被覆盖或更新，不应把它当作不可变归档区。
- `OLLAMA_URL` 只影响可选的本地 Ollama 图表解读；Docker 访问宿主机服务时使用 `host.docker.internal`。
- `st.dataframe()` 的内部渲染由 Streamlit/Glide Data Grid 控制，不能用全局 CSS 强行覆盖其 canvas、单元格和内部类名，否则可能出现“有工具栏但表格空白”。
- 当前仓库没有自动化测试套件；修改清洗、分析或报告模块后，至少使用示例数据运行一次 Web 或完整流水线。

## 10. 对接手人的叮嘱

先看 `project_paths.py`，所有输入、输出和报告路径都应从这里取得，不要在新脚本里硬编码路径。修改数据清洗或字段识别逻辑时，同时检查通用模式和 `legacy` 模式。涉及 AI 的改动要同时验证“有 API Key”和“无 API Key”两种情况。部署相关改动请同步更新 `DOCKER_DEPLOY.md` 与 `AGENTS.md`，并在提交前确认 `.env` 和真实数据没有进入版本库。

---

更多容器部署细节见 [DOCKER_DEPLOY.md](./DOCKER_DEPLOY.md)；维护规约和故障记录见 [AGENTS.md](./AGENTS.md)。
