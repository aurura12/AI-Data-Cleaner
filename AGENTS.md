# Agents Instructions

本文档是本项目的维护、开发和交接规约。项目是 Python + Streamlit 数据分析平台，不适用 ERP 项目中的前后端镜像版本号或线上容器名规则。

## 修改前的基本检查

1. 先确认当前目录和 Git 工作区状态：

   ```powershell
   Get-Location
   git status --short
   ```

2. 先阅读 `README.md`、`project_paths.py`、`docker-compose.yml` 以及要修改模块的入口文件。
3. 不覆盖用户已有的未提交修改；尤其不要删除或重置未跟踪的输入数据、报告和调试记录。
4. 涉及 API、数据或部署配置时，确认 `.env` 未被提交，也不要在日志、代码和文档中写入真实 API Key。

## 路径与数据约定

- 项目根目录由 `project_paths.py` 统一计算；新代码应复用其中的路径常量。
- 默认批处理输入文件是 `data/input/梳理版.csv`。
- Web 上传支持 CSV、Excel、TSV 和 TXT；上传文件可以选择替换默认数据或合并数据。
- 所有分析结果写入 `output/`，包括清洗数据、图表、HTML/PDF 报告和 JSON 中间结果。
- 不要把真实业务数据、API Key、临时输出和本地环境文件加入 Git。

## 运行与验证规则

### 本地 Web

```powershell
python -m venv venv
venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run web_app/app.py
```

默认访问 `http://localhost:8502`。

### 本地流水线

```powershell
python analysis_pipeline/run_pipeline.py --input path/to/data.csv
```

流水线顺序固定为：数据清洗、DataSchema、EDA、机器学习、位置效应、AI 文本解读、HTML 报告。修改其中一个步骤后，至少运行对应脚本或完整流水线一次，并检查 `output/report.html` 是否生成。

### Docker

```bash
docker compose up -d --build web
docker compose run --rm --build pipeline
docker compose ps
docker compose logs -f web
```

容器内工作目录是 `/app`，`data` 和 `output` 通过 Compose 挂载到宿主机。变更 Dockerfile、依赖或入口脚本后必须重新构建镜像。

## 依赖与 AI 配置规则

- Python 依赖统一维护在 `requirements.txt`，新增依赖后同步验证本地和 Docker 安装。
- AI 配置优先使用 `DASHSCOPE_API_KEY`、`DASHSCOPE_API_BASE`、`DASHSCOPE_TEXT_MODEL` 和 `DASHSCOPE_CODER_MODEL`。
- `API_KEY`、`BASE_URL`、`TEXT_MODEL` 和 `CODER_MODEL` 是兼容别名；不要在新代码中创建另一套变量名。
- Ollama 是可选能力，地址由 `OLLAMA_URL` 配置；不能把 Ollama 当作必需依赖。
- AI 调用失败时应保留基础数据分析能力，优先提供清晰提示或静态回退，不要让整个报告流程无故中止。

## 代码修改约定

- UI 入口在 `web_app/app.py`；数据清洗、统计、AI 和导出逻辑分别放在对应模块，不要把大量业务逻辑继续堆入入口文件。
- 批处理入口在 `analysis_pipeline/run_pipeline.py`；新步骤要明确输入、输出和失败时的回退行为。
- 图表和报告需要兼容中文字体；使用 `assets/fonts/SimHei.ttf`，不要依赖开发机上的字体路径。
- 数据处理应尽量支持动态列名和缺失字段；不要把单个样例数据的列名、类别或行业术语硬编码为唯一前提。
- 修改 Streamlit 样式时，只覆盖稳定的外层组件选择器。不要覆盖 `st.dataframe` 底层 Glide Data Grid 的 canvas、内部哈希类名或 CSS 变量。
- 修改上传、清洗或数据指纹逻辑时，确认旧分析结果不会错误地复用于新数据。

## 版本与发布记录

当前项目没有远程镜像仓库，也没有前后端分别递增的发布版本规则。发布以 Git 提交和 Docker 镜像构建时间为准：

1. 提交前运行 `git status --short`，确认没有敏感文件。
2. 记录本次变更涉及的模块、输入格式和输出格式。
3. 按需要重新构建镜像：`docker compose build web`。
4. 启动 Web 并运行一次最小样例或完整流水线。
5. 在提交说明中写清楚是否改变了环境变量、输出路径或数据格式。

不要使用时间戳冒充应用版本号，也不要在没有实际部署信息时虚构服务器、镜像标签或访问地址。

## 已知问题记录

### 2026-07-22 — Streamlit DataFrame 表格显示空白

**现象：** `st.dataframe()` 只显示工具栏，数据行和列不可见；点击单元格后内容可能短暂出现。

**原因：** Streamlit 1.59.2 使用 Glide Data Grid 绘制表格。全局 CSS 使用 `!important` 覆盖了旧版内部类名、canvas 背景和文字颜色，导致 JavaScript 绘制的文字与背景颜色冲突。

**处理：** 仅保留 `div[data-testid="stDataFrame"]` 的边框和圆角样式，让 Streamlit/Glide 原生管理表格内部主题。今后修改主题时不要重新加入内部元素覆盖。

### 2026-07-23 — 通用数据与半导体专用流程兼容

**注意：** 项目已支持通用智能清洗，但位置效应分析和部分传统清洗逻辑仍更适合具有晶圆/位置字段的半导体数据。使用食品或其他行业数据时，应重点检查目标列识别、位置分析是否有可用字段，以及报告内容是否需要人工复核。

### 持续性注意 — AI 子进程环境变量

报告和描述性统计模块可能通过子进程运行分析脚本。若 AI 功能表现为“主页面能配置、子进程却未调用模型”，先检查 `DASHSCOPE_API_KEY`、`DASHSCOPE_API_BASE` 和模型变量是否被传入子进程，再检查 API 网络连通性。

## 最低验收清单

- [ ] `python -m py_compile` 可通过修改过的 Python 文件。
- [ ] Web 页面可以启动并访问健康检查/首页。
- [ ] 使用一个小型 CSV 完成上传、清洗和统计展示。
- [ ] 批处理能生成 `output/report.html`，或在 AI 不可用时给出明确回退信息。
- [ ] 中文图表或报告不出现明显乱码。
- [ ] Docker 修改已通过 `docker compose config` 和一次实际启动验证。
- [ ] `git status --short` 中没有 `.env`、API Key 或不应提交的业务数据。
