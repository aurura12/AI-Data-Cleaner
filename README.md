# AI 半导体生产数据分析平台

基于 AI 的半导体生产数据分析系统，提供从数据清洗、探索性分析、机器学习归因到智能问答与报告导出的完整流程。

‼️需要自行配置api

**Docker 部署**请参阅 [DOCKER_DEPLOY.md](./DOCKER_DEPLOY.md)（含代码路径、启动脚本、环境配置）。
---

## 快速开始

### 1. 环境准备

需要 Python 3.8 或更高版本。

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动方式（二选一）

**方式 A：Web 交互界面（推荐）**

```bash
streamlit run web_app/app.py
```

或执行：

```bash
bash scripts/start_web_app.sh
```

浏览器访问 `http://localhost:8502`，可进行数据上传、清洗、统计、深度挖掘与 AI 对话。

**方式 B：一键批处理分析**

将原始数据放入 `data/input/梳理版.csv`，然后运行：

```bash
python analysis_pipeline/run_pipeline.py
```

或执行：

```bash
bash scripts/run_full_analysis.sh
```

完成后查看 `output/report.html` 完整分析报告。

---

## 项目结构

```
ai_semiconductor_production/
│
├── README.md                      # 本说明文档
├── requirements.txt               # Python 依赖
├── project_paths.py               # 统一路径配置（所有脚本共用）
│
├── scripts/                       # 快捷启动脚本
│   ├── start_web_app.sh           # 启动 Web 界面
│   └── run_full_analysis.sh       # 运行完整分析流水线
│
├── data/
│   └── input/
│       └── 梳理版.csv             # 原始输入数据（示例）
│
├── output/                        # 所有分析输出结果
│   ├── cleaned_chip_data_final.csv    # 清洗后数据
│   ├── analysis_report/               # EDA 探索性分析图表
│   ├── ml_report/                     # 机器学习归因图表
│   ├── position_analysis_v2/            # 位置效应分析图表
│   ├── report.html                    # 完整 HTML 分析报告
│   └── *.json                         # AI 分析中间结果
│
├── web_app/                       # Web 交互界面（Streamlit）
│   ├── app.py                     # ★ 应用入口
│   ├── config.py                  # 界面配置与中文字体
│   ├── data_cleaning.py           # 数据清洗模块
│   ├── descriptive_stats.py       # 描述性统计与报告展示
│   ├── ai_chat.py                 # AI 智能对话分析
│   └── report_export.py           # PDF / HTML 报告导出
│
├── analysis_pipeline/             # 批处理分析流水线
│   ├── run_pipeline.py            # ★ 一键运行全流程
│   ├── eda_analysis.py            # 步骤1：探索性数据分析
│   ├── ml_analysis.py             # 步骤2：机器学习归因
│   ├── position_analysis.py       # 步骤3：位置效应分析
│   ├── ai_text_analysis.py        # 步骤4：AI 文本解读
│   ├── ai_chart_analysis.py       # 步骤5：AI 图表解读（可选）
│   ├── generate_html_report.py    # 步骤6：生成 HTML 报告
│   ├── chart_data_extractor.py    # 图表数据提取工具
│   └── static_report_generator.py # 静态报告内容生成
│
└── assets/
    └── fonts/
        └── SimHei.ttf             # 中文字体
```

---

## 功能说明

| 模块 | 功能 |
|------|------|
| 数据清洗 | 自动处理缺失值、异常值，标准化字段格式 |
| 统计概览 | 生成描述性统计报告，支持按芯片型号筛选 |
| EDA 分析 | 生产状态分布、参数相关性、趋势与漂移分析 |
| 机器学习 | XGBoost + SHAP 归因，决策树规则提取 |
| 位置效应 | 晶圆空间位置对良率的影响分析 |
| AI 智能看板 | 对话式数据问答与工艺优化建议 |
| 报告导出 | 支持 HTML / PDF 格式完整报告下载 |

---

## 数据说明

- **输入**：将 CSV 原始数据放入 `data/input/` 目录，默认文件名为 `梳理版.csv`
- **输出**：所有分析结果统一保存在 `output/` 目录
- Web 界面也支持直接上传 CSV 文件，无需手动放置

---

## 常见问题

**Q: 图表中文显示乱码？**  
A: 确认 `assets/fonts/SimHei.ttf` 存在；macOS 会自动回退到 PingFang SC。

**Q: AI 功能无法使用？**  
A: 需配置相应的 API Key（如 DashScope / OpenAI），详见 `web_app/config.py` 中的模型配置。

**Q: 如何只运行某一步分析？**  
A: 直接运行对应脚本，例如 `python analysis_pipeline/eda_analysis.py`（需先完成数据清洗）。
