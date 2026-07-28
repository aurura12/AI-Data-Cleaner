import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import shutil
import glob
import config as utils
import data_cleaning as analysis
import descriptive_stats
import ai_chat as ai_analysis
import report_export as full_report
import schema_analyzer

# 1. 语言配置和主题设置
st.set_page_config(page_title="AI Semiconductor Production Assistant", layout="wide", initial_sidebar_state="expanded")

# 主题切换（在侧边栏顶部，仅影响AI报告页面）
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

theme_option = st.sidebar.radio(
    "🌓 主题 / Theme",
    ["浅色 / Light", "深色 / Dark"],
    index=0 if st.session_state.theme == 'light' else 1,
    key="theme_selector"
)
st.session_state.theme = 'light' if theme_option == "浅色 / Light" else 'dark'

# 全局主题CSS注入：让切换立即生效（不跟随系统）
if st.session_state.theme == "dark":
    _bg = "#0e1117"
    _fg = "#e6edf3"
    _card = "#161b22"
    _border = "#30363d"
    _accent = "#2f81f7"
else:
    _bg = "#ffffff"
    _fg = "#0b1220"
    _card = "#ffffff"
    _border = "#e5e7eb"
    _accent = "#1d4ed8"

st.markdown(
    f"""
<style>
  :root {{
    color-scheme: {'dark' if st.session_state.theme == 'dark' else 'light'};
    --theme-bg: {_bg};
    --theme-fg: {_fg};
    --theme-card: {_card};
    --theme-border: {_border};
    --theme-accent: {_accent};
  }}
  html, body {{
    background: {_bg} !important;
    color: {_fg} !important;
  }}
  .stApp {{
    background: {_bg} !important;
    color: {_fg} !important;
    color-scheme: {'dark' if st.session_state.theme == 'dark' else 'light'};
  }}
  div[data-testid="stAppViewContainer"],
  section.main,
  main {{
    background: {_bg} !important;
    color: {_fg} !important;
  }}
  .stApp * {{
    color: {_fg} !important;
  }}
  /* 语法高亮保护：撤销通配符强覆盖，让代码块 span 保留内联颜色 */
  div[data-testid="stCodeBlock"] pre code span {{
    color: revert !important;
  }}
  section[data-testid="stSidebar"] {{
    background: {_card};
    border-right: 1px solid {_border};
    color: {_fg} !important;
  }}
  section[data-testid="stSidebar"] * {{
    color: {_fg} !important;
  }}
  header[data-testid="stHeader"] {{
    background: transparent;
  }}
  .stMarkdown, .stMarkdown * {{
    color: {_fg} !important;
  }}
  .stText, .stText * {{
    color: {_fg} !important;
  }}
  .stCaption, .stCaption * {{
    color: {_fg} !important;
  }}
  /* 输入框和文本区域 */
  .stTextInput > div > div > input {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  .stTextArea > div > div > textarea {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  /* 选择框和下拉菜单 */
  .stSelectbox > div > div {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stSelectbox > div > div > div {{
    color: {_fg} !important;
  }}
  .stSelectbox label {{
    color: {_fg} !important;
  }}
  /* 单选框和复选框 */
  .stRadio > label {{
    color: {_fg} !important;
  }}
  .stRadio > label > div {{
    color: {_fg} !important;
  }}
  .stCheckbox > label {{
    color: {_fg} !important;
  }}
  .stCheckbox > label > div {{
    color: {_fg} !important;
  }}
  /* 滑块 */
  .stSlider > label {{
    color: {_fg} !important;
  }}
  /* 数字输入框 */
  .stNumberInput > div > div > input {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  .stNumberInput label {{
    color: {_fg} !important;
  }}
  /* 按钮 */
  .stButton > button,
  .stDownloadButton > button {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border: 1px solid {_border} !important;
    border-radius: 8px !important;
    box-shadow: none !important;
  }}
  .stButton > button:hover,
  .stDownloadButton > button:hover {{
    background-color: {_border} !important;
    border-color: {_fg} !important;
  }}
  .stButton > button[kind="primary"],
  .stDownloadButton > button[kind="primary"] {{
    background-color: {_accent} !important;
    color: white !important;
    border-color: {_accent} !important;
  }}
  .stButton > button[kind="primary"]:hover,
  .stDownloadButton > button[kind="primary"]:hover {{
    background-color: {_accent} !important;
    filter: brightness(1.05);
  }}
  .stButton > button:focus,
  .stDownloadButton > button:focus {{
    outline: 2px solid {_accent} !important;
    outline-offset: 2px;
  }}
  /* 数据框外框：内部由 Streamlit/Glide canvas 原生绘制，避免覆盖导致空白 */
  div[data-testid="stDataFrame"] {{
    border: 1px solid {_border};
    border-radius: 8px;
    overflow: hidden;
  }}
  /* 警告和信息框 */
  .stAlert {{
    color: {_fg} !important;
  }}
  .stAlert * {{
    color: {_fg} !important;
  }}
  .stSuccess {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stSuccess * {{
    color: {_fg} !important;
  }}
  .stInfo {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stInfo * {{
    color: {_fg} !important;
  }}
  .stWarning {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stWarning * {{
    color: {_fg} !important;
  }}
  .stError {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stError * {{
    color: {_fg} !important;
  }}
  /* 标签和标题 */
  label {{
    color: {_fg} !important;
  }}
  h1, h2, h3, h4, h5, h6 {{
    color: {_fg} !important;
  }}
  /* 代码块 - 深色模式：深底浅字，保留语法高亮 */
  div[data-testid="stCodeBlock"],
  div[data-testid="stCodeBlock"] > div,
  div[data-testid="stCodeBlock"] > div > div {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stCodeBlock pre,
  div[data-testid="stCodeBlock"] pre,
  .stCodeBlock code,
  div[data-testid="stCodeBlock"] code {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  /* 展开器 */
  .streamlit-expanderHeader {{
    color: {_fg} !important;
  }}
  .streamlit-expanderHeader * {{
    color: {_fg} !important;
  }}
  /* 标签页 */
  .stTabs [data-baseweb="tab"] {{
    color: {_fg} !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    color: {_fg} !important;
  }}
  /* 文件上传 */
  .stFileUploader > label {{
    color: {_fg} !important;
  }}
  .stFileUploader > label > div {{
    color: {_fg} !important;
  }}
  .stFileUploader > section {{
    background-color: {_card} !important;
    border-color: {_border} !important;
    color: {_fg} !important;
  }}
  .stFileUploader > section > div {{
    color: {_fg} !important;
  }}
  .stFileUploader div[data-testid="stFileUploaderDropzone"] {{
    background-color: {_card} !important;
    border-color: {_border} !important;
    color: {_fg} !important;
    box-shadow: none !important;
  }}
  .stFileUploader div[data-testid="stFileUploaderDropzone"] * {{
    color: {_fg} !important;
  }}
  .stFileUploader div[data-testid="stFileUploaderDropzone"] small {{
    color: {_fg} !important;
  }}
  .stFileUploader div[data-testid="stFileUploaderDropzone"] button {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  .stFileUploader div[data-testid="stFileUploaderDropzone"] button:hover {{
    border-color: {_fg} !important;
  }}
  .stFileUploader div[data-testid="stFileUploaderFileName"] {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {{
    color: {_fg} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] * {{
    color: {_fg} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderUploadButton"] button {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border: 1px solid {_border} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderUploadButton"] button:hover {{
    background-color: {_border} !important;
  }}
  /* 文件上传侧边栏 */
  section[data-testid="stSidebar"] .stFileUploader > section {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  section[data-testid="stSidebar"] .stFileUploader > section > div {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  section[data-testid="stSidebar"] .stFileUploader div[data-testid="stFileUploaderDropzone"] {{
    background-color: {_card} !important;
    border-color: {_border} !important;
    color: {_fg} !important;
  }}
  /* ===== 文件上传器完整子层级兜底 ===== */
  .stFileUploader section {{
    background-color: {_card} !important;
    border-color: {_border} !important;
  }}
  .stFileUploader section > div > div {{
    background-color: {_card} !important;
  }}
  .stFileUploader section > div > div > div {{
    background-color: {_card} !important;
  }}
  /* 上传后文件列表项 */
  [data-testid="stFileUploaderFileList"] {{
    background-color: {_card} !important;
  }}
  [data-testid="stFileUploaderFileList"] > div {{
    background-color: {_card} !important;
  }}
  [data-testid="stFileUploaderFileList"] > div > div {{
    background-color: {_card} !important;
  }}
  [data-testid="stFileUploaderFileList"] > div > div > div {{
    background-color: {_card} !important;
  }}
  /* 侧边栏：所有文件上传器内部元素背景兜底 */
  section[data-testid="stSidebar"] .stFileUploader > div {{
    background-color: {_card} !important;
  }}
  section[data-testid="stSidebar"] .stFileUploader > div > div {{
    background-color: {_card} !important;
  }}
  section[data-testid="stSidebar"] .stFileUploader > div > div > div {{
    background-color: {_card} !important;
  }}
  /* 文件上传器 - 通用覆盖所有内部元素背景 */
  .stFileUploader {{
    background-color: transparent !important;
  }}
  .stFileUploader > div {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stFileUploader > div > div {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stFileUploader * {{
    color: {_fg} !important;
  }}
  /* 上传后文件列表容器 */
  .stFileUploader [data-testid="stFileUploaderFileList"] {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderFileList"] > div {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  /* 文件上传器成功状态 */
  .stFileUploader [data-testid="stFileUploaderFileName"] {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderFileName"] * {{
    color: {_fg} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderFileSize"] {{
    color: rgba(230, 237, 243, 0.6) !important;
  }}
  .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] * {{
    color: {_fg} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderUploadButton"] button {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border: 1px solid {_border} !important;
  }}
  .stFileUploader [data-testid="stFileUploaderUploadButton"] button:hover {{
    border-color: {_fg} !important;
  }}
  /* 修复文件上传器已上传文件行的白色背景（更深层级兜底） */
  .stFileUploader div[data-testid="stFileUploaderFile"],
  .stFileUploader div[data-testid="stFileUploaderFile"] > div,
  .stFileUploader div[data-testid="stFileUploaderFile"] > div > div,
  .stFileUploader div[data-testid="stFileUploaderFile"] > div > div > div,
  .stFileUploader div[data-testid="stFileUploaderFile"] button,
  .stFileUploader div[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileName"],
  .stFileUploader div[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileName"] > div,
  .stFileUploader div[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileSize"],
  .stFileUploader [data-testid="stFileUploaderFileList"] > div > div > div > div,
  .stFileUploader [data-testid="stFileUploaderFileList"] > div > div > div > div > div,
  .stFileUploader [data-testid="stFileUploaderFileList"] > div > div > div > div > div > div {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  .stFileUploader div[data-testid="stFileUploaderFile"] button:hover {{
    background-color: {_border} !important;
  }}
  .stFileUploader div[data-testid="stFileUploaderFile"] svg,
  .stFileUploader div[data-testid="stFileUploaderFile"] svg path {{
    fill: {_fg} !important;
  }}
  /* 文件上传器所有内部 div 最终兜底 */
  .stFileUploader div,
  .stFileUploader section,
  .stFileUploader button {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  /* 指标卡片 */
  .stMetric {{
    color: {_fg} !important;
    background-color: {_card} !important;
    border: 1px solid {_border} !important;
    border-radius: 8px !important;
    padding: 8px !important;
  }}
  .stMetric * {{
    color: {_fg} !important;
  }}
  .stMetric label {{
    color: {_fg} !important;
  }}
  .stMetric div[data-testid="stMetricValue"] {{
    color: {_fg} !important;
  }}
  .stMetric div[data-testid="stMetricDelta"] {{
    color: {_fg} !important;
  }}
  /* 进度条 */
  .stProgress > div > div {{
    background-color: {_card} !important;
  }}
  /* 下载按钮 */
  .stDownloadButton > button {{
    color: white !important;
  }}
  /* 顶部工具栏（三点菜单） */
  div[data-testid="stToolbar"] {{
    background-color: transparent !important;
  }}
  div[data-testid="stToolbar"] button {{
    color: {_fg} !important;
    background-color: transparent !important;
  }}
  div[data-testid="stToolbar"] button:hover {{
    background-color: {_border} !important;
  }}
  /* 工具栏状态/菜单弹出层 */
  div[data-testid="stStatusWidget"] {{
    background-color: transparent !important;
  }}
  /* 三点菜单弹出层 - 覆盖所有可能的class名 */
  div[data-testid="stStatusWidget"] + div[role="menu"],
  [role="menu"] {{
    background-color: {_card} !important;
    border: 1px solid {_border} !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
  }}
  [role="menu"] * {{
    color: {_fg} !important;
  }}
  [role="menu"] > div, [role="menu"] li, [role="menuitem"] {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  [role="menu"] > div:hover, [role="menu"] li:hover, [role="menuitem"]:hover {{
    background-color: {_border} !important;
  }}
  /* 菜单分隔线 */
  [role="menu"] hr, [role="separator"] {{
    border-color: {_border} !important;
  }}
  /* 所有弹出层/浮层 */
  [data-testid^="stPopover"], 
  [data-testid="stPopoverBody"],
  [role="dialog"],
  [data-testid="stDialog"] {{
    background-color: {_card} !important;
    border-color: {_border} !important;
    color: {_fg} !important;
  }}
  [data-testid^="stPopover"] *,
  [role="dialog"] *,
  [data-testid="stDialog"] * {{
    color: {_fg} !important;
  }}
  /* 右键菜单 */
  .stMainMenu {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  /* 下拉选项菜单 */
  div[data-baseweb="popover"] > div {{
    background-color: {_card} !important;
    border-color: {_border} !important;
  }}
  div[data-baseweb="popover"] li {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  div[data-baseweb="popover"] li:hover {{
    background-color: {_border} !important;
  }}
  div[data-baseweb="popover"] li[aria-selected="true"] {{
    background-color: {_border} !important;
  }}
  /* 多行选择框 */
  div[data-baseweb="select"] > div {{
    background-color: {_card} !important;
    border-color: {_border} !important;
  }}
  div[data-baseweb="select"] > div * {{
    color: {_fg} !important;
  }}
  /* 数字输入 */
  .stNumberInput > div > div > input {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  .stNumberInput label {{
    color: {_fg} !important;
  }}
  /* 多列布局 - 修复对齐 */
  [data-testid="column"] {{
    color: {_fg} !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
  }}
  [data-testid="column"] > div {{
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
  }}
  [data-testid="column"] * {{
    color: {_fg} !important;
  }}
  /* 容器 - 修复对齐 */
  .element-container {{
    color: {_fg} !important;
    width: 100% !important;
    display: block !important;
  }}
  .element-container > div {{
    width: 100% !important;
  }}
  .element-container * {{
    color: {_fg} !important;
  }}
  /* 按钮容器对齐 */
  .stButton {{
    width: 100% !important;
    display: block !important;
  }}
  .stButton > button {{
    width: 100% !important;
  }}
  /* 卡片和按钮组对齐 */
  div[data-testid="stVerticalBlock"] > div {{
    width: 100% !important;
  }}
  /* 空状态 */
  .stEmpty {{
    color: {_fg} !important;
  }}
  .stEmpty * {{
    color: {_fg} !important;
  }}
  /* 图表容器 */
  [data-testid="stPlotlyChart"] {{
    background-color: {_card} !important;
  }}
  /* 链接 */
  a {{
    color: #5dade2 !important;
  }}
  /* 确保所有文本元素可见 */
  p, span, div, li, td, th {{
    color: {_fg} !important;
  }}

  /* 三点菜单 - 更精确的Streamlit 1.28+ 结构 */
  button[title="View main menu"] {{
    color: {_fg} !important;
  }}

  /* 通用：所有浮层容器（按 data-testid 匹配） */
  [data-testid$="menu"], [data-testid$="Menu"],
  [data-testid$="popover"], [data-testid$="Popover"],
  [data-testid$="overlay"], [data-testid$="Overlay"],
  [data-testid$="dialog"], [data-testid$="Dialog"] {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  [data-testid$="menu"] *, [data-testid$="Menu"] *,
  [data-testid$="popover"] *, [data-testid$="Popover"] *,
  [data-testid$="overlay"] *, [data-testid$="Overlay"] *,
  [data-testid$="dialog"] *, [data-testid$="Dialog"] * {{
    color: {_fg} !important;
  }}

  /* 文件上传器内部 "Browse files" 按钮 */
  .stFileUploader button {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border: 1px solid {_border} !important;
  }}
  .stFileUploader button:hover {{
    border-color: {_fg} !important;
  }}
  /* 修复选择框下拉菜单背景 */
  div[data-baseweb="select"] > div {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  div[data-baseweb="popover"] {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  div[data-baseweb="popover"] * {{
    color: {_fg} !important;
  }}
  /* 修复输入框占位符颜色 */
  .stTextInput > div > div > input::placeholder {{
    color: rgba(230, 237, 243, 0.6) !important;
  }}
  .stTextArea > div > div > textarea::placeholder {{
    color: rgba(230, 237, 243, 0.6) !important;
  }}
  /* 修复单选框选中状态 */
  .stRadio [data-baseweb="radio"] {{
    color: {_fg} !important;
  }}
  .stRadio [data-baseweb="radio"]:checked {{
    background-color: {_card} !important;
  }}
  /* 修复复选框 */
  .stCheckbox [data-baseweb="checkbox"] {{
    background-color: {_card} !important;
    border-color: {_border} !important;
  }}
  /* 修复滑块轨道和手柄 */
  .stSlider [data-baseweb="slider-track"] {{
    background-color: {_border} !important;
  }}
  .stSlider [data-baseweb="slider-handle"] {{
    background-color: {_fg} !important;
    border-color: {_border} !important;
  }}
  /* 修复标签页 */
  .stTabs [data-baseweb="tab-list"] {{
    background-color: {_bg} !important;
    border-bottom: 1px solid {_border} !important;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {_fg} !important;
    background-color: transparent !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    background-color: {_card} !important;
  }}
  .stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: {_fg} !important;
    border-bottom: 2px solid #5dade2 !important;
  }}
  /* 修复展开器内容区域 */
  .streamlit-expanderContent {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .streamlit-expanderContent * {{
    color: {_fg} !important;
  }}
  /* 修复主内容区域 */
  .main .block-container {{
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
  }}
  /* 确保所有块级元素正确对齐 */
  .block-container {{
    max-width: 100% !important;
  }}
  /* 修复垂直块对齐 */
  [data-testid="stVerticalBlock"] {{
    width: 100% !important;
  }}
  [data-testid="stVerticalBlock"] > div {{
    width: 100% !important;
  }}
  /* 加载旋转器背景 */
  .stSpinner > div {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stSpinner > div * {{
    color: {_fg} !important;
  }}
  /* 侧边栏的单选框/复选框容器背景 */
  section[data-testid="stSidebar"] .stRadio > div {{
    background-color: {_card} !important;
    border-radius: 6px !important;
    padding: 4px !important;
  }}
  section[data-testid="stSidebar"] .stRadio label {{
    background-color: transparent !important;
    color: {_fg} !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
  }}
  /* 表单容器边框 */
  .stForm {{
    background-color: {_card} !important;
    border-color: {_border} !important;
  }}
  .stForm * {{
    color: {_fg} !important;
  }}
  /* 分割线颜色 */
  hr {{
    border-color: {_border} !important;
  }}
  .stDivider {{
    color: {_border} !important;
  }}
  /* 通知/提示消息 */
  [data-testid="stNotification"] {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  .stToast {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border: 1px solid {_border} !important;
  }}
  /* 状态容器 (st.status) */
  .stStatus {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  .stStatus * {{
    color: {_fg} !important;
  }}
  /* About/设置对话框 */
  div[role="dialog"] {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border: 1px solid {_border} !important;
  }}
  div[role="dialog"] * {{
    color: {_fg} !important;
  }}
  /* DataFrame分页控件 */
  div[data-testid="stDataFrame"] button {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  div[data-testid="stDataFrame"] button:hover {{
    background-color: {_border} !important;
  }}
  /* 侧边栏展开器 */
  section[data-testid="stSidebar"] .streamlit-expanderHeader {{
    background-color: {_bg} !important;
    color: {_fg} !important;
    border-radius: 4px !important;
  }}
  section[data-testid="stSidebar"] .streamlit-expanderContent {{
    background-color: {_card} !important;
  }}
  /* 侧边栏警告/成功/信息框 */
  section[data-testid="stSidebar"] .stAlert,
  section[data-testid="stSidebar"] .stSuccess,
  section[data-testid="stSidebar"] .stInfo,
  section[data-testid="stSidebar"] .stWarning,
  section[data-testid="stSidebar"] .stError {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  section[data-testid="stSidebar"] .stAlert *,
  section[data-testid="stSidebar"] .stSuccess *,
  section[data-testid="stSidebar"] .stInfo *,
  section[data-testid="stSidebar"] .stWarning *,
  section[data-testid="stSidebar"] .stError * {{
    color: {_fg} !important;
  }}
  /* 自定义滚动条（暗色） */
  ::-webkit-scrollbar {{
    width: 8px !important;
    height: 8px !important;
  }}
  ::-webkit-scrollbar-track {{
    background: {_bg} !important;
  }}
  ::-webkit-scrollbar-thumb {{
    background: {_border} !important;
    border-radius: 4px !important;
  }}
  ::-webkit-scrollbar-thumb:hover {{
    background: #555 !important;
  }}
  /* 列配置/列排序弹窗 */
  div[data-testid="stDataFrameResizeHandle"] {{
    background-color: {_border} !important;
  }}
  /* 图形标题背景 */
  .stPlotlyChart {{
    background-color: {_card} !important;
  }}
</style>
""",
    unsafe_allow_html=True,
)

lang_option = st.sidebar.selectbox("Language / 语言", ["中文", "English"])
t = utils.TRANSLATIONS[lang_option]
st.title(t['page_title'])

# 2. 初始化
utils.configure_chinese_font()
client = utils.get_ai_client()

# 3. 加载数据
df = utils.load_data_sidebar(t)
df_active = utils.get_active_analysis_df(df)

# ==========================================
# 3.5 Schema 自动分析（由LLM驱动）
# ==========================================
# 检测数据是否变化（通过列名+形状hash，避免每次rerun的id变化触发重新分析）
data_fingerprint = f"{df.shape[0]}_{df.shape[1]}_{'_'.join(df.columns)}" if df is not None else None

if df is not None:
    # 如果数据指纹变了，重新分析Schema
    if data_fingerprint != st.session_state.get('data_fingerprint'):
        if 'data_schema' in st.session_state:
            del st.session_state['data_schema']
        st.session_state['data_fingerprint'] = data_fingerprint
        # 新数据上传时清除所有旧分析结果，确保 pipeline 子进程用新数据重建
        _output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        # 1) 删除 schema.json
        _schema_json = os.path.join(_output_dir, 'schema.json')
        if os.path.exists(_schema_json):
            os.remove(_schema_json)
        # 2) 清理 ML 报告残留（防止旧数据特征名污染新报告）
        _ml_dir = os.path.join(_output_dir, 'ml_report')
        if os.path.isdir(_ml_dir):
            shutil.rmtree(_ml_dir)
        # 3) 清理位置分析残留图表
        _pos_dir = os.path.join(_output_dir, 'position_analysis_v2')
        if os.path.isdir(_pos_dir):
            shutil.rmtree(_pos_dir)
        # 4) 清理 EDA 分析残留图表
        for pattern in ['*.png', '*.json', '*.csv']:
            for fp in glob.glob(os.path.join(_output_dir, 'analysis_report', pattern)):
                try:
                    os.remove(fp)
                except OSError:
                    pass
        # 5) 清理旧 AI 文本分析结果
        for fname in ['ai_text_analysis_results.json', 'ai_chart_analysis_results.json',
                       'ai_chart_analysis_intermediate.json', 'analysis_summary.json']:
            _path = os.path.join(_output_dir, fname)
            if os.path.exists(_path):
                os.remove(_path)
        print(f"[app.py] 已清除所有旧分析结果（新数据指纹: {data_fingerprint[:16]}...）")
    
    if 'data_schema' not in st.session_state:
        with st.spinner("🤖 AI 正在分析数据结构，识别列含义..."):
            try:
                schema = schema_analyzer.analyze_schema(df, client, utils.TEXT_MODEL)
                st.session_state['data_schema'] = schema
            except Exception as e:
                st.warning(f"LLM分析失败，使用规则推断: {e}")
                schema = schema_analyzer._fallback_schema(df)
                st.session_state['data_schema'] = schema

# 展示Schema分析结果（可折叠展开）
if 'data_schema' in st.session_state and df is not None:
    schema = st.session_state['data_schema']
    with st.sidebar.expander("📋 数据结构识别结果", expanded=False):
        st.caption(f"识别到 {len(schema.columns)} 列，{schema.raw_data_shape[0]} 行")
        
        # 目标列展示
        if schema.target_column:
            st.success(f"🎯 目标列: **{schema.target_column}**")
            if schema.target_mapping:
                mapping_text = ", ".join([f"{k}→{v}" for k, v in schema.target_mapping.items()])
                st.caption(f"值映射: {mapping_text}")
        else:
            st.warning("未识别到目标列")
        
        # 列角色总览表格
        overview = []
        for col in schema.columns:
            overview.append({
                "列名": col.raw_name,
                "角色": col.role,
                "类型": col.dtype,
                "置信度": f"{col.confidence:.0%}",
                "单位": col.physical_unit or "-"
            })
        st.dataframe(pd.DataFrame(overview), width="stretch", hide_index=True)
        
        # 如果有不确定项，提供用户确认
        uncertain = schema.get_uncertain_columns()
        if uncertain:
            st.warning(f"以下 {len(uncertain)} 列识别置信度较低，可在后续步骤中手动调整")
        
        # 重新分析按钮
        if st.button("🔄 重新分析", key="rerun_schema"):
            for key in ['data_schema', 'data_fingerprint']:
                st.session_state.pop(key, None)
            st.rerun()


# ==========================================
# 4. 变量映射设置 (优先使用Schema，回退到规则)
# ==========================================
target_col = None
id_col = None

if df is not None:
    schema = st.session_state.get('data_schema')
    if schema and schema.target_column:
        # 使用 Schema 自动识别结果
        target_col = schema.target_column
        id_col = schema.id_column
    else:
        # 回退到关键词匹配
        columns = list(df.columns)
        possible_targets = ['结果', '状态', '良率', 'grade', 'quality',
                            'class', 'label', 'target', 'Pass', 'Fail', 'defect', 'outcome']
        default_target_index = len(columns) - 1
        for pt in possible_targets:
            for col in columns:
                if pt.lower() in col.lower():
                    default_target_index = columns.index(col)
                    break
            else:
                continue
            break
        target_col = columns[default_target_index] if columns else None

        possible_ids = ['编号', '芯片号', 'ID', 'id', 'PatientID', 'No', '序号', 'serial']
        id_col = next((c for c in columns if c in possible_ids), None)

# ==========================================

# 5. 构建选项卡
if df is not None:
    # 修改：解包出 5 个 Tab (对应 utils_config 中新增的 tab_names)
    tab1, tab_stat, tab2, tab3, tab_report = st.tabs(t['tab_names'])

    with tab1:
        # Tab 1 负责清洗并更新 st.session_state['df_clean']
        analysis.render_data_cleaning(df, t, id_col)

    # --- Tab: 统计 + 描述性报告（不影响其他模块/页面） ---
    with tab_stat:
        descriptive_stats.render_ai_report(t, df_active)
    # ---------------------------

    with tab2:
        # --- 修改开始：判断使用清洗数据还是原始数据 ---
        if df_active is not None:
            df_tab2 = df_active
            filter_label = st.session_state.get('analysis_chip_filter_applied', utils.CHIP_FILTER_ALL)
            if 'df_clean' in st.session_state:
                status_msg = f"{t.get('status_using_clean')} (N={len(df_tab2)})"
                if filter_label != utils.CHIP_FILTER_ALL:
                    status_msg += f" | 当前筛选: {filter_label}"
                st.success(status_msg)
            else:
                st.info(t.get('status_using_raw', "ℹ️ Using Raw Data"))
        else:
            df_tab2 = df
            st.info(t.get('status_using_raw', "ℹ️ Using Raw Data")) # 可选，如果没有定义 status_using_raw 就显示默认英文
        # ---------------------------------------------

        # 传入判定后的 df_tab2
        analysis.render_deep_mining(df_tab2, t, target_col, id_col)
        
        # st.divider()
        # analysis.render_simple_tree_viz(df_tab2, t, target_col, id_col)
    
    with tab3:
    # 只需一行代码调用，传入必要的参数
        ai_analysis.render_ai_dashboard(
            df_active if df_active is not None else df, t, client, target_col,
            schema=st.session_state.get('data_schema')
        )
    
    with tab_report:
        full_report.render_download_section(t)
