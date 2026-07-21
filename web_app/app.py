import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import config as utils
import data_cleaning as analysis
import descriptive_stats
import ai_chat as ai_analysis
import report_export as full_report

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
else:
    _bg = "#ffffff"
    _fg = "#0b1220"
    _card = "#ffffff"
    _border = "#e5e7eb"

st.markdown(
    f"""
<style>
  .stApp {{
    background: {_bg};
    color: {_fg} !important;
  }}
  .stApp * {{
    color: {_fg} !important;
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
  .stButton > button {{
    color: white !important;
  }}
  /* 数据框和表格 */
  .stDataFrame {{
    color: {_fg} !important;
    background-color: {_card} !important;
  }}
  .stDataFrame * {{
    color: {_fg} !important;
  }}
  .stDataFrame table {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stDataFrame table td, .stDataFrame table th {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
  }}
  /* pandas dataframe 表格 */
  div[data-testid="stDataFrame"] table {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  div[data-testid="stDataFrame"] table td, 
  div[data-testid="stDataFrame"] table th {{
    background-color: {_card} !important;
    color: {_fg} !important;
    border-color: {_border} !important;
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
  /* 代码块 */
  .stCodeBlock {{
    background-color: {_card} !important;
    color: {_fg} !important;
  }}
  .stCodeBlock * {{
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
  /* 指标 */
  .stMetric {{
    color: {_fg} !important;
  }}
  .stMetric * {{
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
# 4. 关键修改: 变量映射设置 (适配不同列名)
# ==========================================
target_col = None
id_col = None

if df is not None:
    columns = list(df.columns)

    # 智能预选（不展示侧边栏设置）
    possible_targets = ['是否患冠心病', 'HeartDisease', 'Target', 'outcome', 'Diagnosis']
    default_target_index = len(columns) - 1
    for pt in possible_targets:
        if pt in columns:
            default_target_index = columns.index(pt)
            break
    target_col = columns[default_target_index] if columns else None

    # ID 列自动选择（可为空）
    possible_ids = ['ID', 'PatientID', 'No', '序号']
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
            df_active if df_active is not None else df, t, client, target_col
        )
    
    with tab_report:
        full_report.render_download_section(t)
