import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import re
import sys
import io
import os
import platform
import config as utils
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import FONT_FILE

def _init_chinese_font():
    """
    初始化中文字体，逻辑与 base_function/main/eda.py 保持一致
    """
    sys_name = platform.system()
    
    # 尝试加载
    font_name = "sans-serif" # 默认回退
    if os.path.exists(FONT_FILE):
        fm.fontManager.addfont(FONT_FILE)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        font_name = "SimHei"
    elif sys_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        font_name = "SimHei"
    elif sys_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
        font_name = "PingFang SC"
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        font_name = "WenQuanYi Micro Hei"
        
    return font_name

def get_data_info(df):
    """
    提供最基础的数据信息，保证代码不报错
    """
    info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            unique_vals = list(df[col].dropna().unique()[:3]) # 只取前3个示例
            info.append(f"{col} (Text/Cat, {n_unique} unique): e.g. {unique_vals}")
        else:
            # 对于数值列，也提供一些基础统计量，帮助 AI 了解数据范围
            desc = df[col].describe()
            info.append(f"{col} (Numeric, range: {desc['min']:.2f}~{desc['max']:.2f}, mean: {desc['mean']:.2f})")
    return "\n".join(info)

def capture_output(code, local_vars):
    """
    执行代码并同时捕获绘图和 print 的文本输出
    """
    # 捕获文本输出 (print 的内容)
    output_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_capture
    
    fig = None
    exec_error = None
    
    try:
        # 清除旧图
        plt.clf()
        plt.figure(figsize=(10, 5)) 
        
        # 执行代码
        exec(code, globals(), local_vars)
        
        # 获取图表对象
        fig = plt.gcf()
    except Exception as e:
        exec_error = str(e)
    finally:
        # 恢复标准输出，否则 Streamlit 后面的 log 都会看不见
        sys.stdout = original_stdout
        
    return fig, output_capture.getvalue(), exec_error

def render_ai_dashboard(df, t, client, target_col=None):
    # 确保中文字体已注册，并获取推荐字体名称
    font_name = _init_chinese_font()

    # [防御性编程] 安全获取配置文本，防止 utils_config.py 键值丢失导致报错
    def safe_get(key, default_val):
        return t.get(key, default_val)

    st.subheader(safe_get('ai_title', 'AI 智能分析'))

    # 1. 简单的数据源判断与安检
    df_curr = utils.get_active_analysis_df(df)

    # [防御性编程] 数据完整性检查
    if df_curr is None or df_curr.empty:
        st.warning(safe_get('ai_warn_no_data', "⚠️ 数据尚未加载，请先在左侧上传文件。"))
        return

    # 2. 输入区域
    col1, col2 = st.columns([5, 1])
    with col1:
        # [i18n] 输入框占位符
        user_query = st.text_input(
            safe_get('ai_input_label', '告诉 AI 你想分析什么...'), 
            placeholder=safe_get('ai_placeholder_input', '例如：分析良率随时间的变化趋势')
        )
    with col2:
        st.write("")
        st.write("")
        # [i18n] 按钮文字
        btn_run = st.button(safe_get('btn_run_ai', '运行分析'), type="primary")

    if btn_run and user_query:
        
        # --- 第一轮：让 AI 写代码 (Coder) ---
        # --- Domain Knowledge Injection ---
        domain_context = (
            "Domain Context & Column Meanings:\n"
            "- **Target Variable**: 'Label_Pass' (1=Pass/Good, 0=Fail/Bad). Use this to analyze yield/quality.\n"
            "- **Key Process Parameters (Inputs)**: 'Force_kg' (Bonding Force), 'Equipment_Temp' (Temp), 'Vacuum_Level', 'Leveling_Mode' (Laser vs Cross).\n"
            "- **Key Physical Measurements**: 'Total_Indium_Height', 'Indium_Taper_Zscore', '铟柱上底(CD)' (Upper Diameter), '铟柱下底(CD)' (Lower Diameter).\n"
            "- **Time/Batch**: 'Process_Date', 'Time_Seq_Day', 'Batch_ID', 'Wafer_Index'.\n"
            "- **Analysis Tips**: \n"
            "  - To analyze 'Yield' or 'Failure', compare Process Parameters distribution between Pass(1) and Fail(0) groups (e.g., Boxplots).\n"
            "  - 'Leveling_Mode' is categorical. Use Bar charts to compare pass rates between modes.\n"
        )

        prompt_coder = (
            f"Data columns: {list(df_curr.columns)}\n"
            f"Detailed Info:\n{get_data_info(df_curr)}\n\n"
            f"{domain_context}\n\n"
            f"User Requirement: {user_query}\n\n"
            f"Coding Rules:\n"
            f"1. **Environment**: Use `df` variable. Import `seaborn as sns`, `matplotlib.pyplot as plt`.\n"
            f"2. **Style**: Use `sns.set(style='whitegrid', font='{font_name}')` to make it look good and support Chinese.\n"
            f"3. **CRITICAL - DATA PREPARATION**:\n"
            f"   - **Target Variable**: 'Label_Pass' is numeric (1=Pass, 0=Fail). If using it as a grouping variable (hue), you MUST map it to strings first to avoid np.int64 errors: `df['Status'] = df['Label_Pass'].map({{1:'Pass', 0:'Fail'}})`.\n"
            f"   - **Color Palette**: NEVER use integers as palette keys. ALWAYS use string keys. Example: `palette={{'Pass': 'green', 'Fail': 'red'}}`.\n"
            f"   - **Numeric Safety**: Ensure physical parameters ('Equipment_Temp', 'Vacuum_Level', 'Force_kg', 'Total_Indium_Height', 'Calc_Circuit_Range') are float. Use `pd.to_numeric(df['col'], errors='coerce')` to be safe.\n"
            f"   - **Drop NaNs**: If plotting a specific column, drop NaNs for that column to avoid errors.\n"

            f"4. **CRITICAL - AVOID OVERPLOTTING (High Cardinality)**:\n"
            f"   - Columns like '芯片号' (Chip ID), 'Batch_ID', 'Time_Seq_Day' might have too many unique values.\n"
            f"   - **DO NOT** use them as x-axis directly unless you aggregate them.\n"
            f"   - **Strategy**: \n"
            f"     a) Group by the column and calculate mean/sum (e.g., `df.groupby('Batch_ID')['Label_Pass'].mean()`).\n"
            f"     b) Or filter to Top 10 / Bottom 10 items if specific IDs are requested.\n"
            f"     c) Or use Histograms/Boxplots to show the overall distribution instead of individual points.\n"
            
            f"5. **CRITICAL - AGGREGATION RULES**:\n"
            f"   - When using `.mean()`, `.sum()`, etc., ALWAYS specify `numeric_only=True`.\n"
            f"   - Better yet, select explicit columns: `df[['Equipment_Temp', 'Label_Pass']].groupby('Equipment_Temp').mean()`.\n"
            
            f"6. **CRITICAL - PRINT STATS (Input for Analysis)**:\n"
            f"   - You **MUST** `print()` the statistics derived from the chart. \n"
            f"   - Example: If plotting Failure Rate vs Temp, print: `print(df.groupby('Equipment_Temp')['Label_Pass'].mean())`.\n"
            f"   - This text output is REQUIRED for the AI analyst to generate the report.\n"

            f"7. **Layout & Formatting**:\n"
            f"   - Use `plt.figure(figsize=(10, 6))` (adjust if necessary).\n"
            f"   - Use `plt.tight_layout()` before `plt.show()`.\n"
            f"   - If x-axis labels overlap, add `plt.xticks(rotation=45, ha='right')`.\n"

            f"8. **Code Block Only**:\n"
            f"   - Output ONLY valid Python code inside a ```python``` block. No explanations outside the block."
        )

        messages_coder = [
            {
                "role": "system",
                "content": "You are a data analyst focusing on semiconductor device manufacturing data. "
                           "Always print the statistics you visualize."
            },
            {"role": "user", "content": prompt_coder}
        ]
        
        code_to_run = ""
        
        # [i18n] 加载状态：正在写代码
        with st.spinner(safe_get('ai_status_gen_code', '正在生成分析代码...')):
            try:
                resp = client.chat.completions.create(
                    model=utils.CODER_MODEL,
                    messages=messages_coder,
                    temperature=0.1
                )
                content = resp.choices[0].message.content
                match = re.search(r"```python(.*?)```", content, re.DOTALL)
                code_to_run = match.group(1).strip() if match else content.replace("```", "").strip()
            except Exception as e:
                # [i18n] API 报错
                st.error(f"{safe_get('ai_err_api', 'API调用失败')}: {e}")

        # --- 执行代码并捕获结果 ---
        if code_to_run:
            local_vars = {'df': df_curr, 'pd': pd, 'plt': plt, 'sns': sns, 'np': np}
            
            # 使用自定义函数同时拿图和拿字
            fig, text_output, error = capture_output(code_to_run, local_vars)

            if error:
                # [i18n] 代码执行报错
                st.error(f"{safe_get('ai_err_exec', '代码执行出错')}: {error}")
                # [i18n] 调试代码 Expander
                with st.expander(safe_get('ai_exp_debug', '查看生成的代码')):
                    st.code(code_to_run)
            else:
                # 1. 展示代码及图表 ([i18n] 原有的 view_code)
                with st.expander(safe_get('view_code', '查看代码')):
                    st.code(code_to_run)
                
                if fig and fig.get_axes():
                    st.pyplot(fig)
                else:
                    # [i18n] 警告：没画出图
                    st.warning(safe_get('ai_warn_no_chart', '代码运行成功，但没有生成图表。'))

                # 2. 展示具体的统计数字 (Debug用，也可给用户看)
                if text_output.strip():
                    # [i18n] 查看原始数据 Expander
                    with st.expander(safe_get('ai_exp_raw_data', '查看统计数据')):
                        st.text(text_output)
                
                # --- 第二轮：基于数字做结论 (Insight) ---
                # 只有当不仅有代码，而且真的 print 出了数字时，分析才准确
                if text_output.strip():
                    # [i18n] 加载状态：正在分析结果
                    with st.spinner(safe_get('ai_status_analyzing', '正在解读分析结果...')):
                        prompt_insight = (
                            f"User Question: {user_query}\n"
                            f"The data analysis code output the following statistics:\n"
                            f"START DATA\n{text_output}\nEND DATA\n\n"
                            # [i18n] 关键点：让 prompt 的要求也变成变量，
                            # 这样在中文模式下，可以要求 AI "请用中文简要总结"
                            f"{safe_get('ai_prompt_insight_req', 'Please summarize the analysis results in Chinese briefly.')}"
                        )
                        
                        try:
                            resp_insight = client.chat.completions.create(
                                model=utils.TEXT_MODEL, 
                                messages=[{"role": "user", "content": prompt_insight}],
                                temperature=0.5
                            )
                            insight = resp_insight.choices[0].message.content
                            
                            # 漂亮的展示结果
                            st.divider()
                            # [i18n] 结果标题
                            st.info(f"**{safe_get('ai_insight_title', 'AI 智能解读')}:** {insight}")
                            
                        except Exception as e:
                            # [i18n] 解释生成报错
                            st.error(f"{safe_get('ai_err_insight', '解读生成失败')}: {e}")
                else:
                    # 如果 AI 偷懒没 print 东西，就给一个通用提示
                    st.divider()
                    # [i18n] Caption：无数据输出
                    st.caption(safe_get('ai_msg_no_stats', '（AI 未输出具体的统计数据，无法提供进一步解读）'))
                
