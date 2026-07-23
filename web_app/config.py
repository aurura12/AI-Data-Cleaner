import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import re
import time
import sys
import json
import streamlit.components.v1 as _st_comp
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件（支持 API Key、模型配置等）
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import ROOT_DIR, OUTPUT_DIR, CLEANED_DATA_FILE, FONT_FILE

# 1. 后端设置
matplotlib.use('Agg')

# ==========================================
# 2. 语言包字典 (报错就是因为这一大段你还没拷进去)
# ==========================================
# 找到 TRANSLATIONS 字典，更新或替换对应部分
# ... (前面的代码保持不变)
# ==========================================
# 文件: utils_config.py (只需替换这一个字典变量)
# ==========================================

TRANSLATIONS = {
    '中文': {
        # --- 基础/侧边栏 ---
        'page_title': "AI半导体生产助手",
        'sidebar_header': "1. 数据源",
        'sidebar_settings': "2. 变量设置",
        'select_target': "选择目标变量 (Y)",
        'select_id': "选择各类ID/流水号 (不分析)",
        'col_not_selected': "无 (不排除)",
        'load_success': "已加载",
        'load_fail': "加载失败",
        'upload_hint': "请上传 Excel 或 CSV 文件",
        'upload_success': "文件上传成功",
        'upload_guide': "⬅️ 请在左侧上传数据文件",
        'data_compare_title': "数据对比",
        'data_compare_dataset': "数据集",
        'data_compare_old': "旧数据",
        'data_compare_new': "新数据",
        'data_compare_rows': "行数",
        'data_compare_date_col': "日期列",
        'data_compare_date_range': "日期范围",
        'data_compare_overlap': "重叠检查",
        'data_compare_overlap_yes': "可能存在重叠",
        'data_compare_overlap_no': "未发现重叠",
        'data_compare_overlap_unknown': "无法判断",
        'data_choice_title': "数据选择",
        'data_choice_desc': "检测到已上传新数据，请选择后续分析的数据范围。",
        'data_choice_option_merge': "A. 合并旧数据 + 新数据（去重后分析）",
        'data_choice_option_new': "B. 仅分析最新上传数据",
        'data_choice_confirm': "确认",
        'data_choice_wait': "请先选择数据处理方式",
        'data_choice_merge_summary': "合并后数据",
        'data_choice_new_summary': "仅最新数据",
        'data_choice_label_prefix': "时间范围",
        'data_choice_no_date': "无日期信息",
        'data_choice_upload_time': "上传时间",
        'data_choice_file_time': "文件时间",
        'tab_names': ["🧹 数据自动清洗", "📊 描述性统计分析", "🔍 深度挖掘", "🤖 个性化分析", "📥 完整报告"],

        # --- Tab 1: 半导体数据清洗 ---
        'clean_title': "数据清洗中心 (Data Cleaning)",
        'clean_intro': "💡 针对半导体生产数据设计的智能清洗模块，支持表头识别、特征提取和质量控制。",

        # === 新增模块 1: 重复处理 ===
        'sect_dupe': "0. 数据去重 (Deduplication)",
        'dupe_info': "检测到 **{}** 行完全重复的记录。",
        'btn_dupe': "🗑️ 一键删除重复行",
        'dupe_clean': "✅ 没有发现重复记录。",

        # === 新增模块 2: 类型修正 ===
        'sect_type': "0.5 数据类型修正",
        'type_info': "警告：下列半导体参数看起来是数字，但存储为文本（可能是因为混入了字符）：",
        'btn_type_fix': "🔧 强制转为数值 (非法字符变为空值)",
        'type_clean': "✅ 所有数值型参数格式正常。",

        # === 新增模块 3: 逻辑检查 ===
        'sect_logic': "1. 半导体参数逻辑检查 (负值检测)",
        'logic_info': "警告：在半导体物理参数中发现 **{}** 个负数值（通常属于录入错误）。",
        'btn_logic_fix': "🔧 将负数设为空值 (NaN) 以便后续填补",
        
        # 缺失值部分
        'sect_missing': "1. 缺失值诊断与修复",
        'missing_info': "📊 缺失热力图 (黄色代表缺失)",
        'btn_impute': "🚀 启动智能填补",
        'impute_help': "使用多重插补法(Iterative Imputer)。系统利用其他完整的参数（如高度、压力）建立回归模型，预测缺失的数值。",
        'impute_success': "✅ 修复完成！共填补了 {} 个空缺值。",
        'download_imputed': "📥 下载修复后的数据",
        
        # 异常值部分
        'sect_outlier': "2. 异常参数检测",
        'outlier_method': "检测算法选择",
        'method_iqr': "IQR 四分位距法 (推荐/鲁棒)",
        'method_if': "孤立森林 (AI/多维异常)",
        'method_z': "Z-Score (仅适合正态分布)",
        'outlier_res': "🔍 检测结果：[{}] 列发现 {} 个异常样本",

        # 新增方法论解释框的标题
        'method_expander_title': "📘 方法论说明：传统数据清洗策略",
        
        # 用 Markdown 写一段稍微长一点的解释
        'method_explanation': """
        **1. 非结构化字段解析**
        针对设备日志中混合字符的非标准字段（如“真空度：-866 T:19.8℃”），能确保环境参数被数字化为有效特征。

        **2. 物理与几何特征工程**
        基于工艺原理构建了高阶特征：
        *   **总铟柱高度**：合成电路端与芯片端高度，输出决定连接质量的 Total_Indium_Height。
        *   **平面度量化**：计算电路端高度极差，量化基板表面的倾斜风险。
        *   **形态异常检测**：引入 Z-score 标准分 算法，自动识别铟柱锥度偏离统计均值的异常样本。

        **3. 多维溯源**
        对 **芯片号** 进行结构化解构，自动提取 **源产地、批次ID、空间坐标** 以及 **时间序列**。将单一的ID字符串转化为可用于时空相关性分析的独立维度。

        **4. 标签标准化与降噪**
        *   **目标构建**：将离散的压连情况（-1, 0, 1, 2）映射为标准的二分类 **良率标签 (Pass/Fail)** ，为机器学习模型提供明确的监督信号。
        *   **特征选择**：根据业务相关性，自动剔除“设备报警代码”、“开关机状态”等低方差或全空列，有效降低数据噪声，提升后续分析的信噪比。
        """,

        # --- 新增：数据流转与确认 ---
        'data_flow_panel': "📊 当前分析数据集状态",
        'rows_info': "当前样本量: **{}** 片 | 原始样本量: {} 片",
        'btn_apply_remove': "✂️ 确认剔除选中的异常样本 (并保存)",
        'apply_success': "✅ 数据已更新！已剔除 {} 片半导体数据。现在用这版干净数据进行下一步分析。",
        'btn_reset': "🔄 重置回原始数据",
        'reset_success': "数据已重置为上传时的初始状态。",
        'confirm_tip': "💡 重要提示：不同的检测方法结果不同。请仔细查看下方详情，确认是你想要剔除的样本，再点击“确认剔除”。",

        # --- 新增：Tab 2 描述性报告 ---
        'ai_report_title': "📊 描述性报告",
        'ai_report_not_found': "ℹ️ 尚未生成描述性报告。请先运行完整的数据分析流程。",
        
        # --- 保留：描述性统计 (Table 1) 用于其他场景 ---
        't1_title': "📊 半导体生产统计分析",
        't1_group_by': "📌 分组变量: **{}** (用于计算 P-value)",
        't1_warning_multiclass': "⚠️ 注意：自动 P 值计算仅支持 **二分类** 变量 (如: 合格 vs 不合格)。多分类仅展示均值/频率。",
        't1_info_no_target': "ℹ️ 未选择目标变量，仅展示整体描述性统计 (无法计算 P 值)。",
        't1_col_feature': "参数 ",
        't1_col_type': "类型",
        't1_col_total': "整体 ",
        't1_type_cont': "连续 (Mean±SD)",
        't1_type_cat': "分类 N(%)",
        't1_val_prefix': "值:",
        't1_p_val': "P-value",
        't1_footer': """
        > **说明**：
        > *   **连续变量**：使用 均值 ± 标准差 表示，比较采用 Welch's T-test。
        > *   **分类变量**：使用 数量 (占比%) 表示，比较采用 Chi-square 卡方检验。
        > *   对于多分类变量，默认展示占比最高的类别。
        """,
        
        # --- Tab 3: XGBoost 分析 (修复这里) ---
        'status_using_clean': "✅ 正在使用清洗后的半导体数据进行分析",
        'cart_title': "半导体生产质量建模",
        'cart_title_suffix': "(含交叉验证与SHAP)",  # <--- 新增
        'missing_target': "⚠️ 请在左侧侧边栏选择正确的目标变量",
        'settings_expander': "⚙️ 分析参数设置",        # <--- 新增
        'train_test_label': "测试集比例 (验证数据占比)", # <--- 新增
        'random_seed': "随机种子 (Random Seed)",         # <--- 新增
        'btn_run_cart': "开始建模分析 (XGBoost + SHAP)",
        'spinner_msg': "正在训练模型并计算 SHAP 值...",
        
        # 结果指标
        'metrics_title': "1. 模型预测效能 (Test Set)",
        'auc_label': "AUC (区分度)",
        'auc_help': "0.8-1.0为优秀。反映模型区分合格和不合格半导体的能力。",
        'acc_label': "Accuracy (准确率)",
        'sens_label': "敏感度 (Sensitivity)",
        'sens_help': "能检测出多少真正不合格的半导体（不漏检）",
        'spec_label': "特异度 (Specificity)",
        'spec_help': "能准确识别多少合格的半导体（不误判）",
        'cm_title': "查看详细混淆矩阵",
        
        # SHAP
        'shap_title': "2. 质量影响因子全览 (SHAP)",
        'shap_info': "**图例说明：**\n\n🔴 **红色** = 数值较高\n\n🔵 **蓝色** = 数值较低\n\n➡️ **X轴向右** = 不合格风险增加 (+Risk)\n\n⬅️ **X轴向左** = 不合格风险降低 (-Risk)",
        'dep_title': "3. 参数非线性分析",
        'dep_select': "选择参数查看拐点/阈值:",

        # --- Tab 3: 决策树可视化 (修复这里) ---
        'tree_viz_title': "🌳 质量决策逻辑可视化 (白盒模型)",
        'tree_viz_info': "💡 提示：XGBoost 是黑盒模型。为了方便生产解释，这里额外训练了一棵单颗决策树来模拟分类逻辑，生成的规则类似生产质量控制流程图。",
        'tree_depth': "决策树深度 (Depth)",
        'btn_tree': "生成决策路径图",
        'tree_path_header': "🔍 质量决策路径图",

        # --- Tab 4: AI ---
        'ai_title': """
        💡 使用说明
        本模块支持 **个性化数据分析**。请在下方输入您的具体需求，AI 将自动编写 Python 代码并生成图表。
        
        **示例指令：**
        *   “请帮我画出铟柱高度的直方图”
        *   “分析一下温度和良率之间有没有关系”
        """,
        'ai_input_label': "请输入分析指令:",   # <--- 新增
        'ai_placeholder': "请输入分析需求 (例如: 画图展示铟柱高度分布)",
        'ai_powered_clean': "🚀 由阿里云大模型驱动 (正在分析 {len} 片清洗后的半导体数据)", # <--- 新增
        'ai_powered_raw': "🚀 由阿里云大模型驱动 (正在分析原始半导体数据)", # <--- 新增
        'btn_run_ai': "执行分析",
        'ai_thinking_msg': "AI 正在编写代码...",
        'view_code': "👀 查看 AI 生成的代码",
        'run_success': "运行成功",
        'ai_sys_prompt': "你是一个Python数据分析助手。只能输出Python代码。",
        'ai_user_prompt_suffix': "请用中文注释，图表标题和标签建议使用中文",

        # ... (现有代码) ...
        'clean_title': "数据清洗中心 (Data Cleaning)",
        
        # 1. 重复
        'dupe_clean': "✅ 没有发现重复记录。",  # 确保这句存在
        
        # 2. 类型
        'type_clean': "✅ 所有数值型参数格式正常。", # 确保这句存在

        # 3. 逻辑检查 (新增翻译)
        'logic_clean': "✅ 未发现不合逻辑的负值 (所有参数数值正常)。",

        # 4. 缺失值 (新增翻译)
        'missing_clean': "✅ 数据完整，未发现缺失值。",

        # 5. 异常值 (新增翻译)
        'outlier_clean': "✅ 未检测到明显的异常值。",

         # --- 新增/补全以下 5 行 ---
        'missing_found': "⚠️ 共发现 {num} 个缺失值",
        'if_error': "⚠️ 孤立森林算法要求数据不能有缺失值，请先执行第4步修复。",
        'btn_scan': "🔍 开始扫描",
        'outlier_found': "🔴 发现 **{num}** 片异常半导体数据",
        'show_details': "展开查看详情",

        # --- 新增的通用字段 ---
        'target_prefix': "🎯 目标变量:",
        'settings_expander': "参数设置 (Settings)",
        
        # --- XGBoost 分析相关 ---
        'mode_label': "分析模式 (Analysis Mode)",
        'mode_binary': "二分类 (合格 vs 不合格)",
        'mode_multi': "多分类 (0, 1, 2...)",
        'cat_enc_info': "ℹ️ 已自动对分类变量进行编码转换:",
        'f1_label': "F1分数 (Weighted)",
        'cm_true': "真实标签",
        'cm_pred': "预测标签",
        'shap_class_note': "⚠️ 注意：图中展示的是该类别的 SHAP 值 -> 类别",
        'enc_ref': "⚠️ 编码对照表: ",
        
        # --- 决策树相关 ---
        'tree_path_header': "决策树可视化结果",
        'tree_depth': "树的最大深度",
        'btn_tree': "生成决策树",

        "ai_placeholder_input": "例如：分析一下铟柱高度的分布情况",
        "ai_status_gen_code": "🤖 AI 正在编写代码...",
        "ai_err_api": "AI API 调用错误",
        "ai_err_exec": "代码执行错误",
        "ai_exp_debug": "调试代码",
        "ai_warn_no_chart": "代码运行成功，但未生成图表。",
        "ai_exp_raw_data": "📊 查看运行结果数据 (Print Output)",
        "ai_status_analyzing": "💡 AI 正在解读分析结果...",
        # 注意：这一条决定了 AI 回复的语言
        "ai_prompt_insight_req": "请根据以上半导体生产数据给出一个简明扼要的结论或见解（1-2句）。", 
        "ai_insight_title": "AI 洞察",
        "ai_err_insight": "生成分析结论时出错",
        "ai_msg_no_stats": "由于未检测到文本统计输出，暂无文字版深度分析（可能图表已足够直观）。"
    },
    'English': {
        # --- Basic ---
        'page_title': "AI Semiconductor Production Assistant",
        'sidebar_header': "1. Data Source",
        'sidebar_settings': "2. Variable Settings",
        'select_target': "Select Target Variable (Y)",
        'select_id': "Select ID/Serial Col (Exclude)",
        'col_not_selected': "None",
        'load_success': "Loaded",
        'load_fail': "Load Failed",
        'upload_hint': "Upload Excel or CSV File",
        'upload_success': "Upload Successful",
        'upload_guide': "⬅️ Please upload data on the left",
        'data_compare_title': "Data Comparison",
        'data_compare_dataset': "Dataset",
        'data_compare_old': "Old Data",
        'data_compare_new': "New Data",
        'data_compare_rows': "Rows",
        'data_compare_date_col': "Date Column",
        'data_compare_date_range': "Date Range",
        'data_compare_overlap': "Overlap Check",
        'data_compare_overlap_yes': "Possible overlap",
        'data_compare_overlap_no': "No overlap found",
        'data_compare_overlap_unknown': "Unable to determine",
        'data_choice_title': "Data Selection",
        'data_choice_desc': "New data detected. Please choose what to analyze next.",
        'data_choice_option_merge': "A. Merge old + new (deduplicate)",
        'data_choice_option_new': "B. Analyze only the latest uploaded data",
        'data_choice_confirm': "Confirm",
        'data_choice_wait': "Please choose a data handling option first",
        'data_choice_merge_summary': "Merged Data",
        'data_choice_new_summary': "Latest Data Only",
        'data_choice_label_prefix': "Time Range",
        'data_choice_no_date': "No date info",
        'data_choice_upload_time': "Upload Time",
        'data_choice_file_time': "File Time",
        'tab_names': ["🧹 Auto Cleaning", "📊 Stats", "🔍 Mining", "🤖 AI Analysis"],

        # --- Tab 1: Advanced Cleaning ---
        'clean_title': "Data Health Check Center",
        'clean_intro': "💡 Designed for Real World Evidence (RWE) data cleaning.",

         # === New 1: Deduplication ===
        'sect_dupe': "0. Deduplication",
        'dupe_info': "Found **{}** fully duplicated rows.",
        'btn_dupe': "🗑️ Remove Duplicates",
        'dupe_clean': "✅ No duplicates found.",

        # === New 2: Type Conversion ===
        'sect_type': "0.5 Data Type/Format Fix",
        'type_info': "Warning: These columns look numeric but are stored as Text:",
        'btn_type_fix': "🔧 Force Convert to Numeric (Coerce Errors to NaN)",
        'type_clean': "✅ All numeric types look correct.",

        # === New 3: Logic Search ===
        'sect_logic': "1. Medical Logic Check (Negative Values)",
        'logic_info': "Warning: Found **{}** negative values in physiological/numeric columns.",
        'btn_logic_fix': "🔧 Set Negatives to NaN (for Imputation)",
        
        # Missing
        'sect_missing': "1. Missing Data Diagnostics & AI Imputation",
        'missing_info': "📊 Missingness Heatmap (Yellow = Missing)",
        'btn_impute': "🚀 Start AI Imputation (MICE)",
        'impute_help': "Uses Iterative Imputer. Models each feature with missing values as a function of other features to predict values.",
        'impute_success': "✅ Imputation Done! Filled {} missing values.",
        'download_imputed': "📥 Download Cleaned Data",
        
        # Outlier
        'sect_outlier': "2. Outlier/Anomaly Detection",
        'outlier_method': "Detection Method",
        'method_iqr': "IQR (Interquartile Range - Robust)",
        'method_if': "Isolation Forest (AI - Multivariate)",
        'method_z': "Z-Score (Normal Dist Only)",
        'outlier_res': "🔍 Result: [{}] found {} anomalies",

        'method_expander_title': "📘 Methodology: Why AI Imputation & Robust Detection?",
        
        'method_explanation': """
        **1. Missing Data: **
        *   ❌ **Mean Imputation**: Filling missing values with the average artificially reduces variance and biases standard errors, leading to invalid P-values.
        *   ✅ **Our Approach (MICE Algorithm)**: *Multivariate Imputation by Chained Equations*. It models the missing variable (e.g., BP) as a function of valid variables (Age, BMI), effectively "predicting" the value based on patient context. This preserves the statistical structure of RWE data.

        **2. Outliers: Medical Data is Rarely Normal**
        *    **Z-Score (3-SD)**: Assumes a Gaussian distribution. Medical variables (e.g., CRP, Cost) are often skewed. 3-SD removes valid extreme cases (e.g., severe patients).
        *    **IQR Method**: Robust to non-normal distributions (Skewed data).
        *    **Isolation Forest**: An unsupervised AI algorithm that detects **"Multivariate Anomalies"**. E.g., A specific combination of Age and BMI that is extremely rare, even if individual values are within normal ranges.
        """,

         # --- New: Data Flow ---
        'data_flow_panel': "📊 Current Dataset Status",
        'rows_info': "Current Rows: **{}** | Original Rows: {}",
        'btn_apply_remove': "✂️ Confirm Removal & Save",
        'apply_success': "✅ Data Updated! Removed {} rows. We will now use this clean version.",
        'btn_reset': "🔄 Reset to Original",
        'reset_success': "Data reset to initial upload state.",
        'confirm_tip': "💡 Note: Different methods yield different outliers. Review carefully below before confirming removal.",
    

        # --- Tab 2: Table 1 Stats (New) ---
        't1_title': "📊 Baseline Characteristics",
        't1_group_by': "📌 Group By: **{}** (Calculating P-value)",
        't1_warning_multiclass': "⚠️ Note: Auto P-value supports **Binary** targets only. Multi-class shows Mean/Freq only.",
        't1_info_no_target': "ℹ️ No target selected. Showing overall stats only (No P-value).",
        't1_col_feature': "Variable",
        't1_col_type': "Type",
        't1_col_total': "Total",
        't1_type_cont': "Cont. (Mean±SD)",
        't1_type_cat': "Cat. N(%)",
        't1_val_prefix': "Val:",
        't1_p_val': "P-value",
        't1_footer': """
        > **Legend**:
        > *   **Continuous**: Mean ± SD (Welch's T-test).
        > *   **Categorical**: N (%) (Chi-square test).
        > *   For multi-category variables, the most frequent class is shown.
        """,
        
        # --- Tab 3: XGBoost Mining ---
        'status_using_clean': "✅ Using Cleaned Data for Analysis",
        'cart_title': "Multidimensional Modeling",
        'cart_title_suffix': "(w/ Cross-Validation & SHAP)", # <--- English
        'missing_target': "⚠️ Please select a valid target variable in the sidebar",
        'settings_expander': "⚙️ Analysis Settings",         # <--- English
        'train_test_label': "Test Set Ratio (Split)",        # <--- English
        'random_seed': "Random Seed",
        'btn_run_cart': "Run Analysis (XGBoost + SHAP)",
        'spinner_msg': "Training model & Calculating SHAP...",
        
        # Metrics
        'metrics_title': "1. Clinical Performance (Test Set)",
        'auc_label': "AUC (Discrimination)",
        'auc_help': "0.8-1.0 is excellent. Ability to distinguish patients.",
        'acc_label': "Accuracy",
        'sens_label': "Sensitivity",
        'sens_help': "Ability to identify true patients (Recall)",
        'spec_label': "Specificity",
        'spec_help': "Ability to exclude healthy people",
        'cm_title': "View Confusion Matrix",

        # SHAP
        'shap_title': "2. Risk Factor Overview (SHAP)",
        'shap_info': "**Legend:**\n\n🔴 **Red** = High Value\n\n🔵 **Blue** = Low Value\n\n➡️ **Right** = Higher Risk\n\n⬅️ **Left** = Lower Risk",
        'dep_title': "3. Non-linear Analysis",
        'dep_select': "Select Feature to view Thresholds:",

        # --- Tab 3: Tree Viz ---
        'tree_viz_title': "🌳 Decision Tree Logic (Whitebox)",
        'tree_viz_info': "💡 Note: XGBoost is a black-box. Here we trained a separate single Decision Tree surrogate model to visualize the decision logic (like clinical guidelines).",
        'tree_depth': "Tree Depth",
        'btn_tree': "Generate Decision Path",
        'tree_path_header': "🔍 Clinical Decision Path",

        # --- Tab 4: AI ---
        'ai_title': "🤖 AI Data Analyst",
        'ai_input_label': "Input Instruction:", # <--- New
        'ai_placeholder': "Describe analysis (e.g., Plot gender distribution)",
        'ai_powered_clean': "🚀 Powered by Alibaba Cloud API (Analyzing {len} cleaned rows)", # <--- New
        'ai_powered_raw': "🚀 Powered by Alibaba Cloud API (Analyzing Raw Data)", # <--- New
        'btn_run_ai': "Execute Analysis",
        'ai_thinking_msg': "AI is coding...",
        'view_code': "👀 View Generated Code",
        'run_success': "Execution Successful",
        'ai_sys_prompt': "You are a Python coding assistant. Output Python code only.",
        'ai_user_prompt_suffix': "Please use English comments and English labels/titles for plots.",

        # ... (Existing code) ...
        'clean_title': "Data Health Check Center",

        # 1. Duplicates
        'dupe_clean': "✅ No duplicates found.",

        # 2. Types
        'type_clean': "✅ Data types appear normal.",

        # 3. Logic (New)
        'logic_clean': "✅ No illogical negative values found.",

        # 4. Missing (New)
        'missing_clean': "✅ Clean. No missing values.",

        # 5. Outliers (New)
        'outlier_clean': "✅ Clean. No outliers detected.",

        'missing_found': "⚠️ Total missing values: {num}",
        'if_error': "⚠️ Isolation Forest requires data without missing values. Please Impute first (Step 4).",
        'btn_scan': "🔍 Run Scan",
        'outlier_found': "🔴 Found **{num}** anomalous rows.",
        'show_details': "Show details",

         # --- New General Keys ---
        'target_prefix': "🎯 Target:",
        'settings_expander': "Model Settings",

        # --- XGBoost Analysis ---
        'mode_label': "Analysis Mode",
        'mode_binary': "Binary (0 vs 1+)",
        'mode_multi': "Multiclass (0, 1, 2...)",
        'cat_enc_info': "ℹ️ Categorical variables automatically encoded:",
        'f1_label': "F1-Score (Weighted)",
        'cm_true': "True Label",
        'cm_pred': "Predicted Label",
        'shap_class_note': "⚠️ Note: SHAP values shown for Class",
        'enc_ref': "⚠️ Encoding Reference:",

        # --- Decision Tree ---
        'tree_path_header': "Decision Tree Visualization",
        'tree_depth': "Max Depth",
        'btn_tree': "Run Decision Tree",

        "ai_placeholder_input": "e.g. Analysis of Age distribution",
        "ai_status_gen_code": "🤖 AI is generating code...",
        "ai_err_api": "API Error",
        "ai_err_exec": "Code Execution Error",
        "ai_exp_debug": "Debug Code",
        "ai_warn_no_chart": "Code ran but generated no chart.",
        "ai_exp_raw_data": "📊 View Raw Data (Printed by Code)",
        "ai_status_analyzing": "💡 AI is analyzing the results...",
        # Prompt instruction in English
        "ai_prompt_insight_req": "Please provide a concise (1-2 sentences) conclusion or insight based on these numbers.",
        "ai_insight_title": "AI Insight",
        "ai_err_insight": "Insight Generation Error",
        "ai_msg_no_stats": "No text statistics were output mostly because the chart is self-explanatory."
    }
}
TRANSLATIONS['中文'].update({
    'pdf_title': "半导体器件生产助手 - 工艺优化全景报告",
    'pdf_date': "生成日期: ",
    'pdf_sec_1': "1. 核心指标概览",
    'pdf_yield': "整体良品率",
    'pdf_open_fail': "虚焊失效率",
    'pdf_severe_fail': "严重压连率",
    'pdf_baseline': "基准线",
    'pdf_count': "数量: ",
    'pdf_no_data': "暂无核心指标数据",
    'pdf_sec_2': "2. 描述性统计分析",
    'pdf_sec_3': "3. 深度挖掘与归因分析",
    'pdf_key_findings': "<b>📊 关键发现:</b>",
    'pdf_suggestions': "<b>💡 工艺建议:</b>",
    'pdf_detailed': "<b>🔍 详细分析:</b>",
    'pdf_img_fail': "图片加载失败: ",
    'pdf_no_sug': "暂无最终建议。",
    'pdf_sug_default_title': "优化建议",
    'html_title': "半导体器件生产助手 - 工艺优化全景报告",
    'html_sec_1': "第一部分：描述性统计分析",
    'html_sec_2': "第二部分：深度挖掘与归因分析",
    'html_sec_3': "第三部分：总结与工艺优化建议",
    'html_no_mining': "暂无深度挖掘结果。",
    'html_no_stats': "暂无描述性统计报告。",
    'html_no_sug': "暂无最终建议。",
    'dl_title': "📥 完整报告导出",
    'dl_intro': "该模块将整合【描述性统计分析】、【深度挖掘】以及【工艺优化建议】，生成一份包含所有图表和智能分析的可离线阅读报告。",
    'dl_warn_no_data': "⚠️ 暂无分析数据。请先运行【描述性统计分析】或【深度挖掘】模块。",
    'dl_contains': "#### 包含内容：",
    'dl_item_stats': "📊 描述性统计报告",
    'dl_item_mining': "🔍 深度挖掘分析",
    'dl_item_sug': "💡 工艺优化建议",
    'dl_options': "### 导出选项",
    'dl_btn_html': "🚀 生成 HTML 报告",
    'dl_spin_html': "正在生成 HTML 报告...",
    'dl_success_html': "HTML 报告就绪！",
    'dl_fail_html': "HTML 生成失败: ",
    'dl_btn_pdf': "📄 生成 PDF 报告",
    'dl_spin_pdf': "正在生成 PDF 报告...",
    'dl_success_pdf': "PDF 报告就绪！",
    'dl_fail_pdf': "PDF 生成失败: ",
    'dl_down_html': "📥 下载 HTML",
    'dl_down_pdf': "📥 下载 PDF",
})

TRANSLATIONS['English'].update({
    'tab_names': ["🧹 Auto Cleaning", "📊 Stats", "🔍 Mining", "🤖 AI Analysis", "📥 Full Report"],
    'pdf_title': "Semiconductor Device Production Assistant - Process Optimization Panorama Report",
    'pdf_date': "Generated At: ",
    'pdf_sec_1': "1. KPI Overview",
    'pdf_yield': "Overall Yield",
    'pdf_open_fail': "Open Failure Rate",
    'pdf_severe_fail': "Severe Bridging Rate",
    'pdf_baseline': "Baseline",
    'pdf_count': "Count: ",
    'pdf_no_data': "No KPI data available",
    'pdf_sec_2': "2. Descriptive Statistical Analysis",
    'pdf_sec_3': "3. Deep Mining and Attribution Analysis",
    'pdf_key_findings': "<b>📊 Key Findings:</b>",
    'pdf_suggestions': "<b>💡 Process Suggestions:</b>",
    'pdf_detailed': "<b>🔍 Detailed Analysis:</b>",
    'pdf_img_fail': "Image load failed: ",
    'pdf_no_sug': "No final suggestions yet.",
    'pdf_sug_default_title': "Optimization Suggestion",
    'html_title': "Semiconductor Device Production Assistant - Process Optimization Panorama Report",
    'html_sec_1': "Part 1: Descriptive Statistical Analysis",
    'html_sec_2': "Part 2: Deep Mining and Attribution Analysis",
    'html_sec_3': "Part 3: Summary and Process Optimization Suggestions",
    'html_no_mining': "No deep mining results yet.",
    'html_no_stats': "No descriptive report yet.",
    'html_no_sug': "No final suggestions yet.",
    'dl_title': "📥 Full Report Export",
    'dl_intro': "This module integrates descriptive stats, deep mining, and process suggestions into an offline-ready report with all charts and AI analysis.",
    'dl_warn_no_data': "⚠️ No analysis data found. Please run the descriptive stats or deep mining module first.",
    'dl_contains': "#### Contents:",
    'dl_item_stats': "📊 Descriptive Stats Report",
    'dl_item_mining': "🔍 Deep Mining Analysis",
    'dl_item_sug': "💡 Process Suggestions",
    'dl_options': "### Export Options",
    'dl_btn_html': "🚀 Generate HTML Report",
    'dl_spin_html': "Generating HTML Report...",
    'dl_success_html': "HTML Report Ready!",
    'dl_fail_html': "HTML Generation Failed: ",
    'dl_btn_pdf': "📄 Generate PDF Report",
    'dl_spin_pdf': "Generating PDF Report...",
    'dl_success_pdf': "PDF Report Ready!",
    'dl_fail_pdf': "PDF Generation Failed: ",
    'dl_down_html': "📥 Download HTML",
    'dl_down_pdf': "📥 Download PDF",
})

# 3. API 配置
def _get_secret(key: str):
    try:
        return st.secrets.get(key)
    except Exception:
        return None

# 4. 模型配置（优先读取环境变量，便于 Docker / 生产部署）
API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("API_KEY")
    or _get_secret("DASHSCOPE_API_KEY")
    or "自己的api"
)
BASE_URL = (
    os.getenv("DASHSCOPE_API_BASE")
    or os.getenv("BASE_URL")
    or _get_secret("DASHSCOPE_API_BASE")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
TEXT_MODEL = os.getenv("DASHSCOPE_TEXT_MODEL") or os.getenv("TEXT_MODEL") or "qwen-plus"
CODER_MODEL = os.getenv("DASHSCOPE_CODER_MODEL") or os.getenv("CODER_MODEL") or "qwen2.5-coder-7b-instruct"
CHIP_FILTER_ALL = "全部"


def _get_chip_filter_pattern():
    return re.compile(r'(M[1-9])(?!\d)', re.IGNORECASE)

def get_ai_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _extract_chip_group_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="object")

    pattern = _get_chip_filter_pattern()

    if 'Position_Code' in df.columns:
        series = df['Position_Code']
    elif '芯片号' in df.columns:
        series = df['芯片号']
    else:
        return pd.Series([None] * len(df), index=df.index, dtype="object")

    extracted = (
        series.astype(str)
        .str.extract(pattern, expand=False)
        .str.upper()
    )
    return extracted.reindex(df.index)


def get_available_chip_filters(df: pd.DataFrame):
    groups = _extract_chip_group_series(df).dropna().astype(str).str.upper()
    preferred = [f"M{i}" for i in range(1, 10)]
    available = [chip for chip in preferred if chip in set(groups.tolist())]
    return [CHIP_FILTER_ALL] + available


def apply_chip_filter(df: pd.DataFrame, selected_chip: str):
    if df is None or df.empty:
        return df

    selected_chip = (selected_chip or CHIP_FILTER_ALL).upper()
    if selected_chip == CHIP_FILTER_ALL.upper():
        return df.copy()

    chip_groups = _extract_chip_group_series(df)
    filtered = df.loc[chip_groups == selected_chip].copy()
    return filtered


def persist_analysis_dataset(df: pd.DataFrame):
    if df is None or df.empty:
        return

    targets = [CLEANED_DATA_FILE]

    for path in targets:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')


def _clear_downstream_analysis_state():
    keys_to_clear = [
        'deep_mining_results',
        'final_suggestions',
        'full_report_html_bytes',
        'full_report_pdf_bytes',
        'last_analysis',
        'last_job_status',
        'report_generated_for_signature',
        'report_generated_for_label',
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def render_chip_filter_sidebar(df_raw: pd.DataFrame, t):
    clean_state = st.session_state.get('semiconductor_processor_state', {})
    cleaning_completed = bool(clean_state) and all(clean_state.values())

    base_df = st.session_state.get('df_clean') if cleaning_completed and 'df_clean' in st.session_state else df_raw
    if base_df is None:
        return df_raw

    st.sidebar.markdown("### 2. 芯片号筛选")
    st.sidebar.caption("清洗完成后，可按芯片号中的位置编码 M1~M9 筛选后续分析数据。")

    if not cleaning_completed:
        previous_selected = st.session_state.get('analysis_chip_filter_applied')
        if previous_selected not in (None, CHIP_FILTER_ALL):
            _clear_downstream_analysis_state()
        st.session_state['analysis_chip_filter_applied'] = CHIP_FILTER_ALL
        st.session_state['analysis_filter_signature'] = f"{CHIP_FILTER_ALL}|{len(base_df)}|{len(base_df)}"
        st.session_state['analysis_df'] = base_df.copy()
        return base_df

    options = get_available_chip_filters(base_df)
    widget_key = 'analysis_chip_filter'
    if st.session_state.pop('reset_analysis_chip_filter', False):
        st.session_state[widget_key] = CHIP_FILTER_ALL

    current_value = st.session_state.get(widget_key, CHIP_FILTER_ALL)
    if current_value not in options:
        st.session_state[widget_key] = CHIP_FILTER_ALL

    selected_chip = st.sidebar.selectbox(
        "筛选型号",
        options,
        key=widget_key,
        help="默认使用全部清洗后数据；选择 M1~M9 后，后续统计分析、深度挖掘、个性分析与报告导出都将使用筛选后的数据。",
    )

    filtered_df = apply_chip_filter(base_df, selected_chip)
    filter_label = "全部芯片" if selected_chip == CHIP_FILTER_ALL else f"仅 {selected_chip}"
    st.sidebar.caption(f"当前分析范围：{filter_label}（{len(filtered_df)} rows）")

    previous_selected = st.session_state.get('analysis_chip_filter_applied')
    if previous_selected != selected_chip:
        _clear_downstream_analysis_state()
        st.session_state['analysis_chip_filter_applied'] = selected_chip

    st.session_state['analysis_filter_signature'] = f"{selected_chip}|{len(filtered_df)}|{len(base_df)}"
    st.session_state['analysis_df'] = filtered_df.copy()
    return filtered_df


def get_active_analysis_df(df_fallback: pd.DataFrame = None):
    if 'analysis_df' in st.session_state and st.session_state['analysis_df'] is not None:
        return st.session_state['analysis_df'].copy()

    clean_state = st.session_state.get('semiconductor_processor_state', {})
    if bool(clean_state) and all(clean_state.values()) and 'df_clean' in st.session_state:
        selected_chip = st.session_state.get('analysis_chip_filter_applied', CHIP_FILTER_ALL)
        return apply_chip_filter(st.session_state['df_clean'], selected_chip)

    if 'df_clean' in st.session_state and st.session_state['df_clean'] is not None:
        return st.session_state['df_clean'].copy()

    if df_fallback is None:
        return None
    return df_fallback.copy()


def is_descriptive_report_current():
    current_signature = st.session_state.get('analysis_filter_signature')
    generated_signature = st.session_state.get('report_generated_for_signature')

    if generated_signature:
        return generated_signature == current_signature

    selected_chip = st.session_state.get('analysis_chip_filter_applied', CHIP_FILTER_ALL)
    return selected_chip == CHIP_FILTER_ALL

def configure_chinese_font():
    """解决 Linux 中文乱码 (中英通用)"""
    font_filename = FONT_FILE

    if not os.path.exists(font_filename):
        font_filename = os.path.join(ROOT_DIR, "SimHei.ttf")
    
    # 如果还是不存在，尝试下载
    if not os.path.exists(font_filename):
        os.makedirs(os.path.dirname(font_filename), exist_ok=True)
        font_filename = FONT_FILE
        try:
            st.toast("Downloading font resources...", icon="📥")
            os.system(f"wget -O {font_filename} https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf")
        except:
            pass
        
    # 尝试加载字体，如果失败则使用系统默认字体
    if os.path.exists(font_filename):
        try:
            fm.fontManager.addfont(font_filename)
            plt.rcParams['font.sans-serif'] = ['SimHei'] # SimHei 也包含英文字符，通用
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            # 如果字体加载失败，使用系统默认字体
            print(f"⚠️ 字体加载失败: {e}，使用系统默认字体")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

def _detect_date_column(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    columns = list(df.columns)
    candidates = []
    for col in columns:
        col_str = str(col)
        if "日期" in col_str or "date" in col_str.lower():
            candidates.append(col)
    for col in list(dict.fromkeys(candidates + columns)):
        try:
            series = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            continue
        if series.notna().mean() >= 0.6:
            return col
    return None

def _get_date_range(df: pd.DataFrame, date_col: str):
    if df is None or df.empty or not date_col:
        return None, None
    try:
        series = pd.to_datetime(df[date_col], errors='coerce').dropna()
    except Exception:
        return None, None
    if series.empty:
        return None, None
    return series.min(), series.max()

def _format_range_label(start_dt, end_dt, fallback_label: str, t, is_upload: bool):
    if start_dt is not None and end_dt is not None:
        return f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}"
    if fallback_label:
        prefix_key = 'data_choice_upload_time' if is_upload else 'data_choice_file_time'
        return f"{t[prefix_key]}: {fallback_label}"
    return t['data_choice_no_date']

def _get_file_mtime(path: str):
    if not path or not os.path.exists(path):
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))

def _align_columns(base_df: pd.DataFrame, new_df: pd.DataFrame):
    base_cols = list(base_df.columns) if base_df is not None else []
    new_cols = list(new_df.columns) if new_df is not None else []
    all_cols = list(dict.fromkeys(base_cols + new_cols))
    base_aligned = base_df.reindex(columns=all_cols) if base_df is not None else None
    new_aligned = new_df.reindex(columns=all_cols) if new_df is not None else None
    return base_aligned, new_aligned, all_cols

def _select_dedup_keys(all_cols, date_col: str):
    key_candidates = ["编号", "芯片号", "芯片编号", "ID", "id", "样本编号"]
    keys = []
    for key in key_candidates:
        if key in all_cols:
            keys.append(key)
            break
    if date_col and date_col in all_cols and date_col not in keys:
        keys.append(date_col)
    return keys if keys else None

def _merge_datasets(base_df: pd.DataFrame, new_df: pd.DataFrame, date_col: str):
    base_aligned, new_aligned, all_cols = _align_columns(base_df, new_df)
    merged = pd.concat([base_aligned, new_aligned], ignore_index=True)
    before = len(merged)
    dedup_keys = _select_dedup_keys(all_cols, date_col)
    if dedup_keys:
        merged = merged.drop_duplicates(subset=dedup_keys, keep='first')
    else:
        merged = merged.drop_duplicates()
    return merged, before - len(merged), dedup_keys

def _compute_overlap(base_df: pd.DataFrame, new_df: pd.DataFrame, dedup_keys):
    if base_df is None or new_df is None or not dedup_keys:
        return None
    try:
        left = base_df[dedup_keys].dropna()
        right = new_df[dedup_keys].dropna()
        if left.empty or right.empty:
            return 0
        overlap = left.merge(right, on=dedup_keys, how="inner").drop_duplicates()
        return len(overlap)
    except Exception:
        return None

def load_data_sidebar(t):
    """注意：这里接收 t 参数"""
    st.sidebar.header(t['sidebar_header'])

    df = None
    base_df = None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(current_dir, "data", "梳理版.csv")
    fallback_path = os.path.join(current_dir, "梳理版.csv")
    if not os.path.exists(default_path) and os.path.exists(fallback_path):
        default_path = fallback_path

    # 1) 默认优先加载项目内测试数据
    if os.path.exists(default_path):
        try:
            base_df = pd.read_csv(default_path)
            default_name = os.path.basename(default_path)
            st.sidebar.success(f"{t['load_success']}: {default_name} ({len(base_df)} rows)")
        except Exception as e:
            st.sidebar.error(f"{t['load_fail']}: {e}")
    if base_df is not None:
        df = base_df

    # 2) 允许用户上传新数据，与旧数据进行对比选择
    uploaded_file = st.sidebar.file_uploader(
        t['upload_hint'],
        type=['xlsx', 'xls', 'csv', 'tsv', 'txt']
    )
    if uploaded_file:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif name.endswith('.tsv'):
                df = pd.read_csv(uploaded_file, sep='\t')
            elif name.endswith('.txt'):
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            else:
                df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"{t['upload_success']}: {uploaded_file.name} ({len(df)} rows)")
        except Exception as e:
            st.sidebar.error(f"{t['load_fail']}: {e}")
            st.stop()

        upload_context = f"{uploaded_file.name}-{uploaded_file.size}"
        if st.session_state.get("data_choice_context") != upload_context:
            st.session_state.data_choice_context = upload_context
            st.session_state.data_choice = None
            st.session_state.data_upload_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            if "data_choice_radio" in st.session_state:
                del st.session_state["data_choice_radio"]

        upload_ts = st.session_state.get("data_upload_ts")
        base_date_col = _detect_date_column(base_df) if base_df is not None else None
        new_date_col = _detect_date_column(df)

        base_start, base_end = _get_date_range(base_df, base_date_col)
        new_start, new_end = _get_date_range(df, new_date_col)

        base_range_label = _format_range_label(
            base_start, base_end, _get_file_mtime(default_path), t, is_upload=False
        )
        new_range_label = _format_range_label(
            new_start, new_end, upload_ts, t, is_upload=True
        )

        st.sidebar.markdown(f"### {t['data_compare_title']}")
        compare_rows = []
        if base_df is not None:
            compare_rows.append({
                t['data_compare_dataset']: t['data_compare_old'],
                t['data_compare_rows']: len(base_df),
                t['data_compare_date_col']: base_date_col or t['data_choice_no_date'],
                t['data_compare_date_range']: base_range_label
            })
        compare_rows.append({
            t['data_compare_dataset']: t['data_compare_new'],
            t['data_compare_rows']: len(df),
            t['data_compare_date_col']: new_date_col or t['data_choice_no_date'],
            t['data_compare_date_range']: new_range_label
        })
        compare_df = pd.DataFrame(compare_rows)
        # Browser console log
        try:
            _debug_data = {
                'shape': [int(compare_df.shape[0]), int(compare_df.shape[1])],
                'empty': bool(compare_df.empty),
                'columns': [str(c) for c in compare_df.columns],
                'dtypes': {str(k): str(v) for k, v in compare_df.dtypes.items()},
                'rows': [{str(k): str(v) for k, v in row.items()} for _, row in compare_df.iterrows()]
            }
            _st_comp.html(
                f'<script>parent.console.log("[DEBUG config.py] 数据对比:", {json.dumps(_debug_data, ensure_ascii=False)})</script>',
                height=0
            )
        except Exception as _e:
            print(f"[DEBUG config.py] Browser console log failed: {_e}")
        print(f"\n{'='*60}\n[DEBUG config.py] 数据对比 DataFrame:")
        print(f"  shape={compare_df.shape}, empty={compare_df.empty}")
        print(f"  columns={list(compare_df.columns)}")
        print(f"  dtypes={dict(compare_df.dtypes)}")
        print(f"  data={compare_df.to_dict('records')}")
        print(f"{'='*60}\n")
        st.sidebar.dataframe(compare_df, width="stretch", hide_index=True)
        with st.sidebar.expander("DEBUG 数据对比", expanded=False):
            st.write(f"shape={compare_df.shape}, empty={compare_df.empty}")
            st.write(f"columns={list(compare_df.columns)}")
            st.write(f"dtypes={dict(compare_df.dtypes)}")
            st.write(compare_df)

        overlap_text = t['data_compare_overlap_unknown']
        if base_df is not None:
            date_col_for_overlap = base_date_col or new_date_col
            all_cols = list(dict.fromkeys(list(base_df.columns) + list(df.columns)))
            overlap_keys = _select_dedup_keys(all_cols, date_col_for_overlap)
            overlap_count = _compute_overlap(base_df, df, overlap_keys)
            if overlap_count is not None:
                overlap_text = (
                    f"{t['data_compare_overlap_yes']} ({overlap_count})"
                    if overlap_count > 0 else t['data_compare_overlap_no']
                )
            elif base_start is not None and base_end is not None and new_start is not None and new_end is not None:
                if not (base_end < new_start or new_end < base_start):
                    overlap_text = t['data_compare_overlap_yes']
                else:
                    overlap_text = t['data_compare_overlap_no']
            st.sidebar.info(f"{t['data_compare_overlap']}: {overlap_text}")

        if base_df is not None:
            def _render_choice_dialog():
                st.write(t['data_choice_desc'])
                merge_date_col = base_date_col or new_date_col
                merge_preview, _, _ = _merge_datasets(base_df, df, merge_date_col)
                merge_start, merge_end = _get_date_range(merge_preview, merge_date_col)
                merge_range_label = _format_range_label(
                    merge_start, merge_end, upload_ts, t, is_upload=True
                )
                option_merge = f"{t['data_choice_option_merge']} ({t['data_choice_label_prefix']}: {merge_range_label})"
                option_new = f"{t['data_choice_option_new']} ({t['data_choice_label_prefix']}: {new_range_label})"
                choice = st.radio("", [option_merge, option_new], key="data_choice_radio")
                if st.button(t['data_choice_confirm']):
                    st.session_state.data_choice = "merge" if choice == option_merge else "new"
                    st.rerun()

            if st.session_state.get("data_choice") is None:
                if hasattr(st, "dialog"):
                    @st.dialog(t['data_choice_title'])
                    def _dialog():
                        _render_choice_dialog()
                    _dialog()
                else:
                    st.sidebar.warning(t['data_choice_wait'])
                    _render_choice_dialog()
                st.stop()

            if st.session_state.get("data_choice") == "merge":
                merge_date_col = base_date_col or new_date_col
                merged_df, deduped_count, _ = _merge_datasets(base_df, df, merge_date_col)
                st.sidebar.success(f"{t['data_choice_merge_summary']}: {len(merged_df)} rows")
                if deduped_count > 0:
                    st.sidebar.caption(f"{t['data_compare_overlap']}: -{deduped_count} rows")
                render_chip_filter_sidebar(merged_df, t)
                return merged_df
            st.sidebar.success(f"{t['data_choice_new_summary']}: {len(df)} rows")
            render_chip_filter_sidebar(df, t)
            return df

        render_chip_filter_sidebar(df, t)
        return df

    if df is None:
        st.info(t['upload_guide'])
        st.stop()

    render_chip_filter_sidebar(df, t)
    return df
