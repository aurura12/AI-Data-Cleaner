import os
import platform
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适配无显示环境
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import matplotlib.font_manager as fm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import FONT_FILE, CLEANED_DATA_FILE, ML_REPORT_DIR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_app"))
from schema_analyzer import DataSchema

# ================= 配置 =================
INPUT_FILE = CLEANED_DATA_FILE
OUTPUT_DIR = ML_REPORT_DIR

# --- 0. 环境配置：字体与绘图风格 ---
plt.style.use('seaborn-v0_8')
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 自动判断操作系统，选择中文字体
sys_name = platform.system()
font_path = FONT_FILE
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print(f"✅ 成功加载自定义字体: {font_path}")
elif sys_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
elif sys_name == "Darwin":  # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

# =======================================

_SCIENTIFIC_TOKEN_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def _normalize_numeric_series(series):
    """Sanitize odd stringified numeric values such as "[5E-1]"."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')

    def _clean_scalar(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val

        text = str(val).strip()
        if not text:
            return np.nan
        if text.startswith('[') and text.endswith(']'):
            inner = text[1:-1].strip()
            if ',' not in inner:
                text = inner

        match = _SCIENTIFIC_TOKEN_RE.search(text)
        return match.group(0) if match else np.nan

    return pd.to_numeric(series.map(_clean_scalar), errors='coerce')


def _compute_shap_values(xgb_model, X):
    """Use SHAP when possible and fall back to XGBoost native contributions."""
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        method = "shap.TreeExplainer"
    except Exception as e:
        print(f"[warning] SHAP TreeExplainer failed, fallback to XGBoost pred_contribs: {e}")
        booster = xgb_model.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns), missing=np.nan)
        shap_values = booster.predict(dmatrix, pred_contribs=True, validate_features=False)
        if getattr(shap_values, "ndim", 0) == 3:
            shap_values = shap_values[1]
        shap_values = shap_values[:, :-1]
        method = "xgboost.pred_contribs"
    return np.asarray(shap_values), method

def run_ml_analysis(input_path=None, output_path=None, schema=None):
    file_to_read = input_path if input_path else INPUT_FILE
    dir_to_save = output_path if output_path else OUTPUT_DIR

    if not os.path.exists(file_to_read):
        print(f"错误：找不到文件 {file_to_read}")
        return []
        
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
        
    print(f"[读取] 正在加载数据...")
    df = pd.read_csv(file_to_read)

    analysis_results = []
    
    # =====================================================
    # 1. 特征工程 (Feature Engineering)
    # =====================================================
    
    # --- A. 标签定义 (Target Definition) ---
    # 逻辑：我们预测样本是否为"不良/异常"。
    # 当有 schema 时，使用 schema 定义的目标列和值映射。
    # SHAP 值越正，代表该特征越推动样本变为"不良"。

    target_col = None
    target_col_is_pass = False  # 目标列是否为"1=Pass, 0=Fail"的良率模式

    if schema and schema.target_column and schema.target_column in df.columns:
        target_col = schema.target_column
        # 检查是否有明确的 pass/fail 值映射
        if schema.pass_values and schema.fail_values:
            # 构建映射: pass_values → 0 (good), fail_values → 1 (bad)
            s_str = df[target_col].astype(str)
            pass_set = {str(v) for v in schema.pass_values}
            fail_set = {str(v) for v in schema.fail_values}
            # 先标记 pass 为 0 (非缺陷), fail 为 1 (缺陷)
            y = pd.Series(1, index=df.index)  # 默认视为缺陷
            y[s_str.isin(pass_set)] = 0       # pass 值标记为 0
            y[s_str.isin(fail_set)] = 1       # fail 值标记为 1
            # 不在映射表中的行设为 NaN 并丢弃
            mapped_mask = s_str.isin(pass_set) | s_str.isin(fail_set)
            y = y[mapped_mask]
            df = df.loc[mapped_mask].copy()
        else:
            # 没有 pass/fail 映射，检查是否是 0/1 格式
            uniq = sorted(df[target_col].dropna().unique())
            if set(uniq) <= {0, 1} or set(str(v) for v in uniq) <= {'0', '1'}:
                y = df[target_col].astype(int)
                target_col_is_pass = True
            elif set(uniq) <= {-1, 0, 1}:
                y = (df[target_col] <= 0).astype(int)
            else:
                print(f"警告：目标列 {target_col} 取值 {uniq}，尝试将最小值对应为良品")
                # 将取值最少的视为"不良"
                vc = df[target_col].value_counts()
                minority = vc.idxmin()
                y = (df[target_col] == minority).astype(int) if minority else pd.Series(0, index=df.index)
    elif 'Is_Pass' in df.columns:
        target_col = 'Is_Pass'
        target_col_is_pass = True
        y = (1 - df['Is_Pass']).astype(int)
    elif 'Label_Pass' in df.columns:
        target_col = 'Label_Pass'
        target_col_is_pass = True
        y = (1 - df['Label_Pass']).astype(int)
    elif schema and schema.target_column and schema.target_column in df.columns:
        target_col = schema.target_column
        # 尝试二元化
        vc = df[target_col].value_counts()
        if len(vc) >= 2:
            minority = vc.idxmin()
            y = (df[target_col].astype(str) == str(minority)).astype(int)
        else:
            print("错误：目标列取值单一，无法进行分类分析")
            return []
    else:
        print("错误：未找到目标列（尝试 Is_Pass/Label_Pass 或 schema 定义的目标列）")
        return []

    defect_rate = y.mean()
    fail_label = schema.fail_label if (schema and schema.fail_label) else "不良"
    pass_label = schema.pass_label if (schema and schema.pass_label) else "良品"
    print(f"[统计] 样本总数: {len(y)} | {fail_label}样本(1): {y.sum()} | {fail_label}率: {defect_rate:.2%}")
    
    # --- B. 准备特征矩阵 (X) ---
    
    if schema and schema.get_numeric_features():
        # 用 schema 自动选择特征（排除目标列）
        target_name = schema.get_target_column()
        feature_candidates = [
            f for f in schema.get_numeric_features()
            if f != target_name
        ]
        # 位置编码
        pos_col = schema.find_column('position', 'code', '位置')
        if pos_col and pos_col in df.columns:
            df['Position_Code'] = df[pos_col].fillna('Unknown').astype(str)
            df['Position_Code_Enc'] = LabelEncoder().fit_transform(df['Position_Code'])
            feature_candidates.append('Position_Code_Enc')
    else:
        # 无 schema 时：用数据启发式，选所有数值列作为候选特征
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != target_col
        ]
        feature_candidates = list(numeric_cols)

        # 尝试加入位置编码（若有名称含 position/位置/code 的列）
        pos_col = next(
            (c for c in df.columns
             if any(k in c.lower() for k in ['position', '位置', 'code', '工位'])),
            None
        )
        if pos_col and pos_col in df.columns and pos_col != target_col:
            df['Position_Code_Enc'] = LabelEncoder().fit_transform(
                df[pos_col].fillna('Unknown').astype(str)
            )
            if 'Position_Code_Enc' not in feature_candidates:
                feature_candidates.append('Position_Code_Enc')
    
    # 仅保留存在的列
    valid_features = [f for f in feature_candidates if f in df.columns]
    X = df[valid_features].copy()
    for col in X.columns:
        X[col] = _normalize_numeric_series(X[col])
    
    # --- C. 增加交互项（仅在 schema 中有先验知识时启用） ---
    # 无 schema 时不做任何特定领域假设，让模型自行发现交互关系
    
    # --- D. 显示名映射 (用于绘图展示，优先使用 schema display_name) ---
    name_mapping = {}
    if schema:
        for col_s in schema.columns:
            if col_s.raw_name in X.columns:
                name_mapping[col_s.raw_name] = col_s.display_name or col_s.semantic_name or col_s.raw_name
    # 无 schema 时，对特殊加工列做合理命名
    if not schema:
        name_mapping.update({
            'Position_Code_Enc': '位置编码',
        })

    # 按 valid_features 顺序构建中文特征名列表（供后续 importance_df 使用）
    feature_names_cn = [name_mapping.get(f, f) for f in valid_features]

    # --- E. 缺失值填充 (针对 RandomForeest) ---
    # XGBoost 可以自动处理 NaN，但 RF 不行
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # =====================================================
    # 2. 模型训练与分析
    # =====================================================

    # --- 模型 1: XGBoost (主要用于 SHAP 分析) ---
    print("[训练] 正在训练 XGBoost (用于归因分析)...")
    # scale_pos_weight 处理样本不平衡 (Fail样本少，增加权重)
    # 防止除以零
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_weight = neg_count / pos_count if pos_count > 0 else 1
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X, y) # XGB可以直接吃带NaN的X

    # --- SHAP 可解释性分析 ---
    print("[分析] 计算 SHAP 值 (寻找缺陷成因)...")
    shap_values, shap_method = _compute_shap_values(xgb_model, X)
    
    # 图1: SHAP Summary (核心图)
    plt.figure(figsize=(10, 8))
    # 兼容处理: shap_values 可能是 list (multiclass) 或 array
    sv_target = shap_values
    
    shap.summary_plot(sv_target, X, show=False)
    plt.title(f"关键特征对【{fail_label}风险】的影响程度\n(SHAP值>0 代表增加{fail_label}风险)", fontsize=14)
    plt.tight_layout()
    img_path_1 = os.path.join(dir_to_save, '1_SHAP_归因分析.png')
    plt.savefig(img_path_1, dpi=300)
    plt.close()

    mean_abs_shap = np.abs(sv_target).mean(axis=0)
    shap_importance = pd.DataFrame({
        'Feature': X.columns,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    shap_desc = "SHAP归因分析摘要：\n"
    shap_desc += f"模型识别出的最重要影响因素为 {shap_importance.iloc[0]['Feature']}，平均SHAP影响值 {shap_importance.iloc[0]['Mean_Abs_SHAP']:.4f}。\n"
    shap_desc += "前5大关键特征及其影响力排行：\n"
    for idx, row in shap_importance.head(5).iterrows():
        shap_desc += f"- {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}\n"
    shap_desc += "注：SHAP值越高，说明该特征对缺陷风险的影响权重越大。"

    analysis_results.append({
        "chart_name": "1.SHAP归因分析",
        "image_path": img_path_1,
        "data_description": shap_desc
    })
    
    # 图2: 依赖图 - 选择最重要的特征进行非线性分析
    first_feat = X.columns[0] if len(X.columns) > 0 else None
    target_feat_cn = name_mapping.get(first_feat, first_feat) if first_feat else None
    if first_feat:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            first_feat, 
            sv_target, 
            X, 
            display_features=X,
            show=False,
            interaction_index=None
        )
        plt.ylabel("SHAP值 (对结果贡献度)")
        plt.title(f"{target_feat_cn} 对目标的影响趋势", fontsize=14)
        plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
        plt.tight_layout()
        img_path_2 = os.path.join(dir_to_save, '2_主要特征依赖分析.png')
        plt.savefig(img_path_2, dpi=300)
        plt.close()

        dep_desc = f"特征依赖分析摘要（针对 {target_feat_cn}）：\n"
        feat_vals = X[first_feat].values
        feat_idx = list(X.columns).index(first_feat)
        feat_shaps = sv_target[:, feat_idx] if len(sv_target.shape) > 1 else sv_target
        high_risk_mask = feat_shaps > 0
        if np.any(high_risk_mask):
            risk_vals = feat_vals[high_risk_mask]
            dep_desc += f"当 {target_feat_cn} 处于区间 [{risk_vals.min():.2f}, {risk_vals.max():.2f}] 时，{fail_label}风险增加。\n"
        else:
            dep_desc += "该特征在当前观测范围内未显示出明显的风险增加趋势。\n"
        corr = np.corrcoef(feat_vals, feat_shaps)[0, 1]
        dep_desc += f"特征值与风险值相关系数为 {corr:.2f}。"
        
        analysis_results.append({
            "chart_name": "2.单变量依赖分析",
            "image_path": img_path_2,
            "data_description": dep_desc
        })

    # --- 模型 2: 随机森林 (Random Forest) ---
    print("[训练] 正在训练 Random Forest (用于重要性交叉验证)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_imputed, y) # RF 需要填充后的数据
    
    # --- 统计方法: 互信息 (Mutual Information) ---
    print("[分析] 计算互信息 (非线性相关性)...")
    mi_scores = mutual_info_classif(X_imputed, y, random_state=42, discrete_features='auto')
    
    # =====================================================
    # 3. 综合报告生成
    # =====================================================
    print("[整合] 生成参数重要性排行...")
    importance_df = pd.DataFrame({
        'Feature': feature_names_cn,
        'XGBoost': xgb_model.feature_importances_,
        'RandomForest': rf_model.feature_importances_,
        'MutualInfo': mi_scores
    })
    
    # 计算综合得分 (归一化后加权平均)
    scaler = MinMaxScaler()
    norm_df = pd.DataFrame(
        scaler.fit_transform(importance_df[['XGBoost', 'RandomForest', 'MutualInfo']]),
        columns=['Norm_XGB', 'Norm_RF', 'Norm_MI']
    )
    importance_df['Total_Score'] = norm_df.mean(axis=1)
    importance_df = importance_df.sort_values('Total_Score', ascending=False)
    
    print(f"\n=== Top 5 关键特征 ===")
    print(importance_df[['Feature', 'Total_Score']].head(5))
    importance_df.to_csv(os.path.join(dir_to_save, 'feature_importance_ranking.csv'), index=False)

    # 图3: 综合重要性柱状图
    plt.figure(figsize=(12, 6))
    # 绘制堆叠图或者并排图
    plot_df = importance_df.head(8).set_index('Feature')[['XGBoost', 'RandomForest', 'MutualInfo']]
    plot_df = pd.DataFrame(scaler.fit_transform(plot_df), index=plot_df.index, columns=['XGBoost权重', '随机森林权重', '互信息'])
    
    plot_df.plot(kind='barh', figsize=(12, 6), width=0.8, colormap='viridis')
    plt.gca().invert_yaxis() # 排名高的在上面
    plt.title("特征重要性综合排名 (Top 8 Factors)", fontsize=16)
    plt.xlabel("归一化重要性 (0~1)")
    plt.legend(loc='lower right')
    plt.tight_layout()
    img_path_3 = os.path.join(dir_to_save, '3_关键参数排名.png')
    plt.savefig(img_path_3, dpi=300)
    plt.close()
    
    rank_desc = "多模型综合特征重要性排名摘要：\n"
    rank_desc += "综合 XGBoost、RandomForest 和 MutualInfo 结果，排名前5的关键参数如下：\n"
    for i, (idx, row) in enumerate(importance_df.head(5).iterrows(), 1):
        rank_desc += f"{i}. {row['Feature']} (综合得分: {row['Total_Score']:.2f})\n"
    rank_desc += "建议优先关注以上关键参数。"

    analysis_results.append({
        "chart_name": "3.特征重要性综合排名",
        "image_path": img_path_3,
        "data_description": rank_desc
    })

    # 图4: 简单决策树 (生成人类可读的规则)
    print("[可视] 生成决策树规则图...")
    # 只用最重要的2个特征来画树，方便看阈值
    # 注意: importance_df['Feature'] 是 display_name（可能不含单位如(℃)），
    #        X_imputed 列名是 raw_name（含单位），需要反向映射后使用 raw_name 索引
    _feat_to_raw = {name_mapping.get(f, f): f for f in valid_features}
    top_2_display = importance_df['Feature'].head(2).tolist()
    top_2_raw = [_feat_to_raw.get(d, d) for d in top_2_display]
    X_tree = X_imputed[top_2_raw]
    
    tree_viz = DecisionTreeClassifier(max_depth=3, class_weight='balanced', min_samples_leaf=10)
    tree_viz.fit(X_tree, y)
    
    plt.figure(figsize=(16, 9))
    plot_tree(
        tree_viz, 
        feature_names=top_2_display, 
        filled=True, 
        rounded=True,
        class_names=[f'{pass_label}(合格)', f'{fail_label}(风险)'], 
        proportion=True, # 显示比例而不是数量
        fontsize=12
    )
    plt.title(f"基于[{top_2_display[0]}]与[{top_2_display[1]}]的判定规则树", fontsize=18)
    img_path_4 = os.path.join(dir_to_save, '4_决策树阈值规则.png')
    plt.savefig(img_path_4, dpi=300)
    plt.close()
    
    tree_desc = f"决策树规则摘要：基于 {top_2_display} 生成可读的阈值规则图。"

    analysis_results.append({
        "chart_name": "4.决策树规则分析",
        "image_path": img_path_4,
        "data_description": tree_desc
    })
    
    print(f"\n[完成] 机器学习分析结束。请查看目录: {dir_to_save}")
    return analysis_results
    print("建议重点关注: 1_SHAP_归因分析.png (红点越靠右，说明该特征值导致了缺陷)")

if __name__ == "__main__":
    try:
        from domain_adapter import load_or_build_schema
        # 从 pipeline 共享的 schema.json 加载，不传 client 表示仅从文件读取
        _schema = load_or_build_schema()
    except Exception:
        _schema = None
    run_ml_analysis(schema=_schema)
