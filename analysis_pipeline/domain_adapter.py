"""
domain_adapter.py — 领域无关的分析 / 报告适配层

把原先硬编码在 ai_text_analysis / generate_html_report / static_report_generator
里的半导体业务知识，改为完全由 DataSchema（LLM 或规则自动识别）驱动。
这样同一套分析 → 解读 → 报告流水线可以适用于任意行业的数据集。

设计原则：
  1. 业务理解来自 DataSchema（列语义、目标含义、单位），而不是写死的字段名。
  2. LLM 在生成报告时，接收 schema 的语义信息 + 通用统计量，自行产出
     领域适配的叙述，从而彻底摆脱"虚焊/压连/铟柱/M5"这类硬编码知识。
  3. 所有统计计算只依赖 schema 给出的角色（target / feature / datetime / position），
     不依赖任何具体列名。
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "web_app"))

from schema_analyzer import DataSchema, analyze_schema, _fallback_schema
from project_paths import OUTPUT_DIR, CLEANED_DATA_FILE, ML_REPORT_DIR

# Schema 持久化文件，供流水线各步骤共享同一份理解
SCHEMA_FILE = os.path.join(OUTPUT_DIR, "schema.json")


# ─────────────────────────────────────────────────────────────
# 1. Schema 加载 / 构建 / 持久化
# ─────────────────────────────────────────────────────────────

def make_ai_client():
    """构造 OpenAI 兼容客户端（沿用 ai_text_analysis 的环境变量约定）。

    返回 None 表示未配置 API，调用方应退回到基于规则的分析。
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        return None
    base_url = (
        os.getenv("DASHSCOPE_API_BASE")
        or os.getenv("BASE_URL")
        or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def discover_charts(root_dir: Optional[str] = None) -> List[str]:
    """发现当前实际生成的图表文件，返回相对 ROOT_DIR 的路径列表（如 output/analysis_report/0_xxx.png）。

    供 LLM 在报告中真实引用，避免编造不存在的图表路径。
    """
    root_dir = root_dir or _ROOT
    rel_paths = []
    for sub in ("analysis_report", "ml_report", "position_analysis_v2"):
        d = os.path.join(root_dir, "output", sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".png"):
                rel_paths.append(f"output/{sub}/{fn}")
    return rel_paths


def load_or_build_schema(cleaned_csv: Optional[str] = None,
                         client=None,
                         model: Optional[str] = None,
                         force: bool = False) -> DataSchema:
    """加载已持久化的 schema.json；不存在则用规则/LLM 构建并保存。"""
    cleaned_csv = cleaned_csv or CLEANED_DATA_FILE

    if not force and os.path.exists(SCHEMA_FILE):
        try:
            with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
                return DataSchema.from_dict(json.load(f))
        except Exception as e:
            print(f"[domain_adapter] 读取 schema.json 失败，重新构建: {e}")

    if not os.path.exists(cleaned_csv):
        print(f"[domain_adapter] 找不到清洗数据 {cleaned_csv}，返回空 schema")
        return DataSchema()

    df = pd.read_csv(cleaned_csv)
    schema = None
    if client is not None and model:
        try:
            schema = analyze_schema(df, client, model=model)
        except Exception as e:
            print(f"[domain_adapter] LLM 识别 schema 失败，改用规则: {e}")
            schema = None
    if schema is None:
        schema = _fallback_schema(df)

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(SCHEMA_FILE, "w", encoding="utf-8") as f:
            json.dump(schema.to_dict(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[domain_adapter] 保存 schema.json 失败: {e}")

    return schema


# ─────────────────────────────────────────────────────────────
# 2. 通用 KPI（只依赖 schema 的目标列与目标值映射）
# ─────────────────────────────────────────────────────────────

def _resolve_target_column(df: pd.DataFrame, schema: DataSchema) -> Optional[str]:
    """通用目标列解析：优先 schema，其次常见名。"""
    candidates = []
    if schema and schema.target_column:
        candidates.append(schema.target_column)
    candidates += ["Is_Pass", "Label_Pass", "target", "label", "y", "class", "结果", "状态", "良率"]
    for c in candidates:
        if c and c in df.columns:
            return c
    # 最后尝试：取值种类最少的类别列
    cat_cols = [c for c in df.columns if df[c].dtype == object or df[c].nunique() < 20]
    if cat_cols:
        return min(cat_cols, key=lambda c: df[c].nunique())
    return None


def compute_kpi(df: pd.DataFrame, schema: Optional[DataSchema] = None) -> Dict[str, Any]:
    """基于 schema 的目标列与目标值映射，通用计算 KPI。

    返回结构（全部可选，缺失字段表示无法计算）：
        total, target_column, target_label,
        pass_count, fail_count, pass_rate,
        value_distribution (dict: 原始值 -> 数量)
    """
    result: Dict[str, Any] = {"total": len(df)}
    if df.empty:
        return result

    target_col = _resolve_target_column(df, schema)
    if not target_col:
        return result
    result["target_column"] = target_col

    series = df[target_col]
    result["value_distribution"] = {
        str(k): int(v) for k, v in series.value_counts(dropna=True).to_dict().items()
    }

    pass_set, fail_set = _target_value_sets(schema, series)
    if pass_set or fail_set:
        s_str = series.astype(str)
        pass_count = int(s_str.isin(pass_set).sum()) if pass_set else 0
        fail_count = int(s_str.isin(fail_set).sum()) if fail_set else 0
        total = len(series.dropna())
        result["pass_count"] = pass_count
        result["fail_count"] = fail_count
        result["pass_rate"] = (pass_count / total * 100) if total else 0.0
        if schema:
            result["target_label"] = f"{schema.pass_label} / {schema.fail_label}"
    return result


def _target_value_sets(schema: Optional[DataSchema], series: pd.Series) -> Tuple[set, set]:
    """从 schema 解析 pass/fail 值集合；没有则用 0/1 启发式。"""
    pass_set, fail_set = set(), set()
    if schema:
        if schema.pass_values:
            pass_set = {str(v) for v in schema.pass_values}
        if schema.fail_values:
            fail_set = {str(v) for v in schema.fail_values}
    if not pass_set and not fail_set:
        uniq = set(series.dropna().astype(str).unique())
        if uniq <= {"0", "1"} or uniq <= {0, 1}:
            pass_set, fail_set = {"1"}, {"0"}
    return pass_set, fail_set


# ─────────────────────────────────────────────────────────────
# 3. 通用分析数据收集（供 LLM 叙述与静态报告共用）
# ─────────────────────────────────────────────────────────────

def collect_generic_analysis(cleaned_csv: Optional[str] = None,
                             schema: Optional[DataSchema] = None) -> Dict[str, Any]:
    """收集一套领域无关的统计量，作为 LLM 叙述与静态报告的输入。

    只依赖 schema 给出的角色，不引用任何具体列名。
    """
    cleaned_csv = cleaned_csv or CLEANED_DATA_FILE
    analysis: Dict[str, Any] = {"kpi_stats": {}, "feature_stats": [], "correlations": [],
                                "drift": None, "position_stats": {}, "ml_importance": []}

    if not os.path.exists(cleaned_csv):
        return analysis

    df = pd.read_csv(cleaned_csv)
    if df.empty:
        return analysis

    # KPI
    analysis["kpi_stats"] = compute_kpi(df, schema)

    # 数值特征：合格/不合格 对比统计
    target_col = analysis["kpi_stats"].get("target_column")
    numeric_features = schema.get_numeric_features() if schema else \
        [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if target_col and target_col in df.columns:
        pass_set, fail_set = _target_value_sets(schema, df[target_col])
        for col in numeric_features[:12]:
            if col == target_col or col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() == 0:
                continue
            rec = {
                "column": col,
                "display": _display_name(schema, col),
                "unit": _unit_of(schema, col),
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
            }
            if pass_set or fail_set:
                s_str = df[target_col].astype(str)
                mask_pass = s_str.isin(pass_set) if pass_set else None
                mask_fail = s_str.isin(fail_set) if fail_set else None
                if mask_pass is not None and mask_pass.sum() > 0:
                    rec["pass_median"] = round(float(s[mask_pass].median()), 3)
                if mask_fail is not None and mask_fail.sum() > 0:
                    rec["fail_median"] = round(float(s[mask_fail].median()), 3)
                if mask_pass is not None and mask_fail is not None and \
                        mask_pass.sum() > 0 and mask_fail.sum() > 0:
                    rec["delta"] = round(float(s[mask_pass].median() - s[mask_fail].median()), 3)
            analysis["feature_stats"].append(rec)

    # 相关性：top 绝对值对
    corr_cols = [c for c in numeric_features if c in df.columns]
    if target_col and target_col in df.columns:
        corr_cols = corr_cols + [target_col]
    if len(corr_cols) >= 2:
        try:
            corr = df[corr_cols].corr()
            pairs = []
            for i in range(len(corr_cols)):
                for j in range(i + 1, len(corr_cols)):
                    v = corr.iloc[i, j]
                    if pd.notna(v) and abs(v) >= 0.3:
                        pairs.append({
                            "a": _display_name(schema, corr_cols[i]),
                            "b": _display_name(schema, corr_cols[j]),
                            "r": round(float(v), 2),
                        })
            pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
            analysis["correlations"] = pairs[:8]
        except Exception:
            pass

    # 漂移：如有日期列，比较前半/后半段的首个数值特征
    analysis["drift"] = _compute_drift(df, schema)

    # 位置效应：如有位置列
    analysis["position_stats"] = _compute_position_effect(df, schema, target_col)

    # ML 特征重要性（若存在）
    analysis["ml_importance"] = _read_ml_importance()

    return analysis


def _display_name(schema: Optional[DataSchema], col: str) -> str:
    if schema:
        for c in schema.columns:
            if c.raw_name == col:
                return c.display_name or c.semantic_name or col
    return col


def _unit_of(schema: Optional[DataSchema], col: str) -> Optional[str]:
    if schema:
        for c in schema.columns:
            if c.raw_name == col:
                return c.physical_unit
    return None


def _compute_drift(df: pd.DataFrame, schema: Optional[DataSchema]) -> Optional[Dict[str, Any]]:
    date_col = schema.find_column("date", "时间", "process") if schema else None
    if not date_col or date_col not in df.columns:
        for c in df.columns:  # 兜底：找 datetime 类型列
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break
    if not date_col or date_col not in df.columns:
        return None

    numeric_features = schema.get_numeric_features() if schema else \
        [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_features = [c for c in numeric_features if c in df.columns]
    if not numeric_features:
        return None

    feat = numeric_features[0]
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, feat]).sort_values(date_col)
    if len(d) < 10:
        return None
    median_idx = len(d) // 2
    early = pd.to_numeric(d[feat].iloc[:median_idx], errors="coerce").mean()
    late = pd.to_numeric(d[feat].iloc[median_idx:], errors="coerce").mean()
    if pd.isna(early) or pd.isna(late):
        return None
    return {
        "feature": _display_name(schema, feat),
        "unit": _unit_of(schema, feat),
        "early_mean": round(float(early), 3),
        "late_mean": round(float(late), 3),
        "drift_amount": round(float(late - early), 3),
        "drift_trend": "negative" if late < early else "positive",
    }


def _compute_position_effect(df: pd.DataFrame, schema: Optional[DataSchema],
                             target_col: Optional[str]) -> Dict[str, Any]:
    if not target_col or target_col not in df.columns:
        return {}
    pos_col = schema.find_column("position", "位置", "code") if schema else None
    if not pos_col or pos_col not in df.columns:
        return {}
    sub = df.dropna(subset=[pos_col, target_col])
    if sub.empty:
        return {}
    pass_set, fail_set = _target_value_sets(schema, df[target_col])
    grp = sub.groupby(pos_col)[target_col].agg(["mean", "count"]).reset_index()
    grp = grp[grp["count"] > 0]
    if grp.empty:
        return {}
    worst = grp.loc[grp["mean"].idxmin()].to_dict()
    best = grp.loc[grp["mean"].idxmax()].to_dict()
    return {
        "position_column": pos_col,
        "worst": {pos_col: str(worst[pos_col]), "yield": round(float(worst["mean"]) * 100, 1)},
        "best": {pos_col: str(best[pos_col]), "yield": round(float(best["mean"]) * 100, 1)},
        "count": int(grp.shape[0]),
    }


def _read_ml_importance() -> List[Dict[str, Any]]:
    csv_path = os.path.join(ML_REPORT_DIR, "feature_importance_ranking.csv")
    if not os.path.exists(csv_path):
        return []
    try:
        mdf = pd.read_csv(csv_path)
        cols = [c for c in ["Feature", "Total_Score", "mean_importance"] if c in mdf.columns]
        if "Feature" not in cols:
            return []
        keep = ["Feature"] + [c for c in cols if c != "Feature"]
        recs = mdf.head(5)[keep].to_dict("records")
        out = []
        for r in recs:
            out.append({
                "feature": r.get("Feature"),
                "score": round(float(r.get("Total_Score", r.get("mean_importance", 0))), 4),
            })
        return out
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# 4. 把 Schema 语义转为 LLM 可理解的"业务背景"文本
# ─────────────────────────────────────────────────────────────

def describe_schema_for_llm(schema: Optional[DataSchema]) -> str:
    """将 DataSchema 的语义信息整理成给 LLM 的业务背景说明。

    这是"业务理解"的核心：模型不再依赖硬编码字段，而是从这里获得
    目标含义、特征清单与单位。
    """
    if not schema:
        return "（未提供数据模式，请仅基于下方统计结果做客观、通用的分析。）"

    lines = []
    if schema.target_column:
        lines.append(f"预测目标列：{schema.target_column}（{schema.target_type} 类型）")
        if schema.target_mapping:
            mapped = "；".join(f"{k}={v}" for k, v in schema.target_mapping.items())
            lines.append(f"目标值含义：{mapped}")
        if schema.pass_values or schema.fail_values:
            lines.append(f"判定标准：{schema.pass_label} = {schema.pass_values}；"
                         f"{schema.fail_label} = {schema.fail_values}")
    else:
        lines.append("未识别到明确的目标列，请基于整体数据分布做描述性分析。")

    feats = [c for c in schema.columns if c.role == "feature"]
    if feats:
        lines.append("关键特征（语义名 / 显示名 / 单位 / 类型）：")
        for c in feats[:20]:
            unit = f" / 单位:{c.physical_unit}" if c.physical_unit else ""
            lines.append(f"  - {c.semantic_name or c.raw_name}（{c.display_name or c.raw_name}）"
                         f"{unit} / {c.dtype}")
    if schema.id_column:
        lines.append(f"样本标识列：{schema.id_column}")
    if schema.has_column_role("datetime"):
        lines.append("数据包含时间列，可用于趋势/漂移分析。")
    if schema.has_column_role("ignore"):
        ignored = [c.raw_name for c in schema.columns if c.role == "ignore"]
        lines.append(f"已忽略的列：{', '.join(ignored)}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 5. 通用 LLM 报告 prompt（无行业硬编码）
# ─────────────────────────────────────────────────────────────

def _stats_to_text(analysis: Dict[str, Any]) -> str:
    """把通用统计量格式化为 LLM 易读的文字。"""
    parts = []
    kpi = analysis.get("kpi_stats", {})
    if kpi.get("total"):
        parts.append(f"样本总数：{kpi['total']}")
    if "pass_rate" in kpi:
        parts.append(f"整体{kpi.get('target_label', '良品率')}：{kpi['pass_rate']:.2f}%"
                     f"（{kpi.get('pass_count', 0)} 个合格 / {kpi.get('fail_count', 0)} 个不合格）")
    if kpi.get("value_distribution"):
        dist = "，".join(f"{k}:{v}" for k, v in kpi["value_distribution"].items())
        parts.append(f"目标列取值分布：{dist}")

    feats = analysis.get("feature_stats", [])
    if feats:
        parts.append("\n关键特征统计（含合格/不合格中位数对比）：")
        for f in feats[:10]:
            line = f"  - {f['display']}"
            if f.get("unit"):
                line += f"（单位 {f['unit']}）"
            line += f"：均值 {f['mean']}，范围 {f['min']}~{f['max']}"
            if "pass_median" in f and "fail_median" in f:
                line += f"；合格中位数 {f['pass_median']} vs 不合格中位数 {f['fail_median']}"
                if "delta" in f:
                    line += f"（差异 {f['delta']}）"
            parts.append(line)

    corrs = analysis.get("correlations", [])
    if corrs:
        parts.append("\n参数间相关性（|r| >= 0.3）：")
        for c in corrs:
            parts.append(f"  - {c['a']} 与 {c['b']}：r = {c['r']}")

    drift = analysis.get("drift")
    if drift:
        parts.append(f"\n趋势/漂移：{drift['feature']} 前期均值 {drift['early_mean']} → "
                     f"后期均值 {drift['late_mean']}（{'下降' if drift['drift_trend']=='negative' else '上升'} "
                     f"{abs(drift['drift_amount'])}）")

    pos = analysis.get("position_stats", {})
    if pos:
        pcol = pos.get("position_column")
        parts.append(f"\n位置/空间效应（基于 {pcol} 列，共 {pos.get('count')} 个位置）："
                     f"最差位置 {pos.get('worst', {}).get(pcol)}"
                     f"（{pos.get('worst', {}).get('yield')}%）"
                     f"，最佳位置 {pos.get('best', {}).get(pcol)}"
                     f"（{pos.get('best', {}).get('yield')}%）")

    ml = analysis.get("ml_importance", [])
    if ml:
        parts.append("\n机器学习特征重要性（Top）：")
        for m in ml:
            parts.append(f"  - {m['feature']}：{m['score']}")
    return "\n".join(parts)


def build_report_prompt(analysis: Dict[str, Any],
                        schema: Optional[DataSchema] = None,
                        chart_data_text: Optional[str] = None,
                        chart_paths: Optional[List[str]] = None,
                        report_title: str = "数据分析报告") -> str:
    """构建领域无关的 LLM 报告 prompt。

    关键：业务背景来自 describe_schema_for_llm(schema)，
    统计事实来自 _stats_to_text(analysis)。不出现任何半导体专属知识。
    chart_paths：当前实际生成的图表相对路径列表，供 LLM 真实引用。
    """
    business_ctx = describe_schema_for_llm(schema)
    stats_text = _stats_to_text(analysis)

    prompt = f"""【角色设定】
你是一位资深的数据分析与企业决策顾问，擅长跨行业的质量与工艺分析。
我们已完成数据探索（EDA）、位置/空间效应分析、机器学习归因等步骤，现在需要你基于提供的统计事实，生成一份专业、客观的综合分析报告。

【业务背景（由数据模式自动识别，请据此理解字段含义）】
{business_ctx}

【统计数据事实（请严格引用，严禁虚构或臆测数值区间）】
{stats_text}
"""

    if chart_data_text:
        prompt += f"\n【图表数据摘要】\n{chart_data_text}\n"

    if chart_paths:
        chart_list = "\n".join(f"  - {p}" for p in chart_paths)
        prompt += f"""
【可用图表（请只在分析对应内容时引用以下相对路径；格式：
  <div class="chart-wrapper"><img src="相对路径" alt="图表说明"></div>）】
{chart_list}
"""

    prompt += f"""
【任务要求】
1. 输出一份结构完整的 HTML 报告内容（从 <div class="section-card"> 开始，到 </div> 结束），
   可包含 <h2>/<h3>/<p>/<ul>/<li> 以及 <div class="chart-wrapper"><img .../></div>。
2. 报告结构建议：
   - 总体介绍（用一句话概括数据主题，不要臆测行业细节）
   - 核心分布现状（结合目标列分布与关键特征）
   - 关键参数/特征差异（合格 vs 不合格对比，指出风险区间）
   - 参数关联性（若提供相关性）
   - 趋势/漂移（若提供）
   - 位置/空间效应（若提供）
   - 机器学习归因（若提供特征重要性）
   - 总结与优化建议（每条建议以“建议”开头，使用审慎、建议性口吻，避免绝对化指令）
3. 全程使用中文；字段一律使用业务背景中给出的显示名/语义名，不要出现原始英文列名或下划线字段名。
4. 每个图表对应一段分析，每段至少 3-5 句话；建议紧跟对应图表下方。
5. 只能基于【统计数据事实】撰写，严禁编造不存在的数值、百分比或物理区间。
6. 直接输出 HTML 片段，不要包含 ```markdown 或 ```html 标记，不要额外解释。

请开始生成报告："""
    return prompt


# ─────────────────────────────────────────────────────────────
# 6. 通用静态报告（AI 失败时的兜底，同样不依赖行业知识）
# ─────────────────────────────────────────────────────────────

def build_static_report_html(analysis: Dict[str, Any],
                             schema: Optional[DataSchema] = None) -> str:
    """生成领域无关的静态 HTML 报告内容（兜底用）。"""
    kpi = analysis.get("kpi_stats", {})
    total = kpi.get("total", 0)

    html = ['<div class="section-card">', '<h2>核心分布现状</h2>']

    # 目标分布
    if kpi.get("value_distribution"):
        dist_items = "，".join(f"{k}: {v} 个" for k, v in kpi["value_distribution"].items())
        html.append(
            '<div class="chart-wrapper"><img src="output/analysis_report/0_生产状态分布统计.png" '
            'alt="目标分布"></div>'
        )
        html.append(f"<p>基于全量样本（N={total}）的分析显示，目标列取值分布为：{dist_items}。</p>")

    # 关键特征差异
    feats = analysis.get("feature_stats", [])
    if feats:
        html.append('<h3>关键特征差异</h3>')
        html.append(
            '<div class="chart-wrapper"><img src="output/analysis_report/2_核心特征分布_2x3_中文.png" '
            'alt="关键特征分布"></div>'
        )
        html.append("<p>以下为特征在合格与不合格样本间的统计对比：</p><ul>")
        for f in feats[:8]:
            unit = f" {f['unit']}" if f.get("unit") else ""
            if "pass_median" in f and "fail_median" in f:
                html.append(
                    f"<li><strong>{f['display']}{unit}：</strong> 均值 {f['mean']}；"
                    f"合格中位数 {f['pass_median']} vs 不合格中位数 {f['fail_median']}"
                    f"（差异 {f['delta']}）</li>"
                )
            else:
                html.append(
                    f"<li><strong>{f['display']}{unit}：</strong> 均值 {f['mean']}，"
                    f"范围 {f['min']}~{f['max']}</li>"
                )
        html.append("</ul>")

    # 相关性
    corrs = analysis.get("correlations", [])
    if corrs:
        html.append('<h3>参数关联性</h3>')
        html.append(
            '<div class="chart-wrapper"><img src="output/analysis_report/1_参数相关性分析.png" '
            'alt="参数相关性"></div>'
        )
        html.append('<div class="analysis-box"><ul>')
        for c in corrs:
            direction = "正相关" if c["r"] > 0 else "负相关"
            html.append(f"<li><strong>{c['a']} 与 {c['b']}：</strong> {direction}（r = {c['r']}）</li>")
        html.append("</ul></div>")

    html.append("</div>")

    # 位置效应
    pos = analysis.get("position_stats", {})
    if pos:
        pcol = pos.get("position_column")
        worst = pos.get("worst", {})
        best = pos.get("best", {})
        html.append('<div class="section-card"><h2>位置 / 空间效应分析</h2>')
        html.append(
            '<div class="chart-wrapper"><img src="output/position_analysis_v2/1_Position_Yield_Rate.png" '
            'alt="位置良率"></div>'
        )
        html.append(
            f"<p>基于 {pcol} 列的分析显示，共 {pos.get('count')} 个位置。"
            f"最差位置 {worst.get(pcol)}（{worst.get('yield')}%），"
            f"最佳位置 {best.get(pcol)}（{best.get('yield')}%）。"
            f"建议重点排查 {worst.get(pcol)} 对应的工艺/设备一致性。</p>"
        )
        html.append("</div>")

    # ML 归因
    ml = analysis.get("ml_importance", [])
    if ml:
        html.append('<div class="section-card"><h2>归因分析：机器学习模型洞察</h2>')
        html.append(
            '<div class="chart-wrapper"><img src="output/ml_report/1_SHAP_归因分析.png" '
            'alt="SHAP 归因"></div>'
        )
        html.append("<p><strong>Top 关键特征：</strong></p><ul>")
        for m in ml:
            html.append(f"<li><strong>{m['feature']}：</strong> 重要性得分 {m['score']}</li>")
        html.append("</ul></div>")

    # 总结建议
    html.append(
        '<div class="section-card" style="border-top: 5px solid #7cb342;">'
        '<h2 style="color: #33691e; border-bottom-color: #7cb342;">总结与优化建议</h2>'
        "<p>基于全量数据分析，建议围绕上述关键特征与高风险区间建立监控与预警机制：</p>"
        '<div class="advice-list">'
    )
    for f in feats[:3]:
        html.append(
            f'<div class="advice-item"><div class="advice-icon">📌</div>'
            f'<div class="advice-content"><strong>关注 {f["display"]} 的控制区间</strong>'
            f'建议监控该参数的分布漂移，并对偏离合格样本中位数较大的批次进行排查。</div></div>'
        )
    if pos:
        html.append(
            f'<div class="advice-item"><div class="advice-icon">🎯</div>'
            f'<div class="advice-content"><strong>位置特异性整改</strong>'
            f'建议以最优位置 {best.get(pcol)} 为参照，对最差位置 {worst.get(pcol)} 开展针对性工艺校准。</div></div>'
        )
    html.append("</div></div>")

    return "\n".join(html)


if __name__ == "__main__":
    s = load_or_build_schema()
    print("Schema 列数:", len(s.columns))
    print(describe_schema_for_llm(s))
    a = collect_generic_analysis(schema=s)
    print(json.dumps(a, ensure_ascii=False, indent=2)[:2000])
