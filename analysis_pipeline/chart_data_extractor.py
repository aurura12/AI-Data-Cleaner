"""
从EDA、ML、Position的绘图代码中提取关键数据点
转换为结构化文本，供LLM生成报告使用

全行业通用版：所有列名/显示名/单位/值含义均通过 DataSchema 驱动，
不包含任何行业特定的硬编码。
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import ROOT_DIR, OUTPUT_DIR, CLEANED_DATA_FILE, ML_REPORT_DIR
from domain_adapter import _display_name, _unit_of, _target_value_sets
from schema_analyzer import DataSchema


def extract_eda_chart_data(base_dir: str,
                           schema: Optional[DataSchema] = None) -> Dict[str, Any]:
    """从EDA分析中提取图表数据（全行业通用）。"""
    data_file = CLEANED_DATA_FILE
    if not os.path.exists(data_file):
        return {}

    df = pd.read_csv(data_file)

    # ── 1. 目标列 ──────────────────────────────────────────
    target_col = schema.target_column if (schema and schema.target_column) else None
    if not target_col:
        # 启发式：优先选数值型二元列（0/1），避免误选分类列（如产线A/B）
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            uniq = df[col].dropna().unique()
            if len(uniq) <= 2:
                target_col = col
                break
    if not target_col:
        # 回退：严格匹配仅含"0"和"1"的列（允许非数值型）
        for col in df.columns:
            uniq = set(str(u) for u in df[col].dropna().unique())
            if uniq <= {"0", "1"}:
                target_col = col
                break
    if not target_col:
        return {}  # 无目标列，做不了分类分析

    # ── 2. Is_Pass 判定 ────────────────────────────────────
    pass_set, fail_set = _target_value_sets(schema, df[target_col])
    if not pass_set and not fail_set:
        return {}

    def _is_pass(val):
        try:
            return 1 if str(val) in pass_set else 0
        except Exception:
            return 0

    df['_Is_Pass'] = df[target_col].apply(_is_pass)

    chart_data: Dict[str, Any] = {}

    # ── 3. 目标值分布 ──────────────────────────────────────
    status_map = schema.target_mapping if (schema and schema.target_mapping) else {}
    status_counts = {}
    for val in df[target_col].dropna():
        key = str(val)
        display = status_map.get(key, key)
        status_counts[display] = status_counts.get(display, 0) + 1

    chart_data['status_distribution'] = {
        'type': 'bar_chart',
        'title': '目标值分布',
        'data': [{'x': k, 'y': v} for k, v in status_counts.items()],
        'summary': f"总样本{len(df)}项，其中：{', '.join([f'{k} {v}项' for k, v in status_counts.items()])}"
    }

    # ── 4. 参数相关性矩阵 ──────────────────────────────────
    num_feats = schema.get_numeric_features() if schema else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    corr_cols = [c for c in num_feats if c in df.columns and c != target_col][:6]
    if target_col in df.columns and target_col not in corr_cols:
        corr_cols = corr_cols + [target_col]

    if len(corr_cols) >= 2:
        try:
            corr = df[corr_cols].corr()
            correlations = []
            for i, col1 in enumerate(corr_cols):
                for j, col2 in enumerate(corr_cols):
                    if i < j:
                        v = corr.loc[col1, col2]
                        if not np.isnan(v):
                            correlations.append({
                                'feature1': _display_name(schema, col1),
                                'feature2': _display_name(schema, col2),
                                'correlation': float(v)
                            })
            chart_data['correlation_matrix'] = {
                'type': 'heatmap',
                'title': '参数相关性分析',
                'data': correlations,
                'key_findings': [
                    f"{c['feature1']}与{c['feature2']}的相关系数为{c['correlation']:.2f}"
                    for c in correlations if abs(c['correlation']) > 0.3
                ]
            }
        except Exception:
            pass

    # ── 5. 合格/不合格特征对比（前3个数值特征） ────────────
    pass_label = schema.pass_label if (schema and schema.pass_label) else '合格'
    fail_label = schema.fail_label if (schema and schema.fail_label) else '不合格'

    for feat_col in num_feats[:3]:
        if feat_col == target_col or feat_col not in df.columns:
            continue
        col_display = _display_name(schema, feat_col)
        col_unit = _unit_of(schema, feat_col)

        pass_vals = df[df['_Is_Pass'] == 1][feat_col].dropna()
        fail_vals = df[df['_Is_Pass'] == 0][feat_col].dropna()

        chart_data[f'pass_fail_{feat_col}'] = {
            'type': 'boxplot',
            'title': f'{pass_label}/{fail_label} {col_display}对比',
            'data': {
                pass_label: {
                    'median': float(pass_vals.median()) if len(pass_vals) > 0 else 0,
                    'mean': float(pass_vals.mean()) if len(pass_vals) > 0 else 0,
                    'count': len(pass_vals),
                },
                fail_label: {
                    'median': float(fail_vals.median()) if len(fail_vals) > 0 else 0,
                    'mean': float(fail_vals.mean()) if len(fail_vals) > 0 else 0,
                    'count': len(fail_vals),
                },
            },
            'unit': col_unit,
        }

    # ── 6. 位置效应 ────────────────────────────────────────
    pos_col = schema.find_column("position", "位置", "code") if schema else None
    if pos_col and pos_col in df.columns:
        pos_stats = df.groupby(pos_col)['_Is_Pass'].agg(['mean', 'count']).reset_index()
        pos_stats = pos_stats[pos_stats['count'] >= 1].sort_values('mean', ascending=True)

        pos_data = [{
            'position': str(row[pos_col]),
            'yield_rate': float(row['mean']),
            'count': int(row['count']),
        } for _, row in pos_stats.iterrows()]

        chart_data['position_yield'] = {
            'type': 'bar_chart',
            'title': '位置效应',
            'data': pos_data,
            'worst': pos_data[0] if pos_data else None,
            'best': pos_data[-1] if pos_data else None,
            'average': float(df['_Is_Pass'].mean()),
            'position_column': pos_col,
        }

    # ── 7. 加工次序效应 ────────────────────────────────────
    order_col = schema.find_column("order", "sequence", "index", "批次") if schema else None
    if order_col and order_col in df.columns:
        order_stats = df.groupby(order_col)['_Is_Pass'].agg(['mean', 'count']).reset_index()
        order_stats = order_stats.sort_values(order_col)

        order_data = [{
            'order_value': str(row[order_col]),
            'yield_rate': float(row['mean']),
            'count': int(row['count']),
        } for _, row in order_stats.iterrows()]

        chart_data['order_effect'] = {
            'type': 'bar_chart',
            'title': '加工次序效应',
            'data': order_data,
            'worst': min(order_data, key=lambda x: x['yield_rate']) if order_data else None,
        }

    # ── 8. 时间趋势漂移 ────────────────────────────────────
    date_col = schema.find_column("date", "时间", "datetime") if schema else None
    if not date_col:
        for c in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df[c], errors='coerce')):
                    date_col = c
                    break
            except Exception:
                continue

    if date_col and date_col in df.columns and num_feats:
        drift_feat = num_feats[0]
        if drift_feat in df.columns and drift_feat != target_col:
            try:
                d = df.copy()
                d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
                d = d.dropna(subset=[date_col, drift_feat]).sort_values(date_col)
                if len(d) >= 10:
                    mid = len(d) // 2
                    early = pd.to_numeric(d[drift_feat].iloc[:mid], errors='coerce').mean()
                    late = pd.to_numeric(d[drift_feat].iloc[mid:], errors='coerce').mean()
                    if not (np.isnan(early) or np.isnan(late)):
                        feat_display = _display_name(schema, drift_feat)
                        feat_unit = _unit_of(schema, drift_feat)
                        unit_str = f"({feat_unit})" if feat_unit else ""
                        chart_data['time_drift'] = {
                            'type': 'trend',
                            'title': f'{feat_display}趋势漂移',
                            'data': {
                                'early_mean': float(early),
                                'late_mean': float(late),
                                'drift': float(late - early),
                                'feature': feat_display,
                                'unit': feat_unit,
                            },
                        }
            except Exception:
                pass

    return chart_data


def extract_ml_chart_data(base_dir: str) -> Dict[str, Any]:
    """从ML分析中提取图表数据（直接从 feature_importance_ranking.csv 读取，无需 schema）。"""
    ml_csv = os.path.join(ML_REPORT_DIR, 'feature_importance_ranking.csv')
    if not os.path.exists(ml_csv):
        return {}

    df = pd.read_csv(ml_csv)

    chart_data = {}
    feature_data = []
    for _, row in df.head(8).iterrows():
        feature_data.append({
            'feature': str(row['Feature']),
            'xgb_score': float(row.get('XGBoost', 0)),
            'rf_score': float(row.get('RandomForest', 0)),
            'mi_score': float(row.get('MutualInfo', 0)),
            'total_score': float(row.get('Total_Score', 0)),
        })

    chart_data['feature_importance'] = {
        'type': 'bar_chart',
        'title': '特征重要性综合排名',
        'data': feature_data,
        'top3': feature_data[:3] if len(feature_data) >= 3 else feature_data,
    }

    if len(feature_data) >= 2:
        chart_data['decision_tree_threshold'] = {
            'type': 'threshold',
            'title': '决策树阈值规则',
            'data': {
                'primary_feature': feature_data[0]['feature'],
                'threshold_note': '需要从实际决策树模型提取具体阈值',
            },
        }

    return chart_data


def extract_position_chart_data(base_dir: str,
                                schema: Optional[DataSchema] = None) -> Dict[str, Any]:
    """从Position分析中提取图表数据（全行业通用）。"""
    data_file = CLEANED_DATA_FILE
    if not os.path.exists(data_file):
        return {}

    df = pd.read_csv(data_file)

    # 位置列
    pos_col = schema.find_column("position", "位置", "code") if schema else None
    if not pos_col or pos_col not in df.columns:
        return {}

    # 目标列
    target_col = schema.target_column if (schema and schema.target_column) else None
    if not target_col or target_col not in df.columns:
        return {}

    # Is_Pass
    pass_set, fail_set = _target_value_sets(schema, df[target_col])
    if not pass_set and not fail_set:
        return {}

    def _is_pass(val):
        try:
            return 1 if str(val) in pass_set else 0
        except Exception:
            return 0

    df['_Is_Pass'] = df[target_col].apply(_is_pass)

    chart_data = {}

    # ── 位置良率排行 ───────────────────────────────────────
    pos_stats = df.groupby(pos_col)['_Is_Pass'].agg(['mean', 'count']).reset_index()
    pos_stats.columns = [pos_col, 'Yield_Rate', 'Count']
    pos_stats = pos_stats.sort_values('Yield_Rate', ascending=True)

    yield_data = [{
        'position': str(row[pos_col]),
        'yield_rate': float(row['Yield_Rate']),
        'count': int(row['Count']),
    } for _, row in pos_stats.iterrows()]

    chart_data['position_yield_ranking'] = {
        'type': 'bar_chart',
        'title': '位置良率排行',
        'data': yield_data,
        'worst': yield_data[0] if yield_data else None,
        'best': yield_data[-1] if yield_data else None,
    }

    # ── 位置缺陷详情 ───────────────────────────────────────
    status_map = schema.target_mapping if (schema and schema.target_mapping) else {}

    failure_data = []
    for pos in df[pos_col].dropna().unique():
        pos_df = df[df[pos_col] == pos]
        counts = {}
        for val in pos_df[target_col].dropna():
            key = str(val)
            display = status_map.get(key, key)
            counts[display] = counts.get(display, 0) + 1
        failure_data.append({
            'position': str(pos),
            'failure_distribution': counts,
        })

    chart_data['position_failure_detail'] = {
        'type': 'stacked_bar',
        'title': '位置缺陷详情',
        'data': failure_data,
    }

    # ── 位置特征分布（前2个数值特征） ──────────────────────
    num_feats = schema.get_numeric_features() if schema else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    position_feats = [c for c in num_feats if c != target_col][:2]

    for feat_col in position_feats:
        if feat_col not in df.columns:
            continue
        col_display = _display_name(schema, feat_col)
        col_unit = _unit_of(schema, feat_col)
        key = f'position_feature_{feat_col}'

        pos_data = []
        for pos in df[pos_col].dropna().unique():
            pos_df = df[df[pos_col] == pos]
            vals = pos_df[feat_col].dropna()
            if len(vals) > 0:
                pos_data.append({
                    'position': str(pos),
                    'mean': float(vals.mean()),
                    'median': float(vals.median()),
                    'std': float(vals.std()),
                    'count': len(vals),
                })

        chart_data[key] = {
            'type': 'boxplot',
            'title': f'各位置{col_display}分布',
            'data': pos_data,
            'unit': col_unit,
        }

    return chart_data


def format_chart_data_for_llm(chart_data: Dict[str, Any],
                               schema: Optional[DataSchema] = None) -> str:
    """将图表数据格式化为LLM可读的文本（全行业通用）。"""
    pass_label = schema.pass_label if (schema and schema.pass_label) else '合格'
    fail_label = schema.fail_label if (schema and schema.fail_label) else '不合格'

    text_parts = []

    # 目标值分布
    sd = chart_data.get('status_distribution')
    if sd:
        text_parts.append(f"【目标值分布】{sd['summary']}")

    # 相关性分析
    cm = chart_data.get('correlation_matrix')
    if cm:
        text_parts.append("【参数相关性分析】")
        for finding in cm.get('key_findings', [])[:5]:
            text_parts.append(f"  - {finding}")

    # 合格/不合格特征对比（动态遍历所有 pass_fail_ 前缀的 key）
    for key, data in chart_data.items():
        if not key.startswith('pass_fail_'):
            continue
        p = data['data'].get(pass_label, {})
        f = data['data'].get(fail_label, {})
        unit_str = f" {data.get('unit', '')}" if data.get('unit') else ""
        text_parts.append(
            f"【{pass_label}/{fail_label}特征对比】"
            f"{pass_label}中位数{p.get('median', 0):.2f}{unit_str}，"
            f"{fail_label}中位数{f.get('median', 0):.2f}{unit_str}"
        )

    # 位置效应
    py = chart_data.get('position_yield')
    if py and py.get('worst'):
        text_parts.append(
            f"【位置效应】最低位置{py['worst']['position']}"
            f"良率{py['worst']['yield_rate']*100:.1f}%，"
            f"最高位置{py['best']['position']}良率{py['best']['yield_rate']*100:.1f}%"
        )

    # 时间漂移
    td = chart_data.get('time_drift')
    if td:
        d = td['data']
        unit_str = f"({d['unit']})" if d.get('unit') else ""
        text_parts.append(
            f"【趋势漂移】早{d['feature']}均值{d['early_mean']:.2f}{unit_str}，"
            f"后期均值{d['late_mean']:.2f}{unit_str}，"
            f"漂移量{d['drift']:.2f}{unit_str}"
        )

    # 特征重要性（ML）
    fi = chart_data.get('feature_importance')
    if fi:
        text_parts.append("【特征重要性排名】")
        for i, feat in enumerate(fi.get('top3', []), 1):
            text_parts.append(f"  {i}. {feat['feature']}: 综合得分{feat['total_score']:.4f}")

    # 位置分析
    pyr = chart_data.get('position_yield_ranking')
    if pyr and pyr.get('worst'):
        text_parts.append(
            f"【位置分析】最差位置{pyr['worst']['position']}"
            f"良率{pyr['worst']['yield_rate']*100:.1f}%，"
            f"最佳位置{pyr['best']['position']}良率{pyr['best']['yield_rate']*100:.1f}%"
        )

    return "\n".join(text_parts)


def extract_all_chart_data(base_dir: str,
                           schema: Optional[DataSchema] = None) -> Tuple[str, Dict]:
    """提取所有图表数据并格式化为LLM输入（全行业通用）。"""
    eda_data = extract_eda_chart_data(base_dir, schema=schema)
    ml_data = extract_ml_chart_data(base_dir)
    position_data = extract_position_chart_data(base_dir, schema=schema)

    all_data = {
        'eda': eda_data,
        'ml': ml_data,
        'position': position_data,
    }

    merged = {**eda_data, **ml_data, **position_data}
    formatted_text = format_chart_data_for_llm(merged, schema=schema)

    return formatted_text, all_data


if __name__ == "__main__":
    from project_paths import ROOT_DIR
    from domain_adapter import load_or_build_schema

    base_dir = ROOT_DIR
    schema = load_or_build_schema()
    text, data = extract_all_chart_data(base_dir, schema=schema)
    print("=" * 60)
    print("提取的图表数据（LLM可读格式）")
    print("=" * 60)
    print(text)
    print("\n" + "=" * 60)
    print("原始数据结构（JSON格式）")
    print("=" * 60)
    import json
    print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
