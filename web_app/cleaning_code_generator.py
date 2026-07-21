"""
cleaning_code_generator.py — Schema驱动的通用数据清洗模块

架构（两层）：
  Layer 1: Schema驱动通用清洗 — 基于DataSchema对每列按角色做标准处理
  Layer 2: LLM增强清洗 — 对复杂文本列，调用Code模型生成定制清洗代码

当前实现 Layer 1，Layer 2 预留接口。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from schema_analyzer import DataSchema


# ── Layer 1: Schema驱动通用清洗 ────────────────────────────────────

def layer1_generic_clean(df: pd.DataFrame, schema: DataSchema,
                         target_column_override: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Schema驱动通用清洗：基于DataSchema对每列按角色做标准处理。

    参数:
        df: 原始DataFrame
        schema: DataSchema（来自schema_analyzer）
        target_column_override: 可选，手动指定目标列名（schema未识别到时使用）

    返回:
        (cleaned_df, stats_dict)
    """
    stats = {'steps': []}
    df = df.copy()

    # ── Step 1: 丢弃 ignore 列 ──
    ignore_cols = [c.raw_name for c in schema.columns if c.role == 'ignore']
    if ignore_cols:
        cols_before = list(df.columns)
        df = df.drop(columns=[c for c in ignore_cols if c in df.columns])
        stats['steps'].append({
            'step': 'drop_ignore',
            'dropped': [c for c in ignore_cols if c in cols_before],
            'cols_removed': len([c for c in ignore_cols if c in cols_before])
        })

    # ── Step 2: 丢弃全空行 ──
    before_dropna = len(df)
    df = df.dropna(how='all')
    stats['steps'].append({
        'step': 'drop_all_empty',
        'rows_before': before_dropna,
        'rows_after': len(df),
        'rows_removed': before_dropna - len(df)
    })

    # ── Step 3: 去重 ──
    before_dedup = len(df)
    if schema.id_column and schema.id_column in df.columns:
        # 基于ID列去重
        id_col = schema.id_column
        before_id_count = df[id_col].nunique()
        df = df.drop_duplicates(subset=[id_col], keep='first')
        stats['steps'].append({
            'step': 'dedup_by_id',
            'id_column': id_col,
            'rows_before': before_dedup,
            'rows_after': len(df),
            'rows_removed': before_dedup - len(df),
            'unique_ids': before_id_count
        })
    else:
        # 全行去重
        df = df.drop_duplicates()
        stats['steps'].append({
            'step': 'dedup_full',
            'rows_before': before_dedup,
            'rows_after': len(df),
            'rows_removed': before_dedup - len(df)
        })

    # ── Step 4: 类型转换 ──
    type_stats = {'converted': [], 'errors': []}
    for col_schema in schema.columns:
        if col_schema.role == 'ignore':
            continue
        col_name = col_schema.raw_name
        if col_name not in df.columns:
            continue

        if col_schema.dtype == 'numeric':
            before_dtype = str(df[col_name].dtype)
            before_nan = int(df[col_name].isna().sum())
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            after_nan = int(df[col_name].isna().sum())
            type_stats['converted'].append({
                'column': col_name,
                'from': before_dtype,
                'to': 'numeric',
                'new_nan': after_nan - before_nan
            })

        elif col_schema.dtype == 'datetime':
            try:
                before_nan = int(df[col_name].isna().sum())
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                after_nan = int(df[col_name].isna().sum())
                type_stats['converted'].append({
                    'column': col_name,
                    'from': 'object',
                    'to': 'datetime',
                    'new_nan': after_nan - before_nan
                })
            except Exception as e:
                type_stats['errors'].append({'column': col_name, 'error': str(e)})

        elif col_schema.dtype == 'categorical':
            before_unique = df[col_name].nunique()
            df[col_name] = df[col_name].astype('category')
            type_stats['converted'].append({
                'column': col_name,
                'from': 'object',
                'to': 'category',
                'unique_values': before_unique
            })

    stats['steps'].append({
        'step': 'type_conversion',
        'details': type_stats
    })

    # ── Step 5: 目标列映射 ──
    # 优先使用schema识别到的目标列，其次使用手动指定的
    target_col = schema.target_column or target_column_override
    if target_col and target_col in df.columns:
        before_mapping = df[target_col].value_counts().to_dict()
        
        # 创建映射: pass_values→1, fail_values→0
        mapping = {}
        if schema.pass_values:
            for v in schema.pass_values:
                # 尝试数值和字符串两种格式
                try:
                    mapping[int(v)] = 1
                except (ValueError, TypeError):
                    pass
                mapping[str(v)] = 1
        if schema.fail_values:
            for v in schema.fail_values:
                try:
                    mapping[int(v)] = 0
                except (ValueError, TypeError):
                    pass
                mapping[str(v)] = 0

        if mapping:
            df[target_col] = df[target_col].map(mapping)
            # 映射后不在映射表中的值 → 视为 NaN
            # 但保留原始值以便用户检查
            unmapped_count = int(df[target_col].isna().sum())
            df = df.dropna(subset=[target_col])  # 删除目标列为空的行
            
            stats['steps'].append({
                'step': 'target_mapping',
                'column': target_col,
                'mapping': {str(k): str(v) for k, v in mapping.items()},
                'before_distribution': {str(k): int(v) for k, v in before_mapping.items()},
                'after_pass_count': int((df[target_col] == 1).sum()),
                'after_fail_count': int((df[target_col] == 0).sum()),
                'after_total': len(df),
                'unmapped_removed': unmapped_count
            })

    # ── Step 6: 文本列标准化 ──
    text_normalized = []
    for col_schema in schema.columns:
        if col_schema.dtype == 'categorical' or col_schema.dtype == 'text':
            col_name = col_schema.raw_name
            if col_name in df.columns and df[col_name].dtype == 'object':
                before = df[col_name].fillna('').astype(str).str.len().sum()
                df[col_name] = df[col_name].astype(str).str.strip().replace('nan', pd.NA).replace('', pd.NA)
                after = df[col_name].fillna('').astype(str).str.len().sum()
                text_normalized.append({
                    'column': col_name,
                    'chars_before': int(before),
                    'chars_after': int(after)
                })

    if text_normalized:
        stats['steps'].append({
            'step': 'text_normalize',
            'details': text_normalized
        })

    return df, stats


# ── 目标列自动检测 ──────────────────────────────────────────────

def _auto_detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    当Schema未识别到目标列时，用启发式规则自动检测。
    检测关键词：压连、结果、状态、良率、class、label、target等
    """
    target_keywords = ['压连', '结果', '状态', '良率', '等级', 'class', 'label', 'target',
                       'Pass', 'Fail', 'defect', 'quality', 'grade', 'outcome']
    for col in df.columns:
        col_str = str(col)
        if any(k in col_str for k in target_keywords):
            if df[col].nunique() < 20:  # 目标列通常取值少
                return col
    # 最后尝试：最后一列（很多数据集的目标列在最后一列）
    if len(df.columns) > 0:
        last_col = df.columns[-1]
        if df[last_col].nunique() < 20:
            return last_col
    return None


# ── 清洗执行入口 ──────────────────────────────────────────────────

def run_cleaning_pipeline(df: pd.DataFrame, schema: DataSchema,
                          enable_llm_enhanced: bool = False,
                          client=None, model: str = None,
                          target_column_override: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    完整清洗入口：先做Layer 1，可选做Layer 2 LLM增强。

    参数:
        df: 原始DataFrame
        schema: DataSchema
        enable_llm_enhanced: 是否启用LLM增强清洗
        client: OpenAI客户端（LLM增强需要）
        model: 模型名（LLM增强需要）
        target_column_override: 可选，手动指定目标列名

    返回:
        (cleaned_df, full_stats)
    """
    full_stats = {}

    # 尝试自动检测目标列（如果schema和override都没有）
    if schema.target_column is None and target_column_override is None:
        auto_target = _auto_detect_target_column(df)
        if auto_target:
            target_column_override = auto_target

    # Layer 1: Schema驱动通用清洗
    cleaned_df, l1_stats = layer1_generic_clean(df, schema, target_column_override=target_column_override)
    full_stats['layer1'] = l1_stats

    # Layer 2: LLM增强清洗（预留，当前仅返回占位）
    if enable_llm_enhanced:
        full_stats['layer2'] = {'status': 'skipped', 'reason': 'Not implemented yet'}
    else:
        full_stats['layer2'] = {'status': 'disabled'}

    return cleaned_df, full_stats


# ── 格式化统计信息为文本（用于 UI 展示） ─────────────────────────

def format_cleaning_stats(stats: Dict[str, Any]) -> str:
    """将清洗统计格式化为可读文本"""
    lines = []

    # 兼容两种模式：直接 stats（内有 steps）或 stats['layer1']
    if 'steps' in stats:
        steps = stats['steps']
    elif 'layer1' in stats and 'steps' in stats['layer1']:
        steps = stats['layer1']['steps']
    else:
        steps = []

    for step in steps:
        step_name = step.get('step', '')
        
        if step_name == 'drop_ignore':
            dropped = step.get('dropped', [])
            if dropped:
                lines.append(f"📌 丢弃了 {len(dropped)} 个无关列: {', '.join(dropped[:5])}")
                if len(dropped) > 5:
                    lines[-1] += f" 等{len(dropped)}列"
        
        elif step_name == 'drop_all_empty':
            removed = step.get('rows_removed', 0)
            lines.append(f"📌 剔除了 {removed} 个全空行")
        
        elif step_name == 'dedup_by_id':
            lines.append(f"📌 按 {step.get('id_column')} 去重，移除了 {step.get('rows_removed')} 个重复行")
        
        elif step_name == 'dedup_full':
            lines.append(f"📌 全行去重，移除了 {step.get('rows_removed')} 个重复行")
        
        elif step_name == 'type_conversion':
            details = step.get('details', {})
            converted = details.get('converted', [])
            for c in converted:
                nn = c.get('new_nan', 0)
                nan_note = f"（新增 {nn} 个NaN）" if nn > 0 else ""
                lines.append(f"📌 {c['column']}: {c['from']} → {c['to']}{nan_note}")
        
        elif step_name == 'target_mapping':
            lines.append(f"📌 目标列 {step.get('column')} 映射完成")
            lines.append(f"   良品(Pass): {step.get('after_pass_count', 0)} | "
                         f"不良(Fail): {step.get('after_fail_count', 0)} | "
                         f"总计: {step.get('after_total', 0)}")
        
        elif step_name == 'text_normalize':
            details = step.get('details', [])
            for d in details:
                saved = d.get('chars_before', 0) - d.get('chars_after', 0)
                if saved > 0:
                    lines.append(f"📌 {d['column']}: 标准化文本，剪除了 {saved} 个空白字符")
    
    return "\n".join(lines) if lines else "清洗完成，未做特殊处理。"
