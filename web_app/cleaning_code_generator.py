"""
cleaning_code_generator.py — Schema驱动的通用数据清洗模块

架构（两层）：
  Layer 1: Schema驱动通用清洗 — 基于DataSchema对每列按角色做标准处理
  Layer 2: LLM增强清洗 — 对复杂文本列，调用Code模型生成定制清洗代码
"""

import pandas as pd
import numpy as np
import re
import json
import signal
from typing import Dict, List, Optional, Tuple, Any
from openai import OpenAI

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
            # 保持为 object 类型（st.dataframe 的 Glide Data Grid 不支持 category dtype）
            type_stats['converted'].append({
                'column': col_name,
                'from': str(df[col_name].dtype),
                'to': 'string',
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

    # Layer 2 统计
    l2 = stats.get('layer2', {})
    l2_status = l2.get('status', 'disabled')
    if l2_status == 'completed':
        lines.append("")
        lines.append(f"🤖 LLM增强清洗完成")
        detected = l2.get('detected_columns', [])
        if detected:
            lines.append(f"   处理了 {len(detected)} 个复杂列: {', '.join(detected)}")
        new_cols = l2.get('new_columns', [])
        if new_cols:
            lines.append(f"   新增 {len(new_cols)} 个衍生列")
    elif l2_status == 'no_complex_columns':
        lines.append("")
        lines.append("ℹ️ LLM增强清洗: 未检测到需要处理的复杂列")
    elif l2_status == 'error':
        lines.append("")
        lines.append(f"⚠️ LLM增强清洗出错: {l2.get('error', '未知错误')}")
    
    return "\n".join(lines) if lines else "清洗完成，未做特殊处理。"


# ── Layer 2: LLM增强清洗 ────────────────────────────────────────

def detect_complex_columns(df: pd.DataFrame, schema: DataSchema) -> List[Dict[str, Any]]:
    """
    检测哪些列需要LLM增强清洗。
    
    返回列表，每个元素包含:
      - 'column': 列名
      - 'reason': 检测原因描述
      - 'samples': 前5个样本值
    """
    complex_cols = []

    for col_schema in schema.columns:
        if col_schema.role == 'ignore':
            continue
        col_name = col_schema.raw_name
        if col_name not in df.columns:
            continue

        # 只处理文本/ID类列
        if col_schema.dtype not in ('text', 'categorical'):
            continue

        sample_vals = df[col_name].dropna().astype(str).head(10).tolist()
        if not sample_vals:
            continue

        # 检查1: 中文+数字混合（如"真空度：-866 T:19.8℃"）
        cn_num_pattern = sum(
            1 for v in sample_vals
            if re.search(r'[\u4e00-\u9fff]', v) and re.search(r'-?\d+\.?\d*', v)
        )
        if cn_num_pattern >= len(sample_vals) * 0.3:
            complex_cols.append({
                'column': col_name,
                'reason': '文本包含中文与数值混合，可能内嵌结构化数据',
                'samples': sample_vals[:5]
            })
            continue

        # 检查2: 复合ID格式（字母+数字+分隔符）
        id_pattern = sum(
            1 for v in sample_vals
            if re.search(r'[A-Za-z]+\d+[-_#]\d+', v)
        )
        if id_pattern >= len(sample_vals) * 0.5:
            complex_cols.append({
                'column': col_name,
                'reason': '复合ID格式，可能包含可解析的子字段',
                'samples': sample_vals[:5]
            })
            continue

        # 检查3: 数值+单位混合（如"12.5μm"）
        num_unit_pattern = sum(
            1 for v in sample_vals
            if re.search(r'\d+\.?\d*[a-zA-Zμ°%]+', v)
        )
        if num_unit_pattern >= len(sample_vals) * 0.3:
            complex_cols.append({
                'column': col_name,
                'reason': '数值包含单位标记，需剥离',
                'samples': sample_vals[:5]
            })
            continue

    return complex_cols


# ── LLM Prompt 模板 ─────────────────────────────────────────────

_LLM_CLEANING_PROMPT = """你是数据清洗专家。分析以下数据集中需要清洗的列，生成Python清洗代码。

【待处理的列】
{column_info}

【DataFrame列名】
{all_columns}

【要求】
请生成一个名为 `enhanced_clean(df) -> pd.DataFrame` 的Python函数：
1. 对每一列做合理的清洗处理（类型转换、值提取、格式统一等）
2. 如果有内嵌数据（如文本中的数值），提取为新列
3. 如果有复合ID格式，解析为多个有意义的新列
4. 删除原始列中已无用或提取过的部分
5. **只输出纯Python代码，不要markdown包裹和额外说明**
6. 代码中只使用 pandas、numpy、re

示例输出格式（直接是代码，不含markdown）：
def enhanced_clean(df):
    import pandas as pd, numpy as np, re
    df = df.copy()
    # 自定义清洗逻辑...
    return df
"""


def generate_cleaning_code(df: pd.DataFrame, schema: DataSchema,
                           complex_columns: List[Dict[str, Any]],
                           client, model: str, coder_model: str = None) -> str:
    """
    调用Code模型生成增强清洗代码。

    参数:
        df: 原始DataFrame
        schema: DataSchema
        complex_columns: detect_complex_columns的返回值
        client: OpenAI客户端
        model: 文本模型
        coder_model: 代码模型（默认使用model）

    返回:
        生成的Python代码字符串
    """
    # 构建列信息
    col_info_lines = []
    for cc in complex_columns:
        col_name = cc['column']
        samples = cc['samples']
        reason = cc['reason']
        col_info_lines.append(
            f"列名: {col_name}\n"
            f"  问题: {reason}\n"
            f"  样本: {samples}\n"
        )

    all_columns = list(df.columns)

    prompt = _LLM_CLEANING_PROMPT.format(
        column_info="\n".join(col_info_lines),
        all_columns=str(all_columns)
    )

    actual_coder_model = coder_model or model

    messages = [
        {"role": "system", "content": "你是一个严谨的数据清洗Python开发者。输出纯Python代码，不含markdown。"},
        {"role": "user", "content": prompt}
    ]

    resp = client.chat.completions.create(
        model=actual_coder_model,
        messages=messages,
        temperature=0.1,
    )
    content = resp.choices[0].message.content
    if not content:
        return ""

    # 提取代码（移除可能的markdown包裹）
    code_match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
    else:
        # 尝试直接找到 def enhanced_clean
        func_match = re.search(r'(def enhanced_clean.*)', content, re.DOTALL)
        if func_match:
            content = func_match.group(1).strip()

    return content


class _TimeoutError(Exception):
    """代码执行超时"""
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("代码执行超时")


def execute_llm_code_safely(code_str: str, df: pd.DataFrame, timeout: int = 30) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    安全执行LLM生成的清洗代码。

    参数:
        code_str: Python代码字符串
        df: 待处理的DataFrame
        timeout: 超时秒数

    返回:
        (结果DataFrame, 错误信息)
    """
    # 安全的 builtins（允许 import）
    safe_builtins = {
        'True': True, 'False': False, 'None': None,
        'int': int, 'float': float, 'str': str, 'bool': bool,
        'len': len, 'range': range, 'list': list, 'dict': dict, 'tuple': tuple,
        'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
        'isinstance': isinstance, 'type': type, 'print': print,
        'ValueError': ValueError, 'TypeError': TypeError,
        'KeyError': KeyError, 'Exception': Exception,
        'object': object, 'enumerate': enumerate, 'zip': zip,
        'map': map, 'filter': filter, 'any': any, 'all': all,
        'sorted': sorted, 'reversed': reversed, 'iter': iter, 'next': next,
        '__import__': __builtins__['__import__'],
    }

    limited_globals = {
        '__builtins__': safe_builtins,
        'pd': pd,
        'np': np,
        're': re,
        'df': df.copy(),
    }

    error_msg = None

    # 设置超时（仅Unix支持signal.alarm）
    try:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        exec(code_str, limited_globals)

        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # 取消定时器

        # 获取结果
        result = limited_globals.get('enhanced_clean', None)
        if result:
            result_df = result(limited_globals['df'])
        else:
            result_df = limited_globals.get('df', df.copy())

    except _TimeoutError as e:
        error_msg = str(e)
        result_df = df.copy()
    except Exception as e:
        error_msg = f"代码执行错误: {e}"
        result_df = df.copy()
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # 确保定时器被取消

    return result_df, error_msg


# ── 更新清洗入口 ────────────────────────────────────────────────

def run_cleaning_pipeline(df: pd.DataFrame, schema: DataSchema,
                          enable_llm_enhanced: bool = False,
                          client=None, model: str = None,
                          coder_model: str = None,
                          target_column_override: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    完整清洗入口：先做Layer 1，可选做Layer 2 LLM增强。

    参数:
        df: 原始DataFrame
        schema: DataSchema
        enable_llm_enhanced: 是否启用LLM增强清洗
        client: OpenAI客户端（LLM增强需要）
        model: 文本模型名
        coder_model: 代码模型名（默认使用model）
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

    # Layer 2: LLM增强清洗
    if enable_llm_enhanced and client:
        # 检测复杂列
        complex_cols = detect_complex_columns(cleaned_df, schema)
        if complex_cols:
            try:
                full_stats['layer2'] = {
                    'status': 'generating',
                    'detected_columns': [c['column'] for c in complex_cols]
                }

                code_str = generate_cleaning_code(
                    cleaned_df, schema, complex_cols,
                    client, model, coder_model=coder_model
                )

                full_stats['layer2']['code'] = code_str

                if code_str:
                    result_df, error = execute_llm_code_safely(code_str, cleaned_df)
                    if error:
                        full_stats['layer2']['status'] = 'error'
                        full_stats['layer2']['error'] = error
                    else:
                        cleaned_df = result_df
                        full_stats['layer2']['status'] = 'completed'
                        full_stats['layer2']['new_columns'] = [
                            c for c in result_df.columns if c not in cleaned_df.columns[:1]
                        ]
                else:
                    full_stats['layer2']['status'] = 'skipped'
                    full_stats['layer2']['reason'] = '代码生成为空'

            except Exception as e:
                full_stats['layer2'] = {'status': 'error', 'error': str(e)}
        else:
            full_stats['layer2'] = {'status': 'no_complex_columns'}
    else:
        full_stats['layer2'] = {'status': 'disabled'}

    return cleaned_df, full_stats
