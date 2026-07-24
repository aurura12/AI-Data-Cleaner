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
                # 对于混合格式日期（YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD），
                # pandas 2.0+ 需要显式指定 format='mixed' 才能正确解析
                try:
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce', format='mixed')
                except TypeError:
                    # 旧版 pandas 不支持 format='mixed' 参数
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
                # 尝试 int/float/str 三种类型，覆盖 LLM 可能返回的各种格式
                try:
                    mapping[int(v)] = 1
                except (ValueError, TypeError):
                    pass
                try:
                    mapping[float(v)] = 1
                except (ValueError, TypeError):
                    pass
                mapping[str(v)] = 1
        if schema.fail_values:
            for v in schema.fail_values:
                try:
                    mapping[int(v)] = 0
                except (ValueError, TypeError):
                    pass
                try:
                    mapping[float(v)] = 0
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
    _common_target_keywords = ['结果', '状态', '良率', '等级', 'class', 'label', 'target',
                               'Pass', 'Fail', 'defect', 'quality', 'grade', 'outcome']
    target_keywords = _common_target_keywords
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
            lines.append(f"📌 良品(Pass): {step.get('after_pass_count', 0)} | "
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
        lines.append(f"🤖 LLM增强清洗完成（基于业务理解生成的定制清洗代码）")
        detected = l2.get('detected_issues', [])
        if detected:
            lines.append(f"📌 检测到 {len(detected)} 个质量问题:")
            for d in detected[:8]:
                col = d.get('column', '') or '全局'
                lines.append(f"    - [{d.get('severity', 'medium')}] {col}: {d.get('reason', '')}")
        if l2.get('code'):
            lines.append("📌 已生成定制清洗代码（可展开查看）")
        new_cols = l2.get('new_columns', [])
        if new_cols:
            lines.append(f"📌 新增 {len(new_cols)} 个衍生列")
    elif l2_status == 'no_complex_columns' or l2_status == 'no_issues_detected':
        lines.append("")
        lines.append("ℹ️ LLM增强清洗: 未检测到需要处理的质量问题")
    elif l2_status == 'error':
        lines.append("")
        lines.append(f"⚠️ LLM增强清洗出错: {l2.get('error', '未知错误')}")
    
    return "\n".join(lines) if lines else "清洗完成，未做特殊处理。"


# ── Layer 2: LLM增强清洗 ────────────────────────────────────────

def detect_complex_columns(df: pd.DataFrame, schema: DataSchema) -> List[Dict[str, Any]]:
    """
    检测哪些列需要LLM增强清洗。
    
    优先使用 Schema 中 LLM 已经识别出的 quality_issues；
    同时用规则补充检测三类复杂文本列作为兜底。

    返回列表，每个元素包含:
      - 'column': 列名（全局问题可填空字符串）
      - 'reason': 检测原因描述
      - 'severity': 'high'|'medium'|'low'
      - 'samples': 前3个样本值（可选）
    """
    all_issues = []

    # 优先使用 Schema 中 LLM 识别出的质量问题
    if schema.quality_issues:
        for issue in schema.quality_issues:
            all_issues.append({
                'column': issue.get('column', '') or '',
                'reason': issue.get('issue', ''),
                'severity': issue.get('severity', 'medium'),
                'samples': [],
            })

    # 规则兜底：检测三类复杂文本列
    for col_schema in schema.columns:
        if col_schema.role == 'ignore':
            continue
        col_name = col_schema.raw_name
        if col_name not in df.columns:
            continue
        if col_schema.dtype not in ('text', 'categorical'):
            continue

        sample_vals = df[col_name].dropna().astype(str).head(10).tolist()
        if not sample_vals:
            continue

        # 检查1: 中文+数字混合
        cn_num_pattern = sum(
            1 for v in sample_vals
            if re.search(r'[\u4e00-\u9fff]', v) and re.search(r'-?\d+\.?\d*', v)
        )
        if cn_num_pattern >= len(sample_vals) * 0.3:
            # 避免重复添加（如果 schema 已经列出了）
            if not any(i['column'] == col_name for i in all_issues):
                all_issues.append({
                    'column': col_name,
                    'reason': '文本包含中文与数值混合，可能内嵌结构化数据',
                    'severity': 'medium',
                    'samples': sample_vals[:3]
                })
            continue

        # 检查2: 复合ID格式
        id_pattern = sum(
            1 for v in sample_vals
            if re.search(r'[A-Za-z]+\d+[-_#]\d+', v)
        )
        if id_pattern >= len(sample_vals) * 0.5:
            if not any(i['column'] == col_name for i in all_issues):
                all_issues.append({
                    'column': col_name,
                    'reason': '复合ID格式，可能包含可解析的子字段',
                    'severity': 'medium',
                    'samples': sample_vals[:3]
                })
            continue

        # 检查3: 数值+单位混合
        num_unit_pattern = sum(
            1 for v in sample_vals
            if re.search(r'\d+\.?\d*[a-zA-Zμ°%]+', v)
        )
        if num_unit_pattern >= len(sample_vals) * 0.3:
            if not any(i['column'] == col_name for i in all_issues):
                all_issues.append({
                    'column': col_name,
                    'reason': '数值包含单位标记，需剥离',
                    'severity': 'medium',
                    'samples': sample_vals[:3]
                })
            continue

    return all_issues


# ── LLM Prompt 模板 ─────────────────────────────────────────────

_LLM_CLEANING_PROMPT = """你是数据清洗专家。基于以下**业务理解**生成定制化的数据清洗代码。

【业务领域】
{business_domain}

【业务背景】
{business_description}

【已识别的数据质量问题】
{quality_issues_text}

【清洗建议】
{cleaning_recommendations_text}

【列定义】
{column_definitions}

【DataFrame列名】
{all_columns}

【要求：必须严格实现以下全部6个清洗步骤，不能省略任何一个】

请生成一个名为 `enhanced_clean(df) -> pd.DataFrame` 的Python函数：

**步骤1：日期格式统一**
- 对 role=datetime 的列，用 pd.to_datetime(df[col], errors='coerce', format='mixed') 统一格式
- 如 2024/01/15、2024-01-16、2024.01.17 等混合格式都要正确解析

**步骤2：缺失值处理**
- 根据每列的 missing_strategy 处理：
  - median_fill → df[col].fillna(df[col].median())
  - mean_fill → df[col].fillna(df[col].mean())
  - mode_fill → df[col].fillna(df[col].mode()[0])
  - drop → 删除该列有缺失的行
  - keep → 保留缺失值不动

**步骤3：异常值检测与处理**
- 对 outlier_check=true 的数值列，用 IQR 方法检测异常值：
  Q1, Q3 = df[col].quantile([0.25, 0.75])
  IQR = Q3 - Q1
  lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
- 将异常值替换为 NaN，然后用该列的中位数填充
- 同时创建标记列 {col}_outlier（0=正常, 1=异常）

**步骤4：检测并删除业务重复行**
- 如果有 id_column，排除 ID 列后检查其余列是否完全重复
- 对重复组保留第一次出现的行，删除其余

**步骤5：目标列映射**
- 如果有 target_column 且设置了 pass_values/fail_values：
  - pass_values 中的值映射为 1
  - fail_values 中的值映射为 0
  - 不在映射表中的值置为 NaN 并删除该行

**步骤6：文本标准化**
- 对所有 object/string 类型的列：df[col] = df[col].astype(str).str.strip()

**重要约束：**
1. 所有操作都用 `df[col] = new_value` 方式赋值，不要用 inplace=True
2. 不要使用已废弃的参数如 `infer_datetime_format`
3. 必须处理列不存在或全部缺失的边界情况
4. **只输出纯Python代码，不要markdown包裹和额外说明**
5. 代码中只使用 pandas、numpy、re

参考实现框架（请按此模式编写完整函数）：
def enhanced_clean(df):
    import pandas as pd, numpy as np, re
    df = df.copy()
    # 步骤1: 日期格式统一
    # ...
    # 步骤2: 缺失值处理
    # ...
    # 步骤3: 异常值检测与处理
    # ...
    # 步骤4: 重复行删除
    # ...
    # 步骤5: 目标列映射
    # ...
    # 步骤6: 文本标准化
    # ...
    return df
"""


def _build_column_definitions_text(schema: DataSchema) -> str:
    """将列定义格式化为文本"""
    lines = []
    for col in schema.columns:
        parts = [f"  - {col.raw_name}"]
        if col.display_name:
            parts.append(f"（{col.display_name}）")
        parts.append(f"role={col.role}, dtype={col.dtype}")
        if col.physical_unit:
            parts.append(f"unit={col.physical_unit}")
        if col.reasonable_range:
            parts.append(f"合理范围=[{col.reasonable_range.get('min', '?')}, {col.reasonable_range.get('max', '?')}]")
        if col.missing_strategy:
            parts.append(f"缺失策略={col.missing_strategy}")
        if col.outlier_check:
            parts.append("需异常检测")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def generate_cleaning_code(df: pd.DataFrame, schema: DataSchema,
                           complex_columns: List[Dict[str, Any]],
                           client, model: str, coder_model: str = None) -> str:
    """
    调用Code模型生成增强清洗代码。
    传入完整的 Schema 业务理解信息，让 Code 模型生成针对性清洗代码。

    参数:
        df: 原始DataFrame
        schema: DataSchema（含LLM业务理解）
        complex_columns: detect_complex_columns的返回值
        client: OpenAI客户端
        model: 文本模型
        coder_model: 代码模型（默认使用model）

    返回:
        生成的Python代码字符串
    """
    all_columns = list(df.columns)

    # 格式化质量问题和清洗建议
    if complex_columns:
        quality_lines = []
        for cc in complex_columns:
            col = cc.get('column', '') or '全局'
            quality_lines.append(f"  - [{cc.get('severity', 'medium')}] {col}: {cc.get('reason', '')}")
        quality_text = "\n".join(quality_lines)
    elif schema.quality_issues:
        quality_text = "\n".join(
            f"  - [{q.get('severity', 'medium')}] {q.get('column', '全局') or '全局'}: {q.get('issue', '')}"
            for q in schema.quality_issues
        )
    else:
        quality_text = "  （未检测到明确的质量问题）"

    # 清洗建议
    if schema.cleaning_recommendations:
        rec_text = "\n".join(f"  - {r}" for r in schema.cleaning_recommendations)
    else:
        rec_text = "  （无特定的清洗建议）"

    # 列定义
    col_defs = _build_column_definitions_text(schema)

    prompt = _LLM_CLEANING_PROMPT.format(
        business_domain=schema.business_domain or '未知',
        business_description=schema.business_description or '未提供',
        quality_issues_text=quality_text,
        cleaning_recommendations_text=rec_text,
        column_definitions=col_defs,
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
        # 过滤 schema：只保留 Layer 1 后仍然存在的列
        schema_l2 = _filter_schema_for_layer2(schema, cleaned_df)

        # 检测需要处理的问题（combine schema LLM分析结果 + 规则兜底）
        all_issues = detect_complex_columns(cleaned_df, schema_l2)
        
        # 只要检测到任何问题就触发LLM增强清洗
        if all_issues:
            try:
                full_stats['layer2'] = {
                    'status': 'generating',
                    'detected_issues': [{'column': i['column'], 'reason': i['reason'], 
                                         'severity': i.get('severity', 'medium')} 
                                        for i in all_issues]
                }

                code_str = generate_cleaning_code(
                    cleaned_df, schema_l2, all_issues,
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
            full_stats['layer2'] = {'status': 'no_issues_detected'}
    else:
        full_stats['layer2'] = {'status': 'disabled'}

    return cleaned_df, full_stats


def _filter_schema_for_layer2(schema: DataSchema, df: pd.DataFrame) -> DataSchema:
    """过滤 schema，只保留 DataFrame 中仍然存在的列（Layer 1 可能丢弃了 ignore 列）"""
    existing_cols = set(df.columns)
    filtered_columns = [c for c in schema.columns if c.raw_name in existing_cols]
    
    # 创建新 schema，只包含现有列
    from dataclasses import replace
    new_schema = DataSchema(
        id_column=schema.id_column if schema.id_column in existing_cols else None,
        target_column=schema.target_column if schema.target_column in existing_cols else None,
        target_mapping=schema.target_mapping,
        pass_values=schema.pass_values,
        fail_values=schema.fail_values,
        pass_label=schema.pass_label,
        fail_label=schema.fail_label,
        columns=filtered_columns,
        target_type=schema.target_type,
        raw_data_shape=schema.raw_data_shape,
        business_domain=schema.business_domain,
        business_description=schema.business_description,
        quality_issues=[q for q in schema.quality_issues 
                       if not q.get('column') or q['column'] in existing_cols],
        cleaning_recommendations=schema.cleaning_recommendations,
    )
    return new_schema
