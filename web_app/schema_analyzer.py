"""
schema_analyzer.py — LLM驱动的数据Schema自动探查模块

功能：
  1. 接收用户上传的 DataFrame
  2. 提取数据摘要（列名、类型、统计量、样本值）
  3. 调用 LLM 自动识别每列的语义角色（ID/目标/特征/忽略/时间）
  4. 识别目标列的值含义（哪些是好/坏）
  5. 输出标准化的 DataSchema 对象
  6. 标记低置信度项，供用户确认
"""

import pandas as pd
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Dict, Any, Tuple
from openai import OpenAI

# ── 类型别名 ──────────────────────────────────────────────
ColumnRole = Literal['id', 'target', 'feature', 'ignore', 'datetime']
ColumnDType = Literal['numeric', 'categorical', 'text', 'datetime']


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class ColumnSchema:
    """单列的Schema定义"""
    raw_name: str                       # 原始列名
    role: ColumnRole = 'feature'        # 列角色
    dtype: ColumnDType = 'numeric'      # 数据类型
    semantic_name: str = ''             # 标准化语义名称 (snake_case)
    display_name: str = ''              # 中文/显示名
    physical_unit: Optional[str] = None # 单位 (如 μm / kg / ℃)
    confidence: float = 0.0             # LLM置信度 0-1
    needs_confirmation: bool = True     # 是否需要用户确认
    # LLM 增强字段（由 LLM 从业务角度提供，不依赖规则）
    reasonable_range: Optional[Dict[str, float]] = None  # 合理值范围 e.g. {"min": 30, "max": 42}
    missing_strategy: str = ''          # 推荐缺失值策略 (median_fill / mean_fill / mode_fill / drop / keep)
    outlier_check: bool = False         # 是否需要异常检测
    date_formats: Optional[List[str]] = None  # 检测到的日期格式列表


@dataclass
class DataSchema:
    """整个数据集的Schema"""
    id_column: Optional[str] = None                # ID列原始名
    target_column: Optional[str] = None            # 目标列原始名
    target_mapping: Optional[Dict[str, str]] = None  # 目标值含义映射 e.g. {"-1": "虚焊", "0": "正常"}
    pass_values: Optional[List[str]] = None        # 良品值 (原始字符串)
    fail_values: Optional[List[str]] = None        # 不良值 (原始字符串)
    pass_label: str = '良品'                       # 良品显示名
    fail_label: str = '不良'                       # 不良显示名
    columns: List[ColumnSchema] = field(default_factory=list)
    target_type: str = 'binary'                    # binary / multiclass / regression
    raw_data_shape: Tuple[int, int] = (0, 0)
    # LLM 业务理解（由 LLM 从业务角度输出，让下游清洗/分析理解数据背景）
    business_domain: str = ''                      # 业务领域描述
    business_description: str = ''                 # 业务详细说明
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)  # 检测到的质量问题
    cleaning_recommendations: List[str] = field(default_factory=list)    # LLM 建议的清洗方案

    def get_column_by_role(self, role: str) -> List[ColumnSchema]:
        """按角色查找列"""
        return [c for c in self.columns if c.role == role]

    def get_feature_columns(self, dtype: Optional[str] = None) -> List[str]:
        """获取所有特征列的原始名"""
        cols = [c for c in self.columns if c.role == 'feature']
        if dtype:
            cols = [c for c in cols if c.dtype == dtype]
        return [c.raw_name for c in cols]

    def get_id_column(self) -> Optional[str]:
        return self.id_column

    def get_target_column(self) -> Optional[str]:
        return self.target_column

    def to_dict(self) -> dict:
        return {
            'id_column': self.id_column,
            'target_column': self.target_column,
            'target_mapping': self.target_mapping,
            'pass_values': self.pass_values,
            'fail_values': self.fail_values,
            'pass_label': self.pass_label,
            'fail_label': self.fail_label,
            'target_type': self.target_type,
            'raw_data_shape': list(self.raw_data_shape),
            'columns': [asdict(c) for c in self.columns],
            'business_domain': self.business_domain,
            'business_description': self.business_description,
            'quality_issues': self.quality_issues,
            'cleaning_recommendations': self.cleaning_recommendations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DataSchema':
        columns = [ColumnSchema(**c) for c in data.get('columns', [])]
        return cls(
            id_column=data.get('id_column'),
            target_column=data.get('target_column'),
            target_mapping=data.get('target_mapping'),
            pass_values=data.get('pass_values'),
            fail_values=data.get('fail_values'),
            pass_label=data.get('pass_label', '良品'),
            fail_label=data.get('fail_label', '不良'),
            target_type=data.get('target_type', 'binary'),
            raw_data_shape=tuple(data.get('raw_data_shape', (0, 0))),
            columns=columns,
            business_domain=data.get('business_domain', ''),
            business_description=data.get('business_description', ''),
            quality_issues=data.get('quality_issues', []),
            cleaning_recommendations=data.get('cleaning_recommendations', []),
        )

    def get_uncertain_columns(self, threshold: float = 0.7) -> List[ColumnSchema]:
        """返回置信度低于阈值的列"""
        return [c for c in self.columns if c.confidence < threshold or c.needs_confirmation]

    def has_uncertainties(self, threshold: float = 0.7) -> bool:
        return len(self.get_uncertain_columns(threshold)) > 0

    def find_column(self, *keywords: str) -> Optional[str]:
        """
        按关键词语义查找列名（优先匹配semantic_name，其次raw_name）。
        使用单词边界匹配避免误匹配。

        示例: schema.find_column('height') → 'Total_Indium_Height'
               schema.find_column('date') → '倒焊日期'
        """
        if not keywords:
            return None
        keywords_pattern = '|'.join(re.escape(k) for k in keywords)
        word_pattern = re.compile(r'(^|[\s_\-\/])({})([\s_\-\/]|$)'.format(keywords_pattern), re.IGNORECASE)
        
        for col in self.columns:
            search_text = (col.semantic_name + ' ' + col.raw_name)
            if word_pattern.search(search_text):
                return col.raw_name
        
        # 第二轮: 宽松匹配（用于中文等无单词边界的语言）
        keywords_lower = [k.lower() for k in keywords]
        for col in self.columns:
            search_text = (col.semantic_name + ' ' + col.raw_name).lower()
            if any(k in search_text for k in keywords_lower):
                return col.raw_name
        return None

    def get_numeric_features(self) -> List[str]:
        """获取所有数值特征列名"""
        return [c.raw_name for c in self.columns
                if c.role == 'feature' and c.dtype == 'numeric']

    def get_categorical_features(self) -> List[str]:
        """获取所有类别特征列名"""
        return [c.raw_name for c in self.columns
                if c.role == 'feature' and c.dtype == 'categorical']

    def has_column_role(self, role: str) -> bool:
        """检查是否存在某角色的列"""
        return any(c.role == role for c in self.columns)


# ── 数据摘要生成 ──────────────────────────────────────────

def build_data_profile(df: pd.DataFrame, max_columns: int = 50) -> dict:
    """
    构建数据摘要（列统计+样本值），供LLM分析使用。
    返回结构化的 dict。
    max_columns: 最多分析的列数（避免prompt太长）
    """
    profile = {
        'row_count': len(df),
        'column_count': min(len(df.columns), max_columns),
        'columns': []
    }

    for col in df.columns[:max_columns]:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing_rate': round(float(df[col].isna().mean()), 4),
            'unique_count': int(df[col].nunique()),
        }

        # 样本值（前2个非NaN）
        sample_values = df[col].dropna().head(2).tolist()
        col_info['samples'] = [str(v) for v in sample_values]

        # 数值列
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['value_type'] = 'numeric'
            desc = df[col].describe()
            col_info['stats'] = {
                'min': round(float(desc['min']), 2) if 'min' in desc else None,
                'max': round(float(desc['max']), 2) if 'max' in desc else None,
                'mean': round(float(desc['mean']), 2) if 'mean' in desc else None,
                'std': round(float(desc['std']), 2) if 'std' in desc else None,
            }
        # 日期时间列
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info['value_type'] = 'datetime'
        else:
            # 文本/类别列
            unique_vals = df[col].dropna().unique()
            col_info['value_type'] = 'categorical'
            col_info['unique_values'] = [str(v) for v in unique_vals[:8]]
            col_info['unique_ratio'] = round(len(unique_vals) / max(len(df), 1), 4)

        profile['columns'].append(col_info)

    return profile


# ── LLM Prompt 模板 ───────────────────────────────────────

_ROLE_ANALYSIS_PROMPT = """你是跨行业的数据分析专家。分析以下数据集的列信息，从**业务角度**理解每一列的含义、数据质量问题和清洗建议。

【你的任务】
1. 识别这是什么业务领域的数据（如半导体制造、食品发酵、金融风控、物流运输等）
2. 对每一列，识别其在业务中的角色和含义
3. 基于样本数据，判断是否存在数据质量问题
4. 给出针对该数据集的清洗建议

【角色定义】
- **id**: 唯一标识符/流水号。每行都不同
- **target**: 目标变量/结果列。取值种类少，列名常含"结果/状态/良率/等级/grade/quality/defect"等
- **feature**: 特征/输入参数。数值型（温度、压力、高度等）或类别型
- **datetime**: 日期/时间列。取值是日期或时间戳格式
- **ignore**: 需要忽略的列。如备注、注释、空列、或无关列

【输出要求】
请输出严格的JSON格式，不要包含任何markdown标记或额外说明：
{{
  "business_domain": "业务领域描述，如'食品发酵/罐装生产'",
  "business_description": "对数据集的业务含义做简要解释",
  "columns": [
    {{
      "name": "列名",
      "role": "id|target|feature|ignore|datetime",
      "dtype": "numeric|categorical|text|datetime",
      "semantic_name": "标准化英文语义名(蛇形命名)",
      "display_name": "中文显示名",
      "physical_unit": "单位(不确定填null)",
      "reasonable_range": {{"min": 最小值, "max": 最大值}},
      "missing_strategy": "median_fill|mean_fill|mode_fill|drop|keep",
      "outlier_check": true或false,
      "date_formats": ["YYYY/MM/DD", "YYYY-MM-DD"],
      "confidence": 0.95,
      "needs_confirmation": false
    }}
  ],
  "id_column": "ID列名(若无填null)",
  "target_column": "目标列名(若无填null)",
  "target_mapping": {{"值1": "含义1", "值2": "含义2"}},
  "pass_values": ["良品对应的值"],
  "fail_values": ["不良对应的值"],
  "pass_label": "良品的总称(如'合格'/'良品')",
  "fail_label": "不良的总称(如'不合格'/'不良')",
  "target_type": "binary|multiclass|regression",
  "quality_issues": [
    {{"column": "有问题的列名(全局问题填null)", "issue": "问题描述", "severity": "high|medium|low"}}
  ],
  "cleaning_recommendations": [
    "具体的清洗建议1",
    "具体的清洗建议2"
  ]
}}

【注意事项】
1. reasonable_range 只在数值列填写，基于样本值和业务常识判断合理区间（填null表示不适用）
2. missing_strategy 推荐每列的缺失值处理策略：median_fill(中位数填充)、mean_fill(均值填充)、mode_fill(众数填充)、drop(直接删除)、keep(保留)
3. outlier_check 设为true的列后续会做异常值检测
4. quality_issues 列出所有质量问题（日期格式不统一、缺失值、异常值、重复行等）
5. cleaning_recommendations 给出具体的清洗操作建议

【列信息】
{data_profile}
"""

_TARGET_VALUE_PROMPT = """你是工业数据分析专家。以下是一个数据集的目标列（结果列）的取值信息。

【任务】
请分析目标列的每个值的业务含义，判断哪些值代表"良品/正常/合格"，哪些代表"不良/异常/不合格"。

【目标列信息】
- 列名: {column_name}
- 唯一值: {unique_values}
- 前20行样本: {samples}

请输出JSON格式：
{{
  "target_mapping": {{
    "值1": "含义描述1",
    "值2": "含义描述2"
  }},
  "pass_values": ["良品对应的值1", "良品对应的值2"],
  "fail_values": ["不良对应的值"],
  "pass_label": "良品的总称(如'良品'/'合格'/'正常')",
  "fail_label": "不良的总称(如'不良'/'不合格'/'异常')",
  "target_type": "binary|multiclass|regression",
  "confidence": 0.9,
  "explanation": "简要解释判断依据"
}}
不要输出任何markdown标记或额外说明，只输出JSON。
"""


# ── 核心分析函数 ─────────────────────────────────────────

def call_llm_json(client: OpenAI, model: str, messages: list, temperature: float = 0.1) -> dict:
    """
    调用LLM并解析JSON返回。
    如果解析失败，抛出 ValueError。
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("LLM返回内容为空")

    # 提取JSON（兼容各种模型输出格式）
    # 策略1: 尝试直接解析纯JSON
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 策略2: 从markdown代码块中提取
    code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if code_match:
        raw = code_match.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            content = raw

    # 策略3: 用大括号计数法提取最外层JSON（支持嵌套{}）
    brace_count = 0
    start = -1
    for i, ch in enumerate(content):
        if ch == '{':
            if start == -1:
                start = i
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0 and start >= 0:
                content = content[start:i + 1]
                break

    return json.loads(content)


def analyze_schema(df: pd.DataFrame, client: OpenAI,
                   model: str = "qwen-plus") -> DataSchema:
    """
    入口函数：分析DataFrame，返回DataSchema（含LLM业务理解）。
    
    参数:
        df: 待分析的DataFrame
        client: OpenAI客户端
        model: 模型名
        
    返回:
        DataSchema对象（含business_domain, quality_issues, cleaning_recommendations等）
    """
    # 1. 构建数据摘要（增加样本量让LLM更好理解业务）
    profile = build_data_profile(df)

    # 2. 调用LLM进行完整的业务理解分析
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
    messages = [
        {
            "role": "system",
            "content": "你是一个严格的JSON输出助手。你只能输出有效的JSON，不含任何markdown、注释或额外文本。"
        },
        {
            "role": "user",
            "content": _ROLE_ANALYSIS_PROMPT.replace("{data_profile}", profile_json)
        }
    ]

    try:
        result = call_llm_json(client, model, messages)
    except Exception as e:
        print(f"[schema_analyzer] LLM 分析失败，回退到规则: {e}")
        return _fallback_schema(df)

    # 3. 解析结果
    columns_raw = result.get('columns', [])
    columns = []
    for col_info in columns_raw:
        columns.append(ColumnSchema(
            raw_name=col_info.get('name', ''),
            role=col_info.get('role', 'feature'),
            dtype=col_info.get('dtype', 'numeric'),
            semantic_name=col_info.get('semantic_name', ''),
            display_name=col_info.get('display_name', ''),
            physical_unit=col_info.get('physical_unit'),
            confidence=col_info.get('confidence', 0.5),
            needs_confirmation=col_info.get('needs_confirmation', True),
            reasonable_range=col_info.get('reasonable_range'),
            missing_strategy=col_info.get('missing_strategy', ''),
            outlier_check=col_info.get('outlier_check', False),
            date_formats=col_info.get('date_formats'),
        ))

    schema = DataSchema(
        id_column=result.get('id_column'),
        target_column=result.get('target_column'),
        target_mapping=result.get('target_mapping'),
        pass_values=result.get('pass_values'),
        fail_values=result.get('fail_values'),
        pass_label=result.get('pass_label', '良品'),
        fail_label=result.get('fail_label', '不良'),
        target_type=result.get('target_type', 'binary'),
        columns=columns,
        raw_data_shape=(len(df), len(df.columns)),
        business_domain=result.get('business_domain', ''),
        business_description=result.get('business_description', ''),
        quality_issues=result.get('quality_issues', []),
        cleaning_recommendations=result.get('cleaning_recommendations', []),
    )

    # 4. 如果第一轮没有返回目标映射，但识别到了目标列，单独调用第二轮
    if schema.target_column and not schema.target_mapping:
        target_col = schema.target_column
        if target_col in df.columns:
            _try_target_mapping_v2(df, target_col, schema, client, model)

    # 5. 如果没有识别到目标列，尝试规则兜底
    if not schema.target_column:
        for col in df.columns:
            col_str = str(col)
            if any(k in col_str for k in ['等级', 'grade', 'quality', '结果', '状态', 'class', 'label']):
                if df[col].nunique() < 20:
                    schema.target_column = col
                    schema.target_type = 'multiclass' if df[col].nunique() > 2 else 'binary'
                    break

    return schema


def _try_target_mapping_v2(df: pd.DataFrame, target_col: str, schema: DataSchema,
                            client: OpenAI, model: str) -> None:
    """第二轮LLM调用：分析目标列的值含义（仅当第一轮未返回映射时）"""
    unique_vals = df[target_col].dropna().unique()
    samples = df[target_col].dropna().head(20).tolist()

    prompt_v2 = _TARGET_VALUE_PROMPT.format(
        column_name=target_col,
        unique_values=str(list(unique_vals[:20])),
        samples=str(samples)
    )

    messages_v2 = [
        {
            "role": "system",
            "content": "你是一个严格的JSON输出助手。你只能输出有效的JSON，不含任何markdown、注释或额外文本。"
        },
        {
            "role": "user",
            "content": prompt_v2
        }
    ]

    try:
        result_v2 = call_llm_json(client, model, messages_v2)
        schema.target_mapping = result_v2.get('target_mapping')
        schema.pass_values = result_v2.get('pass_values', [])
        schema.fail_values = result_v2.get('fail_values', [])
        schema.pass_label = result_v2.get('pass_label', '良品')
        schema.fail_label = result_v2.get('fail_label', '不良')
        schema.target_type = result_v2.get('target_type', 'binary')
    except Exception as e:
        print(f"[schema_analyzer] 目标值映射分析失败（不影响主流程）: {e}")


# ── 规则 Fallback ─────────────────────────────────────────

def _looks_like_date(series: pd.Series, sample_size: int = 50, threshold: float = 0.8) -> bool:
    """检测字符串列是否为日期格式（纯正则，不依赖行业关键词）"""
    date_patterns = [
        r'^\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}$',                    # YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
        r'^\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\s+\d{1,2}:\d{2}',   # YYYY-MM-DD HH:MM
        r'^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}$',                    # DD/MM/YYYY
        r'^\d{4}年\d{1,2}月\d{1,2}日$',                            # 中文日期
    ]
    samples = series.dropna().astype(str).head(sample_size)
    if len(samples) < 2:
        return False
    match_count = sum(
        1 for v in samples
        if any(re.match(p, v.strip()) for p in date_patterns)
    )
    return match_count / len(samples) >= threshold


def _fallback_schema(df: pd.DataFrame) -> DataSchema:
    """当LLM调用失败时，使用基于规则的启发式方法推断Schema"""
    columns = []
    target_col = None
    id_col = None

    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()

        # 判断数据类型（增加字符串日期检测）
        if pd.api.types.is_numeric_dtype(df[col]):
            dtype = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            dtype = 'datetime'
        elif df[col].dtype == 'object' and df[col].nunique() > 100:
            dtype = 'text'
        elif df[col].dtype == 'object' and _looks_like_date(df[col]):
            dtype = 'datetime'
        else:
            dtype = 'categorical'

        # 启发式判断角色
        _common_target_keywords = ['结果', '状态', '良率', '等级', 'class', 'label', 'target',
                                   'Pass', 'Fail', 'defect', 'quality', 'grade', 'outcome']
        is_target_keyword = any(k in col_str for k in _common_target_keywords)
        # 纯ID列（编号/流水号，应排除）
        is_pure_id = (
            any(col_str == k or col_str.startswith(k) for k in ['编号', '序号', 'No.', '流水号'])
            or any(col_str.endswith(k) for k in ['编号', '单号', 'id', 'ID', '号'])
            or col_str.lower() in ('id', '编号', '序号')
        )
        # 语义ID列（包含业务信息，应保留为特征）
        is_semantic_id = any(k in col_str for k in
                             ['芯片号', '芯片编号', '批次', 'Batch', 'batch', 'code'])

        # 唯一值接近行数的 → 可能是ID
        unique_ratio = df[col].nunique() / max(len(df), 1)
        is_high_cardinality = unique_ratio > 0.8

        if is_target_keyword and target_col is None and df[col].nunique() < 20:
            role = 'target'
            target_col = col
        elif is_pure_id and is_high_cardinality:
            role = 'id'
            if id_col is None:
                id_col = col
        elif is_semantic_id:
            role = 'feature'  # 保留为特征，供后续分析使用
        elif dtype == 'datetime':
            role = 'datetime'
        elif is_high_cardinality and df[col].dtype == 'object':
            role = 'ignore'
        else:
            role = 'feature'

        # 合理的missing_strategy（规则引擎的合理默认值）
        if dtype == 'numeric':
            missing_strategy = 'median_fill'
            outlier_check = True
        elif dtype == 'categorical':
            missing_strategy = 'mode_fill'
            outlier_check = False
        else:
            missing_strategy = 'keep'
            outlier_check = False

        columns.append(ColumnSchema(
            raw_name=col,
            role=role,
            dtype=dtype,
            semantic_name=col,
            display_name=col,
            confidence=0.5,
            needs_confirmation=True,
            missing_strategy=missing_strategy,
            outlier_check=outlier_check,
        ))

    return DataSchema(
        id_column=id_col,
        target_column=target_col,
        columns=columns,
        raw_data_shape=(len(df), len(df.columns)),
        business_domain='（规则推断，未使用LLM）',
        business_description='LLM不可用时基于规则的降级分析',
    )


# ── 辅助：建议需要用户确认的事项 ──────────────────────────

def get_confirmation_questions(schema: DataSchema) -> List[str]:
    """生成需要用户确认的事项列表（含LLM检测到的质量问题）"""
    questions = []

    # 业务领域
    if schema.business_domain:
        questions.append(f"LLM 识别业务领域: {schema.business_domain}，是否正确？")

    # 质量问题
    for issue in schema.quality_issues[:5]:
        col = issue.get('column', '') or ''
        desc = issue.get('issue', '')
        sev = issue.get('severity', 'medium')
        sev_label = {'high': '⚠️ 高', 'medium': '🔶 中', 'low': 'ℹ️ 低'}.get(sev, '')
        questions.append(f"{sev_label} 质量问题: {col}{' - ' if col else ''}{desc}")

    if not schema.target_column:
        questions.append("未识别到目标变量，请手动选择一列作为分析目标。")
    else:
        if schema.target_mapping:
            questions.append(
                f"目标列「{schema.target_column}」的值映射为："
                f"良品={schema.pass_values}，不良={schema.fail_values}"
                f"，是否正确？"
            )

    uncertain_cols = schema.get_uncertain_columns(threshold=0.7)
    for col in uncertain_cols:
        questions.append(f"列「{col.raw_name}」的角色判断为「{col.role}」(置信度{col.confidence:.0%})，是否正确？")

    return questions
