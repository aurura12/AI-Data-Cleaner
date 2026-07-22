# ==========================================
# 文件: data_cleaning.py — 数据清洗与深度挖掘
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import sys
import json
import config as utils
import traceback
import warnings
import cleaning_code_generator
import schema_analyzer

# 忽略 Pandas 的一些切片警告
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_paths import AI_TEXT_RESULTS_FILE
try:
    from analysis_pipeline import ml_analysis as ml
except ImportError as e:
    print(f"Warning: Failed to import ml module: {e}")
    ml = None
    ML_IMPORT_ERROR = str(e)
else:
    ML_IMPORT_ERROR = None

# ==========================================
# 核心数据清洗函数（不依赖Streamlit，可独立调用）
# ==========================================

def clean_data_from_file(input_path, output_path=None):
    """
    从文件读取数据并执行完整的数据清洗流程
    不依赖Streamlit，可用于命令行调用
    
    Args:
        input_path: 输入CSV文件路径
        output_path: 输出CSV文件路径，如果为None则使用默认路径
    
    Returns:
        清洗后的DataFrame
    """
    # 读取数据
    try:
        try:
            raw_df = pd.read_csv(input_path, header=None, encoding='utf-8-sig', low_memory=False)
        except UnicodeDecodeError:
            raw_df = pd.read_csv(input_path, header=None, encoding='gbk', low_memory=False)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None
    
    print(f"--- 1. 读取与初始化 ---")
    print(f"正在读取文件: {input_path}")
    print(f"初始读取行数: {len(raw_df)}")
    raw_df.dropna(how='all', inplace=True)
    
    # 执行清洗流程
    df = raw_df.copy()
    
    # 步骤1: 智能表头识别与清洗
    print("\n[步骤 1/5] 智能表头识别与清洗...")
    df, _ = _clean_headers_core(df)
    print(f"✅ 表头清洗完成: {len(df)} 行, {len(df.columns)} 列")
    
    # 步骤2: 解析芯片正则特征
    print("\n[步骤 2/5] 解析芯片正则特征...")
    df, rstats = _process_regex_core(df)
    print(f"✅ 正则解析完成: {rstats['valid']} / {rstats['total']} 条记录解析成功")
    
    # 步骤3: 解析设备环境特征
    print("\n[步骤 3/5] 解析设备环境特征...")
    df, estats = _process_environment_features_core(df)
    if estats['found_col']:
        print(f"✅ 环境特征解析完成: 温度非空 {estats['temp_nonnull']}, 真空度非空 {estats['vacuum_nonnull']}")
    else:
        print("⚠️ 未找到包含'真空度'格式的列，跳过环境特征解析")
    
    # 步骤4: 计算物理与几何特征
    print("\n[步骤 4/5] 计算物理与几何特征...")
    df, fstats = _process_features_core(df)
    print(f"✅ 物理特征计算完成: 高度 {fstats['height_nonnull']}, 平整度 {fstats['range_nonnull']}, Taper {fstats['taper_nonnull']}, 时间序列 {fstats['time_nonnull']}")
    
    # 步骤5: 生成标签与类别编码
    print("\n[步骤 5/5] 生成标签与类别编码...")
    df, lstats = _process_labels_core(df)
    if not np.isnan(lstats['pass_rate']):
        print(f"✅ 标签处理完成: 样本量 {lstats['rows']}, 良品率 {lstats['pass_rate']:.2%}")
    else:
        print(f"✅ 标签处理完成: 样本量 {lstats['rows']}")
    
    # 保存结果
    if output_path:
        df_final = _reorder_final_core(df)
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 最终数据已保存至: {output_path}")
        print(f"   数据行数: {len(df_final)}, 列数: {len(df_final.columns)}")
        if 'Label_Pass' in df_final.columns:
            print(f"良品率(Pass Rate): {df_final['Label_Pass'].mean():.2%}")
    
    return df

def _clean_headers_core(df):
    """核心表头清洗函数"""
    df0 = df.copy()
    df0.dropna(how='all', inplace=True)
    
    if '芯片号' in df0.columns:
        df0.columns = [str(c).replace('（', '(').replace('）', ')').replace(' ', '').strip() for c in df0.columns]
        before = len(df0)
        df0 = df0[df0['芯片号'].notna() & (df0['芯片号'].astype(str) != '')]
        return df0, {'before': before, 'after': len(df0), 'removed': before - len(df0), 'cols': len(df0.columns)}
    
    first_row_str = str(df0.iloc[0].values)
    if ('倒焊日期' in first_row_str) and ('芯片号' in first_row_str):
        df1 = df0.iloc[1:].copy()
        df1.columns = df0.iloc[0].astype(str).str.strip()
    else:
        header_main = df0.iloc[0].replace('', np.nan).ffill()
        header_sub = df0.iloc[1].fillna('')
        new_columns = []
        for m, s in zip(header_main, header_sub):
            m = str(m).strip().replace('\n', '')
            s = str(s).strip().replace('\n', '')
            if s and s in ['1', '2', '3', '4', '5']:
                new_columns.append(f"{m}_{s}")
            elif m == 'nan':
                new_columns.append(s if s else "Unnamed")
            else:
                if s and s not in ['1', '2', '3', '4', '5']:
                    new_columns.append(f"{m}({s})")
                else:
                    new_columns.append(m)
        df1 = df0.iloc[2:].copy()
        df1.columns = new_columns
    
    df1.columns = [str(c).replace('（', '(').replace('）', ')').replace(' ', '').strip() for c in df1.columns]
    df1.reset_index(drop=True, inplace=True)
    
    if '芯片号' in df1.columns:
        before = len(df1)
        df1 = df1[df1['芯片号'].notna() & (df1['芯片号'].astype(str) != '')]
        return df1, {'before': before, 'after': len(df1), 'removed': before - len(df1), 'cols': len(df1.columns)}
    
    return df1, {'before': len(df1), 'after': len(df1), 'removed': 0, 'cols': len(df1.columns)}

def _process_regex_core(df):
    """核心正则解析函数"""
    pattern = r'([A-Z])(\d{6})-(\d+)-(\d+).*?(M\d+)'
    def extract_info(val):
        val = str(val).strip()
        m = re.search(pattern, val)
        if m:
            return pd.Series({
                'Semiconductor_Source': m.group(1),
                'Date_String': m.group(2),
                'Batch_ID': int(m.group(3)),
                'Wafer_Index': int(m.group(4)),
                'Position_Code': m.group(5)
            })
        return pd.Series([None, None, None, None, None], index=['Semiconductor_Source', 'Date_String', 'Batch_ID', 'Wafer_Index', 'Position_Code'])
    
    if '芯片号' in df.columns:
        meta_info = df['芯片号'].apply(extract_info)
        df2 = pd.concat([df, meta_info], axis=1)
        valid = df2['Position_Code'].notna().sum()
        return df2, {'valid': valid, 'total': len(df2)}
    return df, {'valid': 0, 'total': len(df)}

def _process_environment_features_core(df):
    """核心环境特征解析函数"""
    target_col = None
    for col in df.columns:
        sample_series = df[col].dropna()
        if len(sample_series) > 0 and sample_series.astype(str).str.contains('真空度', na=False).any():
            target_col = col
            break
    
    stats = {'temp_nonnull': 0, 'vacuum_nonnull': 0, 'found_col': None}
    
    if target_col:
        stats['found_col'] = target_col
        env_text = df[target_col].astype(str)
        df['Vacuum_Level'] = pd.to_numeric(
            env_text.str.extract(r'真空度[^\d-]*(-?\d+(?:\.\d+)?)', expand=False),
            errors='coerce'
        )
        df['Equipment_Temp'] = pd.to_numeric(
            env_text.str.extract(r'T\D*(-?\d+(?:\.\d+)?)', expand=False),
            errors='coerce'
        )
        stats['temp_nonnull'] = int(df['Equipment_Temp'].notna().sum())
        stats['vacuum_nonnull'] = int(df['Vacuum_Level'].notna().sum())
    
    return df, stats

def _process_features_core(df):
    """核心物理特征计算函数"""
    numeric_keywords = ['高度', '激光', 'CD', '压力', 'Force', '温度']
    for col in df.columns:
        if any(k in col for k in numeric_keywords):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    parts = ['电路', '芯片']
    part_heights = {}
    for part in parts:
        summary_col = next((c for c in df.columns if part in c and 'Height' in c and not any(x in c for x in ['1','2','3','4'])), None)
        if not summary_col:
            summary_col = next((c for c in df.columns if part in c and '高度' in c and not re.search(r'[_\d]$', c)), None)
        if summary_col:
            part_heights[part] = df[summary_col]
        else:
            point_cols = [c for c in df.columns if part in c and '高度' in c and re.search(r'[1-4]', c)]
            if len(point_cols) >= 3:
                part_heights[part] = df[point_cols].mean(axis=1)
            else:
                part_heights[part] = pd.Series([np.nan]*len(df))
    
    df['Total_Indium_Height'] = part_heights.get('电路', pd.Series([np.nan]*len(df))) + part_heights.get('芯片', pd.Series([np.nan]*len(df)))
    
    c_cols = [c for c in df.columns if '电路' in c and '高度' in c and re.search(r'[1-4]$', c)]
    if len(c_cols) >= 4:
        df['Calc_Circuit_Range'] = df[c_cols].max(axis=1) - df[c_cols].min(axis=1)
    
    cd_up = next((c for c in df.columns if 'CD' in c and '上' in c), None)
    cd_down = next((c for c in df.columns if 'CD' in c and '下' in c), None)
    if cd_up and cd_down:
        taper = df[cd_down] - df[cd_up]
        df['Indium_Taper_Zscore'] = (taper - taper.mean()) / (taper.std() + 1e-6)
    
    date_col = next((c for c in df.columns if '日期' in c), None)
    if date_col:
        df[date_col] = df[date_col].astype(str).str.replace('.', '-').str.replace('/', '-')
        df['Process_Date'] = pd.to_datetime(df[date_col], errors='coerce')
        min_date = df['Process_Date'].min()
        df['Time_Seq_Day'] = (df['Process_Date'] - min_date).dt.days
    
    return df, {
        'height_nonnull': int(pd.Series(df['Total_Indium_Height']).notna().sum()),
        'range_nonnull': int(pd.Series(df.get('Calc_Circuit_Range')).notna().sum()),
        'taper_nonnull': int(pd.Series(df.get('Indium_Taper_Zscore')).notna().sum()),
        'time_nonnull': int(pd.Series(df.get('Time_Seq_Day')).notna().sum())
    }

def _process_labels_core(df):
    """核心标签处理函数"""
    target_col = next((c for c in df.columns if '压连' in c), None)
    if target_col:
        def encode_label(x):
            try:
                val = int(float(x))
                if val in [0, 1]: return 1
                if val in [-1, 2]: return 0
                return 0
            except:
                return np.nan
        df['Label_Pass'] = df[target_col].apply(encode_label)
        df = df[df['Label_Pass'].notna()]
    
    p_col = next((c for c in df.columns if '压力' in c and 'kg' in str(c).lower()), None)
    if not p_col:
        p_col = next((c for c in df.columns if '压力' in c), None)
    if p_col:
        df['Force_kg'] = df[p_col]
    else:
        df['Force_kg'] = np.nan
    
    level_col = next((c for c in df.columns if '调平' in c), None)
    if level_col:
        df['Leveling_Mode'] = df[level_col].astype(str).apply(lambda x: 1 if '激光' in x or 'Laser' in x else 0)
    
    pass_rate = float(pd.Series(df.get('Label_Pass')).mean()) if 'Label_Pass' in df.columns else np.nan
    return df, {'rows': len(df), 'pass_rate': pass_rate}

def _reorder_final_core(df):
    """核心最终排序函数"""
    columns_to_drop = [
        '倒焊机状态是否变更', 
        '设备报警代码', 
        '设备开关机状态'
    ]
    df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
    
    cols = list(df.columns)
    priority_cols = ['芯片号', 'Label_Pass', 'Total_Indium_Height', 'Force_kg', 
                     'Equipment_Temp', 'Vacuum_Level',
                     'Position_Code', 'Wafer_Index', 'Batch_ID', 
                     'Calc_Circuit_Range', 'Indium_Taper_Zscore', 'Time_Seq_Day']
    final_cols = [c for c in priority_cols if c in cols] + [c for c in cols if c not in priority_cols]
    return df[final_cols]

# ==========================================
# Streamlit UI 数据清洗流程
# ==========================================

# 1. 数据清洗流程
def render_data_cleaning(df_raw, t, id_col=None):
    def _fix_duplicate_columns(df):
        """修复重复列名，避免Arrow转换错误"""
        if df.columns.duplicated().any():
            cols = pd.Series(df.columns)
            for dup in cols[cols.duplicated()].unique():
                dup_indices = cols[cols == dup].index.values.tolist()
                cols[dup_indices] = [dup if i == 0 else f"{dup}_{i}" for i in range(len(dup_indices))]
            df.columns = cols
        return df
    
    def _reorder_final(df):
        # Drop unused columns
        columns_to_drop = [
            '倒焊机状态是否变更', 
            '设备报警代码', 
            '设备开关机状态'
        ]
        df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')

        cols = list(df.columns)
        # ##### 【修改】: 添加环境参数列到优先级列表 #####
        priority_cols = ['芯片号', 'Label_Pass', 'Total_Indium_Height', 'Force_kg', 
                         'Equipment_Temp', 'Vacuum_Level',
                         'Position_Code', 'Wafer_Index', 'Batch_ID', 
                         'Calc_Circuit_Range', 'Indium_Taper_Zscore', 'Time_Seq_Day']
        final_cols = [c for c in priority_cols if c in cols] + [c for c in cols if c not in priority_cols]
        return df[final_cols]
    def _clean_headers(df):
        df0 = df.copy()
        df0.dropna(how='all', inplace=True)
        if '芯片号' in df0.columns:
            df0.columns = [str(c).replace('（', '(').replace('）', ')').replace(' ', '').strip() for c in df0.columns]
            before = len(df0)
            df0 = df0[df0['芯片号'].notna() & (df0['芯片号'].astype(str) != '')]
            return df0, {'before': before, 'after': len(df0), 'removed': before - len(df0), 'cols': len(df0.columns)}
        first_row_str = str(df0.iloc[0].values)
        if ('倒焊日期' in first_row_str) and ('芯片号' in first_row_str):
            df1 = df0.iloc[1:].copy()
            df1.columns = df0.iloc[0].astype(str).str.strip()
        else:
            header_main = df0.iloc[0].replace('', np.nan).ffill()
            header_sub = df0.iloc[1].fillna('')
            new_columns = []
            for m, s in zip(header_main, header_sub):
                m = str(m).strip().replace('\n', '')
                s = str(s).strip().replace('\n', '')
                if s and s in ['1', '2', '3', '4', '5']:
                    new_columns.append(f"{m}_{s}")
                elif m == 'nan':
                    new_columns.append(s if s else "Unnamed")
                else:
                    if s and s not in ['1', '2', '3', '4', '5']:
                        new_columns.append(f"{m}({s})")
                    else:
                        new_columns.append(m)
            df1 = df0.iloc[2:].copy()
            df1.columns = new_columns
        df1.columns = [str(c).replace('（', '(').replace('）', ')').replace(' ', '').strip() for c in df1.columns]
        df1.reset_index(drop=True, inplace=True)
        if '芯片号' in df1.columns:
            before = len(df1)
            df1 = df1[df1['芯片号'].notna() & (df1['芯片号'].astype(str) != '')]
            return df1, {'before': before, 'after': len(df1), 'removed': before - len(df1), 'cols': len(df1.columns)}
        return df1, {'before': len(df1), 'after': len(df1), 'removed': 0, 'cols': len(df1.columns)}

    def _process_regex(df):
        pattern = r'([A-Z])(\d{6})-(\d+)-(\d+).*?(M\d+)'
        def extract_info(val):
            val = str(val).strip()
            m = re.search(pattern, val)
            if m:
                return pd.Series({
                    'Semiconductor_Source': m.group(1),
                    'Date_String': m.group(2),
                    'Batch_ID': int(m.group(3)),
                    'Wafer_Index': int(m.group(4)),
                    'Position_Code': m.group(5)
                })
            return pd.Series([None, None, None, None, None], index=['Semiconductor_Source', 'Date_String', 'Batch_ID', 'Wafer_Index', 'Position_Code'])
        if '芯片号' in df.columns:
            meta_info = df['芯片号'].apply(extract_info)
            df2 = pd.concat([df, meta_info], axis=1)
            valid = df2['Position_Code'].notna().sum()
            return df2, {'valid': valid, 'total': len(df2)}
        return df, {'valid': 0, 'total': len(df)}

    ##### 【新增】解析设备环境参数 #####
    def _process_environment_features(df):
        """
        解析设备环境参数
        针对数据格式: "真空度：-866 T:19.8℃"
        """
        target_col = None
        # 使用更灵活的查找方式
        for col in df.columns:
            sample_series = df[col].dropna()
            if len(sample_series) > 0 and sample_series.astype(str).str.contains('真空度', na=False).any():
                # 只要这一列任意位置包含 "真空度" 就认定为环境参数列
                target_col = col
                break
        
        stats = {'temp_nonnull': 0, 'vacuum_nonnull': 0, 'found_col': None}
        
        if target_col:
            stats['found_col'] = target_col

            env_text = df[target_col].astype(str)

            # 1. 提取真空度并安全转数值，兼容尾部单位或标点
            df['Vacuum_Level'] = pd.to_numeric(
                env_text.str.extract(r'真空度[^\d-]*(-?\d+(?:\.\d+)?)', expand=False),
                errors='coerce'
            )
            
            # 2. 提取温度并安全转数值，兼容 "20.2." 这类尾部多余标点
            df['Equipment_Temp'] = pd.to_numeric(
                env_text.str.extract(r'T\D*(-?\d+(?:\.\d+)?)', expand=False),
                errors='coerce'
            )
            
            stats['temp_nonnull'] = int(df['Equipment_Temp'].notna().sum())
            stats['vacuum_nonnull'] = int(df['Vacuum_Level'].notna().sum())
        
        return df, stats

    def _process_features(df):
        numeric_keywords = ['高度', '激光', 'CD', '压力', 'Force', '温度']
        for col in df.columns:
            if any(k in col for k in numeric_keywords):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        parts = ['电路', '芯片']
        part_heights = {}
        for part in parts:
            summary_col = next((c for c in df.columns if part in c and 'Height' in c and not any(x in c for x in ['1','2','3','4'])), None)
            if not summary_col:
                summary_col = next((c for c in df.columns if part in c and '高度' in c and not re.search(r'[_\d]$', c)), None)
            if summary_col:
                part_heights[part] = df[summary_col]
            else:
                point_cols = [c for c in df.columns if part in c and '高度' in c and re.search(r'[1-4]', c)]
                if len(point_cols) >= 3:
                    part_heights[part] = df[point_cols].mean(axis=1)
                else:
                    part_heights[part] = pd.Series([np.nan]*len(df))
        df['Total_Indium_Height'] = part_heights.get('电路', pd.Series([np.nan]*len(df))) + part_heights.get('芯片', pd.Series([np.nan]*len(df)))
        c_cols = [c for c in df.columns if '电路' in c and '高度' in c and re.search(r'[1-4]$', c)]
        if len(c_cols) >= 4:
            df['Calc_Circuit_Range'] = df[c_cols].max(axis=1) - df[c_cols].min(axis=1)
        cd_up = next((c for c in df.columns if 'CD' in c and '上' in c), None)
        cd_down = next((c for c in df.columns if 'CD' in c and '下' in c), None)
        if cd_up and cd_down:
            taper = df[cd_down] - df[cd_up]
            df['Indium_Taper_Zscore'] = (taper - taper.mean()) / (taper.std() + 1e-6)
        date_col = next((c for c in df.columns if '日期' in c), None)
        if date_col:
            df[date_col] = df[date_col].astype(str).str.replace('.', '-').str.replace('/', '-')
            df['Process_Date'] = pd.to_datetime(df[date_col], errors='coerce')
            min_date = df['Process_Date'].min()
            df['Time_Seq_Day'] = (df['Process_Date'] - min_date).dt.days
        return df, {
            'height_nonnull': int(pd.Series(df['Total_Indium_Height']).notna().sum()),
            'range_nonnull': int(pd.Series(df.get('Calc_Circuit_Range')).notna().sum()),
            'taper_nonnull': int(pd.Series(df.get('Indium_Taper_Zscore')).notna().sum()),
            'time_nonnull': int(pd.Series(df.get('Time_Seq_Day')).notna().sum())
        }

    def _process_labels(df):
        target_col = next((c for c in df.columns if '压连' in c), None)
        if target_col:
            def encode_label(x):
                try:
                    val = int(float(x))
                    if val in [0, 1]: return 1
                    if val in [-1, 2]: return 0
                    return 0
                except:
                    return np.nan
            df['Label_Pass'] = df[target_col].apply(encode_label)
            df = df[df['Label_Pass'].notna()]
        p_col = next((c for c in df.columns if '压力' in c and 'kg' in str(c).lower()), None)
        if not p_col:
            p_col = next((c for c in df.columns if '压力' in c), None)
        if p_col:
            df['Force_kg'] = df[p_col]
        else:
            df['Force_kg'] = np.nan
        level_col = next((c for c in df.columns if '调平' in c), None)
        if level_col:
            df['Leveling_Mode'] = df[level_col].astype(str).apply(lambda x: 1 if '激光' in x or 'Laser' in x else 0)
        pass_rate = float(pd.Series(df.get('Label_Pass')).mean()) if 'Label_Pass' in df.columns else np.nan
        return df, {'rows': len(df), 'pass_rate': pass_rate}

    # ==========================================
    # LLM 通用智能清洗 UI（新增）
    # ==========================================
    def _render_llm_cleaning(df_raw, t, df_active):
        st.subheader("✨ 通用智能清洗")

        schema = st.session_state.get('data_schema')
        if schema is None:
            st.warning("⚠️ 尚未完成数据 Schema 分析，请先上传数据等待分析完成。")
            return

        st.success(f"📋 Schema 已就绪：{len(schema.columns)} 列，目标列: {schema.target_column or '未识别'}")

        # 展示schema概览
        with st.expander("查看数据 Schema 详情"):
            overview = []
            for col in schema.columns:
                overview.append({
                    "列名": col.raw_name,
                    "角色": col.role,
                    "类型": col.dtype,
                    "置信度": f"{col.confidence:.0%}"
                })
            st.dataframe(pd.DataFrame(overview), use_container_width=True, hide_index=True)

        # ── LLM增强清洗选项 ──
        enable_llm = st.checkbox(
            "🔧 启用LLM增强清洗（处理复杂文本列、复合ID解析等）",
            value=False,
            key="llm_enhance_checkbox"
        )

        # 检测复杂列（如果勾选了LLM增强）
        detected_complex = []
        if enable_llm:
            with st.spinner("正在检测需要增强处理的列..."):
                detected_complex = cleaning_code_generator.detect_complex_columns(df_raw, schema)
            if detected_complex:
                st.info(f"检测到 {len(detected_complex)} 个列需要LLM增强处理:")
                for cc in detected_complex:
                    st.caption(f"  - {cc['column']}: {cc['reason']}")
                    st.caption(f"    样本: {cc['samples'][:3]}")
            else:
                st.success("未检测到需要LLM增强处理的列")

        # 清洗按钮
        col1, col2 = st.columns([1, 3])
        run_clean = False
        with col1:
            run_clean = st.button("🚀 开始通用清洗", type="primary", use_container_width=True)

        if run_clean:
            with st.spinner("正在执行通用清洗..."):
                try:
                    client = utils.get_ai_client()
                    target_override = schema.target_column
                    # 如果schema没有识别到目标列，尝试用启发式
                    if target_override is None:
                        target_override = cleaning_code_generator._auto_detect_target_column(df_raw)

                    coder_model = utils.CODER_MODEL
                    text_model = utils.TEXT_MODEL

                    cleaned_df, stats = cleaning_code_generator.run_cleaning_pipeline(
                        df_raw, schema,
                        enable_llm_enhanced=enable_llm,
                        client=client if enable_llm else None,
                        model=text_model,
                        coder_model=coder_model,
                        target_column_override=target_override
                    )

                    # Layer 2: 如果有生成的代码，展示给用户
                    l2_data = stats.get('layer2', {})
                    if l2_data.get('code'):
                        with st.expander("🔍 查看LLM生成的清洗代码"):
                            st.code(l2_data['code'], language='python')

                    # 保存到 session_state
                    st.session_state['df_clean'] = cleaned_df
                    st.session_state['semiconductor_processor_state'] = {
                        'initialized': True,
                        'headers_cleaned': True,
                        'regex_processed': True,
                        'features_calculated': True,
                        'labels_processed': True,
                        'final_saved': True
                    }

                    st.success("✅ 通用清洗完成！")

                    # 展示清洗统计
                    stats_text = cleaning_code_generator.format_cleaning_stats(stats)
                    st.info(stats_text)

                    # 数据预览
                    st.markdown("#### 清洗后数据预览")
                    df_display = cleaned_df.head(50).copy()
                    # category 类型会导致 Glide Data Grid 渲染空白，转回 object
                    for col in df_display.select_dtypes(include=['category']).columns:
                        df_display[col] = df_display[col].astype(str)
                    if df_display.columns.duplicated().any():
                        cols = pd.Series(df_display.columns)
                        for dup in cols[cols.duplicated()].unique():
                            cols[cols[cols == dup].index.values.tolist()] = [
                                dup if i == 0 else f"{dup}_{i}" for i in range(sum(cols == dup))
                            ]
                        df_display.columns = cols
                    st.dataframe(df_display, use_container_width=True, height=420)

                    # 目标列分布
                    target = target_override or schema.target_column
                    if target and target in cleaned_df.columns:
                        st.markdown("#### 目标列分布")
                        col1_val, col2_val = st.columns(2)
                        pass_count = int((cleaned_df[target] == 1).sum())
                        fail_count = int((cleaned_df[target] == 0).sum())
                        with col1_val:
                            st.metric("良品 (Pass)", pass_count)
                        with col2_val:
                            st.metric("不良 (Fail)", fail_count)

                    # 下载按钮
                    csv_data = cleaned_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载清洗后数据",
                        data=csv_data,
                        file_name="cleaned_data.csv",
                        mime="text/csv",
                        type="primary"
                    )

                    st.rerun()

                except Exception as e:
                    st.error(f"清洗失败: {e}")
                    import traceback as tb
                    st.code(tb.format_exc())

    # ==========================================
    
    # 0. 初始化
    if 'df_clean' not in st.session_state:
        st.session_state['df_clean'] = df_raw.copy()
    if 'semiconductor_processor_state' not in st.session_state:
        st.session_state['semiconductor_processor_state'] = {
            'initialized': True,
            'headers_cleaned': False,
            'regex_processed': False,
            'features_calculated': False,
            'labels_processed': False,
            'final_saved': False
        }
    
    df_work = st.session_state['df_clean']
    df_active = utils.get_active_analysis_df(df_raw)
    processor_completed = all(st.session_state['semiconductor_processor_state'].values())

    # --- 顶栏 ---
    st.markdown(f"### {t.get('data_flow_panel')}")
    col_info_1, col_info_2 = st.columns([3, 1])
    with col_info_1:
        current_rows = len(df_active) if df_active is not None else len(df_work)
        st.info(t.get('rows_info').format(current_rows, len(df_raw)))
        current_filter = st.session_state.get('analysis_chip_filter_applied', utils.CHIP_FILTER_ALL)
        if processor_completed and current_filter != utils.CHIP_FILTER_ALL:
            st.caption(f"当前芯片筛选: {current_filter}")
    with col_info_2:
        if st.button(t.get('btn_reset'), type="secondary"):
            st.session_state['df_clean'] = df_raw.copy()
            st.session_state['semiconductor_processor_state'] = {
                'initialized': True,
                'headers_cleaned': False,
                'regex_processed': False,
                'features_calculated': False,
                'labels_processed': False,
                'final_saved': False
            }
            st.session_state['reset_analysis_chip_filter'] = True
            st.session_state['analysis_chip_filter_applied'] = utils.CHIP_FILTER_ALL
            st.session_state.pop('analysis_df', None)
            st.session_state.pop('analysis_filter_signature', None)
            st.session_state.pop('deep_mining_results', None)
            st.session_state.pop('final_suggestions', None)
            st.session_state.pop('full_report_html_bytes', None)
            st.session_state.pop('full_report_pdf_bytes', None)
            st.session_state.pop('report_generated_for_signature', None)
            st.session_state.pop('report_generated_for_label', None)
            st.success(t.get('reset_success'))
            st.rerun()

    if processor_completed:
        st.success("🎉 芯片数据清洗流程已全部完成！")

        df_export = df_active if df_active is not None else df_work
        df_out = _reorder_final(df_export)
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(current_dir, 'cleaned_chip_data_final.csv')
            df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
        except Exception:
            pass

        csv = df_out.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载清洗后的芯片数据",
            data=csv,
            file_name="cleaned_chip_data_final.csv",
            mime="text/csv",
            type="primary",
            key="download_final"
        )
    
    st.divider()

    # --- 方法论说明 ---
    with st.expander(t.get('method_expander_title'), expanded=False):
        st.info(t.get('method_explanation'))

    # ==========================================
    # 清洗模式选择（传统 vs 通用智能清洗）
    # ==========================================
    clean_mode = st.radio(
        "清洗模式",
        ["🔧 传统半导体制程清洗", "✨ 通用智能清洗"],
        index=0,
        horizontal=True,
        key="clean_mode_selector"
    )

    if clean_mode == "✨ 通用智能清洗":
        _render_llm_cleaning(df_raw, t, df_active)
        return

    st.subheader(t.get('clean_title'))

    # --- 自动清洗按钮 ---
    if not all(st.session_state['semiconductor_processor_state'].values()):
        st.markdown("\n")
        col_auto, col_spacer = st.columns([1, 3])
        run_auto_clean = False
        with col_auto:
            run_auto_clean = st.button("开始清洗", type="primary", use_container_width=True)

        live_output = st.container()
        if run_auto_clean:
            with live_output:
                with st.spinner("正在执行自动清洗流程..."):
                    try:
                        # --- 步骤 1: 读取与初始化 ---
                        st.info("正在执行步骤 1: 读取与初始化...")
                        
                        # 这里使用传入的 df_raw 作为原始数据
                        raw_df = df_raw.copy()
                        
                        # 剔除全空行
                        initial_rows = len(raw_df)
                        raw_df.dropna(how='all', inplace=True)
                        cleaned_rows = len(raw_df)
                        removed_rows = initial_rows - cleaned_rows
                        
                        # 更新会话状态
                        st.session_state['df_clean'] = raw_df.copy()
                        st.session_state['init_stats'] = {
                            'initial_rows': initial_rows,
                            'cleaned_rows': cleaned_rows,
                            'removed_rows': removed_rows
                        }
                        
                        # 显示初始化结果
                        st.success("✅ 步骤 1 完成: 读取与初始化")
                        st.info(f"初始数据行数: {initial_rows}")
                        st.info(f"剔除全空行: {removed_rows} 行")
                        st.info(f"处理后数据行数: {cleaned_rows}")
                        df_display = _fix_duplicate_columns(raw_df.head(50).copy())
                        st.dataframe(df_display, use_container_width=True, height=420)
                        st.divider()
                        
                        # --- 步骤 2: 智能表头识别与清洗 ---
                        st.info("正在执行步骤 2: 智能表头识别与清洗...")
                        df2, hstats = _clean_headers(st.session_state['df_clean'])
                        st.session_state['df_clean'] = df2
                        st.session_state['semiconductor_processor_state']['headers_cleaned'] = True
                        st.session_state['headers_stats'] = hstats
                        st.success("✅ 步骤 2 完成")
                        st.info(f"列数: {hstats['cols']}")
                        st.info(f"有效行数: {hstats['after']} (移除 {hstats['removed']})")
                        df_display = _fix_duplicate_columns(df2.head(50).copy())
                        st.dataframe(df_display, use_container_width=True, height=420)
                        st.divider()
                        
                        # --- 步骤 3: 解析芯片正则特征 ---
                        st.info("正在执行步骤 3: 解析芯片正则特征...")
                        df3, rstats = _process_regex(st.session_state['df_clean'])
                        st.session_state['df_clean'] = df3
                        st.session_state['semiconductor_processor_state']['regex_processed'] = True
                        st.session_state['regex_stats'] = rstats
                        st.success("✅ 步骤 3 完成")
                        st.info(f"解析成功: {rstats['valid']} / {rstats['total']}")
                        cols_3 = [c for c in ['芯片号','Chip_Source','Date_String','Batch_ID','Wafer_Index','Position_Code'] if c in df3.columns]
                        df_display = _fix_duplicate_columns(df3[cols_3].head(50).copy())
                        st.dataframe(df_display, use_container_width=True, height=420)
                        st.divider()
                        
                        # --- 步骤 3.5: 解析设备环境特征 (新增) ---
                        st.info("正在执行步骤 3.5: 解析设备环境特征...")
                        df3_5, estats = _process_environment_features(st.session_state['df_clean'])
                        st.session_state['df_clean'] = df3_5
                        st.session_state['environment_stats'] = estats
                        st.success("✅ 步骤 3.5 完成")
                        if estats['found_col']:
                            st.info(f"已定位环境数据列: [{estats['found_col']}]")
                            st.info(f"温度提取非空: {estats['temp_nonnull']}")
                            st.info(f"真空度提取非空: {estats['vacuum_nonnull']}")
                        else:
                            st.warning("未找到包含 '真空度' 格式的列，跳过环境特征解析")
                        st.divider()
                        
                        # --- 步骤 4: 计算物理与几何特征 ---
                        st.info("正在执行步骤 4: 计算物理与几何特征...")
                        df4, fstats = _process_features(st.session_state['df_clean'])
                        st.session_state['df_clean'] = df4
                        st.session_state['semiconductor_processor_state']['features_calculated'] = True
                        st.session_state['features_stats'] = fstats
                        st.success("✅ 步骤 4 完成")
                        st.info(f"总高度非空: {fstats['height_nonnull']}")
                        st.info(f"平整度非空: {fstats['range_nonnull']}")
                        st.info(f"Taper非空: {fstats['taper_nonnull']}")
                        st.info(f"时间序列非空: {fstats['time_nonnull']}")
                        # 【修改】: 添加环境参数到显示列表
                        cols_4 = [c for c in ['芯片号','Total_Indium_Height','Calc_Circuit_Range','Indium_Taper_Zscore','Time_Seq_Day','Equipment_Temp','Vacuum_Level'] if c in df4.columns]
                        df_display = _fix_duplicate_columns(df4[cols_4].head(50).copy())
                        st.dataframe(df_display, use_container_width=True, height=420)
                        st.divider()
                        
                        # --- 步骤 5: 生成标签与类别编码 ---
                        st.info("正在执行步骤 5: 生成标签与类别编码...")
                        df5, lstats = _process_labels(st.session_state['df_clean'])
                        st.session_state['df_clean'] = df5
                        st.session_state['semiconductor_processor_state']['labels_processed'] = True
                        st.session_state['labels_stats'] = lstats
                        st.success("✅ 步骤 5 完成")
                        if not np.isnan(lstats['pass_rate']):
                            st.info(f"良品率: {lstats['pass_rate']:.2%}")
                        st.info(f"样本量: {lstats['rows']}")
                        cols_5 = [c for c in ['芯片号','Label_Pass','Force_kg','Leveling_Mode'] if c in df5.columns]
                        df_display = _fix_duplicate_columns(df5[cols_5].head(50).copy())
                        st.dataframe(df_display, use_container_width=True, height=420)
                        st.divider()
                        
                        # --- 步骤 6: 保存最终结果 ---
                        st.session_state['semiconductor_processor_state']['final_saved'] = True
                        
                        st.success("🎉 自动清洗流程已完成！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"自动清洗失败: {str(e)}")
    
    st.divider()

    # ==========================
    # 芯片数据专用清洗流程 (6 个步骤)
    # ==========================
    
    # --- 步骤 1: 读取与初始化 ---
    st.markdown(f"### 1. 读取与初始化")
    
    if st.session_state['semiconductor_processor_state']['initialized']:
        st.success("✅ 数据已成功读取并初始化")
        stats = st.session_state.get('init_stats')
        if stats:
            st.info(f"初始数据行数: {stats['initial_rows']}")
            st.info(f"剔除全空行: {stats['removed_rows']} 行")
            st.info(f"处理后数据行数: {stats['cleaned_rows']}")
            # 处理重复列名（避免Arrow转换错误）
            df_display = st.session_state['df_clean'].head(50).copy()
            if df_display.columns.duplicated().any():
                cols = pd.Series(df_display.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup if i == 0 else f"{dup}_{i}" for i in range(sum(cols == dup))]
                df_display.columns = cols
            st.dataframe(df_display, use_container_width=True, height=420)
        else:
            st.info(f"初始数据行数: {len(df_raw)}")
    
    st.divider()
    
    # --- 步骤 2: 智能表头识别与清洗 ---
    st.markdown(f"### 2. 智能表头识别与清洗")
    
    if st.session_state['semiconductor_processor_state']['headers_cleaned']:
        st.success("✅ 表头已成功识别和清洗")
        h = st.session_state.get('headers_stats')
        if h:
            st.info(f"列数: {h['cols']}")
            st.info(f"有效行数: {h['after']} (移除 {h['removed']})")
            # 处理重复列名（避免Arrow转换错误）
            df_display = st.session_state['df_clean'].head(50).copy()
            if df_display.columns.duplicated().any():
                cols = pd.Series(df_display.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup if i == 0 else f"{dup}_{i}" for i in range(sum(cols == dup))]
                df_display.columns = cols
            st.dataframe(df_display, use_container_width=True, height=420)
    else:
        st.info("⏳ 等待执行...")
    
    st.divider()
    
    # --- 步骤 3: 解析芯片正则特征 ---
    st.markdown(f"### 3. 解析芯片正则特征")
    
    if st.session_state['semiconductor_processor_state']['regex_processed']:
        st.success("✅ 芯片号信息已成功解析")
        r = st.session_state.get('regex_stats')
        if r:
            st.info(f"解析成功: {r['valid']} / {r['total']}")
            cols_3b = [c for c in ['芯片号','Chip_Source','Date_String','Batch_ID','Wafer_Index','Position_Code'] if c in st.session_state['df_clean'].columns]
            df_display = _fix_duplicate_columns(st.session_state['df_clean'][cols_3b].head(50).copy())
            st.dataframe(df_display, use_container_width=True, height=420)
    else:
        st.info("⏳ 等待执行...")
    
    st.divider()
    
    # --- 步骤 4: 计算物理与几何特征 ---
    st.markdown(f"### 4. 计算物理与几何特征")
    
    if st.session_state['semiconductor_processor_state']['features_calculated']:
        st.success("✅ 物理特征已成功计算")
        f = st.session_state.get('features_stats')
        if f:
            st.info(f"总高度非空: {f['height_nonnull']}")
            st.info(f"平整度非空: {f['range_nonnull']}")
            st.info(f"Taper非空: {f['taper_nonnull']}")
            st.info(f"时间序列非空: {f['time_nonnull']}")
            # 【修改】: 添加环境参数到显示列表
            cols_4b = [c for c in ['芯片号','Total_Indium_Height','Calc_Circuit_Range','Indium_Taper_Zscore','Time_Seq_Day','Equipment_Temp','Vacuum_Level'] if c in st.session_state['df_clean'].columns]
            # 处理重复列名（避免Arrow转换错误）
            df_display = st.session_state['df_clean'][cols_4b].head(50).copy()
            if df_display.columns.duplicated().any():
                cols = pd.Series(df_display.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup if i == 0 else f"{dup}_{i}" for i in range(sum(cols == dup))]
                df_display.columns = cols
            st.dataframe(df_display, use_container_width=True, height=420)
    else:
        st.info("⏳ 等待执行...")
    
    st.divider()
    
    # --- 步骤 5: 生成标签与类别编码 ---
    st.markdown(f"### 5. 生成标签与类别编码")
    
    if st.session_state['semiconductor_processor_state']['labels_processed']:
        st.success("✅ 标签与调平方式已成功处理")
        l = st.session_state.get('labels_stats')
        if l:
            if not np.isnan(l['pass_rate']):
                st.info(f"良品率: {l['pass_rate']:.2%}")
            st.info(f"样本量: {l['rows']}")
            cols_5b = [c for c in ['芯片号','Label_Pass','Force_kg','Leveling_Mode'] if c in st.session_state['df_clean'].columns]
            # 处理重复列名（避免Arrow转换错误）
            df_display = st.session_state['df_clean'][cols_5b].head(50).copy()
            if df_display.columns.duplicated().any():
                cols = pd.Series(df_display.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup if i == 0 else f"{dup}_{i}" for i in range(sum(cols == dup))]
                df_display.columns = cols
            st.dataframe(df_display, use_container_width=True, height=420)
    else:
        st.info("⏳ 等待执行...")
    
    st.divider()
    
    # --- 步骤 6: 保存最终结果 ---
    st.markdown(f"### 6. 保存最终结果")
    
    if st.session_state['semiconductor_processor_state']['final_saved']:
        st.success("✅ 最终结果已成功保存")
        df_work = st.session_state['df_clean']
        st.info(f"最终数据行数: {len(df_work)}")
        
        df_out = _reorder_final(df_work)
        df_display = _fix_duplicate_columns(df_out.copy())
        st.dataframe(df_display)
    else:
        st.info("⏳ 等待执行...")
    
    st.divider()
    
def render_deep_mining(df, t, target_col, id_col=None):
    st.subheader("深度挖掘与归因分析 (Deep Mining)")
    
    # Check if analysis has been run
    if 'deep_mining_results' not in st.session_state:
        st.session_state['deep_mining_results'] = None

    if st.button("开始深度挖掘分析", type="primary"):
        with st.spinner("正在进行机器学习挖掘..."):
            # Save current df to temp file for ml.py to read
            temp_input = os.path.join(os.path.dirname(__file__), 'data', 'temp_mining_input.csv')
            os.makedirs(os.path.dirname(temp_input), exist_ok=True)
            df.to_csv(temp_input, index=False)
            
            # Run ML analysis
            try:
                if ml is None:
                    raise ImportError(f"无法加载 analysis_pipeline.ml_analysis 模块: {ML_IMPORT_ERROR or '未知原因'}")

                # Reload ml module to ensure latest changes are picked up
                import importlib
                importlib.reload(ml)
                results = ml.run_ml_analysis(input_path=temp_input)
                st.session_state['deep_mining_results'] = results
                st.success("挖掘完成！")
            except Exception as e:
                st.error(f"挖掘失败: {str(e)}")
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc())

    if st.session_state['deep_mining_results']:
        results = st.session_state['deep_mining_results']
        
        for res in results:
            st.divider()
            st.markdown(f"### {res['chart_name']}")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.image(res['image_path'])
                
            with col2:
                # Generate analysis if not already present
                if 'analysis_text' not in res:
                     with st.spinner(f"正在分析 {res['chart_name']}..."):
                        prompt = f"""你现在是一位半导体良率优化专家。

【任务目标】
基于以下统计数据和分析摘要，分析该图表所反映的工艺特征，识别可能与良率下降相关的物理参数特征和数据模式。

【业务背景与数据字典】
1. 预测目标：Is_Pass (0=不良/Fail, 1=良品/Pass)。**重点关注导致 Is_Pass=0 的参数区间。**
2. 关键参数含义：
   - Equipment_Temp：设备温度（影响焊料熔融与流动性）。
   - Vacuum_Level：真空度（影响气泡排出与空洞率）。
   - Force_kg：倒焊压力（影响接合致密性）。
   - Total_Indium_Height：铟柱总高度。
   - Calc_Circuit_Range：电路表面平整度。
   - Time_Seq_Day：设备连续运行天数（漂移指标）。
注意铟柱总高度由上一道工艺确定，与倒焊压力无关。
【数据摘要】
{res['data_description']}

【分析要求】
1. 根据相应数据识别潜在模式（是否存在**温度/真空/压力**等物理参数的特定区间与低良率强相关）。
3. 关联业务逻辑（结合物理意义，分析为何某些参数组合会导致Fail）。
4. 提出参考建议（柔性表述）。

【输出格式要求】
请严格按照以下JSON格式输出，不要添加任何其他文字：
{{
"key_findings": ["数据观察1", "数据观察2"],
"process_suggestions": "针对工艺参数（特别是温度、真空、压力等）的排查建议",
"detailed_analysis": "详细分析文本"
}}
"""
                        try:
                            response = utils.get_ai_client().chat.completions.create(
                                model=utils.TEXT_MODEL,
                                messages=[{'role': 'user', 'content': prompt}],
                                temperature=0.3
                            )
                            res['analysis_text'] = response.choices[0].message.content
                        except Exception as e:
                            res['analysis_text'] = f"分析失败: {str(e)}"
                
                # Try to parse JSON
                try:
                    # Clean potential markdown code blocks
                    content = res['analysis_text'].replace('```json', '').replace('```', '').strip()
                    analysis_json = json.loads(content)
                    
                    st.markdown("#### 💡 智能分析报告")
                    st.markdown("**关键发现:**")
                    for finding in analysis_json.get('key_findings', []):
                        st.markdown(f"- {finding}")
                        
                    st.markdown("**工艺建议:**")
                    st.info(analysis_json.get('process_suggestions', '暂无建议'))
                    
                    st.markdown("**详细分析:**")
                    st.write(analysis_json.get('detailed_analysis', ''))
                    
                except:
                    st.warning("原始输出（非标准JSON）:")
                    st.write(res['analysis_text'])

        # --- 新增：总结与工艺优化建议 ---
        st.divider()
        st.subheader("📝 总结与工艺优化建议")

        if st.button("生成最终汇总建议", type="primary", key="btn_final_summary"):
            with st.spinner("正在综合分析全量数据与挖掘结果..."):
                # 1. 获取描述性统计分析报告
                descriptive_report = ""
                try:
                    text_results_file = AI_TEXT_RESULTS_FILE
                    
                    if os.path.exists(text_results_file) and utils.is_descriptive_report_current():
                        with open(text_results_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 尝试获取纯文本或HTML内容
                            descriptive_report = data.get('comprehensive_report', '')
                            # 简单去除HTML标签以节省token
                            descriptive_report = re.sub(r'<[^>]+>', '', descriptive_report)
                    elif os.path.exists(text_results_file):
                        st.warning("当前描述性报告与筛选条件不一致，本次汇总建议将仅基于当前深度挖掘结果。")
                except Exception as e:
                    print(f"Error loading descriptive report: {e}")

                # 2. 获取深度挖掘分析结果
                deep_mining_text = ""
                if st.session_state.get('deep_mining_results'):
                    for res in st.session_state['deep_mining_results']:
                        deep_mining_text += f"\n[图表分析: {res.get('chart_name', '')}]\n"
                        deep_mining_text += str(res.get('analysis_text', '')) + "\n"

                # 3. 构建 Prompt
                prompt = f"""
你是一位资深半导体工艺专家。请结合以下两部分分析内容，总结出核心的工艺优化建议。

【输入1：描述性统计分析结论】
{descriptive_report[:3000]} 

【输入2：深度挖掘与归因分析结论】
{deep_mining_text[:3000]}

【任务目标】
请综合分析，识别导致良率下降的核心原因（如高度、压力、温度等参数的异常），并提出3-5条具体的、可执行的工艺优化建议。
风格要求：专业、简洁、直接。

【输出格式要求】
请严格返回以下JSON格式（不要包含markdown代码块标记）：
{{
    "suggestions": [
        {{
            "title": "建议标题 (例如：设立拦截红线：高度 > 11.65 μm)",
            "icon": "🛑", 
            "content": "详细的建议描述，包含数据支持和具体行动项。",
            "type": "critical"
        }},
        {{
            "title": "建议标题",
            "icon": "📉",
            "content": "内容...",
            "type": "warning"
        }}
    ]
}}
注意：
- icon请使用emoji，如 🛑 (critical), 📉 (trend), 🎯 (target), ⚖️ (balance), ⚙️ (adjust)。
- type字段可选 'critical', 'warning', 'info'。
"""
                # 4. 调用 LLM
                try:
                    response = utils.get_ai_client().chat.completions.create(
                        model=utils.TEXT_MODEL,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0.3
                    )
                    res_content = response.choices[0].message.content
                    # 清理可能的markdown标记
                    res_content = res_content.replace('```json', '').replace('```', '').strip()
                    st.session_state['final_suggestions'] = json.loads(res_content)
                except Exception as e:
                    st.error(f"生成建议失败: {str(e)}")

        # 5. 渲染展示
        if 'final_suggestions' in st.session_state and st.session_state['final_suggestions']:
            suggestions = st.session_state['final_suggestions'].get('suggestions', [])
            
            st.markdown("""
            <style>
            .suggestion-card {
                background-color: var(--secondary-background-color);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 15px;
                border-left: 5px solid #67c23a;
                box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
            }
            .suggestion-header {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            .suggestion-icon {
                font-size: 24px;
                margin-right: 12px;
            }
            .suggestion-title {
                font-size: 18px;
                font-weight: 600;
                color: var(--text-color);
            }
            .suggestion-content {
                color: var(--text-color);
                font-size: 15px;
                line-height: 1.6;
                margin-left: 36px;
                opacity: 0.9;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(f"#### 基于全量数据分析，我们提出以下 {len(suggestions)} 条针对性改进策略：")
            
            for sug in suggestions:
                st.markdown(f"""
                <div class="suggestion-card">
                    <div class="suggestion-header">
                        <span class="suggestion-icon">{sug.get('icon', '💡')}</span>
                        <span class="suggestion-title">{sug.get('title', '优化建议')}</span>
                    </div>
                    <div class="suggestion-content">
                        {sug.get('content', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

def _unused_render_cart_analysis(df, t, target_col, id_col=None):
    # 标题拼接
    full_title = f"{t.get('cart_title')} {t.get('cart_title_suffix', '')}"
    st.subheader(full_title)
    
    if not target_col or target_col not in df.columns:
        st.warning(t.get('missing_target'))
        return

    # [i18n] 替换 "Target:"
    st.write(f"{t.get('target_prefix', '🎯 Target:')} **{target_col}** ") 

    # --- 参数设置 ---
    with st.expander(t.get('settings_expander', 'Settings'), expanded=False):
        # [i18n] 模式选择：先获取语言包中的选项文本
        opt_bin = t.get('mode_binary', 'Binary (0 vs 1+)')
        opt_multi = t.get('mode_multi', 'Multiclass (0, 1, 2...)')
        
        analysis_mode = st.radio(
            t.get('mode_label', 'Analysis Mode'), 
            options=[opt_bin, opt_multi],
            index=0
        )
        # 逻辑：直接判断选中的是不是二分类那个选项字符串
        is_binary = (analysis_mode == opt_bin)

        test_size = st.slider(t.get('train_test_label'), 0.1, 0.4, 0.2, 0.05)
        default_seed_label = t.get('random_seed', 'Random Seed')
        random_seed = st.number_input(default_seed_label, value=42)

    if st.button(t.get('btn_run_cart')):
        with st.spinner(t.get('spinner_msg')):
            
            # 1. 预处理与编码记录
            df_model = df.copy()
            if id_col and id_col in df_model.columns:
                df_model = df_model.drop(columns=[id_col])
            
            # --- 文本变量编码 ---
            encoding_map = {} 
            le = LabelEncoder()
            cat_cols = df_model.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols) > 0:
                # [i18n] 替换提示信息
                st.info(t.get('cat_enc_info', 'ℹ️ Categorical Encoding Executed:'))
                map_cols = st.columns(2)
                for idx, col in enumerate(cat_cols):
                    df_model[col] = le.fit_transform(df_model[col].astype(str))
                    mapping = dict(zip(le.transform(le.classes_), le.classes_))
                    encoding_map[col] = mapping
                    with map_cols[idx % 2]:
                        st.caption(f"**{col}**: {mapping}")

            # 根据模式处理目标变量
            if is_binary:
                if df_model[target_col].dtype != 'object':
                     df_model[target_col] = df_model[target_col].apply(lambda x: 1 if x > 0 else 0)
            
            X = df_model.drop(columns=[target_col])
            y = df_model[target_col]

            # 2. 划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=y
            )

            # 3. 训练
            if is_binary:
                pos_ratio = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss', 
                    random_state=random_seed,
                    use_label_encoder=False,
                    scale_pos_weight=pos_ratio,
                    max_depth=4
                )
            else:
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    random_state=random_seed,
                    use_label_encoder=False,
                    max_depth=4
                )

            model.fit(X_train, y_train)
            
            # 4. 指标计算
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            
            if is_binary:
                try: auc = roc_auc_score(y_test, y_prob[:, 1])
                except: auc = 0.5
                sens = recall_score(y_test, y_pred)
                try:
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    spec = tn / (tn + fp)
                except: spec = 0.0
                metric_4_label = t.get('spec_label', 'Specificity')
                metric_4_val = spec
            else:
                try: 
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                except: auc = 0.5
                sens = recall_score(y_test, y_pred, average='weighted')
                
                f1_val = f1_score(y_test, y_pred, average='weighted')
                # [i18n] 多分类下的 F1 指标名称
                metric_4_label = t.get('f1_label', 'F1-Score (Weighted)')
                metric_4_val = f1_val

            # --- 结果展示 1: 指标 ---
            st.markdown(f"### {t.get('metrics_title')}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t.get('auc_label', 'AUC'), f"{auc:.3f}")
            c2.metric(t.get('acc_label', 'Accuracy'), f"{acc:.1%}")
            c3.metric(t.get('sens_label', 'Sensitivity'), f"{sens:.1%}")
            c4.metric(metric_4_label, f"{metric_4_val:.1%}")
            
            with st.expander(t.get('cm_title', 'Confusion Matrix'), expanded=True):
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax_cm)
                # [i18n] 坐标轴文字
                ax_cm.set_ylabel(t.get('cm_true', 'True Label'))
                ax_cm.set_xlabel(t.get('cm_pred', 'Predicted Label'))
                st.pyplot(fig_cm); plt.close(fig_cm)

            st.divider()

            # --- 结果展示 2: SHAP ---
            st.markdown(f"### {t.get('shap_title')}")
            explainer = shap.TreeExplainer(model)
            X_metrics = X.sample(2000, random_state=random_seed) if len(X) > 2000 else X
            shap_values = explainer.shap_values(X_metrics)
            
            if isinstance(shap_values, list):
                target_idx = 1 if len(shap_values) > 1 else 0
                shap_vals_vis = shap_values[target_idx]
                # [i18n] 提示 SHAP 类别
                st.caption(f"{t.get('shap_class_note', '⚠️ SHAP values shown for Class')} {target_idx}")
            else:
                shap_vals_vis = shap_values

            c_shp1, c_shp2 = st.columns([3, 1])
            with c_shp1:
                fig_summary, ax = plt.subplots(figsize=(8, len(X.columns) * 0.4 + 2))
                shap.summary_plot(shap_vals_vis, X_metrics, show=False, plot_type="dot")
                st.pyplot(fig_summary); plt.close(fig_summary)
            with c_shp2:
                st.info(t.get('shap_info'))
                if encoding_map:
                    # [i18n] 编码参考
                    st.markdown(f"**{t.get('enc_ref', '⚠️ Encoding Reference:')}**")
                    for col, mapping in encoding_map.items():
                        st.markdown(f"*{col}*: `{mapping}`")

            st.divider()

            # --- 结果展示 3: 依赖图 ---
            st.markdown(f"### {t.get('dep_title')}")
            mean_abs_shap = np.abs(shap_vals_vis).mean(axis=0)
            top_feat = X_metrics.columns[mean_abs_shap.argmax()]
            
            # [i18n] 下拉框提示
            sel_feat = st.selectbox(
                t.get('dep_select', 'Select Feature:'), 
                options=X_metrics.columns, 
                index=list(X_metrics.columns).index(top_feat)
            )
            
            fig_dep, ax = plt.subplots(figsize=(8, 4))
            shap.dependence_plot(sel_feat, shap_vals_vis, X_metrics, show=False, interaction_index='auto', ax=ax, alpha=0.7)
            st.pyplot(fig_dep); plt.close(fig_dep)

def _unused_render_simple_tree_viz(df, t, target_col, id_col=None):
    # [i18n] 替换标题
    st.subheader(t.get('tree_viz_title', 'Tree Visualization'))
    st.info(t.get('tree_viz_info'))

    if not target_col or target_col not in df.columns:
        return

    col_tree_1, col_tree_2 = st.columns([1, 3])
    with col_tree_1:
        # [i18n] 替换滑块标签
        tree_depth = st.slider(t.get('tree_depth', 'Depth'), 2, 5, 3)
        # [i18n] 替换按钮文字
        run_tree = st.button(t.get('btn_tree', 'Run Decision Tree'))

    if run_tree:
        df_model = df.copy()
        if id_col and id_col in df_model.columns:
            df_model = df_model.drop(columns=[id_col])
        
        le = LabelEncoder()
        for col in df_model.select_dtypes(include=['object', 'category']).columns:
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            
        X = df_model.drop(columns=[target_col])
        y = df_model[target_col]
        
        if y.dtype == float:
            y = y.astype(int)

        dt = DecisionTreeClassifier(max_depth=tree_depth, criterion="entropy", random_state=42)
        dt.fit(X, y)
        
        # [i18n] 替换结果小标题
        st.markdown(f"#### {t.get('tree_path_header', 'Visualization Result')}")
        
        fig, ax = plt.subplots(figsize=(14, 6 + tree_depth))
        plot_tree(
            dt, 
            feature_names=X.columns.tolist(), 
            class_names=[str(c) for c in sorted(y.unique())], 
            filled=True, 
            rounded=True, 
            fontsize=10, 
            ax=ax
        )
        st.pyplot(fig); plt.close(fig)
