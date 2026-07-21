"""
从EDA、ML、Position的绘图代码中提取关键数据点
转换为结构化文本，供LLM生成报告使用
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import ROOT_DIR, OUTPUT_DIR, CLEANED_DATA_FILE, ML_REPORT_DIR

def extract_eda_chart_data(base_dir: str) -> Dict[str, Any]:
    """从EDA分析中提取图表数据"""
    data_file = CLEANED_DATA_FILE
    if not os.path.exists(data_file):
        return {}
    
    df = pd.read_csv(data_file)
    
    # 确保有必要的列
    target_col = next((c for c in df.columns if '压连' in c), None)
    if not target_col:
        return {}
    
    # 处理标签（如果Is_Pass不存在）
    if 'Is_Pass' not in df.columns:
        def get_status(val):
            try:
                v = int(float(val))
            except:
                return 0
            if v == 0 or v == 1:
                return 1  # Pass
            return 0  # Fail
        
        df['Is_Pass'] = df[target_col].apply(get_status)
    
    chart_data = {}
    
    # 1. 生产状态分布（柱状图数据）
    status_map = {-1: '虚焊(-1)', 0: '正常(0)', 1: '轻微压连(1)', 2: '严重压连(2)'}
    status_counts = {}
    for val in df[target_col].dropna():
        try:
            v = int(float(val))
            status_name = status_map.get(v, f'其他({v})')
            status_counts[status_name] = status_counts.get(status_name, 0) + 1
        except:
            pass
    
    chart_data['status_distribution'] = {
        'type': 'bar_chart',
        'title': '生产状态分布',
        'data': [{'x': k, 'y': v} for k, v in status_counts.items()],
        'summary': f"总样本{len(df)}颗，其中：{', '.join([f'{k} {v}颗' for k, v in status_counts.items()])}"
    }
    
    # 2. 相关性矩阵数据
    target_cols = ['Total_Indium_Height', '倒焊压力（kg）', 'Indium_Taper_Zscore', 'Calc_Circuit_Range', 'Is_Pass']
    valid_cols = []
    col_mapping = {}
    for col in target_cols:
        if col == 'Total_Indium_Height' and col in df.columns:
            valid_cols.append(col)
            col_mapping[col] = '总铟柱高度'
        elif col == '倒焊压力（kg）':
            pressure_col = next((c for c in df.columns if '压力' in c), None)
            if pressure_col:
                valid_cols.append(pressure_col)
                col_mapping[pressure_col] = '倒焊压力'
        elif col in df.columns:
            valid_cols.append(col)
            if col == 'Indium_Taper_Zscore':
                col_mapping[col] = '铟柱形状异常度'
            elif col == 'Calc_Circuit_Range':
                col_mapping[col] = '电路端平整度'
            elif col == 'Is_Pass':
                col_mapping[col] = '良率'
    
    if len(valid_cols) >= 2:
        corr = df[valid_cols].corr()
        correlations = []
        for i, col1 in enumerate(valid_cols):
            for j, col2 in enumerate(valid_cols):
                if i < j:  # 只取上三角
                    val = corr.loc[col1, col2]
                    if not np.isnan(val):
                        correlations.append({
                            'feature1': col_mapping.get(col1, col1),
                            'feature2': col_mapping.get(col2, col2),
                            'correlation': float(val)
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
    
    # 3. 良次品特征对比（箱线图统计）
    if 'Total_Indium_Height' in df.columns:
        pass_data = df[df['Is_Pass'] == 1]['Total_Indium_Height'].dropna()
        fail_data = df[df['Is_Pass'] == 0]['Total_Indium_Height'].dropna()
        
        chart_data['pass_fail_comparison'] = {
            'type': 'boxplot',
            'title': '良次品总高度对比',
            'data': {
                '良品': {
                    'median': float(pass_data.median()) if len(pass_data) > 0 else 0,
                    'mean': float(pass_data.mean()) if len(pass_data) > 0 else 0,
                    'q25': float(pass_data.quantile(0.25)) if len(pass_data) > 0 else 0,
                    'q75': float(pass_data.quantile(0.75)) if len(pass_data) > 0 else 0,
                    'count': len(pass_data)
                },
                '不良品': {
                    'median': float(fail_data.median()) if len(fail_data) > 0 else 0,
                    'mean': float(fail_data.mean()) if len(fail_data) > 0 else 0,
                    'q25': float(fail_data.quantile(0.25)) if len(fail_data) > 0 else 0,
                    'q75': float(fail_data.quantile(0.75)) if len(fail_data) > 0 else 0,
                    'count': len(fail_data)
                }
            }
        }
    
    # 4. 位置良率分析（柱状图数据）
    if 'Position_Code' in df.columns:
        pos_stats = df.groupby('Position_Code')['Is_Pass'].agg(['mean', 'count']).reset_index()
        pos_stats = pos_stats[pos_stats['count'] >= 1].sort_values('mean', ascending=True)
        
        position_data = []
        for _, row in pos_stats.iterrows():
            position_data.append({
                'position': str(row['Position_Code']),
                'yield_rate': float(row['mean']),
                'count': int(row['count'])
            })
        
        chart_data['position_yield'] = {
            'type': 'bar_chart',
            'title': '位置良率排行',
            'data': position_data,
            'worst': position_data[0] if position_data else None,
            'best': position_data[-1] if position_data else None,
            'average': float(df['Is_Pass'].mean())
        }
    
    # 5. 晶圆次序效应
    if 'Wafer_Index' in df.columns:
        wafer_stats = df.groupby('Wafer_Index')['Is_Pass'].agg(['mean', 'count']).reset_index()
        wafer_stats = wafer_stats.sort_values('Wafer_Index')
        
        wafer_data = []
        for _, row in wafer_stats.iterrows():
            wafer_data.append({
                'wafer_index': float(row['Wafer_Index']),
                'yield_rate': float(row['mean']),
                'count': int(row['count'])
            })
        
        chart_data['wafer_effect'] = {
            'type': 'bar_chart',
            'title': '晶圆加工次序效应',
            'data': wafer_data,
            'worst_wafer': min(wafer_data, key=lambda x: x['yield_rate']) if wafer_data else None
        }
    
    # 6. 时间趋势（高度漂移）
    if 'Time_Seq_Day' in df.columns and 'Total_Indium_Height' in df.columns:
        # 计算早期和后期的高度均值
        median_day = df['Time_Seq_Day'].quantile(0.5)
        early_mean = df[df['Time_Seq_Day'] < median_day]['Total_Indium_Height'].mean()
        late_mean = df[df['Time_Seq_Day'] >= median_day]['Total_Indium_Height'].mean()
        
        chart_data['height_drift'] = {
            'type': 'trend',
            'title': '高度长期漂移',
            'data': {
                'early_mean': float(early_mean) if not np.isnan(early_mean) else 0,
                'late_mean': float(late_mean) if not np.isnan(late_mean) else 0,
                'drift': float(late_mean - early_mean) if not (np.isnan(early_mean) or np.isnan(late_mean)) else 0
            }
        }
    
    return chart_data

def extract_ml_chart_data(base_dir: str) -> Dict[str, Any]:
    """从ML分析中提取图表数据"""
    ml_csv = os.path.join(ML_REPORT_DIR, 'feature_importance_ranking.csv')
    if not os.path.exists(ml_csv):
        return {}
    
    df = pd.read_csv(ml_csv)
    
    chart_data = {}
    
    # 特征重要性排名（柱状图数据）
    feature_data = []
    for _, row in df.head(8).iterrows():
        feature_data.append({
            'feature': str(row['Feature']),
            'xgb_score': float(row.get('XGBoost', 0)),
            'rf_score': float(row.get('RandomForest', 0)),
            'mi_score': float(row.get('MutualInfo', 0)),
            'total_score': float(row.get('Total_Score', 0))
        })
    
    chart_data['feature_importance'] = {
        'type': 'bar_chart',
        'title': '工艺参数重要性综合排名',
        'data': feature_data,
        'top3': feature_data[:3] if len(feature_data) >= 3 else feature_data
    }
    
    # 决策树阈值（从特征重要性推断，实际应该从决策树模型提取）
    if len(feature_data) >= 2:
        top_feature = feature_data[0]['feature']
        chart_data['decision_tree_threshold'] = {
            'type': 'threshold',
            'title': '决策树阈值规则',
            'data': {
                'primary_feature': top_feature,
                'threshold_note': '需要从实际决策树模型提取具体阈值'
            }
        }
    
    return chart_data

def extract_position_chart_data(base_dir: str) -> Dict[str, Any]:
    """从Position分析中提取图表数据"""
    data_file = CLEANED_DATA_FILE
    if not os.path.exists(data_file):
        return {}
    
    df = pd.read_csv(data_file)
    
    if 'Position_Code' not in df.columns:
        return {}
    
    target_col = next((c for c in df.columns if '压连' in c), None)
    if not target_col:
        return {}
    
    # 确保Is_Pass列存在
    if 'Is_Pass' not in df.columns:
        def encode_yield(x):
            try:
                v = int(float(x))
            except:
                return 0
            return 1 if v in [0, 1] else 0
        df['Is_Pass'] = df[target_col].apply(encode_yield)
    
    chart_data = {}
    
    # 位置良率排行
    pos_stats = df.groupby('Position_Code')['Is_Pass'].agg(['mean', 'count']).reset_index()
    pos_stats.columns = ['Position_Code', 'Yield_Rate', 'Count']
    pos_stats = pos_stats.sort_values('Yield_Rate', ascending=True)
    
    position_yield_data = []
    for _, row in pos_stats.iterrows():
        position_yield_data.append({
            'position': str(row['Position_Code']),
            'yield_rate': float(row['Yield_Rate']),
            'count': int(row['Count'])
        })
    
    chart_data['position_yield_ranking'] = {
        'type': 'bar_chart',
        'title': '位置良率排行',
        'data': position_yield_data,
        'worst': position_yield_data[0] if position_yield_data else None,
        'best': position_yield_data[-1] if position_yield_data else None
    }
    
    # 位置缺陷详情（堆叠柱状图）
    status_map = {-1: '虚焊', 0: '正常', 1: '轻微压连', 2: '严重压连'}
    position_failure_data = []
    
    for pos in df['Position_Code'].dropna().unique():
        pos_df = df[df['Position_Code'] == pos]
        failure_counts = {}
        for val in pos_df[target_col].dropna():
            try:
                v = int(float(val))
                status_name = status_map.get(v, '其他')
                failure_counts[status_name] = failure_counts.get(status_name, 0) + 1
            except:
                pass
        
        position_failure_data.append({
            'position': str(pos),
            'failure_distribution': failure_counts
        })
    
    chart_data['position_failure_detail'] = {
        'type': 'stacked_bar',
        'title': '位置缺陷详情',
        'data': position_failure_data
    }
    
    # 位置物理特征分布
    if 'Total_Indium_Height' in df.columns:
        pos_physical_data = []
        for pos in df['Position_Code'].dropna().unique():
            pos_df = df[df['Position_Code'] == pos]
            height_data = pos_df['Total_Indium_Height'].dropna()
            
            if len(height_data) > 0:
                pos_physical_data.append({
                    'position': str(pos),
                    'height_mean': float(height_data.mean()),
                    'height_median': float(height_data.median()),
                    'height_std': float(height_data.std()),
                    'count': len(height_data)
                })
        
        chart_data['position_physical_features'] = {
            'type': 'boxplot',
            'title': '位置物理特征分布',
            'data': pos_physical_data
        }
    
    return chart_data

def format_chart_data_for_llm(chart_data: Dict[str, Any]) -> str:
    """将图表数据格式化为LLM可读的文本"""
    text_parts = []
    
    # EDA数据
    if 'status_distribution' in chart_data:
        data = chart_data['status_distribution']
        text_parts.append(f"【生产状态分布】{data['summary']}")
    
    if 'correlation_matrix' in chart_data:
        data = chart_data['correlation_matrix']
        text_parts.append("【参数相关性分析】")
        for finding in data.get('key_findings', [])[:5]:
            text_parts.append(f"  - {finding}")
    
    if 'pass_fail_comparison' in chart_data:
        data = chart_data['pass_fail_comparison']
        pass_stats = data['data']['良品']
        fail_stats = data['data']['不良品']
        text_parts.append(f"【良次品总高度对比】良品中位数{pass_stats['median']:.2f}μm，不良品中位数{fail_stats['median']:.2f}μm，差异{pass_stats['median']-fail_stats['median']:.2f}μm")
    
    if 'position_yield' in chart_data:
        data = chart_data['position_yield']
        if data.get('worst'):
            text_parts.append(f"【位置良率】最低位置{data['worst']['position']}良率{data['worst']['yield_rate']*100:.1f}%，最高位置{data['best']['position']}良率{data['best']['yield_rate']*100:.1f}%")
    
    if 'height_drift' in chart_data:
        data = chart_data['height_drift']
        drift = data['data']['drift']
        text_parts.append(f"【高度漂移】早期均值{data['data']['early_mean']:.2f}μm，后期均值{data['data']['late_mean']:.2f}μm，漂移{drift:.2f}μm")
    
    # ML数据
    if 'feature_importance' in chart_data:
        data = chart_data['feature_importance']
        text_parts.append("【特征重要性排名】")
        for i, feat in enumerate(data.get('top3', []), 1):
            text_parts.append(f"  {i}. {feat['feature']}: 综合得分{feat['total_score']:.4f}")
    
    # Position数据
    if 'position_yield_ranking' in chart_data:
        data = chart_data['position_yield_ranking']
        if data.get('worst'):
            text_parts.append(f"【位置分析】最差位置{data['worst']['position']}良率{data['worst']['yield_rate']*100:.1f}%，最佳位置{data['best']['position']}良率{data['best']['yield_rate']*100:.1f}%")
    
    return "\n".join(text_parts)

def extract_all_chart_data(base_dir: str) -> str:
    """提取所有图表数据并格式化为LLM输入"""
    eda_data = extract_eda_chart_data(base_dir)
    ml_data = extract_ml_chart_data(base_dir)
    position_data = extract_position_chart_data(base_dir)
    
    all_data = {
        'eda': eda_data,
        'ml': ml_data,
        'position': position_data
    }
    
    # 格式化为文本
    formatted_text = format_chart_data_for_llm({**eda_data, **ml_data, **position_data})
    
    return formatted_text, all_data

if __name__ == "__main__":
    base_dir = ROOT_DIR
    text, data = extract_all_chart_data(base_dir)
    print("=" * 60)
    print("提取的图表数据（LLM可读格式）")
    print("=" * 60)
    print(text)
    print("\n" + "=" * 60)
    print("原始数据结构（JSON格式）")
    print("=" * 60)
    import json
    print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
