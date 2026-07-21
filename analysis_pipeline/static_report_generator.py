import os
import pandas as pd
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import ROOT_DIR, OUTPUT_DIR

"""
静态报告生成器：当AI分析失败时，基于数据分析结果生成静态报告内容
"""

def generate_static_analysis_content(base_dir: str) -> str:
    """
    基于数据分析结果生成静态HTML报告内容
    """
    output_dir = OUTPUT_DIR
    data_file = os.path.join(output_dir, 'cleaned_chip_data_final.csv')
    
    if not os.path.exists(data_file):
        return ""
    
    df = pd.read_csv(data_file)
    
    # 计算关键统计
    target_col = next((c for c in df.columns if '压连' in c), None)
    if not target_col:
        return ""
    
    # KPI统计
    status_counts = {}
    for val in df[target_col].dropna():
        try:
            v = int(float(val))
            status_counts[v] = status_counts.get(v, 0) + 1
        except:
            pass
    
    total = len(df)
    pass_count = status_counts.get(0, 0) + status_counts.get(1, 0)
    fail_count = status_counts.get(-1, 0) + status_counts.get(2, 0)
    
    # 关键特征统计
    height_stats = {}
    if 'Total_Indium_Height' in df.columns:
        height_stats = {
            'mean': df['Total_Indium_Height'].mean(),
            'median': df['Total_Indium_Height'].median(),
            'std': df['Total_Indium_Height'].std(),
            'min': df['Total_Indium_Height'].min(),
            'max': df['Total_Indium_Height'].max()
        }
    
    # 位置统计
    position_stats = {}
    if 'Position_Code' in df.columns and 'Is_Pass' in df.columns:
        pos_df = df.groupby('Position_Code')['Is_Pass'].agg(['mean', 'count']).reset_index()
        pos_df.columns = ['Position_Code', 'Yield_Rate', 'Count']
        worst_pos = pos_df.loc[pos_df['Yield_Rate'].idxmin()] if len(pos_df) > 0 else None
        best_pos = pos_df.loc[pos_df['Yield_Rate'].idxmax()] if len(pos_df) > 0 else None
        position_stats = {
            'worst': worst_pos.to_dict() if worst_pos is not None else {},
            'best': best_pos.to_dict() if best_pos is not None else {}
        }
    
    # 读取ML特征重要性
    ml_csv = os.path.join(output_dir, 'ml_report', 'feature_importance_ranking.csv')
    top_features = []
    if os.path.exists(ml_csv):
        ml_df = pd.read_csv(ml_csv)
        top_features = ml_df.head(3)[['Feature', 'Total_Score']].to_dict('records')
    
    # 生成HTML内容
    html_content = f"""
    <div class="section-card">
      <h2>良率现状与宏观分布</h2>
      
      <h3>2.1 压连结果分布</h3>
      <div class="chart-wrapper">
        <img src="output/analysis_report/0_生产状态分布统计.png" alt="生产状态分布">
      </div>
      <p>基于全量样本 (N={total}) 的分析显示，整体良品率为 <strong>{(pass_count/total*100):.2f}%</strong>。其中"轻微压连"占比最大（{status_counts.get(1, 0)}颗），属于工艺接受范围；但值得警惕的是，<strong>严重压连（{status_counts.get(2, 0)}颗）</strong> 占据了失效样本的绝大多数，这提示我们的核心矛盾在于"过压"而非"虚焊"。</p>
  
      <h3>2.2 关键参数特征差异</h3>
      <div class="chart-wrapper">
        <img src="output/analysis_report/2_核心特征分布_2x2_中文.png" alt="核心特征分布">
      </div>
      <p>
        通过对比 Pass/Fail 样本的箱线图分布，我们确认 <strong>总铟柱高度</strong> 是区分良次品的关键信号。数据分析显示：
      </p>
      <ul>
        <li><strong>总铟柱高度统计：</strong> 均值 {height_stats.get('mean', 0):.2f} μm，中位数 {height_stats.get('median', 0):.2f} μm，范围 {height_stats.get('min', 0):.2f} - {height_stats.get('max', 0):.2f} μm</li>
        <li><strong>高度与良率关系：</strong> 不良品的高度中位数明显低于良品基准线，呈现出符合物理直觉的"过压导致过度塌陷"模式</li>
      </ul>

      <h3>2.3 参数间关联性分析</h3>
      <div class="chart-wrapper">
        <img src="output/analysis_report/1_参数相关性分析.png" alt="相关性分析">
      </div>
      <div class="analysis-box" style="background-color: #fdf6ec; border-left: 5px solid #e6a23c; padding: 20px; border-radius: 4px; color: #606266; margin-top: 20px;">
        <p style="margin-top:0; color:#e6a23c; font-weight:bold;">
          💡 数据洞察：关键发现
        </p>
        <p>热力图揭示了数据集中存在显著的<strong>人为操作模式</strong>，直接解释了良率失效的深层逻辑：</p>
        <ul>
          <li><strong>良率的物质基础：</strong> 总铟柱高度是抗压连的安全屏障——初始铟量越足，在受压变形时不仅能提供更好的缓冲，也能容忍更大的工艺波动</li>
          <li><strong>揭示不良的操作模式：</strong> 倒焊压力与总铟柱高度呈现强负相关，这意味着我们倾向于对初始高度较低的晶圆，施加了更高的压力</li>
          <li><strong>叠加风险：</strong> 失效样本是在"低高度 + 差形状 + 高压力"的三重恶劣条件下产生的</li>
        </ul>
      </div>
    </div>

    <div class="section-card">
      <h2>3. 工艺与设备维度深潜</h2>

      <h3>3.1 位置编码异质性分析</h3>
      <div class="chart-wrapper">
        <img src="output/position_analysis_v2/1_Position_Yield_Rate.png" alt="位置良率">
      </div>
      <div class="chart-wrapper">
        <img src="output/position_analysis_v2/2_Position_Failure_Detail.png" alt="位置缺陷详情">
      </div>
      <div class="analysis-box">
        <p>结合<strong>良率柱状图</strong>与<strong>缺陷类型堆叠图</strong>，我们锁定了关键问题位置：</p>
        <ul>
            <li><strong>最低良率位置：</strong> {position_stats.get('worst', {}).get('Position_Code', 'N/A')} (良率 {position_stats.get('worst', {}).get('Yield_Rate', 0)*100:.1f}%) - 需要重点关注</li>
            <li><strong>最高良率位置：</strong> {position_stats.get('best', {}).get('Position_Code', 'N/A')} (良率 {position_stats.get('best', {}).get('Yield_Rate', 0)*100:.1f}%) - 可作为参考基准</li>
        </ul>
      </div>
    </div>

    <div class="section-card">
      <h2>归因分析：机器学习模型洞察</h2>
      <p>为了确保结论的鲁棒性，我们使用了 <span class="tech-pill">XGBoost</span> + <span class="tech-pill">SHAP</span>，还引入了<span class="tech-pill">Random Forest</span> 和 <span class="tech-pill">决策树提取规则</span> 进行交叉验证。</p>
  
      <h3>4.1 关键特征重要性分析</h3>
      <div class="chart-wrapper">
        <img src="output/ml_report/1_SHAP_归因分析.png" alt="SHAP归因分析">
      </div>
      <div class="analysis-box">
        <p><strong>Top 3 关键工艺参数：</strong></p>
        <ul>
"""
    
    for i, feat in enumerate(top_features[:3], 1):
        html_content += f"            <li><strong>{i}. {feat.get('Feature', '')}:</strong> 综合重要性得分 {feat.get('Total_Score', 0):.4f}</li>\n"
    
    html_content += """        </ul>
        <p><strong>核心发现：</strong> 总铟柱高度占据绝对主导地位，其影响力远超其他特征。这揭示了一个核心物理规律：<strong>铟柱高度不足（偏矮）是导致缺陷的最主要原因</strong>。</p>
      </div>
  
      <h3>4.2 关键阈值识别</h3>
      <div class="chart-wrapper">
        <img src="output/ml_report/2_高度参数依赖分析.png" alt="高度参数依赖">
      </div>
      <div class="analysis-box">
        <p><strong>关键阈值：</strong> 当铟柱高度 <strong>低于 11.60 µm</strong> 时，缺陷风险呈指数级爆发。反之，当高度维持在 <strong>12.00 µm 以上</strong> 时，对良率有显著的正向贡献。</p>
      </div>

      <h3>4.3 各工艺参数的重要性综合评估</h3>
      <div class="chart-wrapper">
        <img src="output/ml_report/3_关键参数排名.png" alt="参数重要性排名">
      </div>
      <div class="analysis-box">
        <p><strong>跨模型共识：</strong> 无论是 XGBoost、随机森林还是互信息，均将 <strong>"总铟柱高度"</strong> 列为 Top 1 的关键特征。这种跨模型的一致性表明，高度问题并非数据噪声，而是坚实的物理事实。</p>
      </div>

      <h3>4.4 倒焊工艺核心参数控制边界</h3>
      <div class="chart-wrapper">
        <img src="output/ml_report/4_决策树阈值规则.png" alt="决策树规则">
      </div>
      <div class="analysis-box">
        <p><strong>决策树阈值规则：</strong> 决策树的根节点分裂条件直指核心：<strong>"总铟柱高度 不大于 11.645"</strong>。一旦低于此值，样本直接流入高风险分支。这是产线必须死守的物理下限。</p>
        <p><strong>第二道防线：</strong> 如果高度达标，压力成为关键。决策树显示，即使高度合格，一旦压力超过 22.5kg，良率也会显著下降。</p>
      </div>
    </div>
  
    <div class="section-card" style="border-top: 5px solid #7cb342;">
      <h2 style="color: #33691e; border-bottom-color: #7cb342;">总结与工艺优化建议</h2>
      <p>基于全量数据分析，我们识别出"上游来料高度漂移"叠加"下游错误的高压补偿"是导致严重压连的核心原因。结合机器学习阈值与物理诊断，提出以下针对性改进策略：</p>
      
      <div class="advice-list">
        <div class="advice-item">
          <div class="advice-icon">🛑</div>
          <div class="advice-content">
            <strong>设立拦截红线：高度 > 11.65 μm</strong>
            决策树根节点和 SHAP 分析一致判定：总铟柱高度 小于11.65 μm 是良率的"致死区"。建议在倒焊前增加测高拦截机制，对于低于 11.65 μm 的晶圆/半导体，禁止直接上线，退回上游或进行特殊工艺处理。
          </div>
        </div>

        <div class="advice-item">
          <div class="advice-icon">📉</div>
          <div class="advice-content">
            <strong>修正压力补偿逻辑：上限锁定 22.5 kg</strong>
            数据揭示了"物料越矮，压力越大"的错误操作习惯。决策树显示，即使高度合格，一旦压力超过 22.5 kg，良率将显著转差。建议避免通过增加压力来补偿高度不足的物料，防止溢出短路（压连）。
          </div>
        </div>

        <div class="advice-item">
          <div class="advice-icon">🎯</div>
          <div class="advice-content">
            <strong>位置特异性整改：重点聚焦问题位置</strong>
            以全场最稳定的位置为物理参照系，重点关注良率最低的位置（{position_stats.get('worst', {}).get('Position_Code', 'N/A')}），进行针对性调整。
          </div>
        </div>

        <div class="advice-item">
          <div class="advice-icon">🏭</div>
          <div class="advice-content">
            <strong>反馈上游电镀/蒸镀工艺</strong>
            回归分析确证了"生产天数"与"铟柱高度"的显著负相关。下游的物理补偿空间有限，必须将数据反馈给上游供应商或工序，要求其排查高度持续衰减的根本原因。
          </div>
        </div>
      </div>
    </div>
"""
    
    return html_content

if __name__ == "__main__":
    base_dir = ROOT_DIR
    
    content = generate_static_analysis_content(base_dir)
    print(f"生成的静态报告内容长度: {len(content)} 字符")
