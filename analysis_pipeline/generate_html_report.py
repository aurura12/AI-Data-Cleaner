import os
import json
import base64
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import (
    ROOT_DIR,
    OUTPUT_DIR,
    CLEANED_DATA_FILE,
    AI_CHART_RESULTS_FILE,
    AI_TEXT_RESULTS_FILE,
    HTML_REPORT_FILE,
)

# ==========================================
# 1. 核心工具函数
# ==========================================
def get_base64_image(image_path):
    """
    读取本地图片并转换为 Base64 字符串
    """
    if not os.path.exists(image_path):
        print(f"⚠️ 警告: 找不到图片 {image_path}，将保持原样或显示裂图")
        return image_path  # 返回原路径，避免替换出错
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

def calculate_kpi_from_data(data_file: str) -> dict:
    """从数据文件动态计算KPI指标"""
    try:
        df = pd.read_csv(data_file)
        
        # 查找压连情况列
        target_col = next((c for c in df.columns if '压连' in c), None)
        if not target_col:
            return {}
        
        # 计算各类状态数量
        status_counts = {}
        for val in df[target_col].dropna():
            try:
                v = int(float(val))
                if v not in status_counts:
                    status_counts[v] = 0
                status_counts[v] += 1
            except:
                pass
        
        total = len(df)
        pass_count = status_counts.get(0, 0) + status_counts.get(1, 0)  # 正常+轻微压连
        fail_count = status_counts.get(-1, 0) + status_counts.get(2, 0)  # 虚焊+严重压连
        
        pass_rate = (pass_count / total * 100) if total > 0 else 0
        open_rate = (status_counts.get(-1, 0) / total * 100) if total > 0 else 0
        severe_rate = (status_counts.get(2, 0) / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'pass_rate': pass_rate,
            'open_rate': open_rate,
            'severe_rate': severe_rate,
            'open_count': status_counts.get(-1, 0),
            'severe_count': status_counts.get(2, 0),
            'status_counts': status_counts
        }
    except Exception as e:
        print(f"⚠️ 计算KPI失败: {e}")
        return {}

def load_ai_analysis_results(results_file: str) -> dict:
    """加载AI分析结果"""
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"⚠️ 加载AI分析结果失败: {e}")
        return {}

# ==========================================
# 2. 定义 HTML 内容 (拼接你提供的三部分)
# ==========================================

# 第一部分：头部、CSS、KPI
part_1 = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>倒焊工艺良率预测与穿透分析报告</title>
  <style>
    /* 全局重置与字体 */
    :root {
      --primary-blue: #3498db;
      --dark-text: #2c3e50;
      --light-text: #7f8c8d;
      --bg-color: #eff2f7;
      --card-bg: #ffffff;
      --green-bg: #dff0d8;
      --green-border: #d6e9c6;
      --green-text: #3c763d;
      --accent-green: #e1f3d8; /* 用于建议条目的背景 */
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg-color);
      color: var(--dark-text);
      margin: 0;
      padding: 40px 20px;
      line-height: 1.6;
    }

    .container {
      max-width: 1100px;
      margin: 0 auto;
    }

    /* 标题区域 - 仿照参考图居中风格 */
    .report-header {
      text-align: center;
      margin-bottom: 50px;
    }
    .report-header h1 {
      font-size: 36px;
      color: var(--dark-text);
      margin-bottom: 10px;
      font-weight: 700;
      letter-spacing: 1px;
    }
    .report-header p {
      color: var(--light-text);
      font-size: 16px;
    }

    /* 通用卡片容器 */
    .section-card {
      background: var(--card-bg);
      border-radius: 8px;
      padding: 30px;
      margin-bottom: 30px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* 二级标题 - 仿照参考图带下划线 */
    h2 {
      font-size: 24px;
      color: var(--primary-blue);
      margin-top: 0;
      margin-bottom: 25px;
      padding-bottom: 15px;
      border-bottom: 2px solid var(--primary-blue);
      position: relative;
    }

    h3 {
      font-size: 18px;
      color: #444;
      margin-top: 30px;
      margin-bottom: 15px;
      font-weight: 600;
      border-left: 4px solid var(--primary-blue);
      padding-left: 10px;
    }

    /* KPI 核心指标 - 仿照参考图2布局 */
    .kpi-container {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 40px;
    }
    .kpi-card {
      flex: 1;
      background: #fff;
      border-radius: 12px;
      padding: 30px 20px;
      text-align: center;
      box-shadow: 0 4px 15px rgba(0,0,0,0.05);
      transition: transform 0.2s;
    }
    .kpi-card:hover {
      transform: translateY(-5px);
    }
    .kpi-title {
      font-size: 16px;
      color: var(--dark-text);
      font-weight: 600;
      margin-bottom: 15px;
    }
    .kpi-value {
      font-size: 42px;
      font-weight: bold;
      color: var(--primary-blue);
      margin-bottom: 10px;
    }
    .kpi-sub {
      font-size: 13px;
      color: var(--light-text);
    }

    /* 文本段落 */
    p {
      color: #555;
      font-size: 15px;
      margin-bottom: 15px;
    }

    /* 图片容器 - 模拟图表卡片 */
    .chart-wrapper {
      background: #fcfcfc;
      border: 1px solid #eee;
      border-radius: 8px;
      padding: 10px;
      margin: 20px 0;
      text-align: center;
    }
    img {
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }

    /* 结论与建议 - 深度复刻参考图3的绿色条状 */
    .advice-list {
      display: flex;
      flex-direction: column;
      gap: 15px;
    }

    .advice-item {
      background-color: #dcedc8; /* 浅绿色背景 */
      border-left: 5px solid #7cb342; /* 深绿色左侧边框 */
      color: #33691e;
      padding: 15px 20px;
      border-radius: 6px;
      display: flex;
      align-items: flex-start;
      font-size: 15px;
      line-height: 1.5;
    }

    .advice-icon {
      font-size: 20px;
      margin-right: 15px;
      margin-top: -2px; /* 微调图标对齐 */
      min-width: 24px;
    }

    .advice-content strong {
      display: block;
      margin-bottom: 4px;
      color: #2e5c18;
      font-size: 16px;
    }

    /* 机器/模型解释部分的特殊样式 */
    .tech-pill {
      display: inline-block;
      background: #e3f2fd;
      color: #1976d2;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      margin-right: 5px;
      margin-bottom: 5px;
    }

    /* 响应式调整 */
    @media (max-width: 768px) {
      .kpi-container {
        flex-direction: column;
      }
      .report-header h1 {
        font-size: 28px;
      }
    }
  </style>
</head>
<body>

  <div class="container">

    <!-- 顶部标题区 -->
    <header class="report-header">
      <h1>制造良率与工艺穿透分析报告</h1>
      <p>基于 半导体倒装焊接工艺全量数据的多维度参数关联性洞察 | 报告生成日期：REPORT_DATE_PLACEHOLDER</p>
    </header>
  
    <!-- 总体介绍 -->
    <div class="section-card">
      <h2>总体介绍</h2>
      <p>
        本报告聚焦于倒焊半导体生产过程中的良率表现与关键物理参数之间的内在关系。
        我们采集了包括 <strong>总铟柱高度</strong>、<strong>倒焊压力</strong>、
        以及 <strong>铟柱形貌</strong> 等上维度的工艺数据。
      </p>
      <p>
        本次分析旨在通过机器学习方法 (XGBoost/Random Forest) 穿透表层良率波动，
        识别导致“严重压连”与“虚焊”的核心驱动因素，并为产线提供基于物理阈值的精确控制建议。
      </p>
    </div>
  
    <!-- 核心指标概览 (KPIs) -->
    <div class="section-card" style="background: transparent; padding: 0; box-shadow: none;">
      <h2 style="border-bottom: none; margin-bottom: 10px; padding-left: 10px;">核心指标概览</h2>
      <p style="padding-left: 10px; margin-bottom: 25px; font-size: 14px; color: #666;">
          基于全量样本 (N=KPI_TOTAL_PLACEHOLDER) 的核心质量概况统计。
      </p>
      
      <div class="kpi-container">
        <div class="kpi-card">
          <div class="kpi-title">整体良品率</div>
          <div class="kpi-value">KPI_PASS_RATE_PLACEHOLDER%</div>
          <div class="kpi-sub">基准线，含轻微压连(1)与正常(0)</div>
        </div>
        
        <div class="kpi-card">
          <div class="kpi-title">虚焊失效率</div>
          <div class="kpi-value" style="color: #e74c3c;">KPI_OPEN_RATE_PLACEHOLDER%</div>
          <div class="kpi-sub">虚焊(-1) KPI_OPEN_COUNT_PLACEHOLDER颗｜风险：断路</div>
        </div>
        
        <div class="kpi-card">
          <div class="kpi-title">严重压连率</div>
          <div class="kpi-value" style="color: #f39c12;">KPI_SEVERE_RATE_PLACEHOLDER%</div>
          <div class="kpi-sub">严重压连(2) KPI_SEVERE_COUNT_PLACEHOLDER颗｜风险：短路</div>
        </div>
      </div>
    </div>
"""

# 第二部分：现状与相关性
part_2 = r"""
    <div class="section-card">
      <h2>良率现状与宏观分布</h2>
      
      <h3>2.1 压连结果分布</h3>
      <div class="chart-wrapper">
        <img src="output/analysis_report/0_生产状态分布统计.png" alt="图表: 生产状态分布">
      </div>
      <p>上图清晰展示了四类生产状态的分布。其中“轻微压连”占比最大（140颗），属于工艺接受范围；但值得警惕的是，<strong>严重压连（29颗）</strong> 占据了失效样本的绝大多数，这提示我们的核心矛盾在于“过压”而非“虚焊”。</p>
  
      <h3>2.2 关键参数特征差异</h3>
      <div class="chart-wrapper">
        <img src="output/analysis_report/2_核心特征分布_2x2_中文.png" alt="图表: 核心特征分布">
      </div>
      <p>
        通过对比 Pass/Fail 样本的箱线图分布，我们确认 <strong>倒焊压力 (Pressure)</strong> 是区分良次品的最强信号。不良品的压力分布显著上移，核心箱体集中在 <strong>22kg 至 24kg</strong> 的高压区间，而良品则保有更宽的低压操作窗口。这一高压特征在 <strong>总铟柱高度 (Total Height)</strong> 上得到了物理印证：不良品的高度中位数明显低于良品基准线 (12.00 µm)，呈现出符合物理直觉的“过压导致过度塌陷”模式。
      </p>
      <p>
          此外，<strong>铟柱形状异常分数 (Z-Score)</strong> 揭示了潜在的工艺偏离方向：良品中位数维持在正值 (0.09) 附近，而不良品的中位数显著下探至负值区间。这种<strong>分布重心的偏移</strong>比单纯的离散性更具指示意义，暗示特定类型的形状畸变是潜在风险点。<strong>电路端平整度</strong>在两类样本中分布几乎重叠（中位数均为 0.25），确认该特征在当前数据集中不具备区分能力。
      </p>

       <!-- 新增 3.3 小节 -->
       <h3>2.3 参数间关联性与“实验设计复盘”</h3>
       <div class="chart-wrapper">
         <img src="output/analysis_report/1_参数相关性分析.png" alt="图表: 相关性分析">
       </div>
       
       <!-- 这里增加内联样式，替代缺失的 .analysis-box 类 -->
       <div class="analysis-box" style="background-color: #fdf6ec; border-left: 5px solid #e6a23c; padding: 20px; border-radius: 4px; color: #606266; margin-top: 20px;">
         <p style="margin-top:0; color:#e6a23c; font-weight:bold;">
           💡 数据洞察：存在显著的采样偏差
         </p>
         <p>
           热力图揭示了数据集中存在显著的<strong>人为操作模式</strong>，直接解释了良率失效的深层逻辑：
         </p>
         <ul>
           <li>
             <strong>良率的物质基础：</strong> 
             良率(Pass=1) 与 <strong>总铟柱高度 (0.30)</strong> 呈正相关。这说明<strong>来料的初始高度</strong>是抗压连的安全屏障——初始铟量越足，在受压变形时不仅能提供更好的缓冲，也能容忍更大的工艺波动，从而避免焊料溢出导致短路。
           </li>
           <li>
             <strong>揭示不良的操作模式：</strong> 
             <strong>倒焊压力</strong> 与 <strong>总铟柱高度</strong> 竟呈现出强负相关 <strong>(-0.52)</strong>。这意味着我们倾向于对<strong>初始高度较低</strong>的晶圆，施加了<strong>更高的压力</strong>。
           </li>
           <li>
             <strong>叠加风险：</strong> 
             同样的偏差也出现在形状参数上（压力 vs 锥度 -0.48）。这表明这批数据中的失效样本，是在“<strong>低高度 + 差形状 + 高压力</strong>”的三重恶劣条件下产生的。这提示后续生产必须建立<strong>前馈控制</strong>：当检测到本来料高度不足或形状偏差时，应自动<i>降低</i>压力设定或报警，而非盲目加压。
           </li>
         </ul>
       </div>
    </div>
"""

# 第三部分：工艺深潜、ML、结论与结尾
part_3 = r"""
    <div class="section-card">
      <h2>3. 工艺与设备维度深潜</h2>

      <h3>3.1 位置编码异质性分析：以 M5 为基准的偏差诊断</h3>
      <div class="chart-wrapper">
        <img src="output/position_analysis_v2/1_Position_Yield_Rate.png" alt="图表: 空间效应良率">
      </div>
      <div class="chart-wrapper">
        <img src="output/position_analysis_v2/2_Position_Failure_Detail.png" alt="图表: 空间效应缺陷详情">
      </div>

      <div class="analysis-box">
        <p>结合<strong>良率柱状图</strong>与<strong>缺陷类型堆叠图</strong>，我们锁定了即使在看似正常的生产中也存在的严重短板：</p>
        
        <ul>
            <li>
                <strong>M7 (良率崩塌点 - 57.1%)：</strong><br>
                M7 是全场唯一的绝对故障点。堆叠图中最上方的<strong>红色区域（失效-1/2）占比极高</strong>。结合物理图表看，M7 的高度分布并未像良率那样表现出极端的离散，这种“参数正常但结果失效”的背离，强烈暗示该机台存在<strong>非形变类的硬件故障</strong>。
            </li>
            <li>
                <strong>M8 (系统性次优 - 75.0%)：</strong><br>
                M8 是除 M7 外唯一的显著低良率点。其缺陷构成中包含了大量的 Warning（绿色区域）和 Fail（红色区域），表明该位置的工艺窗口偏离了中心值。
            </li>
            <li>
                <strong>M2 & M4 (边缘合格品风险)：</strong><br>
                虽然这两者的良率数字尚可（>88%），但观察堆叠图发现，代表完美连接的<strong>“浅绿色 Type 0”占比极低</strong>，产品主要由“勉强合格”的深绿色 Type 1 构成。这意味着它们长期运行在规格下限边缘，抗风险能力远低于 M5。
            </li>
        </ul>
      </div>

      <div class="chart-wrapper">
        <img src="output/position_analysis_v2/3_Position_Physical_Features.png" alt="图表: 物理一致性特征">
      </div>
      <div class="analysis-box">
        <p>本图表以<strong>M5（中位红线附近的矮扁箱体）</strong>为稳定性标杆，揭示了其他位置在物理层面上的三种典型偏差：</p>
        
        <ul>
            <li>
                <strong>M8：系统性高度偏移</strong><br>
                与 M5 相比，M8 的箱体形态（大小/扁平度）非常相似，说明其机械重复精度很高。但其<strong>整体位置出现显著的“向上平移”</strong>（中位数从 11.76 漂移至 ~12.00）。这证实 M8 的低良率并非源于震动，而是单纯的 <strong>高度设定偏差</strong>。
            </li>
            <li>
                <strong>M3 & M9：一致性失控</strong><br>
                与 M5 收敛的箱体形成鲜明对比，M3 和 M9 的<strong>箱体垂直拉伸极长</strong>（上下须跨度大）。这代表在这些位置生产的半导体“忽高忽低”，极度不稳定。虽然 M3/M9 目前良率尚可（>94%），但这种物理层面的问题是巨大的潜在雷区。
            </li>
            <li>
                <strong>M6：形状异常</strong><br>
                在上方的 Z-Score 锥度分布中，M6 的箱体上边缘延伸最高。这意味着相比于 M5，M6 生产的铟柱<strong>形状偏尖或存在异物</strong>。尽管目前未造成大面积良率下降，但其形状变异度已接近临界值。
            </li>
        </ul>
      </div>
    </div>

    <!-- 机器学习部分 -->
    <div class="section-card">
      <h2>归因分析：机器学习模型洞察</h2>
      <p>
        为了确保结论的鲁棒性，我们不仅使用了 <span class="tech-pill">XGBoost</span> + <span class="tech-pill">SHAP</span>，还引入了<span class="tech-pill">Random Forest</span> 和 <span class="tech-pill">决策树提取规则</span> 进行交叉验证。
      </p>
  
      <h3>4.1 SHAP 全局解释</h3>
      <div class="chart-wrapper">
        <img src="output/ml_report/1_SHAP_归因分析.png" alt="SHAP 归因分析要点">
      </div>
      <div class="analysis-box">
        <p>
          <strong>1. 高度因素占据绝对主导（Total Height）：</strong> 
          SHAP 概要图显示，“总铟柱高度”不仅排在第一位，且其影响力远超其他特征。<strong>注意颜色分布：</strong>代表低数值的<strong>蓝色点</strong>大量集中在X轴右侧（SHAP值>0，高风险区），而红点集中在左侧。这揭示了一个核心物理规律：<strong>铟柱高度不足（偏矮）是导致缺陷的最主要原因</strong>。
        </p>
        <p>
          <strong>2. 设备老化的隐形成本（Production Days）：</strong> 
          排名第三的“生产天数”显示出明显的风险倾向：<strong>红色点（后期天数）</strong>主要分布在右侧风险区。这意味着随着设备运行时间的推移（设备漂移/老化），即使参数未变，产生缺陷的固有概率也在增加，需缩短维护周期。
        </p>
        <p>
          <strong>3. 多维度的次要影响：</strong> 
          “位置编码”和“形状异常度(Z)”虽然紧随其后，但其SHAP值分布不如高度集中，说明它们更多是作为局部扰动因子存在，而非全局决定性因素。
        </p>
      </div>
  
      <h3>4.2 关键阈值识别</h3>
      <div class="chart-wrapper">
        <img src="output/ml_report/2_高度参数依赖分析.png" alt="高度参数依赖分析">
      </div>
      <div class="analysis-box">
        <p>
          <strong>1. 显著的负相关趋势：</strong>
          观察“总铟柱高度”的偏依赖图，我们发现 SHAP 值随高度增加呈近似线性的下降趋势。这再次印证了“越高越安全”的结论。
        </p>
        <p>
          <strong>2. 致命的“低矮”区间：</strong>
          图表清晰地划定了一条红色警戒线：当铟柱高度 <strong>低于 11.60 µm</strong> 时，SHAP 值急剧上升至 +1.0 以上，缺陷风险呈指数级爆发。反之，当高度维持在 <strong>12.00 µm 以上</strong> 时，SHAP 值稳定为负，对良率有显著的正向贡献。
        </p>
      </div>
      

      <h3>4.3 各工艺参数的重要性综合评估</h3>
      <div class="chart-wrapper">
        <img src="output/ml_report/3_关键参数排名.png" alt="关键参数重要性排名">
      </div>
      <div class="analysis-box">
        <p>
          <strong>1. 三大模型的高度共识：</strong> 
          无论是 XGBoost（紫色）、随机森林（绿色）还是互信息（黄色），均将 <strong>“总铟柱高度”</strong> 列为 Top 1 的关键特征。这种跨模型的一致性表明，高度问题并非数据噪声，而是坚实的物理事实。
        </p>
        <p>
          <strong>2. 算法视角的分歧与互补：</strong> 
          有趣的是，XGBoost 给予了 <strong>“倒焊压力”</strong> 极高的权重（第二名），认为压力控制是决定性的物理变量；而随机森林则更看重 <strong>“生产天数”</strong>（设备状态）。这提示我们，在管控高度的同时，不能忽视压力的设定以及设备随时间的周期性校准。
        </p>
      </div>

      <h3>4.4 倒焊工艺核心参数控制边界</h3>
      <div class="chart-wrapper">
        <img src="output/ml_report/4_决策树阈值规则.png" alt="决策树阈值规则">
      </div>
      <div class="analysis-box">
        <p>
          <strong>1. 11.645 µm：</strong> 
          决策树的根节点分裂条件直指核心：<strong>“总铟柱高度 不大于 11.645”</strong>。一旦低于此值，样本直接流入左侧高风险分支，大部分被判定为“缺陷风险(Fail)”。这是产线必须死守的物理下限。
        </p>
        <p>
          <strong>2. 如果高度达标，压力成为关键：</strong> 
          观察右侧分支（False，即高度 > 11.645），虽然高度合格，但风险并未完全消除。此时模型引入了第二道防线：<strong>“倒焊压力 不大于 22.5 kg”</strong>。如果压力超过 22.5kg，即使铟柱高度足够，良率也会显著下降（进入右侧蓝色Fail节点）。
        </p>
      </div>
    </div>
  

    <!-- 总结与建议 - 绿色条状风格 -->
    <div class="section-card" style="border-top: 5px solid #7cb342;">
      <h2 style="color: #33691e; border-bottom-color: #7cb342;">总结与工艺优化建议</h2>
      <p>基于全量数据分析，我们识别出“上游来料高度漂移”叠加“下游错误的高压补偿”是导致严重压连的核心原因。结合机器学习阈值与物理诊断，提出以下五条针对性改进策略：</p>
      
      <div class="advice-list">
        
        <!-- 建议 1: IQC 拦截 (基于 ML 决策树根节点) -->
        <div class="advice-item">
          <div class="advice-icon">🛑</div>
          <div class="advice-content">
            <strong>设立拦截红线：高度 > 11.65 μm</strong>
            决策树根节点和 SHAP 分析一致判定：总铟柱高度 小于11.65 μm 是良率的“致死区”。当前后期生产物料已跌破此线。建议在倒焊前增加测高拦截机制，对于低于 11.65 μm 的晶圆/半导体，禁止直接上线，退回上游或进行特殊工艺处理。
          </div>
        </div>

        <!-- 建议 2: 压力封顶 (基于 决策树第二层分支) -->
        <div class="advice-item">
          <div class="advice-icon">📉</div>
          <div class="advice-content">
            <strong>修正压力补偿逻辑：上限锁定 22.5 kg</strong>
            数据揭示了“物料越矮，压力越大”的错误操作习惯（相关性 -0.52）。决策树显示，即使高度合格，一旦压力超过 22.5 kg，良率将显著转差。建议修避免通过增加压力来补偿高度不足的物料，防止溢出短路（压连）。
          </div>
        </div>

        <!-- 建议 3: 空间靶向治理 (基于 4.3 位置箱线图) -->
        <div class="advice-item">
          <div class="advice-icon">🎯</div>
          <div class="advice-content">
            <strong>位置特异性整改：重点聚焦 M7、M3 与 M8</strong>
            <p style="margin-bottom: 8px; font-size: 0.95em;">
              以全场最稳定的M5为物理参照系，我们锁定了三种不同类型的异常位置：
            </p>
            <ul>
                <li><strong>M7 (良率崩溃点)：</strong> 
                  良率仅 57.1%（全场最低）。尽管其物理高度分布未见极端异常，但如此低的产出率强烈指向隐性硬件故障。
                </li>
                <li><strong>M3 (一致性失控)：</strong> 
                   与 M5 的扁平箱体相比，M3 的方框极长（高度跨度大）。这种极大的离散度说明 M3 存在严重的机械不稳定性（如震动或夹具松动），导致产品忽高忽低，不可预测。
                </li>
                <li><strong>M8 (系统性偏离)：</strong> 
                  M8 的箱体形态尚可，但整体位置比 M5 悬殊地“高出”一截中位数偏差约 0.2μm）。这种“稳定但偏高”的特征说明是参数设定问题，建议调整该位置的铟柱高度，将其基准线向 M5 拉齐。
                </li>
            </ul>
          </div>
        </div>

        <!-- 建议 4: 解决上游漂移  -->
        <div class="advice-item">
          <div class="advice-icon">🏭</div>
          <div class="advice-content">
            <strong>反馈上游电镀/蒸镀工艺</strong>
            回归分析确证了“生产天数”与“铟柱高度”的显著负相关。下游的物理补偿空间有限，必须将数据反馈给上游供应商或工序，要求其排查自9月以来高度持续衰减的根本原因，恢复至 >12.25μm 的健康水平。
          </div>
        </div>

        <!-- 建议 5: 中段维护 -->
        <div class="advice-item">
          <div class="advice-icon">🧹</div>
          <div class="advice-content">
            <strong>引入“Wafer 5 后”的自动清洁机制</strong>
            针对晶圆批次效应中出现的“中段塌陷”（Wafer 6 良率最低）现象，建议在连续生产每 5 片晶圆后，执行一次快速校准程序，以消除累积误差，填补良率谷底。
          </div>
        </div>
        
      </div>
    </div>
  </div>
</body> <!-- ✅ 补全：闭合 body -->
</html> <!-- ✅ 补全：闭合 html -->

"""

# ==========================================
# 3. 合并与图片替换逻辑
# ==========================================
print("🔄 正在生成完整 HTML 报告...")

base_dir = ROOT_DIR

# 计算KPI（动态）
data_file = CLEANED_DATA_FILE
kpi_data = calculate_kpi_from_data(data_file)
if not kpi_data:
    # 使用默认值
    kpi_data = {
        'total': 210,
        'pass_rate': 85.71,
        'open_rate': 0.48,
        'severe_rate': 13.81,
        'open_count': 1,
        'severe_count': 29
    }

# 加载AI分析结果（优先视觉分析，其次文本分析）
ai_results_file = AI_CHART_RESULTS_FILE
ai_analysis = load_ai_analysis_results(ai_results_file)

# 如果视觉分析结果不存在或为空，尝试加载文本分析结果
if not ai_analysis or 'comprehensive_report' not in ai_analysis or not ai_analysis.get('comprehensive_report'):
    text_results_file = AI_TEXT_RESULTS_FILE
    text_analysis = load_ai_analysis_results(text_results_file)
    if text_analysis and 'comprehensive_report' in text_analysis and text_analysis.get('comprehensive_report'):
        print("✅ 使用文本分析结果")
        ai_analysis = text_analysis

# 准备替换变量
current_date = datetime.now().strftime('%Y-%m-%d')
replacements = {
    'REPORT_DATE_PLACEHOLDER': current_date,
    'KPI_TOTAL_PLACEHOLDER': str(kpi_data.get('total', 210)),
    'KPI_PASS_RATE_PLACEHOLDER': f"{kpi_data.get('pass_rate', 85.71):.2f}",
    'KPI_OPEN_RATE_PLACEHOLDER': f"{kpi_data.get('open_rate', 0.48):.2f}",
    'KPI_SEVERE_RATE_PLACEHOLDER': f"{kpi_data.get('severe_rate', 13.81):.2f}",
    'KPI_OPEN_COUNT_PLACEHOLDER': str(kpi_data.get('open_count', 1)),
    'KPI_SEVERE_COUNT_PLACEHOLDER': str(kpi_data.get('severe_count', 29)),
}

# 1. 拼接原始字符串
full_html = part_1 + part_2 + part_3

# 替换KPI变量
for key, value in replacements.items():
    full_html = full_html.replace(key, value)

# 如果有AI分析结果，替换分析内容
ai_content_used = False
if ai_analysis and 'comprehensive_report' in ai_analysis and ai_analysis.get('comprehensive_report'):
    print("✅ 检测到AI分析结果，将嵌入到报告中")
    ai_report = ai_analysis['comprehensive_report']
    # 尝试将AI生成的内容替换到part_2中（替换"良率现状与宏观分布"部分）
    # 由于HTML结构复杂，我们直接替换整个part_2
    if ai_report and len(ai_report) > 100:
        # 构建新的part_2，包含AI生成的内容（part_1已包含总体介绍，这里只替换分析内容）
        # 找到part_2的起始位置（"良率现状与宏观分布"），替换从那里开始的内容
        ai_section_start = '<div class="section-card">\n      <h2>良率现状与宏观分布</h2>'
        if ai_section_start in full_html:
            # 找到part_2的结束位置（在part_3之前）
            part_2_end_marker = '    </div>\n\n    <!-- 机器学习部分 -->'
            if part_2_end_marker in full_html:
                # 替换part_2部分
                before_part2 = full_html.split(ai_section_start)[0]
                after_part2 = full_html.split(part_2_end_marker)[1] if part_2_end_marker in full_html else part_3
                full_html = before_part2 + ai_report + '\n' + part_2_end_marker + after_part2
            else:
                # 如果找不到标记，直接替换整个part_2
                full_html = part_1 + ai_report + part_3
        else:
            # 如果找不到起始标记，直接替换整个part_2
            full_html = part_1 + ai_report + part_3
        ai_content_used = True
        print("✅ AI分析内容已嵌入报告")

# 如果AI分析失败，使用静态报告生成器
if not ai_content_used:
    print("⚠️ 未检测到AI分析结果，使用静态分析内容")
    try:
        from static_report_generator import generate_static_analysis_content
        static_content = generate_static_analysis_content(base_dir)
        if static_content and len(static_content) > 100:
            # 替换part_2为静态内容（part_1已包含总体介绍）
            ai_section_start = '<div class="section-card">\n      <h2>良率现状与宏观分布</h2>'
            if ai_section_start in full_html:
                part_2_end_marker = '    </div>\n\n    <!-- 机器学习部分 -->'
                if part_2_end_marker in full_html:
                    before_part2 = full_html.split(ai_section_start)[0]
                    after_part2 = full_html.split(part_2_end_marker)[1] if part_2_end_marker in full_html else part_3
                    full_html = before_part2 + static_content + '\n' + part_2_end_marker + after_part2
                else:
                    full_html = part_1 + static_content + part_3
            else:
                full_html = part_1 + static_content + part_3
            print("✅ 静态分析内容已嵌入报告")
        else:
            print("⚠️ 静态内容生成失败，使用默认报告内容")
    except Exception as e:
        print(f"⚠️ 静态内容生成异常: {e}，使用默认报告内容")

# 2. 建立图片路径映射列表 (HTML中的路径 -> 你的本地路径)
image_map = [
    "output/analysis_report/0_生产状态分布统计.png",
    "output/analysis_report/1_参数相关性分析.png",
    "output/position_analysis_v2/1_Position_Yield_Rate.png",
    "output/ml_report/1_SHAP_归因分析.png",
    "output/ml_report/2_高度参数依赖分析.png",
    "output/analysis_report/2_核心特征分布_2x2_中文.png",
    "output/position_analysis_v2/2_Position_Failure_Detail.png",
    "output/ml_report/3_关键参数排名.png",
    "output/analysis_report/3_周度趋势分析.png",
    "output/position_analysis_v2/3_Position_Physical_Features.png",
    "output/analysis_report/4_高度长期漂移.png",
    "output/ml_report/4_决策树阈值规则.png",
]

# 3. 循环替换图片为Base64
for img_path in image_map:
    full_img_path = os.path.join(ROOT_DIR, img_path)
    target_src = f'src="{img_path}"'
    
    print(f"   - 处理图片: {img_path}")
    base64_data = get_base64_image(full_img_path)
    
    new_src = f'src="{base64_data}"'
    full_html = full_html.replace(target_src, new_src)

# ==========================================
# 4. 保存文件
# ==========================================
output_filename = HTML_REPORT_FILE
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"\n✅ 成功生成独立报告文件：{output_filename}")
print("您可以直接双击打开，所有样式和图片都已完美嵌入。")