import os
import json
import base64
import requests
import subprocess
import time
import sys
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import ROOT_DIR, OUTPUT_DIR, EDA_REPORT_DIR, POSITION_REPORT_DIR
try:
    import ollama
    OLLAMA_LIB_AVAILABLE = True
except ImportError:
    OLLAMA_LIB_AVAILABLE = False
    print("⚠️ ollama库未安装，将使用requests API（可能需要base64编码）")

# ==========================================
# 1. 模型管理与选择
# ==========================================

def get_available_models() -> List[str]:
    """获取Ollama中可用的模型列表"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # 跳过表头
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]  # 第一列是模型名
                    models.append(model_name)
            return models
        return []
    except Exception as e:
        print(f"⚠️ 获取模型列表失败: {e}")
        return []

def select_vision_model(models: List[str]) -> Optional[str]:
    """从模型列表中选择支持视觉的模型"""
    # 优先级：qwen3-vl > qwen2.5-vl > qwen-vl
    vision_keywords = ['qwen3-vl', 'qwen2.5-vl', 'qwen-vl', 'vl']
    
    # 优先选择qwen3-vl
    for model in models:
        if 'qwen3-vl' in model.lower():
            return model
    
    # 其次选择其他视觉模型
    for keyword in vision_keywords:
        for model in models:
            if keyword in model.lower():
                return model
    
    return None

# ==========================================
# 2. 图表收集与配置
# ==========================================

def collect_chart_files(base_dir: str) -> List[Dict[str, str]]:
    """
    收集所有需要分析的图表文件
    返回: [{path, type, description}, ...]
    """
    charts = []
    
    # 定义图表目录和类型映射
    # base_dir 应该是 LZR_Project，所以需要加上 base_function/output
    chart_configs = [
        {
            'dir': EDA_REPORT_DIR,
            'type': 'EDA分析',
            'files': [
                ('0_目标分布统计.png', '目标分布', '展示目标列的分布情况'),
                ('1_参数相关性分析.png', '参数相关性', '关键参数之间的相关性热力图'),
                ('2_核心特征分布_2x3.png', '核心特征分布', '合格/不合格在关键参数上的分布对比'),
                ('3_周度趋势分析.png', '周度趋势', '目标均值和产量的周度变化趋势'),
                ('4_特征漂移分析.png', '特征漂移', '数值特征随时间的变化趋势'),
                ('5_位置效应分析.png', '位置效应', '各位置的效应分析'),
            ]
        },
        # 注意：已移除ML分析部分，只保留EDA和位置分析
        {
            'dir': POSITION_REPORT_DIR,
            'type': '位置分析',
            'files': [
                ('1_Position_Yield_Rate.png', '位置排名', '各位置的均值对比'),
                ('2_Position_Pass_Fail_Ratio.png', '位置占比', '各位置的结果分布堆叠图'),
                ('3_Position_Physical_Features.png', '位置特征', '各位置在数值特征上的分布对比'),
            ]
        }
    ]
    
    for config in chart_configs:
        dir_path = config['dir']
        if not os.path.exists(dir_path):
            continue
            
        for filename, chart_name, description in config['files']:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                charts.append({
                    'path': file_path,
                    'type': config['type'],
                    'name': chart_name,
                    'description': description,
                    'filename': filename
                })
    
    return charts

# ==========================================
# 3. 图片编码
# ==========================================

def image_to_base64(image_path: str, max_size_mb: float = 5.0) -> str:
    """将图片转换为Base64编码，如果图片太大则压缩"""
    try:
        from PIL import Image
        import io
        
        # 读取原始图片
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # 如果图片太大，尝试压缩
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            print(f"   ⚠️ 图片较大 ({size_mb:.1f}MB)，尝试压缩...")
            try:
                img = Image.open(io.BytesIO(image_data))
                # 如果宽度超过2000，按比例缩小
                if img.width > 2000:
                    ratio = 2000 / img.width
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # 转换为字节
                output = io.BytesIO()
                img.save(output, format='PNG', optimize=True, quality=85)
                image_data = output.getvalue()
                print(f"   ✅ 压缩后大小: {len(image_data) / (1024 * 1024):.1f}MB")
            except Exception as e:
                print(f"   ⚠️ 压缩失败，使用原图: {e}")
        
        base64_str = base64.b64encode(image_data).decode('utf-8')
        return base64_str
    except Exception as e:
        print(f"⚠️ 图片编码失败 {image_path}: {e}")
        return ""

# ==========================================
# 4. Ollama API 调用
# ==========================================

def analyze_chart_with_ollama(
    image_path: str,
    chart_name: str,
    chart_description: str,
    model_name: str = "qwen3-vl:2b",
    ollama_url: str = "http://localhost:11434"
) -> Optional[str]:
    """
    使用Ollama视觉模型分析单张图表
    
    返回: AI分析结果文本
    """
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"⚠️ 图片文件不存在: {image_path}")
        return None
    
    # 读取并编码图片（用于requests API回退）
    base64_image = image_to_base64(image_path)
    if not base64_image:
        print(f"⚠️ 图片编码失败: {image_path}")
        return None
    
    # 构建提示词（使用用户提供的模板，确保语气委婉，数据根据实际情况变化）
    prompt = f"""你现在是一位半导体良率优化专家（Yield Optimization Expert）和机器学习工程师。

【任务目标】
分析这张工艺数据分析图表，识别可能与良率下降相关的物理参数特征和数据模式，** 为产线工程师提供决策参考和排查线索**（非绝对执行指令）。

【业务背景】

数据集：清洗后生产数据（cleaned_data.csv）
预测目标：二分类（{{-1虚焊, 2严重压连}} = Fail(0)；{{0正常, 1轻微压连}} = Pass(1)）
关键特征：
铟柱总高度（Total_Indium_Height）：核心特征，上下高度和
Calc_Circuit_Range：电路端平整度（仅激光调平工艺有此值，存在缺失）
Indium_Taper_Zscore：铟柱形状异常度
倒焊压力（kg）：工艺设定压力
Time_Seq_Day：连续生产天数（设备老化/漂移）
Wafer_Index：晶圆加工次序
Position_Code：空间位置（M1-M10）
【图表信息】

图表名称：{chart_name}
图表描述：{chart_description}
【分析要求】

识别图表类型（箱线图/折线图/柱状图/热力图等）。
提取统计特征（观察中位数、分布范围、离群点分布等）。
识别潜在模式（是否存在分布偏移、趋势变化或特定群体的表现差异）。
关联业务逻辑（分析哪些参数特征的变化趋势可能与Fail风险相关）。
提出参考建议：
避免使用绝对化语言（如"必须设定"、"禁止超过"）。
采用建议性话术（如"建议排查区间"、"风险可能显著增加"、"建议关注"）。
指出潜在的高风险参数区间或需要重点监控的异常点。
【输出格式要求】
请严格按照以下JSON格式输出，不要添加任何其他文字：
{{
"chart_type": "图表类型",
"key_findings": ["数据观察1", "数据观察2", "数据观察3"],
"key_values": {{"指标名": "数值", ...}},
"anomalies": "观测到的异常分布或离群特征（如有）",
"process_suggestions": "针对工艺参数的排查建议或优化方向（请使用'建议维持在'、'需关注'等柔性表述）",
"risk_level": "低/中/高（指数据波动带来的潜在风险程度）",
"detailed_analysis": "详细分析文本（2-3段，使用客观、统计学的口吻分析数据与其对良率的潜在影响）"
}}

请开始分析这张图表："""

    # 调用Ollama API（带重试机制）
    max_retries = 3
    for retry in range(max_retries):
        try:
            # 优先使用ollama库（更简单，自动处理图片路径）
            if OLLAMA_LIB_AVAILABLE:
                if retry > 0:
                    print(f"   🔄 重试第 {retry} 次...")
                print(f"   📤 使用ollama库调用模型（图片路径: {image_path}）...")
                
                # 检查图片文件是否存在
                if not os.path.exists(image_path):
                    print(f"   ⚠️ 图片文件不存在: {image_path}")
                    return None
                
                # 显式设置stream=False，确保非流式输出
                response = ollama.chat(
                    model=model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        # ollama库可以直接使用图片路径，不需要base64编码
                        'images': [image_path]
                    }],
                    stream=False,  # 显式设置为非流式，确保完整响应
                    options={
                        'temperature': 0.3,  # 降低温度提高稳定性
                        'num_predict': 8000,  # 增加输出长度限制
                        'num_ctx': 8192,  # 上下文长度
                        'thinking': False  # 禁用thinking模式，确保输出在content中
                    }
                )
                
                # ollama库返回的是ChatResponse对象，需要访问message.content
                # 格式: response.message.content
                if hasattr(response, 'message'):
                    msg = response.message
                    content = None
                    
                    # 优先使用content
                    if hasattr(msg, 'content'):
                        content = msg.content
                        if content and content.strip():
                            print(f"   ✅ 成功获取响应，长度: {len(content)} 字符")
                            return content
                    
                    # 如果content为空或不存在，检查thinking（某些模型可能将输出放在thinking中）
                    if hasattr(msg, 'thinking') and msg.thinking:
                        thinking = msg.thinking
                        if thinking and thinking.strip():
                            print(f"   ⚠️ content为空，从thinking中提取内容，长度: {len(thinking)} 字符")
                            # 尝试从thinking中提取JSON（如果prompt要求JSON输出）
                            import re
                            # 改进的正则表达式，支持嵌套的JSON对象
                            # 从后往前查找JSON（JSON通常在thinking末尾，但也可能在中间）
                            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                            json_matches = list(re.finditer(json_pattern, thinking, re.DOTALL))
                            
                            print(f"   📊 在thinking中找到 {len(json_matches)} 个可能的JSON对象")
                            
                            # 从后往前查找，优先匹配包含必需字段的JSON
                            for match in reversed(json_matches):
                                try:
                                    json_str = match.group(0)
                                    test_dict = json.loads(json_str)
                                    # 检查是否包含必需的字段
                                    if 'chart_type' in test_dict or 'key_findings' in test_dict:
                                        print(f"   ✅ 从thinking中提取到JSON（包含必需字段），位置: {match.start()}-{match.end()}")
                                        return json_str
                                    # 如果JSON对象有多个键，也可能是有效的
                                    elif len(test_dict) >= 3:
                                        print(f"   ✅ 从thinking中提取到JSON（多字段对象），位置: {match.start()}-{match.end()}")
                                        return json_str
                                    # 即使只有1-2个键，如果包含chart_type或key_findings，也接受
                                    elif 'chart_type' in test_dict or 'key_findings' in test_dict:
                                        print(f"   ✅ 从thinking中提取到JSON（包含必需字段，键数较少），位置: {match.start()}-{match.end()}")
                                        return json_str
                                except (json.JSONDecodeError, ValueError) as e:
                                    # 如果JSON解析失败，继续尝试下一个
                                    continue
                            
                            # 如果所有正则匹配都失败，打印调试信息
                            if json_matches:
                                print(f"   ⚠️ 找到 {len(json_matches)} 个匹配，但都无法解析为有效JSON")
                                # 打印最后一个匹配的内容，用于调试
                                if len(json_matches) > 0:
                                    last_match = json_matches[-1]
                                    print(f"   📋 最后一个匹配内容（前200字符）: {last_match.group(0)[:200]}")
                            
                            # 如果正则匹配都失败，尝试更宽松的匹配：查找最后一个完整的JSON对象
                            # 从后往前查找，找到第一个完整的{...}结构
                            last_brace = thinking.rfind('{')
                            if last_brace >= 0:
                                # 从最后一个{开始，尝试提取完整的JSON
                                potential_json = thinking[last_brace:]
                                # 先尝试找到第一个完整的}（简单情况）
                                first_close = potential_json.find('}')
                                if first_close > 0:
                                    simple_json = potential_json[:first_close+1]
                                    try:
                                        test_dict = json.loads(simple_json)
                                        if 'chart_type' in test_dict or 'key_findings' in test_dict or len(test_dict) >= 3:
                                            print(f"   ✅ 从thinking末尾提取到JSON（简单提取），使用JSON部分")
                                            return simple_json
                                    except:
                                        pass
                                
                                # 如果简单提取失败，尝试补全括号
                                open_braces = potential_json.count('{')
                                close_braces = potential_json.count('}')
                                if open_braces > close_braces:
                                    potential_json += '}' * (open_braces - close_braces)
                                    try:
                                        test_dict = json.loads(potential_json)
                                        if 'chart_type' in test_dict or 'key_findings' in test_dict or len(test_dict) >= 3:
                                            print(f"   ✅ 从thinking末尾提取到JSON（补全后），使用JSON部分")
                                            return potential_json
                                    except:
                                        pass
                            
                            # 如果没找到完整JSON，尝试查找最后一个可能的JSON片段（可能被截断）
                            # 查找最后一个 { 开始的位置
                            last_brace = thinking.rfind('{')
                            if last_brace >= 0:
                                # 尝试从最后一个 { 开始提取到文件末尾
                                potential_json = thinking[last_brace:]
                                # 先尝试找到第一个完整的}，如果找不到再补全
                                first_close = potential_json.find('}')
                                if first_close > 0:
                                    # 提取到第一个}为止
                                    simple_json = potential_json[:first_close+1]
                                    try:
                                        test_dict = json.loads(simple_json)
                                        if 'chart_type' in test_dict or 'key_findings' in test_dict or len(test_dict) >= 3:
                                            print(f"   ✅ 从thinking末尾提取到JSON（简单提取），使用JSON部分")
                                            return simple_json
                                    except:
                                        pass
                                
                                # 如果简单提取失败，尝试补全括号
                                open_braces = potential_json.count('{')
                                close_braces = potential_json.count('}')
                                if open_braces > close_braces:
                                    potential_json += '}' * (open_braces - close_braces)
                                    try:
                                        test_dict = json.loads(potential_json)
                                        if 'chart_type' in test_dict or 'key_findings' in test_dict or len(test_dict) >= 3:
                                            print(f"   ✅ 从thinking末尾提取到JSON（补全后），使用JSON部分")
                                            return potential_json
                                    except:
                                        pass
                            
                            # 如果正则匹配都失败，尝试查找包含关键字段的文本片段
                            # 查找包含chart_type或key_findings的文本区域
                            if '"chart_type"' in thinking or "'chart_type'" in thinking or '"key_findings"' in thinking or "'key_findings'" in thinking:
                                # 找到包含这些关键词的位置
                                keyword_pos = max(
                                    thinking.rfind('"chart_type"'),
                                    thinking.rfind("'chart_type'"),
                                    thinking.rfind('"key_findings"'),
                                    thinking.rfind("'key_findings'")
                                )
                                if keyword_pos >= 0:
                                    # 从关键词位置往前找最近的{，往后找最近的}
                                    start_pos = thinking.rfind('{', 0, keyword_pos)
                                    if start_pos >= 0:
                                        # 从{开始，尝试提取到文件末尾或找到完整的}
                                        potential_json = thinking[start_pos:]
                                        # 尝试找到第一个完整的}
                                        first_close = potential_json.find('}')
                                        if first_close > 0:
                                            simple_json = potential_json[:first_close+1]
                                            try:
                                                test_dict = json.loads(simple_json)
                                                if 'chart_type' in test_dict or 'key_findings' in test_dict:
                                                    print(f"   ✅ 从关键词位置提取到JSON（简单提取），使用JSON部分")
                                                    return simple_json
                                            except:
                                                # 如果简单提取失败，尝试补全括号
                                                open_braces = simple_json.count('{')
                                                close_braces = simple_json.count('}')
                                                if open_braces > close_braces:
                                                    simple_json += '}' * (open_braces - close_braces)
                                                    try:
                                                        test_dict = json.loads(simple_json)
                                                        if 'chart_type' in test_dict or 'key_findings' in test_dict:
                                                            print(f"   ✅ 从关键词位置提取到JSON（补全后），使用JSON部分")
                                                            return simple_json
                                                    except:
                                                        pass
                            
                            # 如果没有找到有效JSON，返回整个thinking内容（让后续代码处理）
                            print(f"   ⚠️ thinking中未找到有效JSON，返回整个thinking内容")
                            return thinking
                    
                    # 如果都为空，打印调试信息
                    print(f"   ⚠️ 响应内容为空")
                    print(f"      done_reason: {response.done_reason if hasattr(response, 'done_reason') else 'N/A'}")
                    print(f"      eval_count: {response.eval_count if hasattr(response, 'eval_count') else 'N/A'}")
                    if hasattr(msg, 'thinking'):
                        print(f"      thinking长度: {len(msg.thinking) if msg.thinking else 0}")
                    if hasattr(msg, 'content'):
                        print(f"      content长度: {len(msg.content) if msg.content else 0}")
                else:
                    print(f"   ⚠️ response对象没有message属性")
                
                # 如果第一次尝试失败，打印调试信息
                if retry == 0:
                    print(f"   ⚠️ 无法提取响应内容，响应类型: {type(response)}")
                    if hasattr(response, 'message'):
                        msg = response.message
                        print(f"      message类型: {type(msg)}")
                        print(f"      message属性: {[a for a in dir(msg) if not a.startswith('_')][:10]}")
                
                # 如果还有重试机会，继续
                if retry < max_retries - 1:
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    return None
                    
            else:
                # 回退到requests API（需要base64编码）
                print(f"   📤 使用requests API调用模型（base64编码）...")
                base64_image = image_to_base64(image_path)
                if not base64_image:
                    print(f"   ⚠️ 图片编码失败")
                    if retry < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return None
                
                api_url = f"{ollama_url}/api/chat"
                
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [base64_image]  # 需要base64编码
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 8000,  # 增加输出长度限制
                        "num_ctx": 8192,  # 上下文长度
                        "thinking": False  # 禁用thinking模式
                    }
                }
                
                # 增加超时时间，视觉模型处理较慢
                response = requests.post(api_url, json=payload, timeout=600)  # 10分钟超时
                
                if response.status_code == 200:
                    result = response.json()
                    content = None
                    
                    # 优先使用content
                    if 'message' in result and 'content' in result['message']:
                        content = result['message']['content']
                        if content and content.strip():
                            print(f"   ✅ 成功获取响应，长度: {len(content)} 字符")
                            return content
                    
                    # 如果content为空，检查thinking（某些模型可能将输出放在thinking中）
                    if (not content or not content.strip()) and 'message' in result and 'thinking' in result['message']:
                        thinking = result['message']['thinking']
                        if thinking and thinking.strip():
                            print(f"   ⚠️ content为空，从thinking中提取内容，长度: {len(thinking)} 字符")
                            # 尝试从thinking中提取JSON（如果prompt要求JSON输出）
                            import re
                            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', thinking, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                try:
                                    # 验证是否是有效的JSON
                                    json.loads(json_str)
                                    print(f"   ✅ 从thinking中提取到JSON，使用JSON部分")
                                    return json_str
                                except:
                                    pass
                            # 如果没有JSON，返回整个thinking内容
                            return thinking
                    
                    # 兼容旧格式
                    if 'response' in result:
                        content = result['response']
                        if content and content.strip():
                            print(f"   ✅ 成功获取响应（旧格式），长度: {len(content)} 字符")
                            return content
                
                # 如果失败且还有重试机会
                if retry < max_retries - 1:
                    print(f"   ⚠️ API调用失败，等待2秒后重试...")
                    time.sleep(2)
                    continue
                else:
                    print(f"   ⚠️ API调用失败: {response.status_code if 'response' in locals() else 'N/A'}")
                    return None
                
        except Exception as e:
            print(f"   ⚠️ 调用ollama时出错: {e}")
            if retry < max_retries - 1:
                print(f"   🔄 等待2秒后重试...")
                time.sleep(2)
                continue
            else:
                import traceback
                traceback.print_exc()
                return None
    
    # 如果所有重试都失败
    print(f"   ❌ 所有重试均失败")
    return None

# ==========================================
# 5. 批量分析
# ==========================================

def batch_analyze_charts(
    base_dir: str,
    model_name: Optional[str] = None,
    save_intermediate: bool = True
) -> Dict[str, Dict]:
    """
    批量分析所有图表
    
    返回: {chart_name: analysis_result, ...}
    """
    # 1. 选择模型
    if not model_name:
        models = get_available_models()
        model_name = select_vision_model(models)
        if not model_name:
            print("❌ 未找到可用的视觉模型，请先拉取 qwen3-vl 或 qwen2.5-vl")
            return {}
        print(f"✅ 使用模型: {model_name}")
    
    # 2. 收集图表
    charts = collect_chart_files(base_dir)
    print(f"📊 找到 {len(charts)} 张图表需要分析")
    
    if not charts:
        print("⚠️ 未找到任何图表文件")
        return {}
    
    # 3. 批量分析
    results = {}
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    intermediate_file = os.path.join(output_dir, 'ai_chart_analysis_intermediate.json')
    
    for i, chart in enumerate(charts, 1):
        print(f"\n[{i}/{len(charts)}] 正在分析: {chart['name']} ({chart['filename']})")
        
        analysis_text = analyze_chart_with_ollama(
            chart['path'],
            chart['name'],
            chart['description'],
            model_name
        )
        
        if analysis_text and analysis_text.strip():
            # 尝试解析JSON
            try:
                # 提取JSON部分（去除可能的markdown代码块）
                json_text = analysis_text.strip()
                
                # 方法1: 尝试找到markdown代码块中的JSON
                if '```json' in json_text:
                    json_text = json_text.split('```json')[1].split('```')[0].strip()
                elif '```' in json_text:
                    # 可能是 ``` 包裹的JSON
                    parts = json_text.split('```')
                    if len(parts) >= 3:
                        json_text = parts[1].strip()
                        # 如果第一部分是json，跳过
                        if parts[0].strip().lower().endswith('json'):
                            json_text = parts[1].strip() if len(parts) > 1 else json_text
                
                # 方法2: 尝试直接解析
                analysis_dict = None
                try:
                    analysis_dict = json.loads(json_text)
                except json.JSONDecodeError:
                    # 方法3: 使用正则表达式提取JSON对象（支持嵌套）
                    import re
                    # 改进的正则表达式，支持嵌套的JSON对象
                    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                    json_matches = list(re.finditer(json_pattern, json_text, re.DOTALL))
                    
                    # 从后往前查找（JSON通常在thinking的末尾）
                    for match in reversed(json_matches):
                        try:
                            potential_json = match.group(0)
                            # 验证是否是有效的JSON
                            test_dict = json.loads(potential_json)
                            # 检查是否包含必需的字段（chart_type或key_findings）
                            if 'chart_type' in test_dict or 'key_findings' in test_dict:
                                analysis_dict = test_dict
                                json_text = potential_json
                                print(f"   ✅ 从thinking中提取到有效JSON（包含必需字段）")
                                break
                        except json.JSONDecodeError:
                            continue
                    
                    # 如果还是没找到，尝试匹配任何有效的JSON对象（多字段）
                    if not analysis_dict:
                        for match in reversed(json_matches):
                            try:
                                potential_json = match.group(0)
                                test_dict = json.loads(potential_json)
                                # 如果JSON对象有多个键，可能是有效的分析结果
                                if len(test_dict) >= 3:
                                    analysis_dict = test_dict
                                    json_text = potential_json
                                    print(f"   ✅ 从thinking中提取到有效JSON（多字段对象）")
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    # 如果还是没找到，尝试查找包含关键字段的文本片段
                    if not analysis_dict:
                        # 查找包含chart_type或key_findings的文本区域
                        if '"chart_type"' in json_text or "'chart_type'" in json_text or '"key_findings"' in json_text or "'key_findings'" in json_text:
                            keyword_pos = max(
                                json_text.rfind('"chart_type"'),
                                json_text.rfind("'chart_type'"),
                                json_text.rfind('"key_findings"'),
                                json_text.rfind("'key_findings'")
                            )
                            if keyword_pos >= 0:
                                # 从关键词位置往前找最近的{，往后找最近的}
                                start_pos = json_text.rfind('{', 0, keyword_pos)
                                if start_pos >= 0:
                                    potential_json = json_text[start_pos:]
                                    first_close = potential_json.find('}')
                                    if first_close > 0:
                                        simple_json = potential_json[:first_close+1]
                                        try:
                                            test_dict = json.loads(simple_json)
                                            if 'chart_type' in test_dict or 'key_findings' in test_dict:
                                                analysis_dict = test_dict
                                                json_text = simple_json
                                                print(f"   ✅ 从关键词位置提取到JSON（简单提取）")
                                        except:
                                            # 尝试补全括号
                                            open_braces = simple_json.count('{')
                                            close_braces = simple_json.count('}')
                                            if open_braces > close_braces:
                                                simple_json += '}' * (open_braces - close_braces)
                                                try:
                                                    test_dict = json.loads(simple_json)
                                                    if 'chart_type' in test_dict or 'key_findings' in test_dict:
                                                        analysis_dict = test_dict
                                                        json_text = simple_json
                                                        print(f"   ✅ 从关键词位置提取到JSON（补全后）")
                                                except:
                                                    pass
                    
                    # 如果还是没找到，尝试查找最后一个 { 开始的位置（可能JSON被截断）
                    if not analysis_dict:
                        last_brace = json_text.rfind('{')
                        if last_brace >= 0:
                            potential_json = json_text[last_brace:]
                            # 尝试补全可能的未闭合括号
                            open_braces = potential_json.count('{')
                            close_braces = potential_json.count('}')
                            if open_braces > close_braces:
                                potential_json += '}' * (open_braces - close_braces)
                            try:
                                test_dict = json.loads(potential_json)
                                if 'chart_type' in test_dict or 'key_findings' in test_dict or len(test_dict) >= 3:
                                    analysis_dict = test_dict
                                    json_text = potential_json
                                    print(f"   ✅ 从文本末尾提取到JSON（补全后），使用JSON部分")
                            except:
                                pass
                    
                    # 如果还是没找到，抛出错误
                    if not analysis_dict:
                        raise ValueError("无法找到有效的JSON内容")
                
                results[chart['name']] = {
                    'chart_info': chart,
                    'analysis': analysis_dict
                }
                print(f"✅ 分析完成")
            except (json.JSONDecodeError, ValueError) as e:
                # 分析失败，立即停止并报告错误
                error_msg = f"❌ 图表分析失败: {chart['name']} ({chart['filename']})"
                print(f"\n{error_msg}")
                print(f"   错误类型: {type(e).__name__}")
                print(f"   错误信息: {str(e)}")
                print(f"   原始响应前200字符: {analysis_text[:200]}")
                print(f"\n⚠️ 已停止批量分析，请检查错误信息后重试")
                # 返回错误信息，让调用者知道失败
                return {
                    'error': True,
                    'failed_chart': chart['name'],
                    'failed_file': chart['filename'],
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'partial_results': results
                }
        else:
            # 未获取到有效响应内容，立即停止
            error_msg = f"❌ 图表分析失败: {chart['name']} ({chart['filename']}) - 未获取到有效响应内容"
            print(f"\n{error_msg}")
            print(f"⚠️ 已停止批量分析，请检查模型响应或网络连接")
            return {
                'error': True,
                'failed_chart': chart['name'],
                'failed_file': chart['filename'],
                'error_type': 'EmptyResponse',
                'error_message': '未获取到有效响应内容',
                'partial_results': results
            }
        
        # 保存中间结果
        if save_intermediate:
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 批量分析完成，共分析 {len(results)} 张图表")
    return results

# ==========================================
# 6. 综合报告生成
# ==========================================

def generate_comprehensive_report(
    chart_analyses: Dict[str, Dict],
    base_dir: str,
    model_name: Optional[str] = None,
    ollama_url: str = "http://localhost:11434"
) -> str:
    """
    基于所有图表分析结果，生成综合评估报告
    
    返回: 综合报告文本（HTML格式）
    """
    # 准备所有分析结果，包括图表路径映射
    analysis_summary = []
    chart_path_mapping = {}  # 图表名称到文件路径的映射
    
    for chart_name, data in chart_analyses.items():
        chart_info = data.get('chart_info', {})
        analysis = data.get('analysis', {})
        
        # 获取图表文件路径
        chart_filename = chart_info.get('filename', '')
        chart_path = chart_info.get('path', '')
        
        # 构建相对路径
        if chart_path:
            # 转换为相对路径（从base_dir开始）
            if 'analysis_report' in chart_path:
                relative_path = f"output/analysis_report/{chart_filename}"
            elif 'position_analysis_v2' in chart_path:
                relative_path = f"output/position_analysis_v2/{chart_filename}"
            else:
                relative_path = chart_path.replace(base_dir + '/', '')
        else:
            relative_path = chart_filename
        
        chart_path_mapping[chart_name] = relative_path
        
        summary = {
            'chart_name': chart_name,
            'chart_filename': chart_filename,
            'chart_path': relative_path,
            'chart_type': analysis.get('chart_type', '未知'),
            'key_findings': analysis.get('key_findings', []),
            'risk_level': analysis.get('risk_level', '未知'),
            'detailed_analysis': analysis.get('detailed_analysis', ''),
            'process_suggestions': analysis.get('process_suggestions', ''),
            'anomalies': analysis.get('anomalies', '')
        }
        analysis_summary.append(summary)
    
    # 构建综合分析的提示词
    summary_text = json.dumps(analysis_summary, ensure_ascii=False, indent=2)
    
    prompt = f"""【角色设定】
你现在是一位半导体良率优化专家（Yield Optimization Expert）和机器学习工程师。
我们已经完成了前期的 EDA（探索性数据分析）、核心建模与归因阶段，现在进入给出建议总结分析阶段。

【任务目标】
基于所有图表分析结果，生成综合评估报告，识别可能与良率下降相关的物理参数特征和数据模式，** 为产线工程师提供决策参考和排查线索**（非绝对执行指令）。分析应基于实际数据，使用客观、统计学的口吻，避免绝对化表述。

【数据概况与业务逻辑】
数据集：cleaned_data.csv
预测目标（Label）：
- 原始列：压连情况（-1=虚焊, 0=正常, 1=轻微压连, 2=严重压连）
- 建模目标：二分类（Binary Classification）
- 逻辑：{{-1, 2}} = Fail (0) ；{{0, 1}} = Pass (1)
- 注意：这是一个非平衡数据集，Fail样本较少但成本极高

关键特征（Features）：
- 铟柱总高度（Total_Indium_Height）：[核心特征] 上下高度和，是抗压连的安全屏障
- Calc_Circuit_Range：电路端平整度（注意：存在大量缺失值，仅激光调平工艺有此值）
- Indium_Taper_Zscore：铟柱形状异常度，反映形状畸变风险
- 倒焊压力（kg）：工艺设定压力，是决定性的物理变量
- Time_Seq_Day：连续生产天数（用来捕捉设备老化/漂移）
- Wafer_Index：晶圆加工次序，反映批次效应
- Position_Code：空间位置（M1-M10），反映位置特异性问题

【已完成的图表分析】
{summary_text}

【参考报告结构（必须严格遵循）】
请严格按照以下HTML结构生成报告内容，保持与模板完全一致的格式和风格：

<div class="section-card">
  <h2>良率现状与宏观分布</h2>
  
  <h3>1.1 压连结果分布</h3>
  <div class="chart-wrapper">
    <img src="output/analysis_report/0_目标分布统计.png" alt="图表: 目标分布">
  </div>
  <p>（分析四类生产状态的分布情况，指出严重压连是主要失效原因）</p>

  <h3>1.2 关键参数特征差异</h3>
  <div class="chart-wrapper">
    <img src="output/analysis_report/2_核心特征分布_2x2_中文.png" alt="图表: 核心特征分布">
  </div>
  <p>（对比Pass/Fail样本在关键参数上的分布差异，重点分析总铟柱高度、倒焊压力、铟柱形状异常度等）</p>

  <h3>1.3 参数间关联性与"实验设计复盘"</h3>
  <div class="chart-wrapper">
    <img src="output/analysis_report/1_参数相关性分析.png" alt="图表: 相关性分析">
  </div>
  <div class="analysis-box" style="background-color: #fdf6ec; border-left: 5px solid #e6a23c; padding: 20px; border-radius: 4px; color: #606266; margin-top: 20px;">
    <p style="margin-top:0; color:#e6a23c; font-weight:bold;">💡 数据洞察：存在显著的采样偏差</p>
    <p>（分析相关性热力图，揭示人为操作模式）</p>
    <ul>
      <li><strong>良率的物质基础：</strong> 良率与总铟柱高度的正相关关系</li>
      <li><strong>揭示不良的操作模式：</strong> 压力与高度的负相关关系</li>
      <li><strong>叠加风险：</strong> 失效样本在"低高度+差形状+高压力"三重恶劣条件下产生</li>
    </ul>
  </div>
</div>

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
    <p>（分析各位置的良率差异，锁定M7、M8、M2、M4等异常位置）</p>
  </div>
  <div class="chart-wrapper">
    <img src="output/position_analysis_v2/3_Position_Physical_Features.png" alt="图表: 物理一致性特征">
  </div>
  <div class="analysis-box">
    <p>（以M5为稳定性标杆，分析其他位置的物理偏差）</p>
  </div>
</div>

【输出要求】
1. 必须严格按照上述HTML结构生成，包括所有div、class、style属性
2. 使用图表路径时，必须使用相对路径（如 output/analysis_report/0_目标分布统计.png）
3. **重要：分析内容必须直接插入到对应图像下方**
   - 每张图表下方必须紧跟着该图表的详细分析（包括detailed_analysis、key_findings、process_suggestions等）
   - 工艺优化建议（process_suggestions）必须直接放在对应图表的分析段落中，不要单独总结
   - 异常发现（anomalies）必须直接放在对应图表的分析段落中
   - 不要在报告末尾总结所有建议，而是将每个建议放在对应的图表分析下方
4. 分析内容要求：
   - 必须基于提供的图表分析结果，结合业务逻辑进行深入、专业的分析
   - 每个章节都要有详细的数据支撑和物理机制解释
   - **避免使用绝对化语言**（如"必须设定"、"禁止超过"、"一定导致"、"必须维持在"等）
   - **采用建议性话术**（如"建议关注"、"风险可能显著增加"、"建议维持在"、"需关注区间"、"值得排查"、"建议排查"等）
   - 指出潜在的高风险参数区间（如"高度低于11.65μm时风险可能显著增加"），但以建议和参考的形式呈现
   - 分析要具体、可操作，包含明确的数值和趋势描述，但数据应根据实际情况变化
   - 工艺优化建议要包含具体的参考方向和预期效果，使用建议性表述
5. 保持与模板完全一致的视觉风格和结构
6. 报告内容要充实，每个段落至少3-5句话，详细阐述发现和结论，使用客观、统计学的口吻
7. 直接输出HTML代码，不要添加任何解释文字

【特别强调】
- 不要使用占位符文字，要用实际的分析内容填充
- 每个图表都要有对应的详细分析文字
- 工艺优化建议要使用建议性话术，避免绝对化指令
- 要体现非平衡数据集的特点（Fail样本少但成本高）
- 数据应根据实际情况变化，基于实际统计结果进行分析

请开始生成综合报告："""

    # 调用视觉模型生成报告（处理图像特征信息）
    try:
        if not model_name:
            model_name = "qwen-vl-plus"
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_API_BASE") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        if resp and resp.choices:
            return resp.choices[0].message.content or ""
        return ""
    except Exception as e:
        print(f"⚠️ 视觉模型报告生成异常: {e}")
        return ""


def generate_text_analysis_report(
    base_dir: str,
    text_model_name: str = "qwen3:8b",
    ollama_url: str = "http://localhost:11434"
) -> str:
    """生成文本分析报告 - 已禁用"""
    return "图像识别功能已被禁用，无法生成文本分析报告"
    """
    使用文字LLM模型生成基于统计数据的文本分析报告
    注意：这个函数处理的是文本数据，不是图像
    """
    try:
        # 导入文本分析模块
        from ai_text_analyzer import collect_analysis_text_data, generate_text_based_report
        from chart_data_extractor import extract_all_chart_data
        
        print("📊 收集文本分析数据...")
        # 收集分析数据
        analysis_data = collect_analysis_text_data(base_dir)
        chart_data_text, _ = extract_all_chart_data(base_dir)
        
        print(f"🤖 使用文字模型 {text_model_name} 生成报告...")
        # 使用文字模型生成报告
        text_report = generate_text_based_report(
            analysis_data,
            model_name=text_model_name,
            ollama_url=ollama_url,
            chart_data_text=chart_data_text
        )
        
        if text_report:
            print("✅ 文字模型报告生成完成")
        else:
            print("⚠️ 文字模型报告生成失败")
        
        return text_report
    except Exception as e:
        print(f"⚠️ 文字模型报告生成异常: {e}")
        import traceback
        traceback.print_exc()
        return ""

# ==========================================
# 7. 主函数
# ==========================================

def main():
    """主函数：执行完整的AI图表分析流程"""
    # 注意：图像识别功能已被禁用，速度太慢，延迟太高
    print("=" * 60)
    print("🤖 AI图表视觉分析模块 - 已禁用")
    print("原因：图像识别速度太慢，延迟太高，已取消此功能")
    print("=" * 60)

    # 返回空结果，表示功能不可用
    return None, "图像识别功能已被禁用"

if __name__ == "__main__":
    main()
