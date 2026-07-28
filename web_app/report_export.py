import base64
import io
import json
import os
import re
import sys
import urllib.parse

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup, NavigableString, Tag
import config as utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import ROOT_DIR, OUTPUT_DIR, WEB_APP_DIR, FONT_FILE
from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from weasyprint import CSS, HTML

    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False


_PDF_UNSAFE_CHARS_RE = re.compile(r"[\U00010000-\U0010ffff]")


def _get_paths():
    return WEB_APP_DIR, ROOT_DIR, ROOT_DIR, OUTPUT_DIR


def get_image_base64(image_path):
    if not image_path or not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def _find_image_path(src, output_dir):
    if not src:
        return None

    src = urllib.parse.unquote(src)
    candidates = [
        os.path.join(output_dir, "analysis_report", os.path.basename(src)),
        os.path.join(output_dir, "ml_report", os.path.basename(src)),
        os.path.join(output_dir, "position_analysis_v2", os.path.basename(src)),
    ]

    clean_src = src.replace("output/", "")
    candidates.append(os.path.join(output_dir, clean_src))
    candidates.append(os.path.join(output_dir, "..", src))

    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if os.path.exists(normalized):
            return normalized
    return None


def _load_kpi_stats(text_results_file):
    if not utils.is_descriptive_report_current():
        return {}

    if not os.path.exists(text_results_file):
        return {}

    with open(text_results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    kpi_stats = data.get("analysis_data", {}).get("kpi_stats", {})
    if kpi_stats:
        return kpi_stats

    summary = data.get("summary_stats", {})
    yield_stats = summary.get("eda_analysis", {}).get("yield_stats", {})
    if not yield_stats:
        return {}

    total = yield_stats.get("total", 0)
    return {
        "pass_rate": yield_stats.get("pass_rate", 0) * 100,
        "open_rate": (yield_stats.get("open_count", 0) / total * 100) if total > 0 else 0,
        "severe_rate": (yield_stats.get("severe_count", 0) / total * 100) if total > 0 else 0,
        "open_count": yield_stats.get("open_count", 0),
        "severe_count": yield_stats.get("severe_count", 0),
    }


def _load_descriptive_html(text_results_file):
    if not utils.is_descriptive_report_current():
        return ""

    if not os.path.exists(text_results_file):
        return ""
    try:
        with open(text_results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("comprehensive_report", "")
    except Exception:
        return ""


def _build_kpi_html(t, kpi_stats):
    if not kpi_stats:
        return ""

    return f"""
    <div class="section-card">
        <h3>{t.get('pdf_sec_1', '1. 核心指标概览')}</h3>
        <div style="display:flex; justify-content:space-around; gap:20px; flex-wrap:wrap; margin-top:20px;">
            <div style="background:#e8f6fd; padding:15px; border-radius:8px; width:30%; min-width:220px; text-align:center;">
                <div style="font-weight:bold; margin-bottom:5px;">{t.get('pdf_yield', '整体良品率')}</div>
                <div style="color:#3498db; font-size:24px; font-weight:bold;">{kpi_stats.get('pass_rate', 0):.1f}%</div>
                <div style="color:#666; font-size:12px; margin-top:5px;">{t.get('pdf_baseline', '基准线')}</div>
            </div>
            <div style="background:#fdecec; padding:15px; border-radius:8px; width:30%; min-width:220px; text-align:center;">
                <div style="font-weight:bold; margin-bottom:5px;">{t.get('pdf_open_fail', '不合格率')}</div>
                <div style="color:#e74c3c; font-size:24px; font-weight:bold;">{kpi_stats.get('open_rate', 0):.1f}%</div>
                <div style="color:#666; font-size:12px; margin-top:5px;">{t.get('pdf_count', '数量: ')}{kpi_stats.get('open_count', 0)}</div>
            </div>
            <div style="background:#fef6e4; padding:15px; border-radius:8px; width:30%; min-width:220px; text-align:center;">
                <div style="font-weight:bold; margin-bottom:5px;">{t.get('pdf_severe_fail', '严重缺陷率')}</div>
                <div style="color:#f39c12; font-size:24px; font-weight:bold;">{kpi_stats.get('severe_rate', 0):.1f}%</div>
                <div style="color:#666; font-size:12px; margin-top:5px;">{t.get('pdf_count', '数量: ')}{kpi_stats.get('severe_count', 0)}</div>
            </div>
        </div>
    </div>
    """


def _inline_report_images(html_content, output_dir):
    soup = BeautifulSoup(html_content or "", "html.parser")
    for img in soup.find_all("img"):
        src = img.get("src")
        abs_path = _find_image_path(src, output_dir)
        if abs_path:
            img_b64 = get_image_base64(abs_path)
            if img_b64:
                img["src"] = f"data:image/png;base64,{img_b64}"
    return str(soup)


def _parse_analysis_text(analysis_text):
    if not isinstance(analysis_text, str):
        return None, str(analysis_text)

    clean_json = analysis_text.replace("```json", "").replace("```", "").strip()
    try:
        if clean_json.startswith("{"):
            return json.loads(clean_json), ""
    except Exception:
        pass
    return None, clean_json


def _clean_pdf_text(text):
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = _PDF_UNSAFE_CHARS_RE.sub("", cleaned)
    for token in ["📊", "💡", "🔍", "📋", "📌", "🛑", "📉", "🎯", "⚖️", "⚙️", "🚀", "📥"]:
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace("鈥?", "• ").replace("\u25a1", "")
    return cleaned.strip()


def _build_kpi_cards_story(styles, t, kpi_stats):
    cards = [
        (t.get("pdf_yield", "整体良品率"), f"{kpi_stats.get('pass_rate', 0):.1f}%", t.get("pdf_baseline", "基准线"), "#e8f6fd", "#3498db"),
        (t.get("pdf_open_fail", "不合格率"), f"{kpi_stats.get('open_rate', 0):.1f}%", f"{t.get('pdf_count', '数量: ')}{kpi_stats.get('open_count', 0)}", "#fdecec", "#e74c3c"),
        (t.get("pdf_severe_fail", "严重缺陷率"), f"{kpi_stats.get('severe_rate', 0):.1f}%", f"{t.get('pdf_count', '数量: ')}{kpi_stats.get('severe_count', 0)}", "#fef6e4", "#f39c12"),
    ]
    cells = []
    for title, value, footer, bg, accent in cards:
        card = Table(
            [[
                [
                    Paragraph(f"<b>{_clean_pdf_text(title)}</b>", styles["CN_Normal"]),
                    Spacer(1, 8),
                    Paragraph(f"<font color='{accent}' size='20'><b>{value}</b></font>", styles["CN_Normal"]),
                    Spacer(1, 8),
                    Paragraph(_clean_pdf_text(footer), styles["CN_Normal"]),
                ]
            ]],
            colWidths=[145],
            rowHeights=[95],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
            ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9e2ec")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ]))
        cells.append(card)

    wrap = Table([cells], colWidths=[150, 150, 150])
    wrap.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return wrap


def _build_deep_mining_html(t):
    results = st.session_state.get("deep_mining_results")
    if not results:
        return f"<p>{t.get('html_no_mining', '暂无深度挖掘结果。')}</p>"

    parts = []
    for res in results:
        chart_name = res.get("chart_name", "未命名图表")
        image_path = res.get("image_path", "")
        analysis_text = res.get("analysis_text", "")
        analysis_data, raw_text = _parse_analysis_text(analysis_text)

        analysis_html = ""
        if analysis_data:
            key_findings = analysis_data.get("key_findings", [])
            if key_findings:
                analysis_html += f"<p>{t.get('pdf_key_findings', '<b>📊 关键发现:</b>')}</p><ul>"
                for item in key_findings:
                    analysis_html += f"<li>{item}</li>"
                analysis_html += "</ul>"
            if analysis_data.get("process_suggestions"):
                analysis_html += (
                    f"<p>{t.get('pdf_suggestions', '<b>💡 工艺建议:</b>')} "
                    f"{analysis_data['process_suggestions']}</p>"
                )
            if analysis_data.get("detailed_analysis"):
                analysis_html += (
                    f"<p>{t.get('pdf_detailed', '<b>🔍 详细分析:</b>')} "
                    f"{analysis_data['detailed_analysis']}</p>"
                )
        else:
            analysis_html = f"<p>{raw_text}</p>"

        img_b64 = get_image_base64(image_path)
        img_tag = f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;">' if img_b64 else ""

        parts.append(
            f"""
            <div class="section-card">
                <h3>{chart_name}</h3>
                <div class="chart-wrapper">{img_tag}</div>
                <div class="analysis-box">{analysis_html}</div>
            </div>
            """
        )
    return "".join(parts)


def _build_suggestions_html(t):
    suggestions = st.session_state.get("final_suggestions", {}).get("suggestions", [])
    if not suggestions:
        return f"<p>{t.get('html_no_sug', '暂无最终建议。')}</p>"

    parts = []
    for sug in suggestions:
        parts.append(
            f"""
            <div class="suggestion-card">
                <div class="suggestion-header">
                    <span class="suggestion-icon">{sug.get('icon', '📌')}</span>
                    <span class="suggestion-title">{sug.get('title', t.get('pdf_sug_default_title', '优化建议'))}</span>
                </div>
                <div class="suggestion-content">{sug.get('content', '')}</div>
            </div>
            """
        )
    return "".join(parts)


def generate_full_report_html(t):
    _, _, _, output_dir = _get_paths()
    text_results_file = os.path.join(output_dir, "ai_text_analysis_results.json")

    kpi_html = _build_kpi_html(t, _load_kpi_stats(text_results_file))
    descriptive_html = _load_descriptive_html(text_results_file)
    descriptive_html = _inline_report_images((kpi_html + descriptive_html) if kpi_html else descriptive_html, output_dir)
    if not descriptive_html:
        descriptive_html = f"<p>{t.get('html_no_stats', '暂无描述性统计报告。')}</p>"

    deep_mining_html = _build_deep_mining_html(t)
    suggestions_html = _build_suggestions_html(t)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{t.get('html_title', '半导体器件生产助手 - 工艺优化全景报告')}</title>
        <style>
            body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f4f6f9; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            h1 {{ text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 30px; }}
            .section-card {{ background: #fff; border-radius: 8px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            .chart-wrapper {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .analysis-box {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin-top: 15px; }}
            .suggestion-card {{ background-color: #f0f9eb; border-radius: 8px; padding: 20px; margin-bottom: 15px; border-left: 5px solid #67c23a; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05); }}
            .suggestion-header {{ display: flex; align-items: center; margin-bottom: 10px; }}
            .suggestion-icon {{ font-size: 24px; margin-right: 12px; }}
            .suggestion-title {{ font-size: 18px; font-weight: 600; color: #2c3e50; }}
            .suggestion-content {{ color: #5e6d82; font-size: 15px; margin-left: 36px; }}
        </style>
    </head>
    <body>
        <h1>📋 {t.get('html_title', '半导体器件生产助手 - 工艺优化全景报告')}</h1>
        <p style="text-align:center; color:#666;">{t.get('pdf_date', '生成日期: ')}{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

        <h2>{t.get('html_sec_1', '第一部分：描述性统计分析')}</h2>
        {descriptive_html}

        <h2>{t.get('html_sec_2', '第二部分：深度挖掘与归因分析')}</h2>
        {deep_mining_html}

        <h2>{t.get('html_sec_3', '第三部分：总结与工艺优化建议')}</h2>
        {suggestions_html}

        <div style="text-align:center; margin-top:50px; color:#999; font-size:12px;">
            Generated by AI Semiconductor Device Production Assistant
        </div>
    </body>
    </html>
    """


def _get_font_name():
    current_dir, _, base_dir, _ = _get_paths()
    font_candidates = [
        (os.path.join(current_dir, "assets", "SimHei.ttf"), "SimHeiLocal"),
        (FONT_FILE, "SimHeiRoot"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "NotoSansCJK"),
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", "DroidSansFallback"),
        ("/usr/share/fonts/truetype/arphic/uming.ttc", "UMing"),
        ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "MicroHei"),
    ]

    for font_path, font_family in font_candidates:
        if not os.path.exists(font_path):
            continue
        try:
            pdfmetrics.registerFont(TTFont(font_family, font_path))
            pdfmetrics.registerFont(TTFont(f"{font_family}-Bold", font_path))
            pdfmetrics.registerFont(TTFont(f"{font_family}-Italic", font_path))
            pdfmetrics.registerFont(TTFont(f"{font_family}-BoldItalic", font_path))
            addMapping(font_family, 0, 0, font_family)
            addMapping(font_family, 0, 1, f"{font_family}-Italic")
            addMapping(font_family, 1, 0, f"{font_family}-Bold")
            addMapping(font_family, 1, 1, f"{font_family}-BoldItalic")
            return font_family
        except Exception:
            continue
    return "Helvetica"


def _build_styles():
    font_name = _get_font_name()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CN_Title", parent=styles["Title"], fontName=font_name, fontSize=22, leading=28, spaceAfter=30, alignment=1))
    styles.add(ParagraphStyle(name="CN_Heading1", parent=styles["Heading1"], fontName=font_name, fontSize=16, leading=22, spaceBefore=20, spaceAfter=12, textColor=colors.HexColor("#2c3e50")))
    styles.add(ParagraphStyle(name="CN_Heading2", parent=styles["Heading2"], fontName=font_name, fontSize=14, leading=20, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor("#34495e")))
    styles.add(ParagraphStyle(name="CN_Heading3", parent=styles["Heading3"], fontName=font_name, fontSize=12, leading=16, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#7f8c8d")))
    styles.add(ParagraphStyle(name="CN_Normal", parent=styles["Normal"], fontName=font_name, fontSize=10.5, leading=16, spaceAfter=6))
    styles.add(ParagraphStyle(name="CN_Bullet", parent=styles["Normal"], fontName=font_name, fontSize=10.5, leading=16, leftIndent=10))
    return styles


def parse_html_to_flowables(html_content, styles, output_dir):
    flowables = []
    soup = BeautifulSoup(html_content, "html.parser")

    def extract_clean_text(tag):
        text = ""
        for child in tag.children:
            if isinstance(child, NavigableString):
                text += str(child)
            elif isinstance(child, Tag):
                if child.name in ["strong", "b"]:
                    text += f"<b>{extract_clean_text(child)}</b>"
                elif child.name in ["em", "i"]:
                    text += f"<i>{extract_clean_text(child)}</i>"
                elif child.name == "br":
                    text += "<br/>"
                else:
                    text += extract_clean_text(child)
        return text

    def process_image(img_tag):
        src = img_tag.get("src")
        abs_path = _find_image_path(src, output_dir)
        if abs_path and os.path.exists(abs_path):
            try:
                img = RLImage(abs_path)
                max_width = 460
                aspect = img.imageHeight / img.imageWidth
                img.drawWidth = max_width
                img.drawHeight = max_width * aspect
                img.hAlign = "CENTER"
                flowables.append(Spacer(1, 5))
                flowables.append(img)
                flowables.append(Spacer(1, 10))
            except Exception:
                flowables.append(Paragraph(f"<font color='red'>图片加载失败: {os.path.basename(src or '')}</font>", styles["CN_Normal"]))
        else:
            flowables.append(Paragraph(f"<font color='red'>图片未找到: {os.path.basename(src or '')}</font>", styles["CN_Normal"]))

    def process_element(element):
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                flowables.append(Paragraph(text, styles["CN_Normal"]))
            return

        if element.name in ["h1", "h2"]:
            flowables.append(Paragraph(element.get_text().strip(), styles["CN_Heading2"]))
        elif element.name == "h3":
            flowables.append(Paragraph(element.get_text().strip(), styles["CN_Heading3"]))
        elif element.name == "p":
            img = element.find("img")
            if img:
                process_image(img)
                element_copy = BeautifulSoup(str(element), "html.parser").p
                found = element_copy.find("img")
                if found:
                    found.decompose()
                text_content = extract_clean_text(element_copy)
                if text_content.strip():
                    flowables.append(Paragraph(text_content, styles["CN_Normal"]))
            else:
                text_content = extract_clean_text(element)
                if text_content.strip():
                    flowables.append(Paragraph(text_content, styles["CN_Normal"]))
        elif element.name == "ul":
            items = []
            for li in element.find_all("li", recursive=False):
                text_content = extract_clean_text(li)
                items.append(ListItem(Paragraph(text_content.strip(), styles["CN_Bullet"])))
            if items:
                flowables.append(ListFlowable(items, bulletType="bullet", leftIndent=20))
                flowables.append(Spacer(1, 5))
        elif element.name == "div":
            classes = element.get("class", [])
            if "chart-wrapper" in classes:
                img = element.find("img")
                if img:
                    process_image(img)
            elif "analysis-box" in classes:
                box_content = []
                for child in element.children:
                    if getattr(child, "name", None) == "p":
                        text_content = extract_clean_text(child)
                        if text_content.strip():
                            box_content.append(Paragraph(text_content.strip(), styles["CN_Normal"]))
                    elif getattr(child, "name", None) == "ul":
                        for li in child.find_all("li"):
                            text_content = extract_clean_text(li)
                            if text_content.strip():
                                box_content.append(Paragraph(f"• {text_content.strip()}", styles["CN_Bullet"]))
                if box_content:
                    table = Table([[box_content]], colWidths=[460])
                    table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
                                ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                                ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor("#3498db")),
                                ("PADDING", (0, 0), (-1, -1), 10),
                            ]
                        )
                    )
                    flowables.append(table)
                    flowables.append(Spacer(1, 10))
            else:
                for child in element.children:
                    process_element(child)
        elif element.name == "img":
            process_image(element)

    roots = soup.body.children if soup.body else soup.children
    for child in roots:
        process_element(child)
    return flowables


def generate_full_report_pdf(t):
    current_dir, _, _, output_dir = _get_paths()

    if WEASYPRINT_AVAILABLE:
        try:
            html_content = generate_full_report_html(t)
            simhei_candidates = [
                os.path.join(current_dir, "assets", "SimHei.ttf"),
                os.path.join(os.path.dirname(os.path.dirname(current_dir)), "SimHei.ttf"),
            ]
            simhei_path = next((p for p in simhei_candidates if os.path.exists(p)), None)
            if simhei_path:
                css_str = (
                    f"@font-face {{ font-family: 'SimHei'; src: url('file://{simhei_path}'); }} "
                    "body { font-family: 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK', sans-serif; } "
                    "h1, h2, h3 { font-family: 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK', sans-serif; }"
                )
            else:
                css_str = "body { font-family: 'Noto Sans CJK', 'Microsoft YaHei', sans-serif; }"
            return HTML(string=html_content, base_url=current_dir).write_pdf(stylesheets=[CSS(string=css_str)])
        except Exception:
            pass

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = _build_styles()
    story = []

    _, _, _, output_dir = _get_paths()
    text_results_file = os.path.join(output_dir, "ai_text_analysis_results.json")
    kpi_stats = _load_kpi_stats(text_results_file)

    story.append(Paragraph(t.get("pdf_title", "半导体器件生产助手 - 工艺优化全景报告"), styles["CN_Title"]))
    story.append(Paragraph(f"{t.get('pdf_date', '生成日期: ')}{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles["CN_Normal"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph(t.get("pdf_sec_1", "1. 核心指标概览"), styles["CN_Heading1"]))
    if kpi_stats:
        table_data = [
            [
                Paragraph(f"<b>{t.get('pdf_yield', '整体良品率')}</b>", styles["CN_Normal"]),
                Paragraph(f"<b>{t.get('pdf_open_fail', '不合格率')}</b>", styles["CN_Normal"]),
                Paragraph(f"<b>{t.get('pdf_severe_fail', '严重缺陷率')}</b>", styles["CN_Normal"]),
            ],
            [
                Paragraph(f"<font color='#3498db' size=14><b>{kpi_stats.get('pass_rate', 0):.1f}%</b></font>", styles["CN_Normal"]),
                Paragraph(f"<font color='#e74c3c' size=14><b>{kpi_stats.get('open_rate', 0):.1f}%</b></font>", styles["CN_Normal"]),
                Paragraph(f"<font color='#f39c12' size=14><b>{kpi_stats.get('severe_rate', 0):.1f}%</b></font>", styles["CN_Normal"]),
            ],
            [
                Paragraph(t.get("pdf_baseline", "基准线"), styles["CN_Normal"]),
                Paragraph(f"{t.get('pdf_count', '数量: ')}{kpi_stats.get('open_count', 0)}", styles["CN_Normal"]),
                Paragraph(f"{t.get('pdf_count', '数量: ')}{kpi_stats.get('severe_count', 0)}", styles["CN_Normal"]),
            ],
        ]
        kpi_table = Table(table_data, colWidths=[150, 150, 150])
        kpi_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f6fd")),
                    ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#fdecec")),
                    ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#fef6e4")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("PADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(kpi_table)
    else:
        story.append(Paragraph(t.get("pdf_no_data", "暂无核心指标数据"), styles["CN_Normal"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph(t.get("pdf_sec_2", "2. 描述性统计分析"), styles["CN_Heading1"]))
    descriptive_html = _load_descriptive_html(text_results_file)
    if descriptive_html:
        soup = BeautifulSoup(descriptive_html, "html.parser")
        first_h2 = soup.find("h2")
        if first_h2 and first_h2.get_text().strip().startswith("2."):
            first_h2.decompose()
        story.extend(parse_html_to_flowables(str(soup), styles, output_dir))
    else:
        story.append(Paragraph(t.get("html_no_stats", "暂无描述性统计报告。"), styles["CN_Normal"]))

    story.append(PageBreak())
    story.append(Paragraph(t.get("pdf_sec_3", "3. 深度挖掘与归因分析"), styles["CN_Heading1"]))
    results = st.session_state.get("deep_mining_results")
    if results:
        for res in results:
            header = [Paragraph(res.get("chart_name", "未命名图表"), styles["CN_Heading2"])]
            image_path = res.get("image_path", "")
            if image_path and os.path.exists(image_path):
                try:
                    img = RLImage(image_path)
                    max_width = 460
                    if img.drawWidth > max_width:
                        ratio = max_width / img.drawWidth
                        img.drawWidth = max_width
                        img.drawHeight *= ratio
                    header.append(img)
                    header.append(Spacer(1, 10))
                except Exception:
                    header.append(Paragraph(f"{t.get('pdf_img_fail', '图片加载失败: ')}{image_path}", styles["CN_Normal"]))
            story.append(KeepTogether(header))

            analysis_data, raw_text = _parse_analysis_text(res.get("analysis_text", ""))
            if analysis_data:
                if analysis_data.get("key_findings"):
                    story.append(Paragraph(t.get("pdf_key_findings", "<b>📊 关键发现:</b>"), styles["CN_Normal"]))
                    for item in analysis_data["key_findings"]:
                        story.append(Paragraph(f"• {item}", styles["CN_Bullet"]))
                    story.append(Spacer(1, 6))
                if analysis_data.get("process_suggestions"):
                    story.append(Paragraph(f"{t.get('pdf_suggestions', '<b>💡 工艺建议:</b>')} {analysis_data['process_suggestions']}", styles["CN_Normal"]))
                    story.append(Spacer(1, 6))
                if analysis_data.get("detailed_analysis"):
                    story.append(Paragraph(f"{t.get('pdf_detailed', '<b>🔍 详细分析:</b>')} {analysis_data['detailed_analysis']}", styles["CN_Normal"]))
            else:
                story.append(Paragraph(raw_text, styles["CN_Normal"]))
            story.append(Spacer(1, 15))
    else:
        story.append(Paragraph(t.get("html_no_mining", "暂无深度挖掘结果。"), styles["CN_Normal"]))

    story.append(PageBreak())
    story.append(Paragraph(t.get("html_sec_3", "第三部分：总结与工艺优化建议"), styles["CN_Heading1"]))
    suggestions = st.session_state.get("final_suggestions", {}).get("suggestions", [])
    if suggestions:
        for sug in suggestions:
            title_para = Paragraph(f"{sug.get('icon', '')} <b>{sug.get('title', t.get('pdf_sug_default_title', '优化建议'))}</b>", styles["CN_Heading2"])
            content_para = Paragraph(sug.get("content", ""), styles["CN_Normal"])
            box = Table([[title_para], [content_para]], colWidths=[460])
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f9eb")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e1f3d8")),
                        ("LINEBEFORE", (0, 0), (0, -1), 5, colors.HexColor("#67c23a")),
                        ("PADDING", (0, 0), (-1, -1), 12),
                        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
                        ("TOPPADDING", (0, 1), (0, 1), 5),
                    ]
                )
            )
            story.append(box)
            story.append(Spacer(1, 15))
    else:
        story.append(Paragraph(t.get("pdf_no_sug", "暂无最终建议。"), styles["CN_Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def render_download_section(t):
    st.subheader(t.get("dl_title", "📥 完整报告导出"))
    current_filter = st.session_state.get('analysis_chip_filter_applied', utils.CHIP_FILTER_ALL)
    filter_desc = "全部芯片" if current_filter == utils.CHIP_FILTER_ALL else f"仅 {current_filter}"
    st.caption(f"当前完整报告分析范围：{filter_desc}")
    st.info(
        t.get(
            "dl_intro",
            "该模块将整合描述性统计、深度挖掘和工艺优化建议，生成可离线阅读的完整报告。",
        )
    )

    _, _, _, output_dir = _get_paths()
    text_results_file = os.path.join(output_dir, "ai_text_analysis_results.json")
    has_stats = os.path.exists(text_results_file) and utils.is_descriptive_report_current()
    has_mining = st.session_state.get("deep_mining_results") is not None
    has_suggestions = st.session_state.get("final_suggestions") is not None

    if not (has_stats or has_mining):
        st.warning(
            t.get(
                "dl_warn_no_data",
                "⚠️ 暂无分析数据。请先运行描述性统计或深度挖掘模块。",
            )
        )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(t.get("dl_contains", "#### 包含内容"))
        st.markdown(f"- {t.get('dl_item_stats', '📊 描述性统计报告')}: {'✅' if has_stats else '❌'}")
        st.markdown(f"- {t.get('dl_item_mining', '🔍 深度挖掘分析')}: {'✅' if has_mining else '❌'}")
        st.markdown(f"- {t.get('dl_item_sug', '💡 工艺优化建议')}: {'✅' if has_suggestions else '❌'}")

    with col2:
        st.write(t.get("dl_options", "### 导出选项"))
        html_col, pdf_col = st.columns(2)

        with html_col:
            if st.button(t.get("dl_btn_html", "🚀 生成 HTML 报告"), type="primary", use_container_width=True):
                with st.spinner(t.get("dl_spin_html", "正在生成 HTML 报告...")):
                    try:
                        html_content = generate_full_report_html(t)
                        st.session_state["full_report_html_bytes"] = html_content.encode("utf-8")
                        st.success(t.get("dl_success_html", "HTML 报告就绪！"))
                    except Exception as e:
                        st.session_state.pop("full_report_html_bytes", None)
                        st.error(f"{t.get('dl_fail_html', 'HTML 生成失败: ')}{str(e)}")
            if st.session_state.get("full_report_html_bytes"):
                st.download_button(
                label=t.get("dl_down_html", "📥 下载 HTML"),
                data=st.session_state["full_report_html_bytes"],
                file_name="Semiconductor_Analysis_Report.html",
                mime="text/html",
                use_container_width=True,
            )

        with pdf_col:
            if st.button(t.get("dl_btn_pdf", "📄 生成 PDF 报告"), use_container_width=True):
                with st.spinner(t.get("dl_spin_pdf", "正在生成 PDF 报告...")):
                    try:
                        pdf_data = generate_full_report_pdf(t)
                        st.session_state["full_report_pdf_bytes"] = pdf_data
                        st.success(t.get("dl_success_pdf", "PDF 报告就绪！"))
                    except Exception as e:
                        st.session_state.pop("full_report_pdf_bytes", None)
                        st.error(f"{t.get('dl_fail_pdf', 'PDF 生成失败: ')}{str(e)}")

            if st.session_state.get("full_report_pdf_bytes"):
                st.download_button(
                    label=t.get("dl_down_pdf", "📥 下载 PDF"),
                    data=st.session_state["full_report_pdf_bytes"],
                    file_name="Semiconductor_Analysis_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
