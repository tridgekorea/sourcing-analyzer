"""Tridge 리포트 PDF 공통 테마.

build_report.py(목업)에서 디자인 요소만 뽑아낸 모듈이다. 폰트·색 토큰·로고·페이지 템플릿(표지/본문/클로징)·
SectionHeader·KPI 카드·브랜드 표·배지·증빙 박스·구분선·클로징 블록을 제공하고, 진입점은 make_report(meta, sections).
화면 문구는 여기서 정하지 않는다 — 모든 텍스트는 호출하는 쪽(analysis_tool.py의 TEXTS)에서 넘겨받는다.
PDF 차트는 matplotlib으로 그린다(Pretendard를 파일에서 직접 등록해 폰트가 확실히 적용되도록).
"""
import io
import os
import re
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image as RLImage, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

BASE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE, "assets", "fonts")
LOGO_BLACK = os.path.join(BASE, "assets", "logo", "tridge_logo_black.png")
LOGO_WHITE = os.path.join(BASE, "assets", "logo", "tridge_logo_white.png")

# ── 브랜드 토큰 (이 값 외의 색은 쓰지 않는다. 텍스트는 Black/White만) ─────────────
BLUE = colors.HexColor("#171CFD")
BLACK = colors.HexColor("#0A0A18")
WHITE = colors.white
DARK_SUB = [colors.HexColor(h) for h in ("#0C023D", "#0A0568", "#070793", "#0706CE", "#3E50FE")]  # 위에는 흰 글씨
LIGHT_SUB = [colors.HexColor(h) for h in ("#6583FF", "#9BB3FF", "#CADDFF")]  # 위에는 검은 글씨
MID_BLUE = colors.HexColor("#3E50FE")
PALE_BLUE = colors.HexColor("#9BB3FF")
LIGHT_BLUE = colors.HexColor("#CADDFF")
HEX = {"blue": "#171CFD", "black": "#0A0A18", "white": "#FFFFFF", "mid": "#3E50FE", "pale": "#9BB3FF", "deep": "#0706CE"}
CHART_COLORS = ["#171CFD", "#0A0A18", "#3E50FE", "#9BB3FF", "#0706CE"]
# A/B/C/D 분류: (배경색, 글자색)
CLASS_STYLE = {"A": (BLUE, WHITE), "B": (MID_BLUE, WHITE), "C": (PALE_BLUE, BLACK), "D": (BLACK, WHITE)}
CLASS_HEX = {"A": "#171CFD", "B": "#3E50FE", "C": "#9BB3FF", "D": "#0A0A18"}


def black_a(a):
    """구분선·행 틴트용 (도형에만 사용, 텍스트에는 쓰지 않는다)."""
    return colors.Color(10 / 255, 10 / 255, 24 / 255, alpha=a)


def white_a(a):
    return colors.Color(1, 1, 1, alpha=a)


# ── 폰트 (Pretendard 5종) ─────────────────────────────────
W = {"XL": "Pretendard-ExtraLight", "L": "Pretendard-Light", "R": "Pretendard-Regular",
     "M": "Pretendard-Medium", "SB": "Pretendard-SemiBold"}
for _name in W.values():
    if _name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(_name, os.path.join(FONT_DIR, f"{_name}.ttf")))

# ── 레이아웃 ───────────────────────────────────────────────
PW, PH = A4
MX = 18 * mm
BODY_W = PW - 2 * MX
with Image.open(LOGO_BLACK) as _im:
    LOGO_RATIO = _im.size[0] / _im.size[1]  # 원본 픽셀 비율 유지


def draw_logo(c, path, x, y_top, h):
    c.drawImage(path, x, y_top - h, width=h * LOGO_RATIO, height=h, mask="auto")


# ── 텍스트 정리 ────────────────────────────────────────────
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF⌀-⏿☀-➿⬀-⯿︀-️‍]+ ?")


def clean_text(text):
    """Pretendard에 글리프가 없는 이모지는 지우고 ①~⑳은 '1.'~'20.'으로 바꾼다."""
    text = "" if text is None else str(text)
    text = re.sub("[①-⑳]", lambda m: f"{ord(m.group()) - 0x245F}.", text)
    return re.sub(" {2,}", " ", _EMOJI_RE.sub("", text)).strip()


def spaced(c, text, x, y, font, size, color, char_space=0.0):
    t = c.beginText(x, y)
    t.setFont(font, size)
    t.setFillColor(color)
    t.setCharSpace(char_space)
    t.textLine(clean_text(text))
    t.setCharSpace(0)  # 자간(Tc)은 PDF 그래픽 상태에 남아 이후 Paragraph 폭을 넓히므로 되돌린다
    c.drawText(t)


def fit_lines(text, font, size, width, max_lines=2, min_size=None):
    """한 줄에 안 들어가면 글자를 줄이고(최소 min_size), 그래도 넘치면 max_lines 줄로 나눈다."""
    text = clean_text(text)
    min_size = min_size or size * 0.75
    s = size
    while s > min_size and pdfmetrics.stringWidth(text, font, s) > width:
        s -= 0.5
    lines = simpleSplit(text, font, s, width) or [""]
    return lines[:max_lines], s


# ── 스타일 (목업과 동일) ───────────────────────────────────
def ps(name, font, size, lead=None, color=BLACK, align=TA_LEFT, **kw):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead or size * 1.5,
                          textColor=color, alignment=align, **kw)


S = {
    "body": ps("body", W["L"], 9.5, 15.5),
    "body_w": ps("body_w", W["L"], 10, 16, WHITE),
    "small": ps("small", W["L"], 8, 12.5),
    "cell": ps("cell", W["L"], 8.2, 11),
    "cell_r": ps("cell_r", W["R"], 8.2, 11, align=TA_RIGHT),
    "cell_b": ps("cell_b", W["M"], 8.2, 11),
    "th": ps("th", W["SB"], 7.8, 10.5, WHITE),
    "th_r": ps("th_r", W["SB"], 7.8, 10.5, WHITE, align=TA_RIGHT),
    "kpi_label": ps("kpi_label", W["M"], 8, 11),
    "kpi_val": ps("kpi_val", W["L"], 22, 26),
    "kpi_note": ps("kpi_note", W["L"], 7.6, 11),
    "h3": ps("h3", W["M"], 11, 16),
    "num": ps("num", W["XL"], 26, 28, BLUE),
    "finding": ps("finding", W["M"], 10.5, 15),
    "eyebrow_w": ps("eyebrow_w", W["SB"], 7.6, 10, WHITE),
}


def P(text, style="body", markup=False):
    """Paragraph. markup=False면 &·< 등을 이스케이프한다 (회사명에 '&'가 들어가도 깨지지 않도록)."""
    st = S[style] if isinstance(style, str) else style
    text = clean_text(text)
    return Paragraph(text if markup else escape(text), st)


def h3(text):
    return P(text, "h3")


def small(text):
    return P(text, "small")


# ── 커스텀 Flowable ────────────────────────────────────────
class SectionHeader(Flowable):
    """아이브로우(SemiBold 8pt 대문자, 자간 0.6) + 헤드라인(Light 21pt, 자간 -0.3) + 서브(Light 9.5pt)."""

    def __init__(self, eyebrow, headline, sub=None):
        super().__init__()
        self.eyebrow, self.headline, self.sub = eyebrow, headline, sub

    def wrap(self, aw, ah):
        self._lines, self._size = fit_lines(self.headline, W["L"], 21, aw, max_lines=2, min_size=16)
        self._line_h = self._size * 1.25
        self.height = 22 * mm + (len(self._lines) - 1) * self._line_h + (6 * mm if self.sub else 0)
        return aw, self.height

    def draw(self):
        c, top = self.canv, self.height
        spaced(c, self.eyebrow.upper(), 0, top - 9, W["SB"], 8, BLUE, 0.6)
        y = top - 32
        for line in self._lines:
            spaced(c, line, 0, y, W["L"], self._size, BLACK, -0.3)
            y -= self._line_h
        if self.sub:
            spaced(c, self.sub, 0, y + self._line_h - 18, W["L"], 9.5, BLACK, 0)


class Rule(Flowable):
    def __init__(self, w=None, alpha=0.12, thick=0.5):
        super().__init__()
        self.w, self.alpha, self.thick = w, alpha, thick

    def wrap(self, aw, ah):
        self._w = self.w or aw
        return self._w, 1

    def draw(self):
        self.canv.setStrokeColor(black_a(self.alpha))
        self.canv.setLineWidth(self.thick)
        self.canv.line(0, 0.5, self._w, 0.5)


class Badge(Flowable):
    """상태 배지: Blue(확정/검증) 또는 Black(미정/미검증/관세 미반영) + 라벨. 밝은 배경(C 분류)은 검은 글씨."""

    def __init__(self, text, fill=BLACK, size=7.4, text_color=WHITE):
        super().__init__()
        self.text, self.fill, self.size, self.text_color = clean_text(text), fill, size, text_color
        self.w = pdfmetrics.stringWidth(self.text, W["SB"], size) + 10
        self.h = size + 7

    def wrap(self, aw, ah):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(self.fill)
        c.roundRect(0, 0, self.w, self.h, 2, stroke=0, fill=1)
        c.setFillColor(self.text_color)
        c.setFont(W["SB"], self.size)
        c.drawString(5, 4.2, self.text)


def status_badge(text, confirmed):
    """검증/확정 = Blue, 미검증/미정/관세 미반영 = Black."""
    return Badge(text, BLUE if confirmed else BLACK)


def class_badge(cls, label):
    fill, txt = CLASS_STYLE[cls]
    return Badge(label, fill, text_color=txt)


def kpi_card(label, value, note, w, h=None):
    # 큰 숫자(Light 22pt)가 카드 폭을 넘으면 한 줄에 들어갈 때까지 글자 크기를 줄인다 (최소 14pt)
    size, avail = 22, w - 20
    while size > 14 and pdfmetrics.stringWidth(clean_text(value), W["L"], size) > avail:
        size -= 1
    val_style = S["kpi_val"] if size == 22 else ps(f"kpi_val_{size}", W["L"], size, size * 1.18)
    inner = [P(label, "kpi_label"), Spacer(1, 5), P(value, val_style), Spacer(1, 4), P(note, "kpi_note")]
    t = Table([[inner]], colWidths=[w], rowHeights=[h] if h else None)
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, black_a(0.15)),
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LINEABOVE", (0, 0), (-1, 0), 2, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def kpi_row(cards, gap=4 * mm):
    """[(label, value, note), ...] → 본문 폭에 꼭 맞는 KPI 카드 한 줄."""
    n = len(cards)
    cw = (BODY_W - gap * (n - 1)) / n
    # 설명이 두 줄인 카드가 있어도 한 줄의 카드 높이를 모두 같게 맞춘다
    h = max(kpi_card(label, value, note, cw).wrap(cw, PH)[1] for label, value, note in cards)
    cells, widths = [], []
    for i, (label, value, note) in enumerate(cards):
        if i:
            cells.append("")
            widths.append(gap)
        cells.append(kpi_card(label, value, note, cw, h))
        widths.append(cw)
    row = Table([cells], colWidths=widths)
    row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                             ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return row


def col_widths(ratios):
    """비율 목록 → 본문 폭에 정확히 맞는 열 너비."""
    total = float(sum(ratios))
    return [BODY_W * r / total for r in ratios]


def brand_table(header, rows, widths, right_cols=(), header_fill=BLUE, header_markup=False):
    """헤더 Blue(방법론 표는 Black) + White SemiBold, 본문 White/Black 3% 교차 + Black Light. 숫자 열만 우측 정렬."""
    data = [[P(h, "th_r" if i in right_cols else "th", markup=header_markup) for i, h in enumerate(header)]]
    for r in rows:
        data.append([cell if isinstance(cell, Flowable) else P(cell, "cell_r" if i in right_cols else "cell")
                     for i, cell in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_fill),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, black_a(0.08)),
    ]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), black_a(0.03)))
    t.setStyle(TableStyle(style))
    return t


def methodology_table(rows, label_w=32 * mm, header=("항목", "기준")):
    return brand_table(list(header), rows, [label_w, BODY_W - label_w], header_fill=BLACK)


def findings_block(items):
    """핵심 결론: [(번호, 굵은 한 줄, 설명), ...]."""
    out = []
    for n, head, detail in items:
        t = Table([[P(n, "num"), [P(head, "finding"), Spacer(1, 2), P(detail, "body")]]],
                  colWidths=[18 * mm, BODY_W - 18 * mm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.5, black_a(0.12))]))
        out.append(t)
    return out


def badge_note(badge, text, badge_w=24 * mm):
    """배지 + 설명 한 줄 (예: '관세 미반영' 고지). 배지가 길면 칸을 넓혀 설명과 겹치지 않게 한다."""
    badge_w = max(badge_w, badge.w + 5 * mm)
    t = Table([[badge, P(text, "small")]], colWidths=[badge_w, BODY_W - badge_w])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return t


def proof_box(eyebrow, text):
    """증빙 문구 박스: Blue 배경 + White 텍스트."""
    box = Table([[[P(eyebrow, "eyebrow_w"), Spacer(1, 4), P(text, "body_w")]]], colWidths=[BODY_W])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE),
                             ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                             ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
    return box


def dash_list(items):
    """'이 리포트로 알 수 없는 것' 같은 대시 목록."""
    out = []
    for text in items:
        r = Table([["—", P(text, "body")]], colWidths=[8 * mm, BODY_W - 8 * mm])
        r.setStyle(TableStyle([("FONTNAME", (0, 0), (0, 0), W["L"]), ("TEXTCOLOR", (0, 0), (0, 0), BLUE),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, black_a(0.08))]))
        out.append(r)
    return out


# ── 차트 (matplotlib) ──────────────────────────────────────
_MPL = {}


def _plt():
    """matplotlib을 처음 쓸 때만 불러오고 Pretendard를 등록한다 (앱 시작 속도 영향 없음)."""
    if "plt" not in _MPL:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for name in W.values():
            font_manager.fontManager.addfont(os.path.join(FONT_DIR, f"{name}.ttf"))
        matplotlib.rcParams["font.family"] = "Pretendard"
        matplotlib.rcParams["axes.unicode_minus"] = False
        _MPL["plt"], _MPL["mpl"] = plt, matplotlib
    return _MPL["plt"]


def new_chart(height_in):
    """본문 폭에 맞춘 차트 캔버스 (fig, ax). 흰 배경, 브랜드 색 순서."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(BODY_W / 72, height_in), dpi=300)
    apply_pdf_chart_style(ax)
    return fig, ax


def apply_pdf_chart_style(ax, grid_axis="both"):
    """PDF 차트 공통 스타일: 테두리 없음, 그리드 Black 8%, 축 글자 Black·Pretendard, 차트 제목 없음."""
    from cycler import cycler
    ax.set_prop_cycle(cycler(color=CHART_COLORS))
    for spine in ax.spines.values():
        spine.set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=HEX["black"], alpha=0.08, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("")
    ax.tick_params(axis="both", length=0, colors=HEX["black"], labelsize=7.5)
    ax.figure.set_facecolor("white")
    ax.set_facecolor("white")
    return ax


def chart_image(fig):
    """matplotlib figure → 본문 폭에 꼭 맞는 PDF 이미지 (300dpi)."""
    plt = _plt()
    buf = io.BytesIO()
    w_in, h_in = fig.get_size_inches()
    fig.savefig(buf, format="png", dpi=300, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=BODY_W, height=BODY_W * h_in / w_in)


# ── 페이지 템플릿 ──────────────────────────────────────────
DEFAULT_LABELS = {
    "period": "분석 기간", "data": "데이터", "date": "작성일", "scope": "공개 범위",
    "data_source": "Tridge TDS · 한국 수입 통관 데이터",
    "blind_on": "Blind — 한국 수입사명 비공개", "blind_off": "내부용",
    "confidential": "Confidential · 본 자료는 통관 데이터 기반 추정치이며 투자·계약 판단의 단일 근거로 사용할 수 없습니다.",
    "footer": "Source: Tridge TDS · VWAP(물량가중평균) 기준 · USD/kg",
    "site": "tridge.com",
}


def _page_callbacks(meta):
    L = {**DEFAULT_LABELS, **meta.get("labels", {})}

    def on_cover(c, doc):
        c.saveState()
        c.setFillColor(BLACK)
        c.rect(0, 0, PW, PH, stroke=0, fill=1)
        draw_logo(c, LOGO_WHITE, MX, PH - 20 * mm, 7 * mm)
        y = PH * 0.52
        spaced(c, meta.get("eyebrow", "REPORT").upper(), MX, y + 46 * mm, W["SB"], 9, WHITE, 0.7)
        c.setStrokeColor(BLUE)
        c.setLineWidth(2)
        c.line(MX, y + 40 * mm, MX + 14 * mm, y + 40 * mm)
        # 헤드라인: ExtraLight 40pt (길면 32pt까지 줄이고 줄바꿈), 최대 3줄
        lines = []
        size = 40
        for t in meta.get("title_lines", []):
            ls, s = fit_lines(t, W["XL"], 40, BODY_W, max_lines=2, min_size=32)
            lines += ls
            size = min(size, s)
        lines = lines[:3]
        line_h = size * 1.15
        top_y = y + 22 * mm  # 첫 줄 위치 고정: 줄이 늘면 아래로 내려가 아이브로우·구분선과 겹치지 않는다
        for i, line in enumerate(lines):
            spaced(c, line, MX, top_y - i * line_h, W["XL"], size, WHITE, -0.4)
        sub = " | ".join(p for p in [meta.get("spec"), f"HS {meta['hs']}" if meta.get("hs") else None] if p)
        if sub:
            spaced(c, sub, MX, top_y - (len(lines) - 1) * line_h - 12 * mm, W["L"], 12, WHITE, 0)
        rows = [(L["period"], meta.get("period", "")), (L["data"], L["data_source"]),
                (L["date"], meta.get("date", "")), (L["scope"], L["blind_on"] if meta.get("blind") else L["blind_off"])]
        yy = 46 * mm
        c.setStrokeColor(white_a(0.25))
        c.setLineWidth(0.5)
        c.line(MX, yy + 9 * mm, PW - MX, yy + 9 * mm)
        colw = BODY_W / 4
        for i, (k, v) in enumerate(rows):
            x = MX + i * colw
            spaced(c, k, x, yy + 3 * mm, W["SB"], 7.5, WHITE, 0.4)
            p = Paragraph(escape(clean_text(v)), ps("m", W["L"], 8.6, 12.5, WHITE))
            _, h = p.wrap(colw - 6 * mm, 30 * mm)
            p.drawOn(c, x, yy - h - 1 * mm)
        spaced(c, L["confidential"], MX, 16 * mm, W["L"], 7.2, WHITE, 0)
        c.restoreState()

    def on_body(c, doc):
        c.saveState()
        c.setFillColor(WHITE)
        c.rect(0, 0, PW, PH, stroke=0, fill=1)
        draw_logo(c, LOGO_BLACK, MX, PH - 13 * mm, 4.6 * mm)
        right = " · ".join(p for p in [meta.get("item"), meta.get("spec")] if p)
        lines, size = fit_lines(right, W["L"], 7.5, BODY_W * 0.6, max_lines=1, min_size=6.5)
        txt = lines[0] if lines else ""
        spaced(c, txt, PW - MX - pdfmetrics.stringWidth(txt, W["L"], size), PH - 16.4 * mm, W["L"], size, BLACK)
        c.setStrokeColor(black_a(0.12))
        c.setLineWidth(0.5)
        c.line(MX, 15 * mm, PW - MX, 15 * mm)
        spaced(c, L["footer"], MX, 10.5 * mm, W["L"], 7.2, BLACK)
        pn = f"{doc.page:02d}"
        spaced(c, pn, PW - MX - pdfmetrics.stringWidth(pn, W["M"], 7.6), 10.5 * mm, W["M"], 7.6, BLACK)
        c.restoreState()

    def on_closing(c, doc):
        c.saveState()
        c.setFillColor(BLACK)
        c.rect(0, 0, PW, PH, stroke=0, fill=1)
        draw_logo(c, LOGO_WHITE, MX, PH - 20 * mm, 7 * mm)
        spaced(c, L["site"], MX, 16 * mm, W["L"], 8, WHITE)
        c.restoreState()

    return on_cover, on_body, on_closing


class ClosingBlock(Flowable):
    """마지막 페이지(Black 배경): 아이브로우 + 헤드라인 + 권장 사항 3줄 + 데이터 한계 고지."""

    def __init__(self, eyebrow, headline, items, note):
        super().__init__()
        self.eyebrow, self.headline, self.items, self.note = eyebrow, headline, items, note

    def wrap(self, aw, ah):
        self.aw = aw
        return aw, ah - 1

    def draw(self):
        c = self.canv
        h = PH - 70 * mm
        spaced(c, self.eyebrow.upper(), 0, h - 12, W["SB"], 9, WHITE, 0.7)
        c.setStrokeColor(BLUE)
        c.setLineWidth(2)
        c.line(0, h - 22, 14 * mm, h - 22)
        lines, size = fit_lines(self.headline, W["XL"], 30, self.aw, max_lines=2, min_size=24)
        y_head = h - 58
        for i, line in enumerate(lines):
            spaced(c, line, 0, y_head - i * size * 1.15, W["XL"], size, WHITE, -0.4)
        y = h - 100 - (len(lines) - 1) * size * 1.15
        for n, k, v in self.items:
            c.setStrokeColor(white_a(0.2))
            c.setLineWidth(0.5)
            c.line(0, y + 16, self.aw, y + 16)
            spaced(c, n, 0, y - 8, W["XL"], 22, WHITE)
            spaced(c, k, 22 * mm, y - 1, W["SB"], 9, WHITE, 0.3)
            p = Paragraph(escape(clean_text(v)), ps("cv", W["L"], 11, 16, WHITE))
            _, ph_ = p.wrap(self.aw - 22 * mm, 40)
            p.drawOn(c, 22 * mm, y - 6 - ph_)
            y -= max(62, ph_ + 30)
        c.setStrokeColor(white_a(0.2))
        c.line(0, y + 16, self.aw, y + 16)
        p = Paragraph(escape(clean_text(self.note)), ps("dn", W["L"], 8.4, 13, WHITE))
        _, ph_ = p.wrap(self.aw, 60)
        p.drawOn(c, 0, y - 10 - ph_)


# ── 진입점 ────────────────────────────────────────────────
def make_report(meta, sections):
    """표지 + 본문 섹션들 + Black 클로징으로 PDF를 만들어 bytes로 반환한다.

    meta: eyebrow, title_lines(list), item, spec, hs, period, date, blind(bool), doc_title, labels(dict, 선택)
    sections: [{'kind': 'body', 'flow': [Flowable, ...]}, ...,
               {'kind': 'closing', 'eyebrow', 'headline', 'items': [(번호, 항목, 내용) x3], 'note'}]
    본문 섹션마다 새 페이지에서 시작한다. 섹션 안에서 PageBreak를 넣어 여러 페이지로 나눌 수 있다."""
    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=MX, rightMargin=MX, topMargin=26 * mm, bottomMargin=22 * mm,
                          title=clean_text(meta.get("doc_title", "")), author="Tridge")
    body_frame = Frame(MX, 22 * mm, BODY_W, PH - 48 * mm, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    closing_frame = Frame(MX, 30 * mm, BODY_W, PH - 70 * mm, id="close", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    on_cover, on_body, on_closing = _page_callbacks(meta)
    doc.addPageTemplates([
        PageTemplate("cover", frames=[body_frame], onPage=on_cover),
        PageTemplate("body", frames=[body_frame], onPage=on_body),
        PageTemplate("closing", frames=[closing_frame], onPage=on_closing),
    ])
    story = [Spacer(1, 1)]
    first_body = True
    for sec in sections:
        if sec.get("kind") == "closing":
            story += [NextPageTemplate("closing"), PageBreak(),
                      ClosingBlock(sec.get("eyebrow", "SUMMARY"), sec.get("headline", ""), sec.get("items", []), sec.get("note", ""))]
        else:
            if first_body:
                story += [NextPageTemplate("body"), PageBreak()]
                first_body = False
            else:
                story.append(PageBreak())
            story += list(sec.get("flow", []))
    doc.build(story)
    return buf.getvalue()
