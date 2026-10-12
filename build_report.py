"""Tridge 입찰 대응 리포트 — PDF 목업 (ReportLab + Pretendard, 가상 데이터)"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Table, TableStyle, Image as RLImage, Flowable, PageBreak,
                                NextPageTemplate, KeepTogether)

BASE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE, "assets", "fonts")
LOGO_BLACK = os.path.join(BASE, "assets", "logo", "tridge_logo_black.png")
LOGO_WHITE = os.path.join(BASE, "assets", "logo", "tridge_logo_white.png")
TMP = os.path.join(BASE, "_mock_charts")
os.makedirs(TMP, exist_ok=True)

# ── 브랜드 토큰 ─────────────────────────────────────────────
BLUE = colors.HexColor("#171CFD")
BLACK = colors.HexColor("#0A0A18")
WHITE = colors.white
MID_BLUE = colors.HexColor("#3E50FE")
LIGHT_BLUE = colors.HexColor("#CADDFF")
def black_a(a): return colors.Color(10/255, 10/255, 24/255, alpha=a)
def white_a(a): return colors.Color(1, 1, 1, alpha=a)

# ── 폰트 ───────────────────────────────────────────────────
W = {"XL": "Pretendard-ExtraLight", "L": "Pretendard-Light", "R": "Pretendard-Regular",
     "M": "Pretendard-Medium", "SB": "Pretendard-SemiBold"}
for name in W.values():
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONT_DIR, f"{name}.ttf")))
for name in W.values():
    font_manager.fontManager.addfont(os.path.join(FONT_DIR, f"{name}.ttf"))

# ── 레이아웃 ───────────────────────────────────────────────
PW, PH = A4
MX = 18 * mm
BODY_W = PW - 2 * MX
LOGO_RATIO = (lambda s: s[0] / s[1])(Image.open(LOGO_BLACK).size)  # 원본 비율 유지

def draw_logo(c, path, x, y_top, h):
    w = h * LOGO_RATIO
    c.drawImage(path, x, y_top - h, width=w, height=h, mask="auto")

def spaced(c, text, x, y, font, size, color, char_space=0.0):
    t = c.beginText(x, y)
    t.setFont(font, size)
    t.setFillColor(color)
    t.setCharSpace(char_space)
    t.textLine(text)
    c.drawText(t)

# ── 스타일 ─────────────────────────────────────────────────
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
}

# ── 커스텀 Flowable ────────────────────────────────────────
class SectionHeader(Flowable):
    """아이브로우(대문자, 자간 ≤0.8) + 헤드라인(Light, 자간 -0.3)"""
    def __init__(self, eyebrow, headline, sub=None):
        super().__init__()
        self.eyebrow, self.headline, self.sub = eyebrow, headline, sub
        self.height = 22 * mm + (6 * mm if sub else 0)
    def wrap(self, aw, ah):
        return aw, self.height
    def draw(self):
        c = self.canv
        top = self.height
        spaced(c, self.eyebrow.upper(), 0, top - 9, W["SB"], 8, BLUE, 0.6)
        spaced(c, self.headline, 0, top - 32, W["L"], 21, BLACK, -0.3)
        if self.sub:
            spaced(c, self.sub, 0, top - 50, W["L"], 9.5, BLACK, 0)

class Rule(Flowable):
    def __init__(self, w=None, alpha=0.12, thick=0.5):
        super().__init__(); self.w, self.alpha, self.thick = w, alpha, thick
    def wrap(self, aw, ah):
        self._w = self.w or aw; return self._w, 1
    def draw(self):
        self.canv.setStrokeColor(black_a(self.alpha)); self.canv.setLineWidth(self.thick)
        self.canv.line(0, 0.5, self._w, 0.5)

class Badge(Flowable):
    """상태 배지: Blue(확정/검증) or Black(미정/미반영) + White 라벨"""
    def __init__(self, text, fill=BLACK, size=7.4):
        super().__init__(); self.text, self.fill, self.size = text, fill, size
        self.w = pdfmetrics.stringWidth(text, W["SB"], size) + 10
        self.h = size + 7
    def wrap(self, aw, ah): return self.w, self.h
    def draw(self):
        c = self.canv
        c.setFillColor(self.fill); c.roundRect(0, 0, self.w, self.h, 2, stroke=0, fill=1)
        c.setFillColor(WHITE); c.setFont(W["SB"], self.size)
        c.drawString(5, 4.2, self.text)

def kpi_card(label, value, note, w):
    inner = [Paragraph(label, S["kpi_label"]), Spacer(1, 5),
             Paragraph(value, S["kpi_val"]), Spacer(1, 4), Paragraph(note, S["kpi_note"])]
    t = Table([[inner]], colWidths=[w])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, black_a(0.15)),
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LINEABOVE", (0, 0), (-1, 0), 2, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t

def brand_table(header, rows, col_w, right_cols=(), header_fill=BLUE):
    data = [[Paragraph(h, S["th_r"] if i in right_cols else S["th"]) for i, h in enumerate(header)]]
    for r in rows:
        data.append([cell if isinstance(cell, Flowable) else
                     Paragraph(str(cell), S["cell_r"] if i in right_cols else S["cell"])
                     for i, cell in enumerate(r)])
    t = Table(data, colWidths=col_w, repeatRows=1)
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), header_fill),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, black_a(0.08)),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), black_a(0.03)))
    t.setStyle(TableStyle(st))
    return t

# ── 페이지 템플릿 ──────────────────────────────────────────
META = {
    "item": "냉동 감자튀김", "spec": "스트레이트컷 3/8\" · 코팅",
    "hs": "2004.10.1000", "period": "2025.10 – 2026.09 (12개월)",
    "date": "2026.10.08", "blind": True, "client": "고객사 제출용",
}

def on_cover(c, doc):
    c.saveState()
    c.setFillColor(BLACK); c.rect(0, 0, PW, PH, stroke=0, fill=1)
    draw_logo(c, LOGO_WHITE, MX, PH - 20 * mm, 7 * mm)
    # 아이브로우 / 헤드라인
    y = PH * 0.52
    spaced(c, "BID RESPONSE REPORT", MX, y + 46 * mm, W["SB"], 9, WHITE, 0.7)
    c.setStrokeColor(BLUE); c.setLineWidth(2); c.line(MX, y + 40 * mm, MX + 14 * mm, y + 40 * mm)
    spaced(c, META["item"], MX, y + 22 * mm, W["XL"], 40, WHITE, -0.4)
    spaced(c, "입찰 대응 리포트", MX, y + 6 * mm, W["XL"], 40, WHITE, -0.4)
    spaced(c, f"{META['spec']}  |  HS {META['hs']}", MX, y - 6 * mm, W["L"], 12, WHITE, 0)
    # 메타
    rows = [("분석 기간", META["period"]), ("데이터", "Tridge TDS · 한국 수입 통관 데이터"),
            ("작성일", META["date"]), ("공개 범위", "Blind — 한국 수입사명 비공개" if META["blind"] else "내부용")]
    yy = 46 * mm
    c.setStrokeColor(white_a(0.25)); c.setLineWidth(0.5)
    c.line(MX, yy + 9 * mm, PW - MX, yy + 9 * mm)
    colw = BODY_W / 4
    for i, (k, v) in enumerate(rows):
        x = MX + i * colw
        spaced(c, k, x, yy + 3 * mm, W["SB"], 7.5, WHITE, 0.4)
        p = Paragraph(v, ps("m", W["L"], 8.6, 12.5, WHITE))
        _, h = p.wrap(colw - 6 * mm, 30 * mm)
        p.drawOn(c, x, yy - h - 1 * mm)
    spaced(c, "Confidential · 본 자료는 통관 데이터 기반 추정치이며 투자·계약 판단의 단일 근거로 사용할 수 없습니다.",
           MX, 16 * mm, W["L"], 7.2, WHITE, 0)
    c.restoreState()

def on_body(c, doc):
    c.saveState()
    c.setFillColor(WHITE); c.rect(0, 0, PW, PH, stroke=0, fill=1)
    draw_logo(c, LOGO_BLACK, MX, PH - 13 * mm, 4.6 * mm)
    spaced(c, f"{META['item']} · {META['spec']}", PW - MX - pdfmetrics.stringWidth(
        f"{META['item']} · {META['spec']}", W["L"], 7.5), PH - 16.4 * mm, W["L"], 7.5, BLACK)
    c.setStrokeColor(black_a(0.12)); c.setLineWidth(0.5)
    c.line(MX, 15 * mm, PW - MX, 15 * mm)
    spaced(c, "Source: Tridge TDS · VWAP(물량가중평균) 기준 · USD/kg", MX, 10.5 * mm, W["L"], 7.2, BLACK)
    pn = f"{doc.page:02d}"
    spaced(c, pn, PW - MX - pdfmetrics.stringWidth(pn, W["M"], 7.6), 10.5 * mm, W["M"], 7.6, BLACK)
    c.restoreState()

def on_closing(c, doc):
    c.saveState()
    c.setFillColor(BLACK); c.rect(0, 0, PW, PH, stroke=0, fill=1)
    draw_logo(c, LOGO_WHITE, MX, PH - 20 * mm, 7 * mm)
    spaced(c, "tridge.com", MX, 16 * mm, W["L"], 8, WHITE)
    c.restoreState()

# ── 가상 데이터 ────────────────────────────────────────────
rng = np.random.default_rng(7)
suppliers = [
    # 공급사, 원산지, VWAP, 건수, 톤, 거래월, 수입사수, 검증
    ("Kesterhoff Foods NV", "벨기에", 1.08, 41, 1820, 12, 4, True),
    ("Nordvale Potato BV", "네덜란드", 1.12, 36, 1540, 12, 5, True),
    ("Prairie Gold Frozen", "미국", 1.17, 28, 1210, 11, 3, True),
    ("Polder Fries NV", "벨기에", 1.19, 22, 860, 10, 2, True),
    ("Andes Valley Foods", "칠레", 1.06, 4, 96, 3, 1, False),
    ("Sutlej Agro Processing", "인도", 0.98, 3, 58, 2, 1, False),
]
INCUMBENT = ("공급사 X (현 납품 기준)", "미국", 1.27)
ANNUAL_T = 2000

def chart_origin_range(path):
    origins = ["벨기에", "네덜란드", "미국", "칠레", "인도", "뉴질랜드"]
    vwap = [1.11, 1.12, 1.21, 1.06, 0.98, 1.24]
    p10 = [1.01, 1.04, 1.10, 0.99, 0.91, 1.16]
    p90 = [1.24, 1.21, 1.33, 1.12, 1.04, 1.31]
    tons = [6420, 4980, 3870, 120, 70, 410]
    order = np.argsort(vwap)[::-1]
    fig, ax = plt.subplots(figsize=(BODY_W / 72, 2.75), dpi=300)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(axis="x", color="#0A0A18", alpha=0.08, linewidth=0.6); ax.set_axisbelow(True)
    for yi, i in enumerate(order):
        thin = tons[i] < 500
        ax.plot([p10[i], p90[i]], [yi, yi], color="#0A0A18", alpha=0.25, linewidth=2.2,
                solid_capstyle="round")
        ax.scatter([vwap[i]], [yi], s=46, color="#FFFFFF" if thin else "#171CFD",
                   edgecolor="#171CFD", linewidth=1.4, zorder=3)
        ax.text(vwap[i], yi + 0.3, f"${vwap[i]:.2f}", va="bottom", ha="center", fontsize=7.2,
                fontfamily="Pretendard", fontweight="regular", color="#0A0A18")
    ax.axvline(INCUMBENT[2], color="#0A0A18", linewidth=1, linestyle=(0, (3, 2)))
    ax.text(INCUMBENT[2] + 0.006, len(order) - 0.45, "현 납품 기준 $1.27", fontsize=7.2,
            fontfamily="Pretendard", color="#0A0A18", va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{origins[i]}  ({tons[i]:,}t)" for i in order], fontsize=8,
                       fontfamily="Pretendard", color="#0A0A18")
    ax.tick_params(axis="both", length=0, colors="#0A0A18", labelsize=7.5)
    for lab in ax.get_xticklabels(): lab.set_fontfamily("Pretendard")
    ax.set_xlim(0.85, 1.42); ax.set_ylim(-0.6, len(order) - 0.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    fig.subplots_adjust(left=0.2, right=0.99, top=0.97, bottom=0.11)
    fig.savefig(path, transparent=False, facecolor="white"); plt.close(fig)

def chart_monthly(path, base, gaps=()):
    months = 36
    vals = np.clip(base + rng.normal(0, base * 0.22, months), base * 0.35, None)
    for g in gaps: vals[g] = 0
    labels = []
    y, m = 2023, 10
    for _ in range(months):
        labels.append(f"{str(y)[2:]}.{m:02d}"); m += 1
        if m > 12: m, y = 1, y + 1
    fig, ax = plt.subplots(figsize=(BODY_W / 72, 2.1), dpi=300)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(axis="y", color="#0A0A18", alpha=0.08, linewidth=0.6); ax.set_axisbelow(True)
    ax.bar(range(months), vals, width=0.72, color="#171CFD", edgecolor="white", linewidth=0)
    for g in gaps:
        ax.bar([g], [base * 0.12], width=0.72, color="white", edgecolor="#0A0A18",
               linewidth=0.8, hatch="////")
    ax.set_xticks(range(0, months, 3)); ax.set_xticklabels(labels[::3], fontsize=7, fontfamily="Pretendard")
    ax.tick_params(length=0, colors="#0A0A18", labelsize=7)
    for lab in ax.get_yticklabels(): lab.set_fontfamily("Pretendard")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}t"))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.96, bottom=0.14)
    fig.savefig(path, facecolor="white"); plt.close(fig)
    return vals

def chart_img(path, h_in):
    return RLImage(path, width=BODY_W, height=BODY_W * (h_in / (BODY_W / 72)))

# ── 본문 구성 ──────────────────────────────────────────────
def build(out):
    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=MX, rightMargin=MX,
                          topMargin=26 * mm, bottomMargin=22 * mm,
                          title="냉동 감자튀김 입찰 대응 리포트", author="Tridge")
    body_frame = Frame(MX, 22 * mm, BODY_W, PH - 48 * mm, id="body",
                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    closing_frame = Frame(MX, 30 * mm, BODY_W, PH - 70 * mm, id="close",
                          leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate("cover", frames=[body_frame], onPage=on_cover),
        PageTemplate("body", frames=[body_frame], onPage=on_body),
        PageTemplate("closing", frames=[closing_frame], onPage=on_closing),
    ])
    st = [NextPageTemplate("body"), PageBreak()]

    # 02 요약
    st.append(SectionHeader("01  Executive Summary", "검증된 대체 공급처로 단가 15% 낮출 수 있습니다"))
    st.append(Spacer(1, 3 * mm))
    gap = 4 * mm
    cw = (BODY_W - gap * 3) / 4
    best = suppliers[0]
    save = (INCUMBENT[2] - best[2]) / INCUMBENT[2]
    cards = [
        kpi_card("최저 검증 공급처 단가", f"${best[2]:.2f}", f"USD/kg · CIF · {best[1]}", cw),
        kpi_card("현 납품 기준 대비", f"–{save*100:.1f}%", f"기준 ${INCUMBENT[2]:.2f}/kg", cw),
        kpi_card("연간 절감 추정", f"${(INCUMBENT[2]-best[2])*ANNUAL_T*1000/1000:,.0f}K",
                 f"연 {ANNUAL_T:,}t 가정", cw),
        kpi_card("공급 공백", "0개월", "최근 36개월 · 1순위 공급사", cw),
    ]
    row = Table([[cards[0], "", cards[1], "", cards[2], "", cards[3]]],
                colWidths=[cw, gap, cw, gap, cw, gap, cw])
    row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                             ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    st += [row, Spacer(1, 9 * mm)]

    st.append(Paragraph("핵심 결론", S["h3"])); st.append(Spacer(1, 2 * mm)); st.append(Rule())
    findings = [
        ("01", "벨기에·네덜란드산이 미국산보다 kg당 $0.09~0.15 낮습니다.",
         "같은 스펙(스트레이트컷 3/8\", 코팅) 안에서만 비교했습니다. 두 원산지 모두 월 평균 400톤 이상 꾸준히 들어오고 있습니다."),
        ("02", "검증 통과 공급사 4곳 중 2곳이 현 납품 기준보다 10% 이상 저렴합니다.",
         "검증 기준은 최근 12개월 3건 이상, 10톤 이상, 한국 수입사 2곳 이상 공급입니다."),
        ("03", "칠레·인도산은 단가는 낮지만 아직 검증 전 단계입니다.",
         "거래 실적이 3~4건으로 적어 입찰 제안의 주력 공급처로는 권하지 않습니다. 보조 옵션으로만 검토하세요."),
    ]
    for n, h, d in findings:
        t = Table([[Paragraph(n, S["num"]),
                    [Paragraph(h, ps("fh", W["M"], 10.5, 15)), Spacer(1, 2), Paragraph(d, S["body"])]]],
                  colWidths=[18 * mm, BODY_W - 18 * mm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.5, black_a(0.12))]))
        st.append(t)
    st.append(Spacer(1, 7 * mm))
    note = Table([[Badge("관세 미반영"), Paragraph(
        "원산지별 관세율이 입력되지 않아 CIF 기준으로 비교했습니다. 원산지 간 비교 시 FTA 적용 세율에 따라 순위가 바뀔 수 있습니다.",
        S["small"])]], colWidths=[24 * mm, BODY_W - 24 * mm])
    note.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    st.append(note)

    # 03 대체 공급처
    st.append(PageBreak())
    st.append(SectionHeader("02  Alternative Suppliers", "대체 공급처 비교",
                            "같은 스펙 안에서 원산지·공급사별 VWAP와 검증 지표를 비교했습니다."))
    st.append(Spacer(1, 2 * mm))
    st.append(Paragraph("원산지별 단가 분포", S["h3"]))
    st.append(Paragraph("점 = VWAP, 막대 = P10~P90 범위. 속이 빈 점은 연 500톤 미만 원산지입니다.", S["small"]))
    st.append(Spacer(1, 2 * mm))
    p1 = os.path.join(TMP, "origin.png"); chart_origin_range(p1)
    st.append(chart_img(p1, 2.75)); st.append(Spacer(1, 7 * mm))

    st.append(Paragraph("공급사 순위 (비교단가 기준)", S["h3"])); st.append(Spacer(1, 2.5 * mm))
    header = ["#", "해외 공급사", "원산지", "VWAP<br/>USD/kg", "기준 대비", "건수", "물량(t)", "거래월", "공급<br/>수입사", "상태"]
    rows = []
    for i, s in enumerate(sorted(suppliers, key=lambda x: (not x[7], x[2])), 1):
        diff = (s[2] - INCUMBENT[2]) / INCUMBENT[2] * 100
        rows.append([str(i), s[0], s[1], f"{s[2]:.2f}", f"{diff:+.1f}%", s[3], f"{s[4]:,}", s[5], f"{s[6]}곳",
                     Badge("검증", BLUE) if s[7] else Badge("미검증", BLACK)])
    rows.append(["–", INCUMBENT[0], INCUMBENT[1], f"{INCUMBENT[2]:.2f}", "기준", "–", "–", "–", "–", ""])
    cw_ = [7, 44, 17, 17, 17, 11, 15, 13, 14, 19]
    cw_ = [w * mm * BODY_W / (sum(cw_) * mm) for w in cw_]
    tbl = brand_table(header, rows, cw_, right_cols=(3, 4, 5, 6, 7, 8))
    tbl.setStyle(TableStyle([("LINEABOVE", (0, len(rows)), (-1, len(rows)), 0.8, BLACK)]))
    st.append(tbl)
    st.append(Spacer(1, 3 * mm))
    st.append(Paragraph("검증 = 최근 12개월 3건 이상 · 10톤 이상 · 한국 수입사 2곳 이상 공급. "
                        "공급 수입사는 실명 없이 수만 표기합니다.", S["small"]))

    # 04 공급 안정성
    for idx, (sup, base, gaps) in enumerate([(suppliers[0], 150, ()), (suppliers[1], 128, (5, 19))]):
        st.append(PageBreak())
        st.append(SectionHeader(f"03  Supply Track Record · {idx+1}/2", sup[0],
                                f"{sup[1]} · 최근 36개월 한국향 선적 실적"))
        st.append(Spacer(1, 2 * mm))
        p = os.path.join(TMP, f"monthly_{idx}.png")
        vals = chart_monthly(p, base, gaps)
        total = vals.sum(); n_gap = int((vals == 0).sum())
        cv = vals[vals > 0].std() / vals[vals > 0].mean()
        cw4 = (BODY_W - gap * 3) / 4
        cards = [kpi_card("연평균 공급량", f"{total/3:,.0f}t", "최근 36개월 기준", cw4),
                 kpi_card("공백 월", f"{n_gap}개월", "선적 0건인 달", cw4),
                 kpi_card("거래 한국 수입사", f"{sup[6]}곳", "실명 비공개", cw4),
                 kpi_card("단가 변동성", f"{cv*100:.0f}%", "월별 VWAP 변동계수", cw4)]
        r = Table([[cards[0], "", cards[1], "", cards[2], "", cards[3]]], colWidths=[cw4, gap, cw4, gap, cw4, gap, cw4])
        r.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        st += [r, Spacer(1, 8 * mm)]
        st.append(Paragraph("월별 선적 물량", S["h3"]))
        st.append(Paragraph("빗금 표시는 선적이 없었던 달입니다.", S["small"]) if gaps else
                  Paragraph("36개월 동안 선적이 끊긴 달이 없습니다.", S["small"]))
        st.append(Spacer(1, 2 * mm)); st.append(chart_img(p, 2.1)); st.append(Spacer(1, 8 * mm))
        st.append(Paragraph("연도별 요약", S["h3"])); st.append(Spacer(1, 2.5 * mm))
        yrows = []
        for yi, label in enumerate(["2023.10 – 2024.09", "2024.10 – 2025.09", "2025.10 – 2026.09"]):
            seg = vals[yi * 12:(yi + 1) * 12]
            yrows.append([label, f"{seg.sum():,.0f}", f"{int((seg > 0).sum())}/12",
                          f"{sup[2] * (0.93 + 0.035 * yi):.2f}", f"{max(1, sup[6] - 2 + yi)}곳"])
        st.append(brand_table(["기간", "선적 물량(t)", "선적 월", "VWAP USD/kg", "공급 수입사"], yrows,
                              [BODY_W * 0.28] + [BODY_W * 0.18] * 4, right_cols=(1, 2, 3, 4)))
        st.append(Spacer(1, 8 * mm))
        stmt = (f"{sup[0]}는 최근 3년간 한국에 연평균 {total/3:,.0f}톤을 "
                + ("공백 없이 공급했습니다." if n_gap == 0 else f"공급했으며, 공백 월은 {n_gap}개월입니다.")
                + f" 같은 기간 한국 수입사 {sup[6]}곳에 납품해 통관·검역 실적이 확인됩니다.")
        box = Table([[[spaced_par("PROOF STATEMENT"), Spacer(1, 4), Paragraph(stmt, S["body_w"])]]],
                    colWidths=[BODY_W])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                                 ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
        st.append(box)

    # 05 방법론
    st.append(PageBreak())
    st.append(SectionHeader("04  Methodology", "분석 기준과 데이터 한계"))
    st.append(Spacer(1, 2 * mm))
    meth = [
        ("데이터", "Tridge TDS 한국 수입 통관 데이터, 건별 수입신고 단위"),
        ("기간", "2025.10 – 2026.09 (12개월). 공급 안정성은 36개월"),
        ("스펙 분리", "HS 2004.10.1000 중 품명에 '3/8', 'COATED' 포함, 'CRINKLE'·'WEDGE' 제외"),
        ("중복 제거", "신고일·수입자·공급자·중량·금액이 모두 같은 행 1건으로 처리"),
        ("이상치 제외", "건당 1톤 미만 소량 수입, 단가 P1 미만·P99 초과"),
        ("평균 단가", "VWAP(물량가중평균), USD/kg, CIF 기준"),
        ("관세", "미반영. 원산지별 세율을 입력하면 비교단가 = CIF × (1 + 세율)"),
        ("블라인드", "한국 수입사명 비공개. 해외 공급사명은 공개"),
    ]
    st.append(brand_table(["항목", "기준"], meth, [32 * mm, BODY_W - 32 * mm], header_fill=BLACK))
    st.append(Spacer(1, 8 * mm))
    st.append(Paragraph("이 리포트로 알 수 없는 것", S["h3"])); st.append(Spacer(1, 2 * mm)); st.append(Rule())
    for t in ["국내 통관·검역·운송·보관비 등 도착 이후 비용",
              "수입자명이 유통사·대행사로 기록된 경우의 실수요자",
              "공급사의 품질 규격 세부(수분율, 길이 분포 등)와 인증 현황",
              "향후 가격 예측. 시세 전망이 필요하면 Tridge Eye와 함께 보세요"]:
        r = Table([["—", Paragraph(t, S["body"])]], colWidths=[8 * mm, BODY_W - 8 * mm])
        r.setStyle(TableStyle([("FONTNAME", (0, 0), (0, 0), W["L"]), ("TEXTCOLOR", (0, 0), (0, 0), BLUE),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, black_a(0.08))]))
        st.append(r)

    # 06 클로징
    st += [NextPageTemplate("closing"), PageBreak()]
    st.append(ClosingBlock())
    doc.build([Spacer(1, 1)] + st)

def spaced_par(text):
    return Paragraph(text, ps("ey", W["SB"], 7.6, 10, WHITE))

class ClosingBlock(Flowable):
    def wrap(self, aw, ah): self.aw = aw; return aw, ah - 1
    def draw(self):
        c = self.canv; top = self._ah if hasattr(self, "_ah") else 0
        h = PH - 70 * mm
        spaced(c, "SUMMARY", 0, h - 12, W["SB"], 9, WHITE, 0.7)
        c.setStrokeColor(BLUE); c.setLineWidth(2); c.line(0, h - 22, 14 * mm, h - 22)
        spaced(c, "입찰 제안 시 권장 구성", 0, h - 58, W["XL"], 30, WHITE, -0.4)
        items = [("01", "주력 공급처", "Kesterhoff Foods NV (벨기에) — $1.08/kg, 검증 통과, 공백 0개월"),
                 ("02", "백업 공급처", "Nordvale Potato BV (네덜란드) — $1.12/kg, 수입사 5곳 공급"),
                 ("03", "절감 근거", "현 납품 기준 대비 –15.0%, 연 2,000t 기준 약 $380K")]
        y = h - 100
        for n, k, v in items:
            c.setStrokeColor(white_a(0.2)); c.setLineWidth(0.5); c.line(0, y + 16, self.aw, y + 16)
            spaced(c, n, 0, y - 8, W["XL"], 22, WHITE)
            spaced(c, k, 22 * mm, y - 1, W["SB"], 9, WHITE, 0.3)
            p = Paragraph(v, ps("cv", W["L"], 11, 16, WHITE))
            _, ph_ = p.wrap(self.aw - 22 * mm, 40); p.drawOn(c, 22 * mm, y - 6 - ph_)
            y -= 62
        c.setStrokeColor(white_a(0.2)); c.line(0, y + 16, self.aw, y + 16)
        p = Paragraph("관세·국내 물류비 미반영. 도착 원가 기준 최종 판단 전 원산지별 FTA 세율과 국내 비용을 확인하세요. "
                      "본 자료의 수치는 통관 데이터 기반 추정치입니다.", ps("dn", W["L"], 8.4, 13, WHITE))
        _, ph_ = p.wrap(self.aw, 60); p.drawOn(c, 0, y - 10 - ph_)

if __name__ == "__main__":
    out = os.path.join(BASE, "Tridge_입찰대응리포트_목업.pdf")
    build(out)
    print(out)
