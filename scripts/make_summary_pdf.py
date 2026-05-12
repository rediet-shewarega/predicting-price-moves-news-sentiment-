"""Generate summary.pdf using fpdf2 with Arial TTF for full Unicode support."""
from fpdf import FPDF, XPos, YPos

FONT_DIR  = "/System/Library/Fonts/Supplemental"
ARIAL_REG = f"{FONT_DIR}/Arial.ttf"
ARIAL_B   = f"{FONT_DIR}/Arial Bold.ttf"
ARIAL_I   = f"{FONT_DIR}/Arial Italic.ttf"
ARIAL_BI  = f"{FONT_DIR}/Arial Bold Italic.ttf"

BLUE  = (30, 58, 138)
MID   = (37, 99, 235)
LIGHT = (239, 246, 255)
GRAY  = (100, 116, 139)
BLACK = (15, 23, 42)
WHITE = (255, 255, 255)
ROW2  = (248, 250, 252)
GREEN = (5, 150, 105)


class PDF(FPDF):
    def header(self):
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 210, 8, "F")

    def footer(self):
        self.set_y(-14)
        self.set_font("Arial", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(
            0, 8,
            f"Page {self.page_no()} / {{nb}}  \u00b7  Nova Financial Solutions \u2014 Interim Submission, May 2026",
            align="C",
        )

    def section_bar(self, title: str):
        self.ln(4)
        self.set_fill_color(*MID)
        self.set_text_color(*WHITE)
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(*BLACK)
        self.ln(2)

    def sub(self, title: str):
        self.set_font("Arial", "B", 10)
        self.set_text_color(*BLUE)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)

    def body(self, text: str, indent: float = 0):
        self.set_font("Arial", "", 9.5)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def kv_row(self, key: str, value: str):
        self.set_font("Arial", "B", 9.5)
        self.set_x(self.l_margin + 4)
        self.cell(52, 6, key)
        self.set_font("Arial", "", 9.5)
        self.multi_cell(0, 6, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def table(self, headers: list, rows: list, col_widths: list | None = None):
        usable = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            w = usable / len(headers)
            col_widths = [w] * len(headers)

        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Arial", "B", 8.5)
        for h, cw in zip(headers, col_widths):
            self.cell(cw, 7, f"  {h}", border=0, fill=True)
        self.ln()

        self.set_text_color(*BLACK)
        self.set_font("Arial", "", 8.5)
        for r_idx, row in enumerate(rows):
            fill_color = LIGHT if r_idx % 2 == 0 else ROW2
            self.set_fill_color(*fill_color)
            x0, y0 = self.l_margin, self.get_y()
            row_h = 6.5
            for cell_txt, cw in zip(row, col_widths):
                self.set_xy(x0, y0)
                self.multi_cell(
                    cw, 6.5, f"  {cell_txt}",
                    border=0, fill=True,
                    new_x=XPos.RIGHT, new_y=YPos.TOP,
                )
                x0 += cw
            # Measure actual height by checking how far each cell went
            heights = []
            tmp_x = self.l_margin
            for cell_txt, cw in zip(row, col_widths):
                lines = max(1, int(self.get_string_width(str(cell_txt)) / (cw - 6)) + 1)
                heights.append(lines * row_h)
                tmp_x += cw
            self.set_xy(self.l_margin, y0 + max(heights))
        self.ln(3)

    def callout(self, text: str):
        self.set_fill_color(*LIGHT)
        self.set_font("Arial", "I", 9)
        self.set_text_color(30, 64, 175)
        self.multi_cell(
            0, 5.5, f"  \u2192  {text}",
            fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.set_text_color(*BLACK)
        self.ln(2)


pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_font("Arial",  "",  ARIAL_REG)
pdf.add_font("Arial",  "B", ARIAL_B)
pdf.add_font("Arial",  "I", ARIAL_I)
pdf.add_font("Arial",  "BI", ARIAL_BI)
pdf.add_page()
pdf.set_margins(18, 18, 18)

# Title block
pdf.ln(4)
pdf.set_font("Arial", "B", 18)
pdf.set_text_color(*BLUE)
pdf.cell(0, 10, "Predicting Price Moves from News Sentiment",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font("Arial", "", 10)
pdf.set_text_color(*GRAY)
pdf.cell(0, 6, "Week 1 Interim Summary  \u00b7  Nova Financial Solutions",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 6, "Rediet Shewarega  \u00b7  10 May 2026",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(*BLACK)
pdf.ln(3)
pdf.set_draw_color(*MID)
pdf.set_line_width(0.6)
pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
pdf.ln(4)

# What Was Built
pdf.section_bar("What Was Built")
pdf.body(
    "A two-notebook analytical pipeline linking financial news sentiment to stock "
    "price movements, covering Tasks 1 and 2 of the Nova Financial Solutions challenge. "
    "The pipeline is fully reproducible, CI-tested via GitHub Actions, and ready to "
    "be re-run on the full FNSPID dataset."
)

# Task 1
pdf.section_bar("Task 1 \u2014 Exploratory Data Analysis")
pdf.body("Dataset: 800 news articles \u00b7 3 tickers (AAPL, MSFT, GOOGL) \u00b7 Jan 2023 onwards")

pdf.table(
    ["Finding", "Detail"],
    [
        ["Headline length",
         "Mean 138 chars / 19 words \u2014 uniform, wire-service style"],
        ["Top publishers",
         "benzinga.com 17%, reuters.com 16%, thefly.com 16%, marketwatch.com 15% "
         "\u2014 top 4 cover 64% of articles"],
        ["Publication timing",
         "Bimodal: peaks 07:00\u201309:00 ET (pre-market) and 15:00\u201316:00 ET (close)"],
        ["Data quality",
         "0% unparseable timestamps after stripping UTC-4 suffix"],
    ],
    col_widths=[46, 128],
)

pdf.sub("4 LDA Topics Identified in Headlines")
pdf.ln(1)
topics = [
    ("1. Biotech catalyst",  "FDA approval, pipeline, stock jumps"),
    ("2. Earnings surprise", "Beat estimates, shares rise, downgrade"),
    ("3. Analyst upgrade",   "Price target raised, bullish"),
    ("4. Income / yield",    "Dividend increased, yield, income"),
]
for name, terms in topics:
    pdf.set_font("Arial", "B", 9.5)
    pdf.set_x(pdf.l_margin + 4)
    pdf.cell(50, 6, name)
    pdf.set_font("Arial", "", 9.5)
    pdf.multi_cell(0, 6, terms, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(1)

pdf.callout(
    "These themes map to known market microstructure patterns and will be used as "
    "categorical features in Task 3 sentiment\u2013return correlation modelling."
)

# Task 2
pdf.section_bar("Task 2 \u2014 Technical Indicators (TA-Lib)")
pdf.body("Dataset: AAPL \u00b7 320 trading days \u00b7 Price range $280 \u2013 $422")

pdf.table(
    ["Indicator", "Parameters", "Last Reading (25 Mar 2024)"],
    [
        ["SMA",  "20-day, 50-day",
         "$381.77 / $370.66 \u2014 price well above both"],
        ["EMA",  "20-day",
         "$389.00 \u2014 faster trend line"],
        ["RSI",  "14-day",
         "67.3 \u2014 pulling back from overbought (peaked 75.2)"],
        ["MACD", "12/26/9",
         "+13.71 \u2014 positive momentum, no reversal signal yet"],
    ],
    col_widths=[28, 34, 112],
)

pdf.callout(
    "Price is in a strong uptrend. RSI recently reached overbought territory (75.2) "
    "and is correcting. MACD histogram expanding positively \u2014 momentum intact."
)

# Final plan
pdf.section_bar("Plan for Final Submission \u2014 12 May 2026")

pdf.table(
    ["Task", "Action"],
    [
        ["Task 1 (full data)",
         "Swap sample CSVs for full FNSPID slice; annotate spikes with FOMC/CPI dates"],
        ["Task 2 (expand)",
         "Add Bollinger Bands and ATR indicators"],
        ["Task 3 (new)",
         "VADER sentiment scores \u2192 date alignment \u2192 Pearson correlation "
         "with daily returns \u2192 investment strategy recommendations"],
    ],
    col_widths=[44, 130],
)

pdf.body(
    "Task 3 is the analytical centrepiece. Articles published after market close will "
    "be aligned to the next trading day. VADER will be the primary sentiment scorer "
    "(robust to financial jargon; no training required); TextBlob scores will serve "
    "as a secondary cross-check."
)

# Footer note
pdf.ln(2)
pdf.set_font("Arial", "I", 8.5)
pdf.set_text_color(*GRAY)
pdf.cell(
    0, 6,
    "GitHub: github.com/rediet-shewarega/predicting-price-moves-news-sentiment-"
    "  \u00b7  Branch: task-1",
    align="C",
)

out = "reports/summary.pdf"
pdf.output(out)
print(f"PDF written -> {out}")
