#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
receipts_report.py
Builds a PDF report from cropped receipts and a CSV table.

Layout: LEFT text column (Chinese-only labels) + RIGHT image box with border.
- Text wraps within the left column width so it never overlaps the image box.
- Image is fit inside a fixed right-side box with padding.
"""
from pathlib import Path
from typing import Optional, List
import csv
import argparse
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------- Fonts (Noto Sans SC + fallbacks) ----------
FONT_REG = None
FONT_BOLD = None

def _try_register_ttf(name: str, path: Path) -> bool:
    try:
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
            return True
    except Exception:
        pass
    return False

def _register_fonts():
    global FONT_REG, FONT_BOLD
    here = Path(__file__).parent
    font_dir = here / "fonts"
    reg_local = font_dir / "NotoSansSC-Regular.ttf"
    bold_local = font_dir / "NotoSansSC-Bold.ttf"

    ok_reg = _try_register_ttf("NotoSansSC", reg_local)
    ok_bold = _try_register_ttf("NotoSansSC-Bold", bold_local)

    if not ok_reg:
        var_path = Path.home() / "Library" / "Fonts" / "NotoSansSC[wght].ttf"
        ok_reg = _try_register_ttf("NotoSansSCVar", var_path)
        if ok_reg:
            FONT_REG = "NotoSansSCVar"

    if ok_reg and FONT_REG is None:
        FONT_REG = "NotoSansSC"
    if ok_bold:
        FONT_BOLD = "NotoSansSC-Bold"

    if FONT_REG is None:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        FONT_REG = "STSong-Light"
        FONT_BOLD = None

_register_fonts()

def set_font(c: canvas.Canvas, size: int, bold: bool = False):
    if bold and FONT_BOLD:
        c.setFont(FONT_BOLD, size)
    elif bold and FONT_BOLD is None:
        c.setFont(FONT_REG, size + 2)
    else:
        c.setFont(FONT_REG, size)

def _strw(c: canvas.Canvas, text: str, size: int) -> float:
    """Measure string width using current canvas font."""
    return pdfmetrics.stringWidth(text, c._fontname, size)

# ---------- Text wrapping helpers (CJK-friendly, char-based) ----------
def _wrap_text(c: canvas.Canvas, text: str, font_size: int, max_width: float) -> List[str]:
    """Character-based greedy wrap: works for CJK (no spaces) and mixed text."""
    set_font(c, font_size, False)
    if not text:
        return [""]
    lines = []
    cur = ""
    for ch in text:
        test = cur + ch
        if _strw(c, test, font_size) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines

def _draw_pair_cn(c: canvas.Canvas, x: float, y: float, label_cn: str, value: str, col_width: float, lh=6*mm, label_gap=3*mm):
    """
    Draw '标签：值' with wrapping for the value within col_width.
    Returns next y (lower).
    """
    label_size = 10
    value_size = 10
    # label
    set_font(c, label_size, False)
    label_text = f"{label_cn}："
    c.drawString(x, y, label_text)
    label_w = _strw(c, label_text, label_size)

    # value (wrapped within remaining width)
    avail_w = max(10.0, col_width - label_w - label_gap)
    lines = _wrap_text(c, value or "", value_size, avail_w)
    vx = x + label_w + label_gap
    for i, line in enumerate(lines):
        set_font(c, value_size, False)
        c.drawString(vx, y - i*lh, line)
    # advance y by number of lines*lh (min 1 line)
    used_lines = max(1, len(lines))
    return y - used_lines*lh

# ---------- Image box ----------
def _draw_image_right(c: canvas.Canvas, img_path: Path, box_x: float, box_y: float, box_w: float, box_h: float, padding: float = 3*mm):
    c.setLineWidth(0.8)
    c.rect(box_x, box_y, box_w, box_h, stroke=1, fill=0)
    inner_x = box_x + padding
    inner_y = box_y + padding
    inner_w = box_w - 2*padding
    inner_h = box_h - 2*padding
    try:
        img = Image.open(img_path)
        img = img.copy()
        img_w, img_h = img.size
        scale = min(inner_w / img_w, inner_h / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        draw_x = inner_x + (inner_w - disp_w) / 2.0
        draw_y = inner_y + (inner_h - disp_h) / 2.0
        c.drawInlineImage(img, draw_x, draw_y, width=disp_w, height=disp_h)
    except Exception as e:
        set_font(c, 9, False)
        c.drawString(inner_x, inner_y + inner_h/2, f"(Image error: {e})")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Build PDF report (Chinese-only labels, right-side image box, wrapped text).")
    ap.add_argument("--crops_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output_pdf", required=True)
    ap.add_argument("--submitted_by", required=True)
    ap.add_argument("--company", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--default_type", default="sales")
    ap.add_argument("--cny_rate", type=float, default=None)

    # Layout knobs
    ap.add_argument("--margin_left_mm", type=float, default=18.0)
    ap.add_argument("--margin_right_mm", type=float, default=18.0)
    ap.add_argument("--margin_top_mm", type=float, default=15.0)
    ap.add_argument("--margin_bottom_mm", type=float, default=15.0)
    ap.add_argument("--image_box_w_mm", type=float, default=90.0)
    ap.add_argument("--image_box_h_mm", type=float, default=175.0)
    ap.add_argument("--image_box_bottom_mm", type=float, default=90.0, help="Distance from bottom to image box bottom")
    ap.add_argument("--gutter_mm", type=float, default=8.0, help="Space between text column and image box")
    args = ap.parse_args()

    crops_dir = Path(args.crops_dir)
    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    page_w, page_h = A4
    left = args.margin_left_mm * mm
    right = page_w - args.margin_right_mm * mm
    top = page_h - args.margin_top_mm * mm
    bottom = args.margin_bottom_mm * mm

    # Right image box
    box_w = args.image_box_w_mm * mm
    box_h = args.image_box_h_mm * mm
    box_x = right - box_w
    box_y = args.image_box_bottom_mm * mm  # bottom offset for the box

    # Text column width: left area up to the gutter before the image box
    gutter = args.gutter_mm * mm
    text_right_limit = box_x - gutter
    text_col_width = max(40*mm, text_right_limit - left)

    c = canvas.Canvas(args.output_pdf, pagesize=A4)

    for row in rows:
        image_file = (row.get("image_file") or "").strip()
        img_path = crops_dir / image_file

        date = (row.get("date") or row.get("date_guess") or "").strip()
        invoice_no = (row.get("invoice_no") or row.get("invoice_no_guess") or "").strip()
        amount_thb = (row.get("amount_thb") or row.get("amount_thb_guess") or "").strip()
        amount_cny = (row.get("amount_cny") or "").strip()
        purpose = (row.get("purpose") or "").strip()
        notes = (row.get("notes") or "").strip()
        itype = (row.get("invoice_type") or args.default_type).strip()

        if not amount_cny and amount_thb and args.cny_rate:
            try:
                amt = float(amount_thb.replace(",", ""))
                amount_cny = f"{amt * args.cny_rate:.2f}"
            except Exception:
                pass

        # Header
        set_font(c, 13, True)
        c.drawString(left, top, "收据记录")

        # Draw right image box
        _draw_image_right(c, img_path, box_x, box_y, box_w, box_h, padding=3*mm)

        # Left text (wrapped within text_col_width)
        y = top - 10*mm
        y = _draw_pair_cn(c, left, y, "日期", date, text_col_width)
        y = _draw_pair_cn(c, left, y, "提交人", args.submitted_by, text_col_width)
        y = _draw_pair_cn(c, left, y, "公司", args.company, text_col_width)
        label_type = "报销发票" if itype.lower().startswith("reimb") else "销售发票"
        y = _draw_pair_cn(c, left, y, "发票类型", label_type, text_col_width)
        y = _draw_pair_cn(c, left, y, "发票号码", invoice_no, text_col_width)
        amt_line = f"{amount_thb or ''} 泰铢"
        if amount_cny:
            amt_line += f"（{amount_cny} 人民币）"
        y = _draw_pair_cn(c, left, y, "金额", amt_line, text_col_width)
        y = _draw_pair_cn(c, left, y, "付款用途/描述", purpose, text_col_width)
        y = _draw_pair_cn(c, left, y, "备注", notes, text_col_width)

        c.showPage()

    c.save()
    print("Wrote", args.output_pdf)

if __name__ == "__main__":
    main()