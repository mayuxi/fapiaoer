#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
receipts_report.py
Builds a PDF report from cropped receipts and a CSV table.

Layout: LEFT text column (Chinese-only labels) + RIGHT image box with border.
- Text wraps within the left column width so it never overlaps the image box.
- Image is fit inside a fixed right-side box with padding.
- 2025-10-08: Multi-currency support.
"""
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import csv
import argparse
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO

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
    return pdfmetrics.stringWidth(text, c._fontname, size)

# ---------- Text wrapping helpers (CJK-friendly, char-based) ----------
def _wrap_text(c: canvas.Canvas, text: str, font_size: int, max_width: float) -> List[str]:
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
    label_size = 10
    value_size = 10
    set_font(c, label_size, False)
    label_text = f"{label_cn}："
    c.drawString(x, y, label_text)
    label_w = _strw(c, label_text, label_size)
    avail_w = max(10.0, col_width - label_w - label_gap)
    lines = _wrap_text(c, value or "", value_size, avail_w)
    vx = x + label_w + label_gap
    for i, line in enumerate(lines):
        set_font(c, value_size, False)
        c.drawString(vx, y - i*lh, line)
    used_lines = max(1, len(lines))
    return y - used_lines*lh

# ---------- Image box ----------
def _draw_image_right(c: canvas.Canvas, img_path: Path, box_x: float, box_y: float, box_w: float, box_h: float, padding: float = 3*mm, jpeg_quality=70):
    c.setLineWidth(0.8)
    c.rect(box_x, box_y, box_w, box_h, stroke=1, fill=0)
    inner_x, inner_y = box_x + padding, box_y + padding
    inner_w, inner_h = box_w - 2*padding, box_h - 2*padding
    try:
        img = Image.open(img_path).convert("RGB")
        # Сжимаем до JPEG в памяти
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70, optimize=True)
        buf.seek(0)
        img = Image.open(buf)
        img_w, img_h = img.size
        scale = min(inner_w / img_w, inner_h / img_h)
        disp_w, disp_h = img_w * scale, img_h * scale
        draw_x = inner_x + (inner_w - disp_w) / 2.0
        draw_y = inner_y + (inner_h - disp_h) / 2.0
        c.drawInlineImage(img, draw_x, draw_y, width=disp_w, height=disp_h)
    except Exception as e:
        c.drawString(inner_x, inner_y + inner_h/2, f"(Image error: {e})")

# ---------- Currency helpers ----------
CURR_CN = {
    "THB": "泰铢",
    "HKD": "港币",
    "TWD": "新台币",
    "CNY": "人民币",
    "USD": "美元",
    "EUR": "欧元",
    "JPY": "日元",
    "GBP": "英镑",
    "SGD": "新加坡元",
    "MOP": "澳门元",
    "KRW": "韩元",
    "VND": "越南盾",
    "MYR": "马来西亚林吉特",
    "IDR": "印尼盾",
}

def parse_fx_to_cny(s: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        code = k.strip().upper()
        try:
            out[code] = float(v.strip())
        except Exception:
            pass
    return out

def coalesce_amount_currency(row: dict, default_currency: str) -> Tuple[Optional[str], str, Optional[str]]:
    amount_cny = (row.get("amount_cny") or "").strip() or None
    amount = (row.get("amount") or "").strip()
    currency = (row.get("currency") or "").strip().upper()
    if amount and not currency:
        currency = default_currency
    if not amount:
        amount_thb = (row.get("amount_thb") or row.get("amount_thb_guess") or "").strip()
        if amount_thb:
            amount = amount_thb
            if not currency:
                currency = "THB"
    if not currency:
        currency = default_currency
    return (amount if amount else None, currency, amount_cny)

def amount_line(amount: Optional[str], currency: str, amount_cny: Optional[str]) -> str:
    label_cn = CURR_CN.get(currency.upper(), currency.upper())
    parts = []
    if amount:
        parts.append(f"{amount} {label_cn}")
    if amount_cny:
        parts.append(f"（{amount_cny} 人民币）")
    return " ".join(parts)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Build PDF report (Chinese-only labels, right-side image box, wrapped text, multi-currency).")
    ap.add_argument("--crops_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output_pdf", required=True)
    ap.add_argument("--submitted_by", required=True)
    ap.add_argument("--company", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--default_type", default="sales")
    ap.add_argument("--default_currency", default="THB", help="Default currency code if CSV has no 'currency'")
    ap.add_argument("--fx_to_cny", default="", help='Mapping like "THB=0.20,HKD=0.93,TWD=0.22,CNY=1" to compute amount_cny when missing')
    # Layout
    ap.add_argument("--margin_left_mm", type=float, default=18.0)
    ap.add_argument("--margin_right_mm", type=float, default=18.0)
    ap.add_argument("--margin_top_mm", type=float, default=15.0)
    ap.add_argument("--margin_bottom_mm", type=float, default=15.0)
    ap.add_argument("--image_box_w_mm", type=float, default=90.0)
    ap.add_argument("--image_box_h_mm", type=float, default=175.0)
    ap.add_argument("--image_box_bottom_mm", type=float, default=90.0)
    ap.add_argument("--gutter_mm", type=float, default=8.0)
    args = ap.parse_args()

    fx_map = parse_fx_to_cny(args.fx_to_cny)

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

    # Right image box
    box_w = args.image_box_w_mm * mm
    box_h = args.image_box_h_mm * mm
    box_x = right - box_w
    box_y = args.image_box_bottom_mm * mm

    # Left text column
    gutter = args.gutter_mm * mm
    text_right_limit = box_x - gutter
    text_col_width = max(40*mm, text_right_limit - left)

    c = canvas.Canvas(args.output_pdf, pagesize=A4)

    for row in rows:
        image_file = (row.get("image_file") or "").strip()
        img_path = crops_dir / image_file

        date = (row.get("date") or row.get("date_guess") or "").strip()
        invoice_no = (row.get("invoice_no") or row.get("invoice_no_guess") or "").strip()
        itype = (row.get("invoice_type") or args.default_type).strip()

        amount, currency, amount_cny = coalesce_amount_currency(row, args.default_currency)

        if not amount_cny and amount and currency:
            try:
                val = float(amount.replace(",", ""))
            except Exception:
                val = None
            rate = fx_map.get(currency.upper())
            if val is not None and rate is not None:
                amount_cny = f"{val * rate:.2f}"

        purpose = (row.get("purpose") or "").strip()
        notes = (row.get("notes") or "").strip()

        set_font(c, 13, True)
        c.drawString(left, top, "收据记录")

        _draw_image_right(c, img_path, box_x, box_y, box_w, box_h, padding=3*mm)

        y = top - 10*mm
        y = _draw_pair_cn(c, left, y, "日期", date, text_col_width)
        y = _draw_pair_cn(c, left, y, "提交人", args.submitted_by, text_col_width)
        y = _draw_pair_cn(c, left, y, "公司", args.company, text_col_width)
        label_type = "报销发票" if itype.lower().startswith("reimb") else "销售发票"
        y = _draw_pair_cn(c, left, y, "发票类型", label_type, text_col_width)
        y = _draw_pair_cn(c, left, y, "发票号码", invoice_no, text_col_width)
        y = _draw_pair_cn(c, left, y, "金额", amount_line(amount, currency, amount_cny), text_col_width)
        y = _draw_pair_cn(c, left, y, "付款用途/描述", purpose, text_col_width)
        y = _draw_pair_cn(c, left, y, "备注", notes, text_col_width)

        c.showPage()

    c.save()
    print("Wrote", args.output_pdf)

if __name__ == "__main__":
    main()
