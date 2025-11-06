#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monthly_receipts_autosplit.py
Automates receipts OCR and builds PDF/CSV reports.
- Strong autosplit (components+contours) with tunable params and support for small/square receipts
- Robust date extraction (month names, BE years, multiple separators)
- CJK font support (Noto Sans SC Regular/Bold; variable TTF fallback; STSong-Light fallback)
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import re
import csv
import argparse
from dateutil import parser as dateparser
from PIL import Image
import pytesseract
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import cv2
import numpy as np
import os

# ====================== FONT REGISTRATION ======================
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
        c.setFont(FONT_REG, size + 2)  # emulate bold
    else:
        c.setFont(FONT_REG, size)

# ====================== OCR/EXTRACTION CONFIG ======================
MONTHS = ("jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec")
DATE_PATTERNS = [
    rf"\b(?:{MONTHS})[a-z]*\s+\d{{1,2}},\s*\d{{4}}\b",     # Sep 01, 2025 / Sept 07, 2025
    rf"\b\d{{1,2}}\s+(?:{MONTHS})[A-Za-z]*\s+\d{{4}}\b",   # 28 SEP 2025
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",                  # 13/09/2025, 05.09.2025, 30-08-2568
]
DATE_HINTS = [r"date", r"วันที่", r"дата", r"日期", r"发票日期"]

AMOUNT_HINTS = [
    r"total", r"amount", r"sum", r"grand\s*total", r"balance",
    r"ยอดสุทธิ", r"รวม", r"จำนวนเงิน",
    r"Сумма", r"Итого",
    r"金额", r"合计", r"总计"
]
AMOUNT_NUMBER = r"(?:\d{1,3}(?:[ ,]\d{3})*|\d+)(?:[.,]\d{2})?"
INV_HINTS = [
    r"invoice\s*no", r"invoice\s*#", r"tax\s*invoice\s*no", r"เลขที่ใบกำกับภาษี", r"เลขที่", r"发票号码", r"发票号", r"发票编号",
    r"Счет[-\s]?фактура\s*№", r"№", r"Номер\s*счета"
]

# ====================== DATA ======================
@dataclass
class Receipt:
    image_path: Path
    date: Optional[str]
    submitted_by: str
    company: str
    invoice_number: Optional[str]
    amount_thb: Optional[str]
    amount_cny: Optional[str]
    purpose: Optional[str]
    notes: Optional[str]

# ====================== DATE HELPERS ======================
def parse_date_string(s: str) -> Optional[str]:
    try:
        m = re.search(r"(\d{4})", s)
        if m:
            y = int(m.group(1))
            if y >= 2400:  # Thai BE -> CE
                s = s.replace(m.group(1), str(y - 543))
        for dayfirst in (False, True):
            dt = dateparser.parse(s, dayfirst=dayfirst, yearfirst=False, fuzzy=True)
            if dt:
                return dt.date().isoformat()
    except Exception:
        return None
    return None

def extract_date(text: str) -> Optional[str]:
    t = re.sub(r"[ \t]+", " ", text)
    candidates = []

    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            raw = m.group(0)
            raw = re.sub(rf"\b({MONTHS})[A-Za-z]*\b", lambda mm: {"sept": "sep"}.get(mm.group(1).lower(), mm.group(1)), raw, flags=re.IGNORECASE)
            iso = parse_date_string(raw)
            if iso:
                candidates.append((m.start(), iso))

    if not candidates:
        lines = t.splitlines()
        for i, line in enumerate(lines):
            low = line.lower()
            if any(h in low for h in DATE_HINTS):
                window = " ".join(lines[i:i+2])
                for pat in DATE_PATTERNS:
                    for m in re.finditer(pat, window, flags=re.IGNORECASE):
                        raw = m.group(0)
                        raw = re.sub(rf"\b({MONTHS})[A-Za-z]*\b", lambda mm: {"sept": "sep"}.get(mm.group(1).lower(), mm.group(1)), raw, flags=re.IGNORECASE)
                        iso = parse_date_string(raw)
                        if iso:
                            candidates.append((i*1000+m.start(), iso))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return None

# ====================== OTHER EXTRACTORS ======================
def run_ocr(img: Image.Image, tesslang: str) -> str:
    return pytesseract.image_to_string(img, lang=tesslang, config="--psm 6")

def extract_amount(text: str) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if any(re.search(h, low) for h in AMOUNT_HINTS):
            window = [line]
            if i + 1 < len(lines): window.append(lines[i+1])
            blob = " ".join(window)
            m = re.search(AMOUNT_NUMBER, blob.replace(",", " "), flags=re.IGNORECASE)
            if m: return m.group(0).replace(" ", "")
    nums = [n.replace(" ", "") for n in re.findall(AMOUNT_NUMBER, text)]
    def to_float(s):
        s = s.replace(",", "")
        if s.count(".") > 1: return None
        try: return float(s)
        except: return None
    vals = [(n, to_float(n)) for n in nums]
    vals = [(n, v) for n, v in vals if v is not None]
    if vals: return max(vals, key=lambda x: x[1])[0]
    return None

def extract_invoice_no(text: str) -> Optional[str]:
    lines = text.splitlines()
    for line in lines:
        low = line.lower()
        if any(re.search(h, low) for h in INV_HINTS):
            m = re.search(r"(?:no\.?|#|เลขที่|号码|编号|№)\s*[:：]?\s*([A-Za-z0-9\-_/]+)", line, flags=re.IGNORECASE)
            if m: return m.group(1)
            toks = re.findall(r"[A-Za-z0-9\-_/]+", line)
            if toks: return toks[-1]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/]{4,}", "\n".join(lines))
    tokens = [t for t in tokens if re.search(r"\d", t)]
    if tokens: return max(tokens, key=len)
    return None

def read_sidecar(path: Path) -> Tuple[Optional[str], Optional[str]]:
    txt = path.with_suffix(".txt")
    if txt.exists():
        try:
            content = txt.read_text(encoding="utf-8").strip()
        except:
            content = txt.read_text(errors="ignore").strip()
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if not lines: return None, None
        if len(lines) == 1: return lines[0], None
        return lines[0], "\n".join(lines[1:])
    return None, None

# ====================== AUTOSPLIT HELPERS ======================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# ====================== AUTOSPLIT (STRONG WHITE-PAPER MODE) ======================
def autosplit_receipts(
    img_bgr: np.ndarray,
    expect_vertical: bool = True,
    min_area_ratio: float = 0.015,
    aspect_min: float = 1.1,
    aspect_max: float = 12.0,
    whiteness_min: float = 0.55,
    rectangularity_min: float = 0.75,
    method: str = "both",          # 'components' | 'contours' | 'both'
    max_crops: int = 20,
    square_bias: bool = False,
    border_mode: bool = False,
    debug: bool = False
) -> List[np.ndarray]:
    """
    Robust splitting for multiple receipts on dark background.
    - HSV.V + CLAHE + Otsu -> white paper segmentation
    - Connected components + minAreaRect warp
    - Optional contours fallback (and border_mode for dark frames)
    - Tunable thresholds; supports small/square receipts via flags
    """
    orig = img_bgr.copy()
    H, W = orig.shape[:2]

    base_scale = 1400.0 / max(H, W)
    if base_scale < 1.0:
        small_base = cv2.resize(orig, (int(W*base_scale), int(H*base_scale)))
    else:
        small_base = orig
        base_scale = 1.0

    us = 1.5  # upsample factor
    small = cv2.resize(small_base, None, fx=us, fy=us) if us != 1.0 else small_base
    total_scale = base_scale * us

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v = hsv[...,2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_eq = clahe.apply(v)
    _, bin_ = cv2.threshold(v_eq, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    cand_rects = []
    area_min_abs = min_area_ratio * (small.shape[0]*small.shape[1])
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if area < area_min_abs: 
            continue

        if expect_vertical:
            ar = h / (w + 1e-6)
        else:
            ar = max(h, w) / (min(h, w) + 1e-6)
        if not (aspect_min <= ar <= aspect_max):
            if not (square_bias and 0.7 <= ar <= 1.6):
                continue

        mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area_cnt = cv2.contourArea(cnt)
        x2,y2,w2,h2 = cv2.boundingRect(cnt)
        rect_ratio = 0 if w2*h2 == 0 else area_cnt / float(w2*h2)
        if rect_ratio < rectangularity_min:
            continue

        crop = gray_small[max(0,y2):y2+h2, max(0,x2):x2+w2]
        if crop.size == 0:
            continue
        _, thr_loc = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        white_frac = float((thr_loc == 255).sum()) / float(crop.size)
        if white_frac < whiteness_min:
            if not (square_bias and white_frac >= (whiteness_min - 0.07)):
                continue

        rot_rect = cv2.minAreaRect(cnt)
        sq_bonus = 0.0
        if square_bias:
            r = max(w2, h2) / (min(w2, h2) + 1e-6)
            sq_bonus = 0.15 if 0.85 <= r <= 1.25 else 0.0
        score = area * (0.6 + 0.4*rect_ratio) * (1.0 + sq_bonus)
        cand_rects.append((score, (x2,y2,w2,h2), rot_rect))

    if method in ("contours", "both") or border_mode:
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5,5), 0)
        edges = cv2.Canny(g, 40, 120)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h
            if area < area_min_abs:
                continue
            if expect_vertical:
                ar = h / (w + 1e-6)
            else:
                ar = max(h,w) / (min(h,w) + 1e-6)
            if not (aspect_min <= ar <= aspect_max or (square_bias and 0.7 <= ar <= 1.6)):
                continue
            area_cnt = cv2.contourArea(cnt)
            rect_ratio = 0 if w*h == 0 else area_cnt / float(w*h)
            if rect_ratio < rectangularity_min:
                continue
            rot_rect = cv2.minAreaRect(cnt)
            cand_rects.append((area, (x,y,w,h), rot_rect))

    cand_rects.sort(key=lambda t: t[0], reverse=True)
    centers = []
    dedup = []
    for score, (x,y,w,h), rot in cand_rects:
        (cx, cy) = rot[0]
        dup = False
        for (px, py) in centers:
            if (cx-px)**2 + (cy-py)**2 < 25**2:
                dup = True; break
        if not dup:
            centers.append((cx, cy))
            dedup.append((score, (x,y,w,h), rot))
        if len(dedup) >= max_crops:
            break

    crops = []
    for _, (x,y,w,h), rot in dedup:
        box = cv2.boxPoints(rot).astype("float32")  # small coords
        box_full = box / total_scale
        try:
            warped = four_point_transform(orig, box_full)
            if warped.shape[0] >= 180 and warped.shape[1] >= 180:
                crops.append(warped); continue
        except Exception:
            pass
        x_full = int(x / total_scale); y_full = int(y / total_scale)
        w_full = int(w / total_scale); h_full = int(h / total_scale)
        crop = orig[y_full:y_full+h_full, x_full:x_full+w_full]
        if crop.shape[0] >= 180 and crop.shape[1] >= 180:
            crops.append(crop)

    return crops if crops else [orig]

# ====================== PDF RENDER ======================
def draw_bilingual_block(c: canvas.Canvas, x: float, y: float, label_en: str, label_zh: str, value: str, line_height=6*mm):
    set_font(c, 10, bold=False)
    c.drawString(x, y, f"{label_en} / {label_zh}:")
    set_font(c, 10, bold=False)
    c.drawString(x + 70*mm, y, value if value else "")
    return y - line_height

def add_receipt_page(c: canvas.Canvas, r: 'Receipt', pil_image: Optional[Image.Image] = None):
    set_font(c, 13, bold=True)
    c.drawString(20*mm, 280*mm, "Receipt Record / 收据记录")

    y = 270*mm
    y = draw_bilingual_block(c, 20*mm, y, "Date", "日期", r.date or "")
    y = draw_bilingual_block(c, 20*mm, y, "Submitted by", "提交人", r.submitted_by)
    y = draw_bilingual_block(c, 20*mm, y, "Company", "公司", r.company)
    y = draw_bilingual_block(c, 20*mm, y, "Invoice Type", "发票类型", "Sales Invoice (销售发票)")
    y = draw_bilingual_block(c, 20*mm, y, "Invoice Number", "发票号码", r.invoice_number or "")
    amt_line = f"{r.amount_thb or ''} Thai Baht 泰铢"
    if r.amount_cny:
        amt_line += f" ({r.amount_cny} 人民币)"
    y = draw_bilingual_block(c, 20*mm, y, "Amount", "金额", amt_line)
    y = draw_bilingual_block(c, 20*mm, y, "Purpose / Description", "付款用途/描述", r.purpose or "")
    y = draw_bilingual_block(c, 20*mm, y, "Additional Notes", "备注", r.notes or "")

    try:
        if pil_image is not None:
            img = pil_image
        else:
            img = Image.open(r.image_path)
        img = img.copy()
        img.thumbnail((1200, 1200))
        img_w, img_h = img.size
        max_w = 120 * mm
        max_h = 160 * mm
        scale = min(max_w / img_w, max_h / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        c.drawInlineImage(img, 80*mm, 90*mm, width=disp_w, height=disp_h)
    except Exception as e:
        set_font(c, 10, bold=False)
        c.drawString(80*mm, 90*mm, f"(Image error: {e})")

    c.showPage()

# ====================== MAIN ======================
def detect_currency_cny_from_thb(amount_thb: Optional[str], rate: Optional[float]) -> Optional[str]:
    if not amount_thb or not rate:
        return None
    amt = float(amount_thb.replace(",", ""))
    cny = amt * rate
    return f"{cny:.2f}"

def main():
    ap = argparse.ArgumentParser(description="Monthly receipt OCR with strong autosplit and robust date extraction.")
    ap.add_argument("--input", required=True, help="Folder with receipt images")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--month", required=True, help="YYYY-MM for output naming")
    ap.add_argument("--submitted_by", required=True, help="Submitted by / 提交人")
    ap.add_argument("--company", required=True, help="Company / 公司")
    ap.add_argument("--tesslang", default="eng", help='Tesseract languages, e.g. "eng+tha+rus+chi_sim"')
    ap.add_argument("--cny_rate", type=float, default=None, help="Optional conversion: 1 THB -> ? CNY")
    ap.add_argument("--autosplit", action="store_true", help="Auto-split multiple receipts per photo")

    # tuning knobs
    ap.add_argument("--min_area_ratio", type=float, default=0.015, help="Minimal area ratio per candidate (e.g., 0.005 for small receipts)")
    ap.add_argument("--aspect_min", type=float, default=1.1, help="Minimal aspect ratio (height/width if expect_vertical else max/min)")
    ap.add_argument("--aspect_max", type=float, default=12.0, help="Max aspect ratio")
    ap.add_argument("--no_expect_vertical", action="store_true", help="Disable vertical bias (detect squares/horizontal)")
    ap.add_argument("--square_bias", action="store_true", help="Prioritize near-square candidates")
    ap.add_argument("--border_mode", action="store_true", help="Extra pass: look for dark rectangular borders")
    ap.add_argument("--debug_preview", action="store_true", help="(No UI) Keep crops/extra files for inspection")

    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    if args.autosplit or args.debug_preview:
        crops_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff"}])
    if not images:
        print("No images found in input folder.")
        return

    receipts: List[dict] = []
    pdf_path = out_dir / f"report_{args.month}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)

    for img_path in images:
        bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        crops = [bgr]
        if args.autosplit:
            crops = autosplit_receipts(
                bgr,
                expect_vertical=not args.no_expect_vertical,
                min_area_ratio=args.min_area_ratio,
                aspect_min=args.aspect_min,
                aspect_max=args.aspect_max,
                method="both",
                square_bias=args.square_bias,
                border_mode=args.border_mode,
                debug=args.debug_preview
            )

        for idx, crop in enumerate(crops):
            crop_path = img_path
            if args.autosplit or args.debug_preview:
                crop_path = crops_dir / f"{img_path.stem}_part{idx+1}.jpg"
                ok, buf = cv2.imencode(".jpg", crop)
                if ok: buf.tofile(str(crop_path))

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 15)
            pil_for_ocr = Image.fromarray(thr)

            text = run_ocr(pil_for_ocr, args.tesslang)
            date = extract_date(text)
            amount_thb = extract_amount(text)
            invoice_no = extract_invoice_no(text)
            purpose, notes = read_sidecar(img_path)  # optional per original image
            amount_cny = detect_currency_cny_from_thb(amount_thb, args.cny_rate)

            rec = Receipt(
                image_path=crop_path,
                date=date,
                submitted_by=args.submitted_by,
                company=args.company,
                invoice_number=invoice_no,
                amount_thb=amount_thb,
                amount_cny=amount_cny,
                purpose=purpose,
                notes=notes
            )

            pil_vis = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            add_receipt_page(c, rec, pil_image=pil_vis)
            d = asdict(rec)
            d["image_path"] = os.path.basename(str(d["image_path"]))
            receipts.append(d)

    c.save()
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path","date","submitted_by","company","invoice_number","amount_thb","amount_cny","purpose","notes"])
        w.writeheader()
        for r in receipts:
            w.writerow(r)

    print(f"Done. PDF: {pdf_path}, CSV: {csv_path}")

if __name__ == "__main__":
    main()
