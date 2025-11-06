#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import re
import csv
import argparse
from datetime import datetime
from dateutil import parser as dateparser
from PIL import Image, ImageOps
import pytesseract
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
import cv2
import numpy as np
import os

# ------------------------ CONFIGURABLE HEURISTICS ------------------------
# Regex patterns for dates (handles DD/MM/YYYY, YYYY-MM-DD, DD.MM.YYYY, etc.)
DATE_PATTERNS = [
    r"(?:19|20|25)\d{2}[-/.年](?:0?[1-9]|1[0-2])[-/.月](?:0?[1-9]|[12]\d|3[01])",
    r"(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20|25)\d{2}",
]

# Thai Buddhist Era year detection (B.E. = C.E. + 543)
# If year >= 2400 we treat it as BE and convert to CE by -543.
def maybe_convert_thai_be(date_str: str) -> str:
    y_m = re.search(r"(?P<y>\d{4})", date_str)
    if not y_m:
        return date_str
    y = int(y_m.group("y"))
    if y >= 2400:
        return date_str.replace(str(y), str(y - 543))
    return date_str

# Keywords to find "Total/Amount" lines in multiple languages
AMOUNT_HINTS = [
    r"total", r"amount", r"sum", r"grand\s*total", r"balance",
    r"ยอดสุทธิ", r"รวม", r"จำนวนเงิน",
    r"Сумма", r"Итого",
    r"金额", r"合计", r"总计"
]

# Numeric patterns for currency lines
AMOUNT_NUMBER = r"(?:\d{1,3}(?:[ ,]\d{3})*|\d+)(?:[.,]\d{2})?"

# Invoice number hints
INV_HINTS = [
    r"invoice\s*no", r"invoice\s*#", r"tax\s*invoice\s*no", r"เลขที่ใบกำกับภาษี", r"เลขที่", r"发票号码", r"发票号", r"发票编号",
    r"Счет[-\s]?фактура\s*№", r"№", r"Номер\s*счета"
]

# ------------------------ DATA STRUCTURES ------------------------
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

# ------------------------ OCR & EXTRACTION ------------------------
def load_image(path: Path) -> Image.Image:
    # Load and pre-process for better OCR: grayscale, contrast, adaptive threshold
    img = Image.open(path).convert("L")
    arr = np.array(img)
    arr = cv2.equalizeHist(arr)
    # Adaptive threshold to improve text
    thr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(thr)

def run_ocr(img: Image.Image, tesslang: str) -> str:
    cfg = "--psm 6"
    text = pytesseract.image_to_string(img, lang=tesslang, config=cfg)
    return text

def extract_date(text: str) -> Optional[str]:
    candidates = []
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            s = maybe_convert_thai_be(m.group(0))
            # Normalize to YYYY-MM-DD if possible
            try:
                dt = dateparser.parse(s, dayfirst=False, yearfirst=False, fuzzy=True)
                candidates.append(dt.date().isoformat())
            except Exception:
                pass
    if candidates:
        # Heuristic: pick the earliest plausible date on the receipt page
        return sorted(candidates)[0]
    return None

def extract_amount(text: str) -> Optional[str]:
    # Search for amount lines near hints
    lines = text.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if any(re.search(h, low) for h in AMOUNT_HINTS):
            # Look on the same or next line for a number
            window = [line]
            if i + 1 < len(lines):
                window.append(lines[i+1])
            blob = " ".join(window)
            m = re.search(AMOUNT_NUMBER, blob.replace(",", " "), flags=re.IGNORECASE)
            if m:
                return m.group(0).replace(" ", "")
    # Fallback: find the largest money-looking number in entire text
    nums = [n.replace(" ", "") for n in re.findall(AMOUNT_NUMBER, text)]
    def to_float(s):
        s = s.replace(",", "")
        if s.count(".") > 1:
            return None
        try:
            return float(s)
        except:
            return None
    vals = [(n, to_float(n)) for n in nums]
    vals = [(n, v) for n, v in vals if v is not None]
    if vals:
        return max(vals, key=lambda x: x[1])[0]
    return None

def extract_invoice_no(text: str) -> Optional[str]:
    # Try to find a hint word, then read a token nearby
    lines = text.splitlines()
    for line in lines:
        low = line.lower()
        if any(re.search(h, low) for h in INV_HINTS):
            # capture alphanumeric groups after hint
            m = re.search(r"(?:no\.?|#|เลขที่|号码|编号|№)\s*[:：]?\s*([A-Za-z0-9\-_/]+)", line, flags=re.IGNORECASE)
            if m:
                return m.group(1)
            # fallback: last token
            toks = re.findall(r"[A-Za-z0-9\-_/]+", line)
            if toks:
                return toks[-1]
    # Fallback: longest plausible alphanumeric token with digits
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/]{4,}", "\n".join(lines))
    tokens = [t for t in tokens if re.search(r"\d", t)]
    if tokens:
        return max(tokens, key=len)
    return None

def read_sidecar(path: Path) -> Tuple[Optional[str], Optional[str]]:
    txt = path.with_suffix(".txt")
    if txt.exists():
        try:
            content = txt.read_text(encoding="utf-8").strip()
        except:
            content = txt.read_text(errors="ignore").strip()
        # first line purpose, rest notes
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if not lines:
            return None, None
        if len(lines) == 1:
            return lines[0], None
        return lines[0], "\n".join(lines[1:])
    return None, None

# ------------------------ PDF RENDERING ------------------------
def draw_bilingual_block(c: canvas.Canvas, x: float, y: float, label_en: str, label_zh: str, value: str, line_height=6*mm):
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"{label_en} / {label_zh}:")
    c.setFont("Helvetica", 10)
    c.drawString(x + 70*mm, y, value if value else "")
    return y - line_height

def add_receipt_page(c: canvas.Canvas, r: 'Receipt'):
    c.setFont("Helvetica-Bold", 13)
    c.drawString(20*mm, 280*mm, "Receipt Record / 收据记录")

    y = 270*mm
    y = draw_bilingual_block(c, 20*mm, y, "Date", "日期", r.date or "")
    y = draw_bilingual_block(c, 20*mm, y, "Submitted by", "提交人", r.submitted_by)
    y = draw_bilingual_block(c, 20*mm, y, "Company", "公司", r.company)
    y = draw_bilingual_block(c, 20*mm, y, "Invoice Type", "发票类型", "Sales Invoice (销售发票)")  # default; adjust if needed
    y = draw_bilingual_block(c, 20*mm, y, "Invoice Number", "发票号码", r.invoice_number or "")
    amt_line = f"{r.amount_thb or ''} Thai Baht 泰铢"
    if r.amount_cny:
        amt_line += f" ({r.amount_cny} 人民币)"
    y = draw_bilingual_block(c, 20*mm, y, "Amount", "金额", amt_line)
    y = draw_bilingual_block(c, 20*mm, y, "Purpose / Description", "付款用途/描述", r.purpose or "")
    y = draw_bilingual_block(c, 20*mm, y, "Additional Notes", "备注", r.notes or "")

    # Place image on the right
    try:
        img = Image.open(r.image_path)
        img.thumbnail((800, 800))
        img_w, img_h = img.size
        # target area ~ (120mm x 160mm)
        max_w = 120 * mm
        max_h = 160 * mm
        scale = min(max_w / img_w, max_h / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        # Position at right side
        c.drawInlineImage(img, 80*mm, 90*mm, width=disp_w, height=disp_h)
    except Exception as e:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(80*mm, 90*mm, f"(Failed to load image: {e})")

    c.showPage()

# ------------------------ MAIN LOGIC ------------------------
def detect_currency_cny_from_thb(amount_thb: Optional[str], rate: Optional[float]) -> Optional[str]:
    if not amount_thb or not rate:
        return None
    # normalize
    amt = float(amount_thb.replace(",", ""))
    cny = amt * rate
    return f"{cny:.2f}"

def main():
    ap = argparse.ArgumentParser(description="Automate monthly receipt OCR and PDF/CSV generation.")
    ap.add_argument("--input", required=True, help="Folder with receipt images (jpg/png/pdf).")
    ap.add_argument("--output", required=True, help="Output folder.")
    ap.add_argument("--month", required=True, help="YYYY-MM for grouping in output filenames.")
    ap.add_argument("--submitted_by", required=True, help="Submitted by / 提交人")
    ap.add_argument("--company", required=True, help="Company / 公司")
    ap.add_argument("--tesslang", default="eng", help='Tesseract languages, e.g. "eng+tha+rus+chi_sim"')
    ap.add_argument("--cny_rate", type=float, default=None, help="Optional conversion rate: 1 THB -> ? CNY")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])
    if not images:
        print("No images found in input folder.")
        return

    receipts: List[dict] = []
    pdf_path = out_dir / f"report_{args.month}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)

    for img_path in images:
        # OCR
        pre = load_image(img_path)
        text = run_ocr(pre, args.tesslang)

        # Extract fields
        date = extract_date(text)
        amount_thb = extract_amount(text)
        invoice_no = extract_invoice_no(text)

        # Optional sidecar purpose/notes
        purpose, notes = read_sidecar(img_path)

        amount_cny = detect_currency_cny_from_thb(amount_thb, args.cny_rate)

        rec = Receipt(
            image_path=img_path,
            date=date,
            submitted_by=args.submitted_by,
            company=args.company,
            invoice_number=invoice_no,
            amount_thb=amount_thb,
            amount_cny=amount_cny,
            purpose=purpose,
            notes=notes
        )

        # Write PDF page
        add_receipt_page(c, rec)
        receipts.append(asdict(rec))

    c.save()

    # CSV summary
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path","date","submitted_by","company","invoice_number","amount_thb","amount_cny","purpose","notes"])
        w.writeheader()
        for r in receipts:
            r["image_path"] = os.path.basename(r["image_path"])
            w.writerow(r)

    print(f"Done. PDF: {pdf_path.name}, CSV: {csv_path.name}")

# ------------------------ Helpers copied locally ------------------------
def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    arr = cv2.equalizeHist(arr)
    thr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(thr)

def run_ocr(img: Image.Image, tesslang: str) -> str:
    cfg = "--psm 6"
    return pytesseract.image_to_string(img, lang=tesslang, config=cfg)

def extract_date(text: str) -> Optional[str]:
    DATE_PATTERNS = [
        r"(?:19|20|25)\d{2}[-/.年](?:0?[1-9]|1[0-2])[-/.月](?:0?[1-9]|[12]\d|3[01])",
        r"(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20|25)\d{2}",
    ]
    def maybe_convert_thai_be(s: str) -> str:
        m = re.search(r"(?P<y>\d{4})", s)
        if not m: return s
        y = int(m.group("y"))
        if y >= 2400: return s.replace(str(y), str(y-543))
        return s
    candidates = []
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            s = maybe_convert_thai_be(m.group(0))
            try:
                dt = dateparser.parse(s, dayfirst=False, yearfirst=False, fuzzy=True)
                candidates.append(dt.date().isoformat())
            except: pass
    if candidates:
        return sorted(candidates)[0]
    return None

def extract_amount(text: str) -> Optional[str]:
    AMOUNT_HINTS = [
        r"total", r"amount", r"sum", r"grand\s*total", r"balance",
        r"ยอดสุทธิ", r"รวม", r"จำนวนเงิน",
        r"Сумма", r"Итого",
        r"金额", r"合计", r"总计"
    ]
    AMOUNT_NUMBER = r"(?:\d{1,3}(?:[ ,]\d{3})*|\d+)(?:[.,]\d{2})?"
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
    INV_HINTS = [
        r"invoice\s*no", r"invoice\s*#", r"tax\s*invoice\s*no", r"เลขที่ใบกำกับภาษี", r"เลขที่", r"发票号码", r"发票号", r"发票编号",
        r"Счет[-\s]?фактура\s*№", r"№", r"Номер\s*счета"
    ]
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

if __name__ == "__main__":
    main()
