#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
receipts_cropper.py
Нарезает несколько чеков с одного фото на отдельные изображения.
Пишет manifest.csv и (по желанию) suggestions.csv с накидками OCR.

Пример:
python receipts_cropper.py \
  --input input_receipts \
  --out_crops out/crops \
  --autosplit \
  --no_expect_vertical \
  --square_bias \
  --border_mode \
  --min_area_ratio 0.005 \
  --aspect_min 0.7 \
  --aspect_max 14 \
  --suggest_csv \
  --tesslang "eng+tha+rus+chi_sim"
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import csv
import argparse
import re
import cv2
import numpy as np
from PIL import Image
import pytesseract

# ---------- Simple OCR helpers (optional suggestions) ----------
MONTHS = ("jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec")
DATE_PATTERNS = [
    rf"\b(?:{MONTHS})[a-z]*\s+\d{{1,2}},\s*\d{{4}}\b",
    rf"\b\d{{1,2}}\s+(?:{MONTHS})[A-Za-z]*\s+\d{{4}}\b",
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",
]
AMOUNT_HINTS = [
    r"total", r"amount", r"sum", r"grand\s*total", r"balance",
    r"ยอดสุทธิ", r"รวม", r"จำนวนเงิน",
    r"Сумма", r"Итого",
    r"金额", r"合计", r"总计"
]
AMOUNT_NUMBER = r"(?:\d{1,3}(?:[ ,]\d{3})*|\d+)(?:[.,]\d{2})?"
INV_HINTS = [
    r"invoice\s*no", r"invoice\s*#", r"tax\s*invoice\s*no",
    r"เลขที่ใบกำกับภาษี", r"เลขที่", r"发票号码", r"发票号", r"发票编号",
    r"Счет[-\s]?фактура\s*№", r"№", r"Номер\s*счета"
]

def _ocr_text(img: Image.Image, tesslang: str) -> str:
    return pytesseract.image_to_string(img, lang=tesslang, config="--psm 6")

def _extract_date(text: str) -> Optional[str]:
    import dateutil.parser as dateparser
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

    t = re.sub(r"[ \t]+", " ", text)
    cands = []
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            raw = m.group(0)
            raw = re.sub(rf"\b({MONTHS})[A-Za-z]*\b",
                         lambda mm: {"sept": "sep"}.get(mm.group(1).lower(), mm.group(1)),
                         raw, flags=re.IGNORECASE)
            iso = parse_date_string(raw)
            if iso:
                cands.append((m.start(), iso))
    if cands:
        cands.sort(key=lambda x: x[0])
        return cands[0][1]
    return None

def _extract_amount(text: str) -> Optional[str]:
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

def _extract_invoice_no(text: str) -> Optional[str]:
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

# ---------- geometry helpers ----------
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
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# ---------- autosplit (white paper mode, tunable) ----------
def autosplit_receipts(
    img_bgr: np.ndarray,
    expect_vertical: bool = True,
    min_area_ratio: float = 0.015,
    aspect_min: float = 1.1,
    aspect_max: float = 12.0,
    whiteness_min: float = 0.55,
    rectangularity_min: float = 0.75,
    method: str = "both",
    max_crops: int = 20,
    square_bias: bool = False,
    border_mode: bool = False
) -> List[np.ndarray]:
    orig = img_bgr.copy()
    H, W = orig.shape[:2]

    base_scale = 1400.0 / max(H, W)
    if base_scale < 1.0:
        small_base = cv2.resize(orig, (int(W*base_scale), int(H*base_scale)))
    else:
        small_base = orig
        base_scale = 1.0

    us = 1.5
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
        ar = (h / (w + 1e-6)) if expect_vertical else (max(h, w) / (min(h, w) + 1e-6))
        if not (aspect_min <= ar <= aspect_max) and not (square_bias and 0.7 <= ar <= 1.6):
            continue
        mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        area_cnt = cv2.contourArea(cnt)
        x2,y2,w2,h2 = cv2.boundingRect(cnt)
        rect_ratio = 0 if w2*h2 == 0 else area_cnt / float(w2*h2)
        if rect_ratio < rectangularity_min:
            continue
        crop = gray_small[max(0,y2):y2+h2, max(0,x2):x2+w2]
        if crop.size == 0: continue
        _, thr_loc = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        white_frac = float((thr_loc == 255).sum()) / float(crop.size)
        if white_frac < whiteness_min and not (square_bias and white_frac >= (whiteness_min - 0.07)):
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
            if area < area_min_abs: continue
            ar = (h / (w + 1e-6)) if expect_vertical else (max(h, w) / (min(h, w) + 1e-6))
            if not (aspect_min <= ar <= aspect_max) and not (square_bias and 0.7 <= ar <= 1.6):
                continue
            area_cnt = cv2.contourArea(cnt)
            rect_ratio = 0 if w*h == 0 else area_cnt / float(w*h)
            if rect_ratio < rectangularity_min: continue
            rot_rect = cv2.minAreaRect(cnt)
            cand_rects.append((area, (x,y,w,h), rot_rect))

    cand_rects.sort(key=lambda t: t[0], reverse=True)
    centers = []; dedup = []
    for score, (x,y,w,h), rot in cand_rects:
        (cx, cy) = rot[0]
        if any((cx-px)**2 + (cy-py)**2 < 25**2 for (px, py) in centers):
            continue
        centers.append((cx, cy)); dedup.append((score, (x,y,w,h), rot))
        if len(dedup) >= max_crops: break

    crops = []
    for _, (x,y,w,h), rot in dedup:
        box = cv2.boxPoints(rot).astype("float32")
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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Crop receipts into separate images; optionally output OCR suggestions.")
    ap.add_argument("--input", required=True, help="Folder with original photos")
    ap.add_argument("--out_crops", required=True, help="Folder to store cropped images")
    ap.add_argument("--autosplit", action="store_true", help="Enable automatic multi-receipt split")
    ap.add_argument("--min_area_ratio", type=float, default=0.015)
    ap.add_argument("--aspect_min", type=float, default=1.1)
    ap.add_argument("--aspect_max", type=float, default=12.0)
    ap.add_argument("--no_expect_vertical", action="store_true")
    ap.add_argument("--square_bias", action="store_true")
    ap.add_argument("--border_mode", action="store_true")
    ap.add_argument("--suggest_csv", action="store_true", help="Write suggestions.csv with OCR guesses")
    ap.add_argument("--tesslang", default="eng", help='Used only if --suggest_csv')

    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out_crops)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff"}])
    if not images:
        print("No images found.")
        return

    manifest_rows = []
    suggest_rows = []

    for img_path in images:
        bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            print("Skip unreadable:", img_path)
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
                border_mode=args.border_mode
            )

        for idx, crop in enumerate(crops, 1):
            out_name = f"{img_path.stem}_part{idx}.jpg"
            out_path = out_dir / out_name
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                buf.tofile(str(out_path))
                manifest_rows.append({
                    "image_file": out_name,
                    "source_file": img_path.name,
                    "index": idx
                })

                if args.suggest_csv:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 31, 15)
                    text = _ocr_text(Image.fromarray(thr), args.tesslang)
                    suggest_rows.append({
                        "image_file": out_name,
                        "date_guess": _extract_date(text) or "",
                        "invoice_no_guess": _extract_invoice_no(text) or "",
                        "amount_thb_guess": _extract_amount(text) or "",
                        "purpose": "",
                        "notes": "",
                        "invoice_type": "sales"  # default; you can change later
                    })

    # write manifest
    mpath = out_dir / "manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_file", "source_file", "index"])
        w.writeheader()
        w.writerows(manifest_rows)
    print("Wrote", mpath)

    if suggest_rows:
        spath = out_dir / "suggestions.csv"
        with open(spath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "image_file","date_guess","invoice_no_guess","amount_thb_guess","purpose","notes","invoice_type"
            ])
            w.writeheader()
            w.writerows(suggest_rows)
        print("Wrote", spath)

if __name__ == "__main__":
    main()