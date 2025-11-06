# 🧾 Fapiaoer — Monthly Receipts OCR & Report Generator

**Fapiaoer** is a lightweight Python automation tool that helps you:
1. **Split photos** containing multiple receipts into individual cropped images.
2. **Extract basic OCR data** (dates, totals, invoice numbers) into a CSV.
3. **Generate a bilingual (Chinese-labeled) PDF report** for accounting submission.

Ideal for freelancers, small teams, or international employees collecting receipts monthly.


## 1. Receipt Cropping — `receipts_cropper.py`

### Purpose
Takes raw receipt photos (possibly multiple receipts per image), detects and crops them into separate images, and generates CSV files (`manifest.csv`, `suggestions.csv`).

### Minimal run
```bash
python receipts_cropper.py   --input input_receipts   --out_crops out/crops   --autosplit
```

### Recommended run (supports vertical & square receipts)
```bash
python receipts_cropper.py   --input input_receipts   --out_crops out/crops   --autosplit   --no_expect_vertical   --square_bias   --border_mode   --min_area_ratio 0.005   --aspect_min 0.7   --aspect_max 14   --suggest_csv   --tesslang "eng+tha+rus+chi_sim"
```

### Key flags
| Flag | Description |
|------|--------------|
| `--input` | Folder with original receipt photos. |
| `--out_crops` | Output folder for cropped receipts. |
| `--autosplit` | Detect multiple receipts per photo. |
| `--no_expect_vertical` | Allow square/horizontal receipts (not strictly vertical). |
| `--square_bias` | Prioritize square-shaped receipts. |
| `--border_mode` | Detect receipts outlined by black frames. |
| `--min_area_ratio` | Minimum contour area ratio (fraction of image). |
| `--aspect_min / --aspect_max` | Allowed aspect ratio range (height/width). |
| `--suggest_csv` | Generate `suggestions.csv` with rough OCR hints (date, total, number). |
| `--tesslang` | Languages for Tesseract OCR (e.g. `"eng+tha+rus+chi_sim"`). |


## 2. Report Generation — `receipts_report.py`

### Purpose
Combines the cropped receipts and CSV data into a clean **PDF report**.  
The left column contains structured text (Chinese labels only), and the right side displays the receipt image in a bordered box.

### CSV format
```
image_file,date,invoice_no,amount,currency,amount_cny,purpose,notes,invoice_type
```
You can edit the auto-generated `suggestions.csv` before building the report.

### Example run
```bash
python receipts_report.py   --crops_dir out/crops   --csv out/crops/suggestions.csv   --output_pdf out/report_2025-09.pdf   --submitted_by "Ma Yuxi"   --company "某没钱买有限公司"   --month 2025-09   --default_currency THB   --fx_to_cny "THB=0.20,HKD=0.93,TWD=0.22,CNY=1"   --default_type sales
```

### Key flags
| Flag | Description |
|------|--------------|
| `--crops_dir` | Folder with cropped receipts (from `receipts_cropper.py`). |
| `--csv` | CSV with invoice data. |
| `--output_pdf` | Output PDF file. |
| `--submitted_by` | Your name or submitter. |
| `--company` | Company name in Chinese. |
| `--month` | Report month (for labeling). |
| `--default_type` | Default invoice type (`sales` or `reimb`). |
| `--default_currency` | Default currency if not provided in CSV (e.g. `THB`). |
| `--fx_to_cny` | Mapping of exchange rates to CNY, e.g. `"THB=0.20,HKD=0.93,TWD=0.22,CNY=1"`. |

### Layout tuning
| Flag | Description |
|------|--------------|
| `--margin_left_mm`, `--margin_right_mm`, `--margin_top_mm`, `--margin_bottom_mm` | Page margins. |
| `--image_box_w_mm`, `--image_box_h_mm` | Size of the right-side image box. |
| `--image_box_bottom_mm` | Distance from bottom to the image box. |
| `--gutter_mm` | Spacing between the text column and the image box. |


## 3. Monthly Workflow

1. 📸 **Take photos** of all receipts and save them to `input_receipts/`.
2. 🪄 **Run** `receipts_cropper.py` with proper flags.  
   → Cropped images and `suggestions.csv` will appear in `out/crops/`.
3. 📝 **Review & edit** `suggestions.csv` — check dates, totals, invoice numbers, and descriptions.
4. 📄 **Generate PDF** with `receipts_report.py`.
5. 📬 **Send** the final PDF report to your accountant. ✅


## 4. Optional: Compressing the PDF with Ghostscript

If your reports are large (e.g. 20–30 MB), you can easily compress them using Ghostscript:

```bash
brew install ghostscript
```

Then run:

```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook    -dNOPAUSE -dQUIET -dBATCH    -sOutputFile=report_small.pdf report_big.pdf
```

**Recommended settings:**
- `/screen` – smallest file size (low quality)
- `/ebook` – balanced (≈150 dpi, ideal for accounting)
- `/printer` – near-lossless (≈300 dpi)


## 5. Dependencies

```bash
pip install opencv-python pillow pytesseract numpy reportlab
```


## 6. Folder Structure Example

```
fapiaoer/
├── receipts_cropper.py
├── receipts_report.py
├── input_receipts/
│   ├── photo1.jpg
│   └── photo2.jpg
├── out/
│   └── crops/
│       ├── crop_001.jpg
│       ├── crop_002.jpg
│       └── suggestions.csv
└── report_2025-09.pdf
```
