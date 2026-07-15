# export.py — writes completed batch job rows out to a downloadable file,
# and owns the "shape" of a result row (field order + pretty labels) since
# that's fundamentally a presentation concern — batch.py's _flatten_result
# imports FIELD_ORDER/COLUMN_LABELS from here rather than the reverse.

import json
import tempfile
from pathlib import Path

# Logical column order for ONE match's fields (primary or an alternative):
# product-identifying details first (what it IS), confidence right after
# the full name (how sure the model is), the rest of the product details
# grouped together, then the three id fields grouped together, then the
# free-text reason, with region last.
FIELD_ORDER = [
    "brand",
    "full_product_name",
    "confidence",
    "products",
    "species",
    "animal_size",
    "pack_size",
    "pack_quantity",
    "animal_age",
    "product_id",
    "product_variant_id",
    "sku_id",
    "reason",
    "region",
]

# Pretty, unambiguous export column names — used for both the primary match
# (no prefix) and alternatives (prefixed "Alternative N " in the flattened
# CSV/JSON row, stripped back off again per-sheet in the Excel export).
COLUMN_LABELS = {
    "brand": "Brand",
    "full_product_name": "Full_Product_Name",
    "confidence": "Confidence",
    "products": "Products",
    "species": "Species",
    "animal_size": "Animal Size",
    "pack_size": "Pack Size",
    "pack_quantity": "Pack Quantity",
    "animal_age": "Animal Age",
    "product_id": "Product ID",
    "product_variant_id": "Product Variant ID",
    "sku_id": "SKU ID",
    "reason": "Reason",
    "region": "Region",
}


_SHEETS = [
    ("Primary Match", ""),
    ("Alternative 1", "Alternative 1 "),
    ("Alternative 2", "Alternative 2 "),
]


def _all_columns() -> list:
    """Every column, in the same logical order used everywhere else:
    Input, then the primary match's fields, then each alternative's —
    each group internally ordered per FIELD_ORDER. Passed explicitly to
    pandas/json rather than trusting dict insertion order, so CSV/JSON
    layout can never drift from the Excel sheets' layout."""
    cols = ["Input"] + [COLUMN_LABELS[f] for f in FIELD_ORDER]
    for title, prefix in _SHEETS[1:]:
        cols += [f"{prefix}{COLUMN_LABELS[f]}" for f in FIELD_ORDER]
    return cols


def export_csv(rows: list) -> Path:
    import pandas as pd

    path = Path(tempfile.gettempdir()) / "veterinary_product_results.csv"
    pd.DataFrame(rows, columns=_all_columns()).to_csv(path, index=False)
    return path


# ─────────────────────────────────────────────────────────────
# EXCEL — one sheet per match (Primary Match / Alternative 1 / Alternative
# 2, rather than one wide sheet repeating every field three times), styled
# for readability: a colored header row, color-coded confidence cells,
# zebra striping, frozen header + autofilter, and generous column widths
# with wrapping so long fields (full product name, reason) are never
# truncated.
# ─────────────────────────────────────────────────────────────

_COLUMN_WIDTHS = {
    "Input": 40,
    "Brand": 16,
    "Full_Product_Name": 55,
    "Confidence": 12,
    "Products": 16,
    "Species": 12,
    "Animal Size": 14,
    "Pack Size": 14,
    "Pack Quantity": 12,
    "Animal Age": 12,
    "Product ID": 10,
    "Product Variant ID": 14,
    "SKU ID": 10,
    "Reason": 50,
    "Region": 18,
}

# Columns long enough that truncation is a real risk — wrapped and given a
# taller row height instead of being cut off at the cell edge.
_WRAP_COLUMNS = {"Input", "Full_Product_Name", "Reason"}

_HEADER_FILL_COLOR = "F97316"   # matches the app's own orange accent
_BAND_FILL_COLOR = "F7F7F7"

# Same green/yellow/orange/red semantic scale used elsewhere in this
# project's spreadsheets (background, then a matching darker font color).
_CONFIDENCE_STYLE = {
    "High":     ("C6EFCE", "006100"),
    "Medium":   ("FFEB9C", "9C6500"),
    "Low":      ("FFDDAA", "9C5700"),
    "Very Low": ("FFC7CE", "9C0006"),
}


def _style_header(ws, headers: list):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill("solid", fgColor=_HEADER_FILL_COLOR)
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = border
        ws.column_dimensions[cell.column_letter].width = _COLUMN_WIDTHS.get(header, 16)

    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "A2"


def _write_sheet(wb, title: str, rows: list, prefix: str):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    ws = wb.create_sheet(title)

    headers = ["Input"] + [COLUMN_LABELS[f] for f in FIELD_ORDER]
    source_keys = ["Input"] + [f"{prefix}{COLUMN_LABELS[f]}" for f in FIELD_ORDER]
    confidence_key = f"{prefix}{COLUMN_LABELS['confidence']}"

    _style_header(ws, headers)

    body_font = Font(size=10)
    wrap_align = Alignment(wrap_text=True, vertical="top")
    plain_align = Alignment(vertical="top")
    center_align = Alignment(horizontal="center", vertical="center")
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    band_fill = PatternFill("solid", fgColor=_BAND_FILL_COLOR)

    for r, row in enumerate(rows, start=2):
        confidence = row.get(confidence_key, "")
        banded = r % 2 == 0

        for c, (header, source_key) in enumerate(zip(headers, source_keys), start=1):
            cell = ws.cell(row=r, column=c, value=row.get(source_key, ""))
            cell.border = border
            cell.alignment = wrap_align if header in _WRAP_COLUMNS else plain_align
            cell.font = body_font

            if header == "Confidence":
                cell.alignment = center_align
                bg, fg = _CONFIDENCE_STYLE.get(confidence, (None, None))
                if bg:
                    cell.fill = PatternFill("solid", fgColor=bg)
                    cell.font = Font(size=10, bold=True, color=fg)
            elif banded:
                cell.fill = band_fill

        ws.row_dimensions[r].height = 30

    ws.auto_filter.ref = ws.dimensions


def export_excel(rows: list) -> Path:
    from openpyxl import Workbook

    path = Path(tempfile.gettempdir()) / "veterinary_product_results.xlsx"

    wb = Workbook()
    wb.remove(wb.active)

    for title, prefix in _SHEETS:
        _write_sheet(wb, title, rows, prefix)

    wb.save(path)
    return path


def _nest_row(row: dict) -> dict:
    """Mirrors the Excel workbook's structure in JSON's own native way:
    JSON has no notion of "sheets", but it does have nested objects, so
    each match gets its own sub-object (same group names as the Excel
    tabs) instead of the flat "Alternative 1 Brand"-style prefixed keys
    used by the single-sheet CSV. Every sub-object is internally ordered
    per FIELD_ORDER, same as everywhere else."""
    nested = {"Input": row.get("Input", "")}

    for title, prefix in _SHEETS:
        nested[title] = {
            COLUMN_LABELS[f]: row.get(f"{prefix}{COLUMN_LABELS[f]}", "")
            for f in FIELD_ORDER
        }

    return nested


def export_json(rows: list) -> Path:
    path = Path(tempfile.gettempdir()) / "veterinary_product_results.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump([_nest_row(row) for row in rows], f, indent=4)

    return path
