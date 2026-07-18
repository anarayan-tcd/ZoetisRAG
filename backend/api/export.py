# export.py — writes completed batch job rows out to a downloadable file,
# and owns the "shape" of a result row (field order + pretty labels) since
# that's fundamentally a presentation concern — batch.py's _flatten_result
# imports FIELD_ORDER/COLUMN_LABELS from here rather than the reverse.
#
# A result is two things: the PRODUCT (what it is — no ids), and every id
# COMBINATION that product maps to (often several, because one real product
# is routinely listed under duplicate/legacy catalog codes). Each export
# format expresses that pairing in its own idiom: Excel gets two sheets,
# CSV keeps one row per query with the combinations summarised into a
# column, and JSON nests them naturally.

import json
import tempfile
from pathlib import Path

# Logical column order for the PRODUCT match: identifying details first
# (what it IS), confidence right after the full name (how sure the model
# is), the rest of the product details grouped together, then the free-text
# reason, with region last. No id fields — those live in ID_FIELD_ORDER.
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
    "reason",
    "region",
]

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
    "reason": "Reason",
    "region": "Region",
}

# One id combination's own columns, for the ID Combinations sheet.
ID_FIELD_ORDER = [
    "product_id",
    "product_variant_id",
    "sku_id",
    "pack_units",
    "ml_volume",
    "occurrences",
    "catalog_description",
]

ID_COLUMN_LABELS = {
    "product_id": "Product ID",
    "product_variant_id": "Product Variant ID",
    "sku_id": "SKU ID",
    "pack_units": "Pack Units",
    "ml_volume": "Ml Volume",
    "occurrences": "Historical Occurrences",
    "catalog_description": "Catalog Description",
}

# Summary columns carried on the product row itself, so a single-sheet CSV
# (or a glance at the Product Match sheet) still shows the ids without
# needing to cross-reference the other sheet.
POSSIBLE_IDS_COLUMN = "Possible IDs (ProductId/VariantId/SkuId)"
ID_COUNT_COLUMN = "ID Combination Count"
ID_REASON_COLUMN = "ID Selection Reason"


def format_combination(combo: dict) -> str:
    return f'{combo.get("product_id", "")}/{combo.get("product_variant_id", "")}/{combo.get("sku_id", "")}'


def _product_columns() -> list:
    return ["Input"] + [COLUMN_LABELS[f] for f in FIELD_ORDER] + [
        ID_COUNT_COLUMN, POSSIBLE_IDS_COLUMN, ID_REASON_COLUMN,
    ]


def _id_columns() -> list:
    return ["Input", "Brand", "Full_Product_Name"] + [ID_COLUMN_LABELS[f] for f in ID_FIELD_ORDER]


def _id_rows(rows: list) -> list:
    """Explodes each result into one row per id combination — the natural
    shape for the ID Combinations sheet, where a query with five possible
    ids should read as five rows, not one crowded cell."""
    out = []
    for row in rows:
        combos = row.get("_id_combinations") or []
        if not combos:
            out.append({
                "Input": row.get("Input", ""),
                "Brand": row.get("Brand", ""),
                "Full_Product_Name": row.get("Full_Product_Name", ""),
                **{ID_COLUMN_LABELS[f]: "" for f in ID_FIELD_ORDER},
            })
            continue
        for combo in combos:
            out.append({
                "Input": row.get("Input", ""),
                "Brand": row.get("Brand", ""),
                "Full_Product_Name": row.get("Full_Product_Name", ""),
                **{ID_COLUMN_LABELS[f]: combo.get(f, "") for f in ID_FIELD_ORDER},
            })
    return out


def _public_row(row: dict) -> dict:
    """Drops the internal _id_combinations payload from a flat row."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


def export_csv(rows: list) -> Path:
    import pandas as pd

    path = Path(tempfile.gettempdir()) / "veterinary_product_results.csv"
    pd.DataFrame([_public_row(r) for r in rows], columns=_product_columns()).to_csv(path, index=False)
    return path


# ─────────────────────────────────────────────────────────────
# EXCEL — two sheets mirroring the app's own two-box output: what the
# product IS, and which ids it could be. Styled for readability: a colored
# header row, color-coded confidence cells, zebra striping, frozen header +
# autofilter, and generous column widths with wrapping so long fields (full
# product name, reason, catalog description) are never truncated.
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
    "Reason": 50,
    "Region": 18,
    ID_COUNT_COLUMN: 10,
    POSSIBLE_IDS_COLUMN: 34,
    ID_REASON_COLUMN: 50,
    "Product ID": 10,
    "Product Variant ID": 14,
    "SKU ID": 10,
    "Pack Units": 10,
    "Ml Volume": 10,
    "Historical Occurrences": 12,
    "Catalog Description": 45,
}

_WRAP_COLUMNS = {
    "Input", "Full_Product_Name", "Reason", POSSIBLE_IDS_COLUMN,
    ID_REASON_COLUMN, "Catalog Description",
}

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


def _write_sheet(wb, title: str, headers: list, records: list):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    ws = wb.create_sheet(title)
    _style_header(ws, headers)

    body_font = Font(size=10)
    wrap_align = Alignment(wrap_text=True, vertical="top")
    plain_align = Alignment(vertical="top")
    center_align = Alignment(horizontal="center", vertical="center")
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    band_fill = PatternFill("solid", fgColor=_BAND_FILL_COLOR)

    for r, record in enumerate(records, start=2):
        confidence = record.get("Confidence", "")
        banded = r % 2 == 0

        for c, header in enumerate(headers, start=1):
            cell = ws.cell(row=r, column=c, value=record.get(header, ""))
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

    _write_sheet(wb, "Product Match", _product_columns(), [_public_row(r) for r in rows])
    _write_sheet(wb, "ID Combinations", _id_columns(), _id_rows(rows))

    wb.save(path)
    return path


def _nest_row(row: dict) -> dict:
    """Mirrors the Excel workbook's structure in JSON's own native way:
    JSON has no notion of "sheets", but it does have nested objects, so the
    product and its id combinations each get their own sub-structure
    (same group names as the Excel tabs) instead of the flat, summarised
    columns the single-sheet CSV has to fall back on."""
    return {
        "Input": row.get("Input", ""),
        "Product Match": {COLUMN_LABELS[f]: row.get(COLUMN_LABELS[f], "") for f in FIELD_ORDER},
        "ID Combinations": [
            {ID_COLUMN_LABELS[f]: combo.get(f, "") for f in ID_FIELD_ORDER}
            for combo in (row.get("_id_combinations") or [])
        ],
        "ID Selection Reason": row.get(ID_REASON_COLUMN, ""),
    }


def export_json(rows: list) -> Path:
    path = Path(tempfile.gettempdir()) / "veterinary_product_results.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump([_nest_row(row) for row in rows], f, indent=4)

    return path
