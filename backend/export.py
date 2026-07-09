# export.py — writes completed batch job rows out to a downloadable file.

import json
import tempfile
from pathlib import Path


def export_csv(rows: list) -> Path:
    import pandas as pd

    path = Path(tempfile.gettempdir()) / "veterinary_product_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_excel(rows: list) -> Path:
    import pandas as pd

    path = Path(tempfile.gettempdir()) / "veterinary_product_results.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def export_json(rows: list) -> Path:
    path = Path(tempfile.gettempdir()) / "veterinary_product_results.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)

    return path
