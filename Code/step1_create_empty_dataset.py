import pandas as pd
from pathlib import Path
COLUMNS = [
    "row_id",
    "paper_id",
    "page_num",

    "chemical_composition",
    "alloy_composition",
    "density",
    "tensile_strength",
    "elongation",
    "thermal_conductivity",
    "thermal_expansion",
    "manufacturing_process",

    "source_label",
    "source_snippet",
    "extracted_from"
]
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "materials_dataset_step1_empty.csv"

df = pd.DataFrame(columns=COLUMNS)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"Empty dataset saved to {OUTPUT_PATH}")
