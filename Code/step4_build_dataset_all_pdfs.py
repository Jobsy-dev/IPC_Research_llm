import os
import json
from pathlib import Path
import re

import pandas as pd
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted   # NEW

# -----------------------
# CONFIG
# -----------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)
MODEL_NAME = "models/gemini-2.5-flash"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
DATASET_PATH = BASE_DIR / "materials_dataset_step1_empty.csv"
OUTPUT_PATH = BASE_DIR / "materials_dataset_final.csv"

# Limit usage (Gemini free tier)
MAX_PDFS = 3              # how many PDFs to process
MAX_PAGES_PER_PDF = 5     # how many pages per PDF


# -----------------------
# Helpers
# -----------------------
def read_pdf_page(pdf_path, page_num: int) -> str:
    """Extract plain text from one page of a PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        text = page.extract_text() or ""
        # Basic cleanup: collapse multiple spaces
        text = re.sub(r"[ \t]+", " ", text)
        return text


def extract_rows_with_gemini(page_text: str):
    """
    Send ONE PAGE (tables + text) to Gemini and ask it to find
    any material-property rows, plus proof info.
    """
    prompt = f"""
You are an expert in materials science. From the text below
(coming from ONE PDF page, including both tables and paragraphs),
extract all rows that describe material properties for a specific alloy.

Return JSON ONLY in exactly this format:

{{
  "rows": [
    {{
      "chemical_composition": "",
      "alloy_composition": "",
      "density": "",
      "tensile_strength": "",
      "elongation": "",
      "thermal_conductivity": "",
      "thermal_expansion": "",
      "manufacturing_process": "",
      "source_label": "",
      "source_snippet": "",
      "extracted_from": ""
    }}
  ]
}}

IMPORTANT FILTER RULE:
- ONLY output a row if you see at least one NUMERIC, material-related value
  on this page. Examples:
  - composition like "Al 0.3%" or "Cr 30 at.%"
  - mechanical properties like "320 MPa", "19%", "345.4 ±1.8 MPa"
  - thermal data like "15.6 W/m·K", "10.6×10^-6 /K"
  - process conditions like "1163 °C", "3 h", "103 MPa"

- If you do NOT find any numeric material information, return:
  {{ "rows": [] }}

Other rules:
- Use empty string "" if a field is not present on this page.
- It is OK if chemical_composition and alloy_composition stay empty
  when this page only has mechanical or thermal properties.

- "tensile_strength":
    * Must clearly state if the value is YS, UTS, or both,
      e.g. "YS = 320 MPa @ 25 °C", "UTS = 450 MPa @ 25 °C",
      or "YS = 320 MPa, UTS = 410 MPa @ 25 °C".
    * Always include units and temperature when they appear.

- "elongation":
    * ONLY fill this when the source explicitly mentions
      words like "elongation", "fracture elongation",
      "El (%)", "El%", "strain to failure", "total strain",
      or a table column clearly labelled with those terms.
    * If the text is about a percentage change in SOME OTHER PROPERTY
      (for example "5–8% increase in thermal conductivity" or
      "70–85% of the thermal conductivity of pure copper"),
      then LEAVE elongation = "".

- "thermal_conductivity":
    * ONLY fill this when the source clearly refers to
      "thermal conductivity", "W/m·K", "W/mK", "% IACS"
      or similar conductivity-related units.
    * Values like "70–85% of pure copper conductivity"
      should go here, **not** in elongation.

- "thermal_expansion":
    * ONLY fill this when the source refers to
      "thermal expansion", "CTE", "/K", "×10^-6 /K", etc.

- You may use information from tables OR sentences in the text.
- For source_label:
    * Use something like "Table 3" if it comes from a table title,
    * Or "Text" if from a paragraph.
- For extracted_from:
    * Use "table" if mainly from a table,
    * Use "text" if mainly from narrative text.
- For source_snippet:
    * Copy the exact line(s) from the page that contain the values.

Do NOT add any commentary, only valid JSON.

Page text:
{page_text}
"""

    model = genai.GenerativeModel(MODEL_NAME)

    try:
        response = model.generate_content(prompt)
    except ResourceExhausted as e:
        # We hit the daily / per-model quota.
        print("\n\n*** GEMINI QUOTA REACHED ***")
        print(str(e))
        # Return a special marker so the caller can stop cleanly
        return None
    except Exception as e:
        print("\n[WARN] Gemini call failed on this page:", e)
        return []

    content = response.text.strip()

    # Remove ```json ... ``` wrappers if Gemini adds them
    if content.startswith("```"):
        content = content.strip()
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()

    try:
        data = json.loads(content)
        rows = data.get("rows", [])
        if not isinstance(rows, list):
            return []
        return rows
    except json.JSONDecodeError:
        return []


def row_has_numeric_info(row: dict) -> bool:
    """
    Return True if this row has at least one digit in a key field.
    This helps us drop pure text notes without numeric properties.
    """
    fields_to_check = [
        "chemical_composition",
        "alloy_composition",
        "density",
        "tensile_strength",
        "elongation",
        "thermal_conductivity",
        "thermal_expansion",
        "manufacturing_process",
    ]

    for f in fields_to_check:
        val = row.get(f, "")
        if isinstance(val, str) and any(ch.isdigit() for ch in val):
            return True

    return False


def clean_elongation(val: str) -> str:
    """
    Keep elongation only if the text clearly refers to elongation/strain.
    Otherwise return empty string.
    """
    if not isinstance(val, str):
        return ""
    text = val.lower()
    keywords = [
        "elongation",
        "elong (%)",
        "el (%)",
        "el%",
        "strain",
        "fracture elongation",
    ]
    if any(k in text for k in keywords):
        return val
    return ""


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    new_records = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))[:MAX_PDFS]

    print("Will process these PDFs:")
    for p in pdf_files:
        print("  -", p.name)

    quota_hit = False

    for pdf_path in pdf_files:
        if quota_hit:
            break

        pdf_file = pdf_path.name
        paper_id = pdf_path.stem

        print(f"\n=== Scanning {pdf_file} ===")

        with pdfplumber.open(pdf_path) as pdf:
            num_pages = min(len(pdf.pages), MAX_PAGES_PER_PDF)

        for page_num in range(num_pages):
            if quota_hit:
                break

            text = read_pdf_page(pdf_path, page_num)
            if not text or len(text.strip()) < 40:
                continue

            print(f"  Page {page_num}: sending to Gemini ...", end="", flush=True)
            rows = extract_rows_with_gemini(text)

            # rows == None means we hit the quota inside extract_rows_with_gemini
            if rows is None:
                quota_hit = True
                print(" quota reached. Stopping further processing.")
                break

            print(f" got {len(rows)} rows (before filtering).", end="")

            filtered_rows = [r for r in rows if row_has_numeric_info(r)]
            print(f" kept {len(filtered_rows)} rows after filtering.")

            for r in filtered_rows:
                record = {
                    "row_id": len(df) + len(new_records) + 1,
                    "paper_id": paper_id,
                    "page_num": page_num,
                    "chemical_composition": r.get("chemical_composition", ""),
                    "alloy_composition": r.get("alloy_composition", ""),
                    "density": r.get("density", ""),
                    "tensile_strength": r.get("tensile_strength", ""),
                    "elongation": r.get("elongation", ""),
                    "thermal_conductivity": r.get("thermal_conductivity", ""),
                    "thermal_expansion": r.get("thermal_expansion", ""),
                    "manufacturing_process": r.get("manufacturing_process", ""),
                    "source_label": r.get("source_label", ""),
                    "source_snippet": r.get("source_snippet", ""),
                    "extracted_from": r.get("extracted_from", ""),
                }
                new_records.append(record)

    if new_records:
        df_new = pd.concat([df, pd.DataFrame(new_records)], ignore_index=True)

        # Clean elongation: keep only real elongation/strain descriptions
        df_new["elongation"] = df_new["elongation"].apply(clean_elongation)

        # fix "Â±"
        obj_cols = df_new.select_dtypes(include="object").columns
        df_new[obj_cols] = df_new[obj_cols].apply(
            lambda s: s.astype(str).str.replace("Â±", "±")
        )

        df_new.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"\nDone. Added {len(new_records)} rows.")
        print(f"Saved final dataset to {OUTPUT_PATH}")
    else:
        print("\nNo rows extracted from any PDF.")
