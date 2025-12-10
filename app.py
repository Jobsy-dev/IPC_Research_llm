import os
import json
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# -----------------------
# CONFIG & PATHS
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DATASET_EMPTY_PATH = BASE_DIR / "materials_dataset_step1_empty.csv"
DATASET_FINAL_PATH = BASE_DIR / "materials_dataset_final.csv"

# Make sure Data folder exists
DATA_DIR.mkdir(exist_ok=True)

# Load API key (only needed for extraction step)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-2.5-flash"

# Columns for the dataset (must match step1 script)
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
    "extracted_from",
]


# -----------------------
# LOW-LEVEL DATA HELPERS
# -----------------------
def load_existing_dataset_from_disk() -> pd.DataFrame:
    """
    Load the dataset from disk (no Streamlit cache).
    - If a final dataset exists, use that.
    - Else fall back to the empty template (step1).
    """
    if DATASET_FINAL_PATH.exists():
        return pd.read_csv(DATASET_FINAL_PATH)
    elif DATASET_EMPTY_PATH.exists():
        return pd.read_csv(DATASET_EMPTY_PATH)
    else:
        return pd.DataFrame(columns=COLUMNS)


# -----------------------
# PDF / GEMINI HELPERS
# -----------------------
def read_pdf_page(pdf_path: Path, page_num: int) -> str:
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
    response = model.generate_content(prompt)
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
    """Return True if this row has at least one digit in a key field."""
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


def run_extraction_on_pdfs(pdf_paths, existing_df: pd.DataFrame):
    """
    Run Gemini extraction on the given list of PDFs.

    Behaviour:
    - ALWAYS scans **all pages** in each PDF.
    - Skips any PDF whose paper_id (filename without .pdf)
      is already present in the existing dataset.
    - Returns a DataFrame containing ONLY the *new* rows
      (does not include existing_df).
    """
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found in .env – cannot call Gemini.")
        return None

    genai.configure(api_key=GEMINI_API_KEY)

    new_records = []
    base_row_count = len(existing_df)

    # Find which paper_ids are already in the dataset
    if not existing_df.empty and "paper_id" in existing_df.columns:
        processed_papers = (
            existing_df["paper_id"].dropna().astype(str).unique().tolist()
        )
        processed_papers = set(processed_papers)
    else:
        processed_papers = set()

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        pdf_file = pdf_path.name
        paper_id = pdf_path.stem  # filename without .pdf

        # Skip PDFs already processed earlier
        if paper_id in processed_papers:
            st.write(
                f"✅ Skipping `{pdf_file}` – "
                f"`paper_id = {paper_id}` already exists in the dataset."
            )
            continue

        st.write(f"**Scanning:** `{pdf_file}`")

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            num_pages = total_pages  # ALWAYS scan all pages

        for page_num in range(num_pages):
            text = read_pdf_page(pdf_path, page_num)
            if not text or len(text.strip()) < 40:
                continue

            st.write(f"- Page {page_num}: sending to Gemini…")
            try:
                rows = extract_rows_with_gemini(text)
            except ResourceExhausted as e:
                st.error("Gemini quota reached. Stopping extraction.")
                st.code(str(e))

                # return whatever new rows we collected so far
                if new_records:
                    df_new_partial = pd.DataFrame(new_records, columns=COLUMNS)

                    # Clean elongation: keep only real elongation/strain descriptions
                    df_new_partial["elongation"] = df_new_partial["elongation"].apply(
                        clean_elongation
                    )

                    # fix "Â±"
                    obj_cols = df_new_partial.select_dtypes(include="object").columns
                    df_new_partial[obj_cols] = df_new_partial[obj_cols].apply(
                        lambda s: s.astype(str).str.replace("Â±", "±")
                    )
                    return df_new_partial
                else:
                    return None

            if not rows:
                continue

            filtered_rows = [r for r in rows if row_has_numeric_info(r)]
            st.write(
                f"  → got {len(rows)} rows, "
                f"kept {len(filtered_rows)} after filtering."
            )

            for r in filtered_rows:
                record = {
                    "row_id": base_row_count + len(new_records) + 1,
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

    if not new_records:
        # No new rows extracted from these PDFs
        return None

    df_new = pd.DataFrame(new_records, columns=COLUMNS)

    # Clean elongation: keep only real elongation/strain descriptions
    df_new["elongation"] = df_new["elongation"].apply(clean_elongation)

    # fix "Â±"
    obj_cols = df_new.select_dtypes(include="object").columns
    df_new[obj_cols] = df_new[obj_cols].apply(
        lambda s: s.astype(str).str.replace("Â±", "±")
    )

    return df_new


# -----------------------
# DATASET LOAD / SAVE (Streamlit)
# -----------------------
@st.cache_data
def load_dataset():
    """Cached loader for use inside the app."""
    return load_existing_dataset_from_disk()


def save_dataset_append(df_new: pd.DataFrame):
    """
    Append new rows to existing dataset and save to disk.
    """
    existing = load_existing_dataset_from_disk()

    if df_new is None or df_new.empty:
        return existing

    df_all = pd.concat([existing, df_new], ignore_index=True)

    # Clean elongation for all rows (including old ones)
    df_all["elongation"] = df_all["elongation"].apply(clean_elongation)

    # Optional: drop exact duplicates
    df_all = df_all.drop_duplicates(
        subset=[
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
            "extracted_from",
        ],
        keep="first",
    )

    df_all.to_csv(DATASET_FINAL_PATH, index=False, encoding="utf-8-sig")
    # clear cache so UI reloads updated data
    load_dataset.clear()
    return df_all


# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(
    page_title="Materials Property Dataset Explorer",
    layout="wide",
)

st.sidebar.title("Filters")

# --- Upload & extraction controls ---
st.sidebar.subheader("Upload & generate dataset")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

run_extraction = st.sidebar.button("Generate dataset (calls Gemini)")

if run_extraction and uploaded_files:
    # Save uploaded PDFs into Data/ folder
    saved_paths = []
    for uf in uploaded_files:
        save_path = DATA_DIR / uf.name
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(save_path)
    st.sidebar.success(f"Saved {len(saved_paths)} PDF(s) to Data/")

    with st.spinner("Running Gemini extraction…"):
        existing_df = load_dataset()
        df_new = run_extraction_on_pdfs(saved_paths, existing_df)
        if df_new is not None:
            df_all = save_dataset_append(df_new)
            st.success(
                f"Extraction finished. Dataset now has {len(df_all)} row(s). "
                "Scroll down to explore."
            )
        else:
            st.info(
                "No new rows were extracted from the uploaded PDFs "
                "(they may already be in the dataset, or contain no numeric material data)."
            )

# -----------------------
# MAIN PAGE CONTENT
# -----------------------
st.title("🧪 Materials Property Dataset Explorer")

st.markdown(
    """
This app lets you **explore, filter, and inspect** the dataset generated
from your research papers.

- Use the **sidebar filters** to narrow down by paper, process, source etc.  
- Use the **search box** to search text in any column.  
- Select a row to see **detailed properties and proof snippet** from the PDF.
"""
)

df = load_dataset()

if df.empty:
    st.warning(
        "The dataset is currently empty. Upload PDFs and click "
        "**Generate dataset (calls Gemini)** in the sidebar."
    )
    st.stop()

# ---- Sidebar filters ----
search_text = st.sidebar.text_input("Search in any column")

paper_options = ["(all)"] + sorted(df["paper_id"].dropna().unique().tolist())
paper_filter = st.sidebar.selectbox("Filter by paper_id", paper_options)

proc_options = ["(all)"] + sorted(df["manufacturing_process"].dropna().unique().tolist())
proc_filter = st.sidebar.selectbox("Filter by manufacturing_process", proc_options)

src_options = ["(all)"] + sorted(df["extracted_from"].dropna().unique().tolist())
src_filter = st.sidebar.selectbox("Filter by source (table/text)", src_options)

# ---- Apply filters ----
df_filtered = df.copy()

if paper_filter != "(all)":
    df_filtered = df_filtered[df_filtered["paper_id"] == paper_filter]

if proc_filter != "(all)":
    df_filtered = df_filtered[df_filtered["manufacturing_process"] == proc_filter]

if src_filter != "(all)":
    df_filtered = df_filtered[df_filtered["extracted_from"] == src_filter]

if search_text:
    mask = pd.Series(False, index=df_filtered.index)
    text_lower = search_text.lower()
    for col in df_filtered.columns:
        mask |= df_filtered[col].astype(str).str.lower().str.contains(text_lower)
    df_filtered = df_filtered[mask]

st.subheader("Filtered dataset")
st.caption(
    f"Showing {len(df_filtered)} of {len(df)} total rows "
    f"from `{DATASET_FINAL_PATH.name if DATASET_FINAL_PATH.exists() else DATASET_EMPTY_PATH.name}`"
)

st.dataframe(df_filtered, use_container_width=True, height=300)

# Download filtered CSV
csv_bytes = df_filtered.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ Download filtered dataset as CSV",
    data=csv_bytes,
    file_name="materials_dataset_filtered.csv",
    mime="text/csv",
)

# -----------------------
# Row inspector
# -----------------------
st.markdown("---")
st.subheader("Inspect a single row (with proof)")

if df_filtered.empty:
    st.info("No rows in the filtered dataset. Adjust your filters above.")
    st.stop()

row_id_options = df_filtered["row_id"].tolist()
selected_row_id = st.selectbox("Select row_id", row_id_options)

row = df_filtered[df_filtered["row_id"] == selected_row_id].iloc[0]

st.markdown("### Selected row summary")
cols_top = st.columns(3)
cols_top[0].markdown(f"**row_id:** {int(row['row_id'])}")
cols_top[1].markdown(f"**paper_id:** `{row['paper_id']}`")
page_display = int(row["page_num"]) + 1 if pd.notna(row["page_num"]) else None
cols_top[2].markdown(
    f"**page_num (PDF page):** {page_display if page_display is not None else 'N/A'}"
)

# Material properties
st.markdown("### Material properties")

prop_cols = [
    "chemical_composition",
    "alloy_composition",
    "density",
    "tensile_strength",
    "elongation",
    "thermal_conductivity",
    "thermal_expansion",
    "manufacturing_process",
]

props_df = pd.DataFrame(
    {"value": [row[c] if pd.notna(row[c]) else "<NA>" for c in prop_cols]},
    index=prop_cols,
)
st.table(props_df)

# Proof section
st.markdown("### Proof from PDF")

st.write(f"**source_label:** {row.get('source_label', '') or '<NA>'}")
st.write(f"**extracted_from:** {row.get('extracted_from', '') or '<NA>'}")

st.write("**source_snippet:**")
st.code(row.get("source_snippet", "") or "<NA>")

# Source PDF info
st.markdown("### Source PDF")

paper_id = str(row["paper_id"])
pdf_guess = None
for p in DATA_DIR.glob("*.pdf"):
    if p.stem == paper_id:
        pdf_guess = p
        break

if pdf_guess is not None:
    st.write(f"Found PDF file: `{pdf_guess.name}`")
    with open(pdf_guess, "rb") as f:
        st.download_button(
            "Download source PDF",
            data=f.read(),
            file_name=pdf_guess.name,
            mime="application/pdf",
        )
else:
    st.info(
        "PDF file not found in Data/ for this paper_id. "
        "Make sure the uploaded file name matches paper_id."
    )
