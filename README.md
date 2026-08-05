# Materials Property Dataset Explorer (LLM-based Extraction)

An LLM-powered pipeline that extracts material property data (composition, tensile strength, thermal properties, etc.) from research paper PDFs using Google's Gemini model, built during an AI Research Internship at Jönköping University.

## Overview

This is an LLM-based approach to automated literature data extraction — using Gemini to read and interpret research paper pages (both tables and narrative text) and pull out structured material property data, with built-in filtering rules to avoid false positives and full traceability back to the source text.

## How it works

1. **Upload PDFs** via the Streamlit sidebar
2. **Page-by-page extraction** — each PDF page's text is sent to Gemini with a carefully engineered prompt that defines exactly which fields to extract (chemical/alloy composition, density, tensile strength, elongation, thermal conductivity, thermal expansion, manufacturing process) and strict rules for when a field should or shouldn't be filled (e.g. elongation is only captured when explicitly labeled as such, to avoid confusing it with unrelated percentage values)
3. **Numeric validation filter** — rows without at least one genuine numeric material value are discarded, reducing noise
4. **Deduplication** — already-processed papers are skipped on re-runs; duplicate rows are removed automatically
5. **Interactive exploration** — filter by paper, manufacturing process, or extraction source (table vs. text), search across all columns, and inspect any row alongside its exact source snippet and downloadable source PDF

## Tech Stack

- **Python**
- **Streamlit** — interactive dashboard
- **Google Gemini API** (`gemini-2.5-flash`) — LLM-based extraction
- **pdfplumber** — PDF text extraction
- **Pandas** — dataset management

## Why an LLM-based approach

This project explores an alternative to traditional rule-based/NLP extraction pipelines — using an LLM's contextual understanding to more flexibly interpret varied table layouts and inconsistent terminology across different papers, while using strict prompt-level rules and post-processing filters to keep extraction accuracy high and reduce hallucinated or misattributed values.

## Setup

Requires a Gemini API key set as an environment variable (`GEMINI_API_KEY`) in a local `.env` file — not included in this repository.
