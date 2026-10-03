# Chart-to-Table Evaluation Pipeline

A lightweight, modular Python framework for extracting underlying data tables from chart images and evaluating table quality using **RMS-F1**, **RNSS**, and **Value-Recall@5%** metrics.

Based on the research paper **DePlot: One-shot visual language reasoning by plot-to-table translation** (*Liu et al., Findings of ACL 2023*).

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+ (Python 3.12 recommended)

### 2. Environment Setup
Clone the repository and set up a virtual environment:

```bash
# Navigate to the project directory
cd chart-to-table

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 How to Run Tests

All unit tests and acceptance criteria are automated with `pytest`.

### 1. Run Gate 1 (Week 4) Mandatory Tests
Verifies the 5 mandatory acceptance test cases for Module B2 metrics:
```bash
pytest tests/test_gate1_rms_f1.py::TestGate1Mandatory -v
```
**Expected outcome:** `5 passed`.

### 2. Run All Metric Tests (RMS-F1, RNSS, Value-Recall, Pairwise)
```bash
pytest tests/test_gate1_rms_f1.py -v
```
**Expected outcome:** `29 passed` (covers exact matches, permutations, numeric errors, text omissions, edge cases, and Week 3 pairwise metric tests).

### 3. Run the Entire Test Suite
```bash
pytest tests/ -v
```
**Expected outcome:** `39 passed` across all modules (B1–B6).

### 4. Run End-to-End Pipeline Smoke Test
```bash
python main.py
```
Runs a complete mock run: loads synthetic charts, simulates extraction, parses tables, and computes metric breakdowns.

---

## 📊 Core Metrics Explained (Module B2)

This repository implements the evaluation metrics defined in Section 3.1 of the DePlot paper:

### 1. `RMS-F1` (Relative Mapping Similarity F1) — Primary Metric
Measures structural and numeric alignment between predicted and ground-truth tables:
- **Canonical Representation:** Tables are represented as sets of cell triplets: `(row, col, value)`.
- **Header Distance ($NL_\tau$):** Normalized Levenshtein distance on column headers with a cutoff threshold $\tau = 0.5$.
- **Value Distance ($D$):** Relative error $D(p, t) = \min(1, |p - t| / |t|)$ for numbers; exact match for text.
- **Pair Similarity:** $\text{sim} = (1 - NL_\tau) \times (1 - D) \in [0.0, 1.0]$.
- **Hungarian Matching:** Computes optimal 1-to-1 bipartite assignment minimizing $(1 - \text{sim})$.
- **Aggregation:** Calculates Precision, Recall, and Harmonic Mean ($F_1$). Invariant to row/column permutations.

### 2. `RNSS` (Relative Number Set Similarity) — Baseline Metric
Evaluates the unordered set of numeric values, ignoring headers and table structure:
$$\text{RNSS} = 1 - \frac{\sum_{\text{matched}} D(p_i, t_j)}{\max(N, M)}$$

### 3. `Value-Recall@5%`
The percentage of ground-truth numbers correctly captured within a $\le 5\%$ relative error margin.

---

## 🎯 Gate 1 Acceptance Criteria (5 Mandatory Cases)

| Test Case | Scenario | Expected Result | Verified By |
| :--- | :--- | :--- | :--- |
| **C1** | Exact match | $F_1 = 1.0$, $P = 1.0$, $R = 1.0$ | `test_c1_identical_tables` |
| **C2** | Shuffled row order | $F_1 = 1.0$ (Row-invariance) | `test_c2_row_permutation` |
| **C3** | ~3% numeric deviation | $F_1 \in [0.90, 0.99]$ | `test_c3_small_numeric_error` |
| **C4** | Missing 50% of rows | $\text{Recall} \approx 0.50$, $F_1 \approx 0.67$ | `test_c4_missing_half_rows` |
| **C5** | Typo in column header | $F_1 \in [0.70, 0.99]$ (Partial credit) | `test_c5_header_typo` |

---

## 📂 Project Structure

```text
chart-to-table/
├── src/
│   ├── format/           # [Module B1] Table representations & parsers (Markdown, CSV, Linearized)
│   │   ├── parser.py     # Cleaners, normalizers, text format parsers
│   │   └── schema.py     # TableSchema & TableTriplet data models
│   ├── metrics/          # [Module B2] Evaluation metrics
│   │   └── table_metrics.py # compute_rms_f1, compute_rnss, compute_value_recall_at_5
│   ├── data_loader/      # [Modules B3 & B4] Synthetic & benchmark data loaders
│   ├── models/           # [Module B5] VLM extraction harness (Mock, Open, API)
│   └── evaluation/       # [Module B6] Evaluator & error categorization
├── tests/
│   ├── test_gate1_rms_f1.py # Gate 1 mandatory tests + RNSS & recall tests (19 tests)
│   └── test_pipeline.py     # End-to-end integration tests (10 tests)
├── data/
│   └── sanity/           # Synthetic chart tables for testing
├── main.py               # Quick demo & smoke test
└── requirements.txt      # Project dependencies
```

---

## 💡 Quick Code Example

You can compute metrics directly in your own scripts:

```python
from src.format.schema import TableSchema
from src.metrics.table_metrics import compute_rms_f1, compute_rnss, compute_value_recall_at_5

# Define Ground-Truth table
gold = TableSchema(
    columns=["Quarter", "Revenue"],
    rows=[["Q1", 100.0], ["Q2", 150.0]]
)

# Define Model Prediction (e.g., with ~3% error)
pred = TableSchema(
    columns=["Quarter", "Revenue"],
    rows=[["Q1", 103.0], ["Q2", 154.5]]
)

# Compute metrics
rms = compute_rms_f1(pred, gold)
print("RMS-F1:         ", round(rms["f1"], 4))
print("RNSS:           ", round(compute_rnss(pred, gold), 4))
print("Value-Recall@5%:", compute_value_recall_at_5(pred, gold))
```

Output:
```text
RMS-F1:          0.97
RNSS:            0.97
Value-Recall@5%: 1.0
```
