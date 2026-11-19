import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.format.parser import parse_csv_table, parse_markdown_table

UNIT_WORDS = {
    "%", "percent", "percentage", "rate", "share", "ratio", "proportion",
    "usd", "dollar", "dollars", "eur", "euro", "euros", "vnd", "gbp",
    "million", "millions", "billion", "billions", "thousand", "thousands",
    "m", "b", "k", "in", "per", "capita", "tonnes", "tons", "kg", "index",
    "total", "average", "avg", "annual", "growth", "gdp", "score"
}

SYNONYMS = {
    "year": {"date", "time", "period", "timeline", "years"},
    "date": {"year", "time", "period", "timeline"},
    "country": {"nation", "state", "region", "territory", "location", "economy", "area", "countries"},
    "nation": {"country", "state", "region", "economy"},
    "region": {"country", "area", "zone", "territory", "location"},
    "sales": {"revenue", "income", "turnover", "earnings"},
    "revenue": {"sales", "income", "turnover", "earnings", "receipts"},
    "value": {"amount", "count", "number", "quantity", "metric", "figure", "total", "score", "level"},
    "amount": {"value", "quantity", "number", "count", "figure", "total", "level"},
    "percentage": {"share", "proportion", "ratio", "rate", "percent", "pct"},
    "share": {"percentage", "proportion", "ratio", "rate", "percent", "slice"},
    "poverty": {"poor", "destitution", "hardship"},
    "population": {"inhabitants", "residents", "people", "citizens"},
    "gender": {"sex"},
    "male": {"men", "boys"},
    "female": {"women", "girls"}
}


def clean_tokens(text: str) -> Set[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return {w for w in cleaned.split() if w}


def clean_units_and_punct(header: str) -> str:
    h = header.lower()
    h = re.sub(r"\([^)]*\)", "", h)
    h = re.sub(r"\[[^\]]*\]", "", h)
    tokens = clean_tokens(h)
    filtered = [t for t in tokens if t not in UNIT_WORDS]
    return " ".join(filtered).strip()


def is_transpose(
    pred_headers: List[str],
    pred_first_col: List[str],
    gold_headers: List[str],
    gold_first_col: List[str]
) -> bool:
    pred_h_set = {str(h).strip().lower() for h in pred_headers[1:]} if len(pred_headers) > 1 else set()
    gold_rows_set = {str(v).strip().lower() for v in gold_first_col}
    gold_h_set = {str(h).strip().lower() for h in gold_headers[1:]} if len(gold_headers) > 1 else set()
    pred_rows_set = {str(v).strip().lower() for v in pred_first_col}

    if pred_h_set and gold_rows_set:
        overlap_1 = len(pred_h_set & gold_rows_set) / max(1, min(len(pred_h_set), len(gold_rows_set)))
        if overlap_1 >= 0.5:
            return True

    if gold_h_set and pred_rows_set:
        overlap_2 = len(gold_h_set & pred_rows_set) / max(1, min(len(gold_h_set), len(pred_rows_set)))
        if overlap_2 >= 0.5:
            return True

    return False


def has_explicit_unit_or_symbol(text: str) -> bool:
    if re.search(r"[\(\[\$€%£¥]", text):
        return True
    lower = text.lower()
    return bool(re.search(r"\b(in\s+(?:million|billion|thousand|usd|eur|percent|tonnes|kg|gdp))\b", lower))


def is_unit_or_title(pred_headers: List[str], gold_headers: List[str]) -> bool:
    for p in pred_headers:
        for g in gold_headers:
            if has_explicit_unit_or_symbol(p) or has_explicit_unit_or_symbol(g):
                p_clean = clean_units_and_punct(p)
                g_clean = clean_units_and_punct(g)
                if p_clean and g_clean and (p_clean == g_clean or p_clean in g_clean or g_clean in p_clean):
                    return True
    return False


def is_synonym_or_paraphrase(pred_headers: List[str], gold_headers: List[str]) -> bool:
    for p in pred_headers:
        p_toks = clean_tokens(p)
        for g in gold_headers:
            g_toks = clean_tokens(g)
            shared = p_toks & g_toks
            if shared:
                return True
            for pt in p_toks:
                syns = SYNONYMS.get(pt, set())
                if syns & g_toks:
                    return True
    return False


def classify_mismatch_case(
    pred_headers: List[str],
    pred_rows: List[List[Any]],
    gold_headers: List[str],
    gold_rows: List[List[Any]]
) -> Tuple[str, str]:
    pred_first_col = [r[0] for r in pred_rows if len(r) > 0]
    gold_first_col = [r[0] for r in gold_rows if len(r) > 0]

    if is_transpose(pred_headers, pred_first_col, gold_headers, gold_first_col):
        return (
            "(b) Bảng bị xoay (transpose)",
            "Headers of pred overlap significantly with row labels of gold table (series converted into rows or vice versa)."
        )

    if is_unit_or_title(pred_headers, gold_headers):
        return (
            "(c) Kèm tiêu đề/đơn vị",
            "Headers differ primarily due to units, percentages, currency, parentheses, or chart captions embedded in column names."
        )

    if is_synonym_or_paraphrase(pred_headers, gold_headers):
        return (
            "(a) Đồng nghĩa khác chữ",
            "Headers use semantic synonyms, paraphrased expressions, or abbreviations for the same underlying concept."
        )

    return (
        "(d) Sai thật (true mismatch)",
        "Headers contain unrelated, hallucinated, or unaligned series names not represented in ground-truth columns."
    )


def analyze_mismatches(report_path: Path, data_dir: Path) -> Dict[str, Any]:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    records = report.get("records", [])
    mismatches = [r for r in records if r.get("error_category") == "header_mismatch"]

    results: List[Dict[str, Any]] = []
    category_counts = {
        "(a) Đồng nghĩa khác chữ": 0,
        "(b) Bảng bị xoay (transpose)": 0,
        "(c) Kèm tiêu đề/đơn vị": 0,
        "(d) Sai thật (true mismatch)": 0,
    }

    for record in mismatches:
        sample_id = record["sample_id"]
        raw_text = record.get("raw_text", "")
        pred_table = parse_markdown_table(raw_text)

        gold_csv_path = data_dir / "tables" / f"{sample_id}.csv"
        if not gold_csv_path.exists():
            continue

        with open(gold_csv_path, "r", encoding="utf-8") as gf:
            gold_table = parse_csv_table(gf.read())

        cat, explanation = classify_mismatch_case(
            pred_headers=pred_table.columns,
            pred_rows=pred_table.rows,
            gold_headers=gold_table.columns,
            gold_rows=gold_table.rows,
        )

        category_counts[cat] += 1

        results.append({
            "sample_id": sample_id,
            "category": cat,
            "qwen_headers": pred_table.columns,
            "gold_headers": gold_table.columns,
            "f1": record.get("rms_f1", 0.0),
            "rnss": record.get("rnss", 0.0),
            "value_recall_5": record.get("value_recall_at_5", 0.0),
            "explanation": explanation
        })

    return {
        "total_mismatches": len(results),
        "category_counts": category_counts,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze and categorize Qwen header mismatch cases.")
    parser.add_argument("--report", default=Path("outputs/gate2_qwen_report.json"), type=Path)
    parser.add_argument("--data-dir", default=Path("data/chartqa_test"), type=Path)
    parser.add_argument("--output-json", default=Path("outputs/qwen_header_mismatch_analysis.json"), type=Path)
    parser.add_argument("--sample-size", default=20, type=int)

    args = parser.parse_args()

    analysis = analyze_mismatches(args.report, args.data_dir)
    total = max(1, analysis["total_mismatches"])

    print("\n" + "=" * 78)
    print(" QWEN2.5-VL HEADER MISMATCH ERROR DIAGNOSIS & CATEGORIZATION REPORT")
    print("=" * 78)
    print(f" Total Header Mismatch Cases Analyzed: {analysis['total_mismatches']}")
    print("-" * 78)
    for cat, cnt in analysis["category_counts"].items():
        pct = (cnt / total) * 100
        print(f" {cat:<35}: {cnt:>3} cases ({pct:>5.1f}%)")
    print("=" * 78)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for item in analysis["results"]:
        c = item["category"]
        by_cat.setdefault(c, []).append(item)

    samples_to_show: List[Dict[str, Any]] = []
    quota_per_cat = max(3, args.sample_size // 4)
    for c, items in by_cat.items():
        samples_to_show.extend(items[:quota_per_cat])

    samples_to_show = samples_to_show[:args.sample_size]

    print("\n" + "=" * 78)
    print(f" REPRESENTATIVE SAMPLE OF {len(samples_to_show)} MISMATCH CASES (SIDE-BY-SIDE AUDIT)")
    print("=" * 78)
    print(f"{'Sample ID':<16} | {'Category':<26} | {'Qwen Headers':<20} | {'Gold Headers'}")
    print("-" * 78)
    for s in samples_to_show:
        q_h = str(s["qwen_headers"])[:20]
        g_h = str(s["gold_headers"])[:35]
        cat_short = s["category"][:26]
        print(f"{s['sample_id']:<16} | {cat_short:<26} | {q_h:<20} | {g_h}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
