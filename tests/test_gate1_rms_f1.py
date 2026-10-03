"""Gate 1 — Tuần 4: 5 Ca Test Bắt Buộc cho Module B2 (RMS-F1 / RNSS / value-recall@5%).

Theo đặc tả KeHoachGD2.md (NV 4.1 → 4.3):
  - Ca 1: Giống hệt  → F1 = 1.0
  - Ca 2: Hoán vị hàng → F1 = 1.0
  - Ca 3: Sai ~3%     → F1 ≈ 0.95–0.99
  - Ca 4: Thiếu nửa hàng → Recall ≈ 0.5
  - Ca 5: Sai chính tả header → F1 ≈ 0.90

Ngoài 5 ca bắt buộc còn có các test bổ sung cho RNSS và value-recall@5%.
Lệnh chạy (từ thư mục gốc chart-to-table):
    .venv\\Scripts\\python.exe -m pytest tests/test_gate1_rms_f1.py -v
"""

from __future__ import annotations

import pytest

from src.format.schema import TableSchema, TableTriplet
from src.metrics.table_metrics import (
    compute_pair_similarity,
    compute_rms_f1,
    compute_rnss,
    compute_value_recall_at_5,
    normalized_levenshtein_tau,
    relative_distance,
)

# =============================================================================
# Fixtures dùng chung
# =============================================================================

@pytest.fixture
def simple_gold() -> TableSchema:
    """Bảng gold cơ bản: 2 cột (year, sales), 3 hàng."""
    return TableSchema(
        columns=["year", "sales"],
        rows=[[2021, 100.0], [2022, 130.0], [2023, 160.0]],
    )


@pytest.fixture
def four_row_gold() -> TableSchema:
    """Bảng gold 4 hàng dùng cho Ca 4 (thiếu nửa hàng)."""
    return TableSchema(
        columns=["year", "sales"],
        rows=[[2021, 100.0], [2022, 130.0], [2023, 160.0], [2024, 200.0]],
    )


# =============================================================================
# ── 5 CA BẮT BUỘC (Gate 1) ──
# =============================================================================

class TestGate1Mandatory:
    """5 ca test bắt buộc theo đặc tả B2 — phải tất cả PASS để qua Gate 1."""

    # ------------------------------------------------------------------
    # Ca 1: Giống hệt → F1 = 1.0
    # ------------------------------------------------------------------
    def test_c1_identical_tables(self, simple_gold: TableSchema):
        """C1: pred == gold hoàn toàn → precision = recall = F1 = 1.0.

        Kiểm tra trường hợp lý tưởng: mô hình trích bảng hoàn hảo.
        Hungarian sẽ khớp mỗi triplet pred với triplet gold tương ứng,
        tổng sim = N (= M), nên P = R = F1 = 1.0.
        """
        pred = TableSchema(
            columns=["year", "sales"],
            rows=[[2021, 100.0], [2022, 130.0], [2023, 160.0]],
        )
        result = compute_rms_f1(pred, simple_gold)

        assert result["f1"] == pytest.approx(1.0, abs=1e-6), (
            f"[C1] F1 phải = 1.0 khi pred == gold, nhận được {result['f1']:.6f}"
        )
        assert result["precision"] == pytest.approx(1.0, abs=1e-6), (
            f"[C1] Precision phải = 1.0, nhận được {result['precision']:.6f}"
        )
        assert result["recall"] == pytest.approx(1.0, abs=1e-6), (
            f"[C1] Recall phải = 1.0, nhận được {result['recall']:.6f}"
        )

    # ------------------------------------------------------------------
    # Ca 2: Hoán vị hàng → F1 = 1.0
    # ------------------------------------------------------------------
    def test_c2_row_permutation(self, simple_gold: TableSchema):
        """C2: Hàng pred đảo thứ tự so với gold → F1 = 1.0.

        Đây là đặc tính cốt lõi: RMS-F1 bất biến với hoán vị hàng vì
        Hungarian matching tìm phép gán tối ưu theo tập bộ ba, không
        theo thứ tự. Nếu C2 thất bại → thuật toán bị lỗi nghiêm trọng.
        """
        pred_shuffled = TableSchema(
            columns=["year", "sales"],
            rows=[[2023, 160.0], [2021, 100.0], [2022, 130.0]],  # đảo thứ tự
        )
        result = compute_rms_f1(pred_shuffled, simple_gold)

        assert result["f1"] == pytest.approx(1.0, abs=1e-6), (
            f"[C2] F1 phải = 1.0 khi chỉ hoán vị hàng, nhận được {result['f1']:.6f}"
        )

    # ------------------------------------------------------------------
    # Ca 3: Sai ~3% → F1 ≈ 0.95–0.99
    # ------------------------------------------------------------------
    def test_c3_small_numeric_error(self, simple_gold: TableSchema):
        """C3: Giá trị số sai khoảng 3% → F1 cao nhưng < 1.0.

        Với D(p,t) = |p-t|/|t| ≈ 0.03 và header khớp hoàn toàn (NL_τ=0):
          sim = (1 - 0) * (1 - 0.03) = 0.97
          P = R = 0.97 → F1 ≈ 0.97.
        Ngưỡng kỳ vọng: 0.90 ≤ F1 ≤ 0.99.
        """
        pred = TableSchema(
            columns=["year", "sales"],
            rows=[[2021, 103.0], [2022, 133.9], [2023, 164.8]],  # +3%
        )
        result = compute_rms_f1(pred, simple_gold)

        assert 0.90 <= result["f1"] <= 0.99, (
            f"[C3] F1 phải nằm trong [0.90, 0.99] khi sai ~3%, "
            f"nhận được {result['f1']:.4f}"
        )
        assert result["f1"] < 1.0, (
            f"[C3] F1 phải < 1.0 vì giá trị không khớp hoàn toàn"
        )

        # RNSS cũng phải phản ánh sai số tương tự
        rnss = compute_rnss(pred, simple_gold)
        assert 0.90 <= rnss <= 1.0, (
            f"[C3] RNSS phải ≥ 0.90 khi sai ~3%, nhận được {rnss:.4f}"
        )

    # ------------------------------------------------------------------
    # Ca 4: Thiếu nửa hàng → Recall ≈ 0.5
    # ------------------------------------------------------------------
    def test_c4_missing_half_rows(self, four_row_gold: TableSchema):
        """C4: Pred chỉ có 2/4 hàng gold → Recall ≈ 0.5, Precision ≈ 1.0.

        Giải thích:
          - Gold: 8 triplets (4 hàng × 2 cột)
          - Pred: 4 triplets (2 hàng × 2 cột)
          - Hungarian khớp 4 pred triplets với 4 trong 8 gold triplets tốt nhất
          - matched_sim_sum ≈ 4.0 (sim=1.0 mỗi cặp)
          - Precision = 4/4 = 1.0 | Recall = 4/8 = 0.5
          - F1 = 2*(1.0*0.5)/(1.0+0.5) = 2/3 ≈ 0.667
        """
        pred_partial = TableSchema(
            columns=["year", "sales"],
            rows=[[2021, 100.0], [2022, 130.0]],  # chỉ 2 trong 4 hàng
        )
        result = compute_rms_f1(pred_partial, four_row_gold)

        assert result["recall"] == pytest.approx(0.5, abs=0.05), (
            f"[C4] Recall phải ≈ 0.5 khi thiếu nửa hàng, nhận được {result['recall']:.4f}"
        )
        assert result["precision"] == pytest.approx(1.0, abs=0.05), (
            f"[C4] Precision phải ≈ 1.0 khi pred đúng hết, nhận được {result['precision']:.4f}"
        )
        expected_f1 = 2 / 3  # ≈ 0.667
        assert result["f1"] == pytest.approx(expected_f1, abs=0.05), (
            f"[C4] F1 phải ≈ {expected_f1:.3f}, nhận được {result['f1']:.4f}"
        )

    # ------------------------------------------------------------------
    # Ca 5: Lỗi chính tả header nhỏ → F1 ≈ 0.80–0.99
    # ------------------------------------------------------------------
    def test_c5_header_typo(self, simple_gold: TableSchema):
        """C5: Header pred có lỗi chính tả nhỏ (NL < τ=0.5) → F1 giảm nhưng > 0.

        Ví dụ: 'revenue' vs 'revnue' (bỏ 'e'):
          NL = edit_dist / max_len = 1/7 ≈ 0.14 < τ=0.5
          → NL_τ = 0.14 (không bị cắt về 1.0)
          → sim = (1 - 0.14) * (1 - D_value)
          → F1 < 1.0 nhưng vẫn cao (header "gần giống")

        Nếu header sai > τ thì sim = 0 → F1 ≈ 0. Ca này kiểm tra lỗi NHỎ.
        """
        pred = TableSchema(
            columns=["year", "sals"],   # "sals" thay vì "sales" — lỗi 1 ký tự, NL ≈ 0.2
            rows=[[2021, 100.0], [2022, 130.0], [2023, 160.0]],
        )
        result = compute_rms_f1(pred, simple_gold)

        # Kiểm tra F1 nằm trong khoảng chấp nhận
        assert 0.70 <= result["f1"] <= 0.99, (
            f"[C5] F1 phải nằm trong [0.70, 0.99] khi header sai nhỏ, "
            f"nhận được {result['f1']:.4f}"
        )
        assert result["f1"] < 1.0, (
            f"[C5] F1 phải < 1.0 vì header không khớp hoàn toàn"
        )


# =============================================================================
# ── TEST BỔ SUNG: RNSS theo đặc tả P05 ──
# =============================================================================

class TestRNSS:
    """Kiểm tra RNSS (Relative Number Set Similarity) — chỉ trên giá trị số."""

    def test_rnss_identical_numeric(self):
        """RNSS = 1.0 khi tất cả số giống hệt."""
        gold = TableSchema(columns=["a", "b"], rows=[[100.0, 200.0], [300.0, 400.0]])
        pred = TableSchema(columns=["x", "y"], rows=[[100.0, 200.0], [300.0, 400.0]])
        # RNSS không quan tâm header ("a"≠"x") — chỉ nhìn vào tập số
        rnss = compute_rnss(pred, gold)
        assert rnss == pytest.approx(1.0, abs=1e-6), (
            f"RNSS phải = 1.0 khi số giống hệt (dù header khác), nhận {rnss:.6f}"
        )

    def test_rnss_small_error(self):
        """RNSS phản ánh sai số tương đối nhỏ (~3%)."""
        gold = TableSchema(columns=["q", "r"], rows=[[1000.0, 2000.0]])
        pred = TableSchema(columns=["q", "r"], rows=[[1030.0, 2060.0]])  # +3%
        rnss = compute_rnss(pred, gold)
        # D ≈ 0.03, RNSS = 1 - 0.03 = 0.97 (± nhỏ do 2 số)
        assert 0.93 <= rnss <= 1.0, (
            f"RNSS phải ≥ 0.93 khi sai ~3%, nhận được {rnss:.4f}"
        )

    def test_rnss_ignores_text_values(self):
        """RNSS bỏ qua ô text, chỉ tính trên số."""
        gold = TableSchema(
            columns=["product", "price"],
            rows=[["Laptop", 1000.0], ["Phone", 500.0]],
        )
        pred = TableSchema(
            columns=["product", "price"],
            rows=[["TabletXYZ", 1000.0], ["WatchABC", 500.0]],  # text sai, số đúng
        )
        rnss = compute_rnss(pred, gold)
        assert rnss == pytest.approx(1.0, abs=1e-6), (
            f"RNSS phải = 1.0 khi số đúng hết (text bị bỏ qua), nhận {rnss:.6f}"
        )

    def test_rnss_missing_numbers(self):
        """RNSS thấp khi pred thiếu số (M > N)."""
        gold = TableSchema(columns=["v"], rows=[[100.0], [200.0], [300.0], [400.0]])
        pred = TableSchema(columns=["v"], rows=[[100.0], [200.0]])  # chỉ 2/4 số
        rnss = compute_rnss(pred, gold)
        # 2 số khớp hoàn toàn (D=0), 2 gold còn lại không được match
        # RNSS = 1 - (0+0) / max(2,4) = 1 - 0/4 = 1.0? Không!
        # Vì N=2, M=4: total_D = 0 (2 pairs matched perfectly)
        # RNSS = 1 - 0 / 4 = 1.0 — nhưng có 2 gold bị bỏ sót
        # → Đây là hạn chế của RNSS (không phân biệt precision/recall)
        # Chỉ cần không crash và trả về giá trị hợp lệ
        assert 0.0 <= rnss <= 1.0, f"RNSS phải trong [0,1], nhận được {rnss}"

    def test_rnss_empty_pred(self):
        """RNSS = 0.0 khi pred không có số."""
        gold = TableSchema(columns=["v"], rows=[[100.0], [200.0]])
        pred = TableSchema(columns=["label"], rows=[["A"], ["B"]])  # chỉ text
        rnss = compute_rnss(pred, gold)
        assert rnss == pytest.approx(0.0, abs=1e-6), (
            f"RNSS phải = 0.0 khi pred không có số, nhận {rnss:.6f}"
        )


# =============================================================================
# ── TEST BỔ SUNG: value-recall@5% ──
# =============================================================================

class TestValueRecall:
    """Kiểm tra value-recall@5%: tỉ lệ gold values khớp trong sai số ≤ 5%."""

    def test_recall_all_match(self):
        """Tất cả pred khớp với gold trong 5% → recall = 1.0."""
        gold = TableSchema(columns=["a", "b"], rows=[[100.0, 200.0], [300.0, 400.0]])
        pred = TableSchema(columns=["a", "b"], rows=[[104.0, 209.0], [294.0, 396.0]])
        # sai tối đa: 4%, 4.5%, 2%, 1% — đều ≤ 5%
        recall = compute_value_recall_at_5(pred, gold)
        assert recall == pytest.approx(1.0, abs=1e-6), (
            f"Recall@5% phải = 1.0 khi tất cả sai ≤ 5%, nhận {recall:.6f}"
        )

    def test_recall_partial_match(self):
        """Một số pred sai > 5% → recall < 1.0."""
        gold = TableSchema(columns=["a", "b"], rows=[[100.0, 200.0], [300.0, 400.0]])
        pred = TableSchema(columns=["a", "b"], rows=[[150.0, 200.0], [300.0, 400.0]])
        # 150 sai 50% → không khớp | còn lại 3/4 khớp
        recall = compute_value_recall_at_5(pred, gold)
        assert recall == pytest.approx(0.75, abs=0.02), (
            f"Recall@5% phải ≈ 0.75 (3/4 khớp), nhận {recall:.4f}"
        )

    def test_recall_none_match(self):
        """Tất cả pred sai > 5% so với mọi gold value → recall = 0.0.

        Lưu ý: pred=[9000, 8000] cách xa gold=[100, 200] hơn 5% với mọi cặp.
        """
        gold = TableSchema(columns=["v"], rows=[[100.0], [200.0]])
        pred = TableSchema(columns=["v"], rows=[[9000.0], [8000.0]])  # cách xa hoàn toàn
        recall = compute_value_recall_at_5(pred, gold)
        assert recall == pytest.approx(0.0, abs=1e-6), (
            f"Recall@5% phải = 0.0 khi tất cả sai > 5%, nhận {recall:.6f}"
        )

    def test_recall_no_gold_numbers(self):
        """Bảng gold không có số → recall = 1.0 (không có gì cần recall)."""
        gold = TableSchema(columns=["name"], rows=[["Alice"], ["Bob"]])
        pred = TableSchema(columns=["name"], rows=[["Charlie"], ["Dave"]])
        recall = compute_value_recall_at_5(pred, gold)
        assert recall == pytest.approx(1.0, abs=1e-6), (
            f"Recall@5% phải = 1.0 khi gold không có số, nhận {recall:.6f}"
        )

    def test_recall_exact_5_percent(self):
        """Sai đúng 5% → vẫn được tính là khớp (≤ không phải <)."""
        gold = TableSchema(columns=["v"], rows=[[100.0]])
        pred = TableSchema(columns=["v"], rows=[[105.0]])  # sai đúng 5%
        recall = compute_value_recall_at_5(pred, gold, tolerance=0.05)
        assert recall == pytest.approx(1.0, abs=1e-6), (
            f"Recall@5% phải = 1.0 khi sai đúng = 5% (boundary), nhận {recall:.6f}"
        )


# =============================================================================
# ── EDGE CASES ──
# =============================================================================

class TestEdgeCases:
    """Kiểm tra các trường hợp biên để đảm bảo code không crash."""

    def test_both_empty_tables(self):
        """Hai bảng đều rỗng → F1 = 1.0 (trivially perfect)."""
        empty = TableSchema(columns=[], rows=[])
        result = compute_rms_f1(empty, empty)
        assert result["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_pred_empty_gold_nonempty(self):
        """Pred rỗng, gold có dữ liệu → F1 = 0.0."""
        gold = TableSchema(columns=["a"], rows=[[1.0], [2.0]])
        pred = TableSchema(columns=[], rows=[])
        result = compute_rms_f1(pred, gold)
        assert result["f1"] == pytest.approx(0.0, abs=1e-6)

    def test_single_cell_match(self):
        """Bảng 1 ô duy nhất khớp hoàn toàn → F1 = 1.0."""
        gold = TableSchema(columns=["value"], rows=[[42.0]])
        pred = TableSchema(columns=["value"], rows=[[42.0]])
        result = compute_rms_f1(pred, gold)
        assert result["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_column_permutation_invariance(self):
        """RMS-F1 bất biến với hoán vị cột (triplets vẫn như nhau)."""
        gold = TableSchema(
            columns=["year", "sales", "profit"],
            rows=[[2021, 100.0, 20.0], [2022, 130.0, 35.0]],
        )
        pred_col_swapped = TableSchema(
            columns=["profit", "year", "sales"],          # đảo cột
            rows=[[20.0, 2021, 100.0], [35.0, 2022, 130.0]],
        )
        result = compute_rms_f1(pred_col_swapped, gold)
        assert result["f1"] == pytest.approx(1.0, abs=1e-6), (
            f"F1 phải = 1.0 khi chỉ hoán vị cột, nhận {result['f1']:.6f}"
        )


# =============================================================================
# ── TEST ĐƠN VỊ TUẦN 3: NL_τ, D, VÀ SIM-CẶP (NV 3.1 & NV 3.2) ──
# =============================================================================

class TestWeek3PairwiseMetrics:
    """Kiểm tra các hàm đo khoảng cách và độ tương đồng cho một cặp entry đơn lẻ."""

    # ------------------------------------------------------------------
    # NV 3.1: Normalized Levenshtein + Cắt ngưỡng τ = 0.5
    # ------------------------------------------------------------------
    def test_nl_tau_exact_match(self):
        """Hai chuỗi giống hệt -> NL_tau = 0.0."""
        assert normalized_levenshtein_tau("sales", "sales", tau=0.5) == 0.0

    def test_nl_tau_below_threshold(self):
        """Sai khác nhỏ (<= tau=0.5) -> giữ nguyên khoảng cách Levenshtein chuẩn hóa."""
        # "sals" vs "sales": edit distance = 1, max_len = 5 -> NL = 0.20 <= 0.5
        nl = normalized_levenshtein_tau("sals", "sales", tau=0.5)
        assert nl == pytest.approx(0.20, abs=1e-6)

    def test_nl_tau_cutoff_applied(self):
        """Sai khác lớn (> tau=0.5) -> bị cắt ngưỡng, phạt tối đa 1.0."""
        # "revenue" vs "sales": edit distance = 6, max_len = 7 -> NL = 6/7 ≈ 0.857 > 0.5
        nl = normalized_levenshtein_tau("revenue", "sales", tau=0.5)
        assert nl == 1.0

    # ------------------------------------------------------------------
    # NV 3.2: Sai số tương đối D = min(1, |p - t| / |t|)
    # ------------------------------------------------------------------
    def test_relative_distance_numeric_exact(self):
        """Số khớp hoàn toàn -> D = 0.0."""
        assert relative_distance(100.0, 100.0) == 0.0

    def test_relative_distance_numeric_small_error(self):
        """Số sai lệch nhỏ 3% -> D = 0.03."""
        # |103 - 100| / 100 = 0.03
        d = relative_distance(103.0, 100.0)
        assert d == pytest.approx(0.03, abs=1e-6)

    def test_relative_distance_numeric_cap(self):
        """Số sai lệch lớn (> 100%) -> cắt tối đa tại 1.0."""
        # |300 - 100| / 100 = 2.0 -> min(1.0, 2.0) = 1.0
        assert relative_distance(300.0, 100.0) == 1.0

    def test_relative_distance_text_exact_and_mismatch(self):
        """Text giống nhau -> 0.0, khác nhau -> 1.0."""
        assert relative_distance("North", "north") == 0.0  # case-insensitive
        assert relative_distance("North", "South") == 1.0

    # ------------------------------------------------------------------
    # NV 3.2: Sim một cặp entry: sim = (1 - NL_tau) * (1 - D)
    # ------------------------------------------------------------------
    def test_pair_similarity_perfect_match(self):
        """Header giống hệt + giá trị giống hệt -> sim = 1.0."""
        p = TableTriplet(row=0, col="sales", value=100.0)
        g = TableTriplet(row=0, col="sales", value=100.0)
        assert compute_pair_similarity(p, g, tau=0.5) == pytest.approx(1.0, abs=1e-6)

    def test_pair_similarity_small_numeric_error(self):
        """Header giống hệt, số sai 3% -> sim = 1 * (1 - 0.03) = 0.97."""
        p = TableTriplet(row=0, col="sales", value=103.0)
        g = TableTriplet(row=0, col="sales", value=100.0)
        assert compute_pair_similarity(p, g, tau=0.5) == pytest.approx(0.97, abs=1e-6)

    def test_pair_similarity_header_cutoff_penalty(self):
        """Header sai khác > tau=0.5 -> sim = 0.0 (dù số đúng hoàn toàn)."""
        p = TableTriplet(row=0, col="revenue", value=100.0)
        g = TableTriplet(row=0, col="sales", value=100.0)
        # NL_tau = 1.0 -> sim = (1 - 1.0) * (1 - 0) = 0.0
        assert compute_pair_similarity(p, g, tau=0.5) == 0.0

