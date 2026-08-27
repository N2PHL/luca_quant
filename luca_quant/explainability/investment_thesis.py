"""
Investment Thesis Engine (Blueprint §17).

Sinh luận điểm đầu tư từ SỐ LIỆU THỰC của phiên nghiên cứu — không phải
template điền tay. Mỗi câu trong thesis đều truy vết được về một bảng kết quả.

Đây chính là phần "Biện luận mô hình trong bài toán phân tích đầu tư"
mà đề bài yêu cầu.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ECONOMIC_STORY = {
    "price": "Lợi suất trễ nắm bắt hiệu ứng phản ứng chậm và đảo chiều ngắn hạn của nhà đầu tư cá nhân — nhóm chiếm phần lớn thanh khoản trên HOSE.",
    "trend": "Khoảng cách giá so với đường trung bình phản ánh hiệu ứng neo giá và hành vi bám xu hướng của dòng tiền.",
    "momentum": "Momentum là dị thường được ghi nhận rộng rãi trong tài chính thực nghiệm, thường gắn với phản ứng dưới mức trước thông tin mới.",
    "volatility": "Biến động có tính cụm; chế độ biến động thấp thường đi kèm phần bù rủi ro ổn định hơn, làm tín hiệu xu hướng đáng tin hơn.",
    "volume": "Khối lượng bất thường là proxy cho dòng tiền có thông tin, thường xuất hiện trước khi giá phản ánh đầy đủ.",
    "fractal": "Hurst > 0.5 hàm ý chuỗi giá có tính dai dẳng, làm tăng hiệu quả của tín hiệu bám xu hướng; Hurst < 0.5 hàm ý quay về trung bình, khi đó tín hiệu xu hướng phản tác dụng.",
    "regime": "Biến chế độ cho phép mô hình phân biệt giai đoạn thị trường, thay vì giả định một quan hệ duy nhất trên toàn mẫu.",
}


class InvestmentThesisEngine:
    def generate(
        self,
        ticker: str,
        ablation_table: pd.DataFrame,
        contribution: pd.DataFrame,
        benchmark_table: pd.DataFrame,
        best_metrics: Dict[str, float],
        gate_result,
        leakage_result,
        n_trials: int = 1,
        fold_metrics: Optional[pd.DataFrame] = None,
    ) -> str:
        L: List[str] = []
        A = L.append

        A(f"L.U.C.A INVESTMENT THESIS — {ticker}")
        A("=" * 68)

        # --- Giả thuyết ---------------------------------------------------
        A("\nGIẢ THUYẾT")
        A("-" * 68)
        A("Thông tin về chế độ xu hướng, biến động và cấu trúc fractal của chuỗi giá")
        A(f"chứa tín hiệu dự báo cho lợi suất kỳ tới của {ticker}, đủ lớn để tồn tại")
        A("sau khi trừ chi phí giao dịch thực tế trên HOSE.")

        # --- Bằng chứng ---------------------------------------------------
        A("\nBẰNG CHỨNG — Ablation Study")
        A("-" * 68)
        if not ablation_table.empty and "Sharpe" in ablation_table.columns:
            base = ablation_table.iloc[0]
            A(f"  Baseline ({base['Experiment']}): Sharpe = {self._f(base.get('Sharpe'))}")
            for _, r in ablation_table.iloc[1:].iterrows():
                d = r.get("Δ Sharpe", np.nan)
                arrow = "↑" if pd.notna(d) and d > 0 else ("↓" if pd.notna(d) and d < 0 else "·")
                A(f"  {arrow} {r['Experiment']:<45} Sharpe = {self._f(r.get('Sharpe'))}"
                  f"  (Δ {self._f(d, sign=True)})")
        else:
            A("  Không có kết quả ablation hợp lệ.")

        # --- Đóng góp lớn nhất --------------------------------------------
        A("\nĐÓNG GÓP LỚN NHẤT")
        A("-" * 68)
        if not contribution.empty:
            top = contribution.iloc[0]
            A(f"  {top['feature_group']}  (Δ Sharpe trung bình = {self._f(top['mean Δ Sharpe'], sign=True)}"
              f" trên {int(top['n_comparisons'])} phép so sánh)")
            for _, r in contribution.iloc[1:].iterrows():
                A(f"    {r['feature_group']:<20} {self._f(r['mean Δ Sharpe'], sign=True)}")
            worst = contribution.iloc[-1]
            if worst["mean Δ Sharpe"] < 0:
                A(f"  Lưu ý: nhóm '{worst['feature_group']}' LÀM GIẢM Sharpe — nên loại khỏi mô hình cuối.")
        else:
            A("  Chưa tính được đóng góp biên.")

        # --- Diễn giải kinh tế --------------------------------------------
        A("\nDIỄN GIẢI KINH TẾ")
        A("-" * 68)
        if not contribution.empty:
            for _, r in contribution.head(3).iterrows():
                g = r["feature_group"]
                A(f"  [{g}] {ECONOMIC_STORY.get(g, 'Chưa có diễn giải kinh tế được đăng ký cho nhóm này.')}")
        A("  Cảnh báo: tương quan thống kê không phải quan hệ nhân quả. Diễn giải trên")
        A("  là giả thuyết kinh tế hậu nghiệm, cần kiểm chứng bằng dữ liệu ngoài mẫu mới.")

        # --- So sánh benchmark ---------------------------------------------
        A("\nSO SÁNH VỚI BASELINE")
        A("-" * 68)
        if not benchmark_table.empty and "Sharpe" in benchmark_table.columns:
            for _, r in benchmark_table.iterrows():
                extra = ""
                if pd.notna(r.get("p-value")):
                    extra = f"   [vs B&H: p = {self._f(r.get('p-value'))}]"
                A(f"  {str(r.get('Model', r.get('Experiment'))):<18} "
                  f"Sharpe {self._f(r.get('Sharpe')):>7}   "
                  f"CAGR {self._pct(r.get('CAGR')):>8}   "
                  f"MDD {self._pct(r.get('Max Drawdown')):>8}{extra}")

        # --- Rủi ro ---------------------------------------------------------
        A("\nRỦI RO VÀ GIỚI HẠN")
        A("-" * 68)
        A(f"  • Nghiên cứu chạy trên MỘT mã ({ticker}). Kết quả chưa chứng minh")
        A("    tính khái quát; cần lặp lại trên rổ VN30 trước khi kết luận.")
        A(f"  • Đã thử {n_trials} cấu hình. Ngưỡng Sharpe sinh ra thuần do may rủi")
        A("    được hiệu chỉnh qua Deflated Sharpe Ratio ở bảng Acceptance Gate.")
        if fold_metrics is not None and not fold_metrics.empty and "Sharpe" in fold_metrics:
            s = fold_metrics["Sharpe"].dropna()
            if len(s):
                A(f"  • Sharpe theo từng fold: {[round(v, 2) for v in s.tolist()]}"
                  f" (độ lệch chuẩn {s.std():.2f}).")
                if s.std() > 1.0:
                    A("    Phân tán lớn -> hiệu năng phụ thuộc mạnh vào chế độ thị trường.")
        A("  • Chi phí giao dịch mô hình hoá theo HOSE (phí mua/bán + thuế bán 0.1%,")
        A("    slippage, ràng buộc T+2, long-only). Thanh khoản thực tế và tác động")
        A("    giá khi quy mô vốn lớn chưa được mô hình hoá.")

        # --- Leakage ---------------------------------------------------------
        A("\nKIỂM ĐỊNH RÒ RỈ DỮ LIỆU")
        A("-" * 68)
        if leakage_result is not None:
            for c in leakage_result.checks:
                mark = "✓" if c["status"] == "PASS" else "✗"
                A(f"  {mark} {c['check']:<28} {c['detail']}")
        else:
            A("  Chưa chạy kiểm định leakage.")

        # --- Kết luận ---------------------------------------------------------
        A("\nKẾT LUẬN")
        A("-" * 68)
        A(f"  QUYẾT ĐỊNH: {gate_result.decision}")
        for c in gate_result.checks:
            mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[c["status"]]
            A(f"  {mark} {c['gate']:<32} {c['value']}  (yêu cầu {c['target']})")
        for n in gate_result.notes:
            A(f"  → {n}")

        if gate_result.decision == "REJECT":
            A("\n  Mô hình chưa đủ điều kiện đưa vào phân bổ vốn. Đây là một KẾT QUẢ")
            A("  NGHIÊN CỨU HỢP LỆ, không phải thất bại: hệ thống đã làm đúng việc")
            A("  của nó là ngăn một chiến lược chưa được chứng minh đi vào thực tế.")

        return "\n".join(L)

    # ------------------------------------------------------------------
    @staticmethod
    def _f(v, sign: bool = False) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
            return "n/a"
        return f"{v:+.2f}" if sign else f"{v:.2f}"

    @staticmethod
    def _pct(v) -> str:
        if v is None or pd.isna(v) or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.1%}"
