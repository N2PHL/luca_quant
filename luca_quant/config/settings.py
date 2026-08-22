"""
Cấu hình trung tâm của L.U.C.A Quant.

Nguyên tắc: KHÔNG hard-code tham số nghiên cứu trong logic.
Mọi ngưỡng (cost, threshold, acceptance gate) đều nằm ở đây và có thể override.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# 1. Chi phí giao dịch — mô hình thị trường Việt Nam
# --------------------------------------------------------------------------
@dataclass
class CostConfig:
    """
    Chi phí thực tế TTCK Việt Nam (long-only, HOSE).

    Lưu ý quan trọng: chi phí MUA và BÁN không đối xứng.
    Bán phải chịu thêm thuế TNCN 0.1% trên giá trị giao dịch.
    Repo cũ dùng một hằng số 0.0015 cho cả hai chiều -> under-estimate ~40% chi phí.
    """
    commission_buy: float = 0.0015          # phí môi giới mua 0.15%
    commission_sell: float = 0.0015         # phí môi giới bán 0.15%
    sell_tax: float = 0.001                 # thuế TNCN khi bán 0.1%
    slippage_bps: float = 5.0               # trượt giá 5 bps mỗi chiều
    allow_short: bool = False               # TTCK VN: không bán khống
    settlement_days: int = 2                # T+2: mua hôm nay, T+2 mới bán được

    @property
    def buy_cost(self) -> float:
        return self.commission_buy + self.slippage_bps / 10_000.0

    @property
    def sell_cost(self) -> float:
        return self.commission_sell + self.sell_tax + self.slippage_bps / 10_000.0


# --------------------------------------------------------------------------
# 2. Cấu hình chia dữ liệu — Train / Valid / Test + Purge + Embargo
# --------------------------------------------------------------------------
@dataclass
class SplitConfig:
    """
    Purged Walk-Forward.

    purge_days  : số phiên bỏ đi giữa các segment, PHẢI >= horizon của label,
                  nếu không nhãn của phiên cuối train sẽ nhìn vào phiên đầu valid.
    embargo_days: chặn thêm sau test để tránh serial correlation rò rỉ ngược.
    """
    n_splits: int = 5
    train_ratio: float = 0.60
    valid_ratio: float = 0.20
    test_ratio: float = 0.20
    purge_days: int = 5
    embargo_days: int = 5
    expanding: bool = True
    min_train_size: int = 250


# --------------------------------------------------------------------------
# 3. Position sizing — ngưỡng xác suất LÀ hyperparameter
# --------------------------------------------------------------------------
@dataclass
class SizingConfig:
    """
    Repo cũ hard-code 0.65 / 0.80 / 0.90.
    Đây là 3 hyperparameter. Chúng phải được TUNE TRÊN VALID, không phải chọn tay
    rồi đem đo trên test (threshold leakage — Blueprint §7).
    """
    grid_entry: List[float] = field(default_factory=lambda: [0.50, 0.52, 0.55, 0.58, 0.60, 0.65])
    max_position: float = 1.0
    mode: str = "step"                      # "step" | "linear" | "binary"
    tune_objective: str = "sharpe"           # metric tối ưu trên VALID


# --------------------------------------------------------------------------
# 4. Ràng buộc rủi ro (KHÔNG tạo alpha — chỉ được phép cắt giảm vị thế)
# --------------------------------------------------------------------------
@dataclass
class RiskConfig:
    max_exposure: float = 1.0
    min_liquidity: float = 50_000           # KLGD tối thiểu để coi là giao dịch được
    target_volatility: float | None = 0.20  # vol targeting năm; None = tắt
    max_drawdown_stop: float | None = 0.25  # kill-switch khi DD vượt ngưỡng
    vol_lookback: int = 20


# --------------------------------------------------------------------------
# 5. Acceptance Gate (Blueprint §16)
# --------------------------------------------------------------------------
@dataclass
class AcceptanceConfig:
    min_oos_sharpe: float = 1.80
    min_sortino: float = 2.00
    max_drawdown: float = 0.20
    min_profit_factor: float = 1.50
    min_deflated_sharpe_pvalue: float = 0.95   # DSR phải > 0.95 mới coi là thật
    require_leakage_clean: bool = True
    require_beat_buyhold: bool = True


@dataclass
class Settings:
    cost: CostConfig = field(default_factory=CostConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    risk_free_rate: float = 0.045            # lãi suất phi rủi ro VN ~4.5%/năm
    random_state: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)


DEFAULT_SETTINGS = Settings()
