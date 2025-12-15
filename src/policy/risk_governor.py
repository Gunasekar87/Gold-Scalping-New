from dataclasses import dataclass

@dataclass
class RiskLimits:
    max_daily_loss_pct: float
    max_total_exposure: float
    max_spread_points: float
    news_lockout: bool

class RiskGovernor:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def veto(self, metrics: dict) -> tuple[bool, str]:
        if metrics.get("daily_pnl_pct", 0.0) <= -self.limits.max_daily_loss_pct:
            return True, "veto:daily_loss"
        if metrics.get("total_exposure", 0.0) > self.limits.max_total_exposure:
            return True, "veto:exposure"
        if metrics.get("spread_points", 0.0) > self.limits.max_spread_points:
            return True, "veto:spread"
        if self.limits.news_lockout and metrics.get("news_now", False):
            return True, "veto:news"
        return False, ""
