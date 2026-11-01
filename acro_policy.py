
"""AURORA: Adaptive Control and Response Orchestration (ACRO)."""

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    MONITOR = "monitor"
    RESOURCE_ADJUST = "resource_adjust"
    REROUTE = "reroute"
    RATE_LIMIT = "rate_limit"
    FILTER = "filter"
    ISOLATE = "isolate"


# Higher values indicate higher enforcement priority.
PRIORITY = {
    Action.MONITOR: 0,
    Action.RESOURCE_ADJUST: 1,
    Action.REROUTE: 2,
    Action.RATE_LIMIT: 3,
    Action.FILTER: 4,
    Action.ISOLATE: 5,
}


@dataclass(frozen=True)
class PolicyConfig:
    low_risk: float = 0.30
    high_risk: float = 0.70
    trust_threshold: float = 0.50
    isolation_windows: int = 3

    def __post_init__(self):
        if not 0 <= self.low_risk < self.high_risk <= 1:
            raise ValueError("Invalid risk thresholds")
        if not 0 <= self.trust_threshold <= 1:
            raise ValueError("Invalid trust threshold")
        if self.isolation_windows < 1:
            raise ValueError("Invalid isolation window count")


@dataclass(frozen=True)
class PolicyDecision:
    uav_id: int
    trust: float
    risk: float
    risk_level: str
    action: Action
    fl_allowed: bool
    consecutive_high_risk: int
    reason: str


class ACROPolicyEngine:
    """Stateful, risk-aware UAV mitigation policy engine."""

    def __init__(self, config=None):
        self.config = config or PolicyConfig()
        self.high_risk_counts = {}

    def evaluate(self, uav_id, trust, *, advance_window=True):
        """Evaluate trust and choose the highest-priority action."""
        trust = float(trust)
        if not 0 <= trust <= 1:
            raise ValueError("Trust must be in [0, 1]")

        risk = 1.0 - trust
        count = self.high_risk_counts.get(uav_id, 0)

        if advance_window:
            count = count + 1 if risk >= self.config.high_risk else 0
            self.high_risk_counts[uav_id] = count

        if risk < self.config.low_risk:
            level, action = "low", Action.MONITOR
        elif risk < self.config.high_risk:
            level, action = "moderate", Action.RATE_LIMIT
        else:
            level, action = "high", Action.FILTER

        if count >= self.config.isolation_windows:
            action = Action.ISOLATE

        fl_allowed = trust >= self.config.trust_threshold

        return PolicyDecision(
            uav_id=uav_id,
            trust=trust,
            risk=risk,
            risk_level=level,
            action=action,
            fl_allowed=fl_allowed,
            consecutive_high_risk=count,
            reason=f"{level} risk; selected {action.value}",
        )

    @staticmethod
    def resolve_conflicts(actions):
        """Choose the highest-priority incompatible action."""
        actions = [Action(a) for a in actions]
        return max(actions, key=PRIORITY.get) if actions else Action.MONITOR

    def evaluate_observation(self, observation):
        """Consume a TrustObservation from aurora.tosa.trust."""
        return self.evaluate(
            observation.uav_id,
            observation.trust,
        )

    def reset(self, uav_id=None):
        """Clear consecutive high-risk counters."""
        if uav_id is None:
            self.high_risk_counts.clear()
        else:
            self.high_risk_counts.pop(uav_id, None)


if __name__ == "__main__":
    engine = ACROPolicyEngine()

    for t, trust in enumerate([0.9, 0.6, 0.2, 0.2, 0.2], 1):
        decision = engine.evaluate(uav_id=1, trust=trust)
        print(
            f"Window {t}: risk={decision.risk:.2f}, "
            f"action={decision.action.value}, "
            f"FL={decision.fl_allowed}"
        )
