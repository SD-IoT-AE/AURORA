
"""AURORA: Window-based UAV monitoring and response pipeline."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np

from aurora.tosa.trust import TrustManager
from aurora.acro.policy import ACROPolicyEngine
from aurora.experiments.attack_injection import AttackInjector


@dataclass
class MonitoringRecord:
    timestamp_s: float
    uav_id: str
    packet_rate_pps: float
    attack_intensity: float
    malicious_probability: float
    trust: float
    risk: float
    action: str
    fl_allowed: bool


class UAVMonitoringPipeline:
    """Monitor synthetic UAV observations in fixed time windows."""

    def __init__(self, num_uavs=50, window_s=2.0,
                 seed=42, scorer=None, logger=None):
        if num_uavs < 1 or window_s <= 0:
            raise ValueError("Invalid UAV count or window size")

        self.num_uavs = num_uavs
        self.window_s = window_s
        self.rng = np.random.default_rng(seed)
        self.scorer = scorer
        self.logger = logger
        self.trust = TrustManager()
        self.policy = ACROPolicyEngine()

        for i in range(num_uavs):
            self.trust.register_uav(f"uav{i}")

    def _score(self, packet_rate, intensity):
        """Estimate threat probability or use an external scorer."""
        if self.scorer is not None:
            probability = self.scorer(packet_rate, intensity)
        else:
            # Synthetic heuristic; not a trained IDS prediction.
            signal = (
                0.004 * max(packet_rate - 100, 0)
                + 4.0 * intensity - 3.0
            )
            probability = 1 / (1 + np.exp(-np.clip(signal, -30, 30)))

        probability = float(probability)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Threat probability must be in [0,1]")
        return probability

    def run(self, scenario_id, duration_s=None):
        """Process scenario attack events over fixed monitoring windows."""
        injector = AttackInjector(
            scenario_id,
            num_uavs=self.num_uavs,
            interval_s=self.window_s,
            seed=int(self.rng.integers(0, 2**31)),
        )
        events = injector.generate()
        duration = (
            float(duration_s) if duration_s is not None
            else injector.scenario.duration_s
        )
        if duration <= 0:
            raise ValueError("Duration must be positive")

        event_map = defaultdict(list)
        for event in events:
            window = int(event.timestamp_s // self.window_s)
            event_map[(window, event.uav_id)].append(event)

        records = []

        for window, timestamp in enumerate(
            np.arange(0, duration, self.window_s)
        ):
            for index in range(self.num_uavs):
                uav_id = f"uav{index}"
                active = event_map.get((window, uav_id), [])

                intensity = max(
                    (event.intensity for event in active),
                    default=0.0,
                )

                baseline = max(
                    0.0, self.rng.normal(80, 8)
                )
                attack_rate = max(
                    (
                        event.metadata.get("packet_rate_pps", 0)
                        for event in active
                    ),
                    default=0,
                )
                packet_rate = baseline + attack_rate

                probability = self._score(
                    packet_rate, intensity
                )

                self.trust.update_from_probability(
                    uav_id, probability,
                    timestamp=float(timestamp),
                )
                trust = self.trust.get_trust(uav_id)
                risk = self.trust.get_risk(uav_id)
                decision = self.policy.evaluate(uav_id, trust)

                record = MonitoringRecord(
                    timestamp_s=round(float(timestamp), 3),
                    uav_id=uav_id,
                    packet_rate_pps=round(packet_rate, 3),
                    attack_intensity=round(intensity, 4),
                    malicious_probability=round(probability, 5),
                    trust=round(trust, 5),
                    risk=round(risk, 5),
                    action=decision.action.value,
                    fl_allowed=decision.fl_allowed,
                )
                records.append(record)

                if self.logger:
                    self.logger.log_tosa(
                        uav_id, probability, trust, risk
                    )
                    self.logger.log_acro(
                        uav_id, decision.action, risk,
                        enforced=False,
                    )

        return records

    @staticmethod
    def export(records, output_dir="outputs/monitoring"):
        """Export monitoring records to CSV and JSON."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        data = [asdict(record) for record in records]

        (path / "monitoring.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

        with (path / "monitoring.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(MonitoringRecord.__dataclass_fields__),
            )
            writer.writeheader()
            writer.writerows(data)

        return path

    @staticmethod
    def summarize(records):
        """Summarize monitoring and mitigation decisions."""
        if not records:
            return {"total_records": 0}

        return {
            "total_records": len(records),
            "monitored_uavs": len({
                record.uav_id for record in records
            }),
            "mean_risk": float(np.mean([
                record.risk for record in records
            ])),
            "fl_eligibility_rate": float(np.mean([
                record.fl_allowed for record in records
            ])),
            "action_counts": {
                action: sum(
                    record.action == action for record in records
                )
                for action in sorted({
                    record.action for record in records
                })
            },
        }


if __name__ == "__main__":
    pipeline = UAVMonitoringPipeline(
        num_uavs=50, window_s=2.0, seed=42
    )

    for scenario in ("S1", "S3", "S7", "S10"):
        # Fresh pipeline avoids trust-state carryover.
        pipeline = UAVMonitoringPipeline(seed=42)
        records = pipeline.run(scenario)

        output = pipeline.export(
            records, f"outputs/monitoring/{scenario}"
        )

        print(scenario, pipeline.summarize(records))
        print("Exported:", output)
