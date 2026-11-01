
"""AURORA: Reproducible offline UAV attack-event simulation."""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np

from aurora.experiments.scenarios import (
    get_scenario, select_malicious_uavs,
)


@dataclass
class AttackEvent:
    scenario_id: str
    timestamp_s: float
    uav_id: str
    attack_type: str
    phase: str
    intensity: float
    metadata: dict


class AttackInjector:
    """Generate scenario-specific synthetic attack events."""

    def __init__(self, scenario_id, num_uavs=50,
                 interval_s=1.0, seed=42):
        self.scenario = get_scenario(scenario_id)
        if num_uavs < 1 or interval_s <= 0:
            raise ValueError("Invalid UAV count or interval")

        self.num_uavs = num_uavs
        self.interval_s = interval_s
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.malicious = select_malicious_uavs(
            self.scenario, num_uavs
        )

    def _phase(self, progress):
        """Determine the attack stage in a monitoring window."""
        if progress < 0.33:
            return "reconnaissance"
        if progress < 0.66:
            return "exploitation"
        return "disruption"

    def _metadata(self, attack, intensity, uav_id, phase):
        """Generate bounded synthetic attack attributes."""
        rate = self.scenario.attack_rate_pps
        metadata = {"synthetic": True}

        if attack in {
            "port_scan", "udp_flood", "control_flood",
            "coordinated_flood", "multi_phase",
        }:
            metadata.update({
                "packet_rate_pps": round(rate * intensity, 2),
                "traffic_multiplier": round(1 + 9 * intensity, 2),
            })

        if attack in {"telemetry_spoofing", "multi_phase"}:
            metadata.update({
                "position_offset_m": round(100 * intensity, 2),
                "speed_bias_mps": round(15 * intensity, 2),
            })

        if attack == "sybil":
            metadata["virtual_identity"] = (
                f"{uav_id}_alias_{int(intensity * 100)}"
            )
            metadata["identity_count"] = max(
                2, int(1 + 5 * intensity)
            )

        if attack == "model_poisoning":
            metadata.update({
                "poisoning_strength": round(
                    0.2 * intensity, 4
                ),
                "poisoning_strategy": "model_delta_perturbation",
                "fl_client_compromised": True,
            })

        if attack == "link_degradation":
            metadata.update({
                "packet_loss_ratio": round(
                    0.4 * intensity, 4
                ),
                "additional_delay_ms": round(
                    150 * intensity, 2
                ),
            })

        if attack == "multi_phase":
            metadata["active_vector"] = {
                "reconnaissance": "port_scan",
                "exploitation": "telemetry_spoofing",
                "disruption": "coordinated_flood",
            }[phase]

        return metadata

    def generate(self):
        """Generate events during the scenario attack interval."""
        scenario = self.scenario
        events = []

        if scenario.attack_type == "none":
            return events

        start = 0.2 * scenario.duration_s
        end = 0.8 * scenario.duration_s
        duration = end - start

        targets = list(self.malicious)

        # Link degradation may affect otherwise benign UAVs.
        if not targets and scenario.attack_type == "link_degradation":
            count = max(1, self.num_uavs // 5)
            targets = list(range(count))

        for timestamp in np.arange(
            start, end, self.interval_s
        ):
            progress = (timestamp - start) / duration
            phase = self._phase(progress)

            for uav_id in targets:
                intensity = float(
                    np.clip(
                        0.5 + 0.5 * progress
                        + self.rng.normal(0, 0.05),
                        0.1, 1.0,
                    )
                )

                events.append(AttackEvent(
                    scenario_id=scenario.scenario_id,
                    timestamp_s=round(float(timestamp), 3),
                    uav_id=f"uav{uav_id}",
                    attack_type=scenario.attack_type,
                    phase=phase,
                    intensity=round(intensity, 4),
                    metadata=self._metadata(
                        scenario.attack_type,
                        intensity,
                        f"uav{uav_id}",
                        phase,
                    ),
                ))

        return events

    @staticmethod
    def export(events, output_dir="outputs/attacks"):
        """Export events to CSV and JSON."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        records = [asdict(event) for event in events]

        with (path / "attack_events.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(records, file, indent=2)

        columns = [
            "scenario_id", "timestamp_s", "uav_id",
            "attack_type", "phase", "intensity", "metadata",
        ]

        with (path / "attack_events.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()

            for record in records:
                record = dict(record)
                record["metadata"] = json.dumps(
                    record["metadata"], sort_keys=True
                )
                writer.writerow(record)

        return path

    @staticmethod
    def summarize(events):
        """Return an attack-event summary."""
        return {
            "total_events": len(events),
            "affected_uavs": len({
                event.uav_id for event in events
            }),
            "attack_types": sorted({
                event.attack_type for event in events
            }),
            "phases": sorted({
                event.phase for event in events
            }),
        }


if __name__ == "__main__":
    for scenario_id in ("S1", "S3", "S7", "S10"):
        injector = AttackInjector(
            scenario_id, num_uavs=50, seed=42
        )
        events = injector.generate()

        output = injector.export(
            events, f"outputs/attacks/{scenario_id}"
        )

        print(scenario_id, injector.summarize(events))
        print("Exported:", output)
