
"""AURORA: Reproducible UAV traffic and attack-event generator."""

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import random

from aurora.experiments.scenarios import build_experiment


@dataclass(frozen=True)
class TrafficConfig:
    num_uavs: int = 50
    benign_rate_pps: int = 10
    interval_s: int = 1
    attack_start_fraction: float = 0.20
    attack_end_fraction: float = 0.80
    seed: int = 42


@dataclass(frozen=True)
class TrafficEvent:
    timestamp: float
    uav_id: int
    src_ip: str
    dst_ip: str
    protocol: str
    packet_size: int
    packets: int
    bytes_total: int
    malicious: bool
    attack_type: str


class UAVTrafficGenerator:
    """Generate aggregated packet events per UAV and interval."""

    NETWORK_ATTACKS = {
        "port_scan", "udp_flood", "control_flood",
        "telemetry_spoofing", "sybil",
        "coordinated_flood", "multi_phase",
    }

    def __init__(self, config=None):
        self.config = config or TrafficConfig()

        if self.config.num_uavs < 1:
            raise ValueError("num_uavs must be positive")
        if self.config.interval_s < 1:
            raise ValueError("interval_s must be positive")
        if self.config.benign_rate_pps < 0:
            raise ValueError("benign_rate_pps must be nonnegative")
        if not (
            0 <= self.config.attack_start_fraction
            < self.config.attack_end_fraction <= 1
        ):
            raise ValueError("Invalid attack scheduling fractions")

    def generate(self, scenario_id):
        """Yield deterministic benign and attack traffic events."""
        cfg = self.config
        experiment = build_experiment(
            scenario_id, cfg.num_uavs, cfg.seed
        )
        scenario = experiment["scenario"]
        attackers = set(experiment["malicious_uavs"])
        duration = scenario["duration_s"]
        attack_type = scenario["attack_type"]

        start = duration * cfg.attack_start_fraction
        end = duration * cfg.attack_end_fraction
        rng = random.Random(cfg.seed)

        for t in range(0, duration, cfg.interval_s):
            window_s = min(cfg.interval_s, duration - t)
            active = start <= t < end

            for uav_id in range(cfg.num_uavs):
                src = f"10.0.{uav_id // 254}.{uav_id % 254 + 1}"
                dst = "10.1.0.1"

                # Benign operational traffic.
                benign_packets = cfg.benign_rate_pps * window_s
                size = rng.randint(128, 1024)

                if benign_packets:
                    yield TrafficEvent(
                        t, uav_id, src, dst, "UDP",
                        size, benign_packets,
                        size * benign_packets,
                        False, "none",
                    )

                if not active or uav_id not in attackers:
                    continue

                # Non-network attacks are handled by their
                # respective experimental modules.
                if attack_type not in self.NETWORK_ATTACKS:
                    continue

                attack_rate = scenario["attack_rate_pps"]
                if attack_rate <= 0:
                    continue

                packets = attack_rate * window_s
                attack_size = rng.randint(64, 256)

                protocol = (
                    "TCP" if attack_type == "port_scan"
                    else "UDP"
                )

                if attack_type == "multi_phase":
                    progress = (t - start) / max(end - start, 1)
                    if progress < 1 / 3:
                        phase = "port_scan"
                        protocol = "TCP"
                    elif progress < 2 / 3:
                        phase = "udp_flood"
                    else:
                        phase = "control_flood"
                    current_attack = phase
                else:
                    current_attack = attack_type

                yield TrafficEvent(
                    t, uav_id, src, dst, protocol,
                    attack_size, packets,
                    attack_size * packets,
                    True, current_attack,
                )

    def summarize(self, scenario_id):
        """Calculate traffic statistics for a scenario."""
        events = list(self.generate(scenario_id))

        return {
            "scenario": scenario_id,
            "events": len(events),
            "total_packets": sum(e.packets for e in events),
            "malicious_packets": sum(
                e.packets for e in events if e.malicious
            ),
            "total_bytes": sum(e.bytes_total for e in events),
            "attacking_uavs": sorted({
                e.uav_id for e in events if e.malicious
            }),
        }

    def export_csv(self, scenario_id, output_path):
        """Export traffic events for reproducible analysis."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(TrafficEvent.__dataclass_fields__)
            )
            writer.writeheader()
            for event in self.generate(scenario_id):
                writer.writerow(asdict(event))

        return path


if __name__ == "__main__":
    generator = UAVTrafficGenerator()

    for scenario_id in ("S1", "S3", "S7", "S9", "S10"):
        summary = generator.summarize(scenario_id)
        print(scenario_id, summary)

    generator.export_csv(
        "S10", "outputs/traffic/S10_traffic.csv"
    )
