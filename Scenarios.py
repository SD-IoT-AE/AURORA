
"""AURORA: Reproducible UAV attack and operational scenarios."""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    name: str
    attack_type: str
    affected_components: tuple
    duration_s: int
    malicious_fraction: float
    attack_rate_pps: int
    mobility_speed_mps: float
    expected_response: str
    seed: int = 42

    def __post_init__(self):
        if self.duration_s <= 0 or self.attack_rate_pps < 0:
            raise ValueError("Invalid duration or attack rate")
        if not 0 <= self.malicious_fraction <= 1:
            raise ValueError("Invalid malicious fraction")
        if self.mobility_speed_mps < 0:
            raise ValueError("Invalid UAV mobility speed")

    def to_dict(self):
        return asdict(self)


SCENARIOS = {
    "S1": Scenario(
        "S1", "Normal UAV Operations", "none",
        ("UAV", "RSU", "Controller"),
        300, 0.00, 0, 10.0, "monitor",
    ),
    "S2": Scenario(
        "S2", "Reconnaissance and Scanning", "port_scan",
        ("UAV", "RSU"),
        300, 0.10, 100, 12.0, "rate_limit",
    ),
    "S3": Scenario(
        "S3", "UAV Flooding Attack", "udp_flood",
        ("UAV", "RSU", "Links"),
        300, 0.20, 2000, 15.0, "filter",
    ),
    "S4": Scenario(
        "S4", "Controller Saturation", "control_flood",
        ("Controller", "SDN_Switch"),
        300, 0.20, 3000, 12.0, "rate_limit",
    ),
    "S5": Scenario(
        "S5", "False Telemetry Injection", "telemetry_spoofing",
        ("UAV", "TOSA"),
        300, 0.20, 200, 10.0, "filter",
    ),
    "S6": Scenario(
        "S6", "Sybil Identity Attack", "sybil",
        ("UAV", "RSU", "TOSA"),
        300, 0.20, 500, 15.0, "isolate",
    ),
    "S7": Scenario(
        "S7", "Federated Model Poisoning", "model_poisoning",
        ("UAV", "RFLD", "Global_Model"),
        600, 0.20, 0, 12.0, "exclude_from_fl",
    ),
    "S8": Scenario(
        "S8", "UAV Mobility and Link Degradation", "link_degradation",
        ("UAV", "RSU", "Wireless_Links"),
        600, 0.00, 0, 25.0, "reroute",
    ),
    "S9": Scenario(
        "S9", "Coordinated Multi-UAV Attack", "coordinated_flood",
        ("UAV", "RSU", "Controller"),
        600, 0.30, 5000, 20.0, "filter",
    ),
    "S10": Scenario(
        "S10", "Multi-Phase Adaptive Attack", "multi_phase",
        ("UAV", "RSU", "TOSA", "RFLD", "ACRO"),
        900, 0.30, 6000, 20.0, "isolate",
    ),
}


def get_scenario(scenario_id):
    """Retrieve a scenario by its identifier."""
    key = scenario_id.upper()
    if key not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_id}")
    return SCENARIOS[key]


def list_scenarios():
    """Return all scenarios in S1-S10 order."""
    return list(SCENARIOS.values())


def select_malicious_uavs(scenario, num_uavs=50):
    """Deterministically select malicious UAV identifiers."""
    if num_uavs <= 0:
        raise ValueError("num_uavs must be positive")

    rng = random.Random(scenario.seed)
    count = round(num_uavs * scenario.malicious_fraction)

    return sorted(rng.sample(range(num_uavs), count))


def build_experiment(scenario_id, num_uavs=50, seed=None):
    """Construct a reproducible experiment configuration."""
    scenario = get_scenario(scenario_id)

    if seed is not None:
        scenario = Scenario(
            **{**scenario.to_dict(), "seed": seed}
        )

    return {
        "scenario": scenario.to_dict(),
        "num_uavs": num_uavs,
        "malicious_uavs": select_malicious_uavs(
            scenario, num_uavs
        ),
        "fl_clients_per_round": min(20, num_uavs),
        "trust_threshold": 0.50,
        "trust_alpha": 0.70,
        "isolation_windows": 3,
    }


def save_experiment(config, output_path):
    """Export the experiment configuration as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    return path


if __name__ == "__main__":
    for scenario in list_scenarios():
        config = build_experiment(scenario.scenario_id)
        save_experiment(
            config,
            f"outputs/scenarios/{scenario.scenario_id}.json",
        )
        print(
            scenario.scenario_id,
            scenario.name,
            "Malicious UAVs:",
            len(config["malicious_uavs"]),
        )
