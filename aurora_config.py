
"""AURORA: Centralized configuration and reproducibility utilities."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
import random


@dataclass
class TOSAConfig:
    """Threat detection and trust assessment."""
    classifier: str = "ensemble"
    trust_alpha: float = 0.70
    initial_trust: float = 1.0
    low_risk_threshold: float = 0.30
    high_risk_threshold: float = 0.70
    fl_trust_threshold: float = 0.50
    isolation_windows: int = 3


@dataclass
class RFLDConfig:
    """Risk-aware federated learning."""
    num_clients: int = 50
    clients_per_round: int = 20
    rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    dirichlet_alpha: float = 0.50
    aggregation_method: str = "aurora"
    malicious_fraction: float = 0.20
    poisoning_strength: float = 0.20


@dataclass
class ACROConfig:
    """Adaptive risk-aware response orchestration."""
    enabled: bool = True
    dry_run: bool = True
    low_risk_threshold: float = 0.30
    high_risk_threshold: float = 0.70
    isolation_windows: int = 3
    rate_limit_kbit: int = 1000


@dataclass
class DatasetConfig:
    """Dataset paths and preprocessing."""
    name: str = "CICIoT2023"
    input_path: str = "data/raw/CICIoT2023.csv"
    output_dir: str = "data/processed"
    label_column: str = "label"
    group_column: str = ""
    test_size: float = 0.15
    val_size: float = 0.15
    max_features: int = 50


@dataclass
class ScenarioConfig:
    """UAV simulation and scenario settings."""
    scenario_id: str = "S1"
    num_uavs: int = 50
    num_rsus: int = 5
    duration_s: int = 300
    mobility_speed_mps: float = 20.0
    controller_ip: str = "127.0.0.1"
    controller_port: int = 6653


@dataclass
class AURORAConfig:
    """Complete AURORA experiment configuration."""
    experiment_name: str = "aurora_experiment"
    seed: int = 42
    output_dir: str = "outputs/experiments"

    tosa: TOSAConfig = field(default_factory=TOSAConfig)
    rfld: RFLDConfig = field(default_factory=RFLDConfig)
    acro: ACROConfig = field(default_factory=ACROConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    def validate(self):
        """Validate configuration consistency and parameter ranges."""
        t, r, a = self.tosa, self.rfld, self.acro
        d, s = self.dataset, self.scenario

        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")

        if not 0 <= t.trust_alpha <= 1:
            raise ValueError("Invalid trust smoothing coefficient")

        if not 0 <= t.initial_trust <= 1:
            raise ValueError("Invalid initial trust")

        for value in (
            t.low_risk_threshold, t.high_risk_threshold,
            t.fl_trust_threshold, a.low_risk_threshold,
            a.high_risk_threshold,
        ):
            if not 0 <= value <= 1:
                raise ValueError("Risk/trust thresholds must be in [0,1]")

        if t.low_risk_threshold >= t.high_risk_threshold:
            raise ValueError("Invalid TOSA risk thresholds")

        if a.low_risk_threshold >= a.high_risk_threshold:
            raise ValueError("Invalid ACRO risk thresholds")

        if t.isolation_windows < 1 or a.isolation_windows < 1:
            raise ValueError("Isolation windows must be positive")

        if r.num_clients < 1 or not (
            1 <= r.clients_per_round <= r.num_clients
        ):
            raise ValueError("Invalid FL client participation")

        if r.rounds < 1 or r.local_epochs < 1 or r.batch_size < 1:
            raise ValueError("Invalid FL training configuration")

        if r.learning_rate <= 0 or r.dirichlet_alpha <= 0:
            raise ValueError("Learning rate and alpha must be positive")

        if not 0 <= r.malicious_fraction <= 1:
            raise ValueError("Invalid malicious-client fraction")

        if r.poisoning_strength < 0:
            raise ValueError("Poisoning strength must be nonnegative")

        if r.aggregation_method not in {
            "aurora", "fedavg", "krum", "trimmed_mean", "median"
        }:
            raise ValueError("Unsupported aggregation method")

        if not 0 < d.test_size < 1 or not 0 < d.val_size < 1:
            raise ValueError("Invalid dataset split fractions")

        if d.test_size + d.val_size >= 1:
            raise ValueError("Validation and test fractions too large")

        if d.max_features < 1:
            raise ValueError("max_features must be positive")

        if s.scenario_id.upper() not in {
            f"S{i}" for i in range(1, 11)
        }:
            raise ValueError("Scenario must be S1-S10")

        if s.num_uavs < 1 or s.num_rsus < 1 or s.duration_s < 1:
            raise ValueError("Invalid topology configuration")

        if s.mobility_speed_mps < 0 or not (
            1 <= s.controller_port <= 65535
        ):
            raise ValueError("Invalid mobility/controller settings")

        if a.rate_limit_kbit < 1:
            raise ValueError("Invalid ACRO rate limit")

        return self

    def to_dict(self):
        return asdict(self)

    def save(self, path=None):
        """Validate and save the full configuration as JSON."""
        self.validate()
        path = Path(
            path or Path(self.output_dir) / "config.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

        return path

    @classmethod
    def load(cls, path):
        """Load JSON configuration into nested dataclasses."""
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)

        for name, config_type in (
            ("tosa", TOSAConfig),
            ("rfld", RFLDConfig),
            ("acro", ACROConfig),
            ("dataset", DatasetConfig),
            ("scenario", ScenarioConfig),
        ):
            if name in data:
                data[name] = config_type(**data[name])

        return cls(**data).validate()

    def set_seed(self):
        """Initialize supported random generators deterministically."""
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)

        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass

        try:
            import torch
            torch.manual_seed(self.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(
                True, warn_only=True
            )
        except ImportError:
            pass

        return self.seed


def load_config(path="configs/aurora.json"):
    """Convenience loader for other AURORA modules."""
    return AURORAConfig.load(path)


if __name__ == "__main__":
    config = AURORAConfig()
    config.validate()
    config.set_seed()

    path = config.save("outputs/config/aurora_default.json")
    restored = AURORAConfig.load(path)

    assert restored.to_dict() == config.to_dict()

    print("Configuration validated:", path)
    print("Scenario:", restored.scenario.scenario_id)
    print("FL clients:", restored.rfld.num_clients)
    print("Random seed:", restored.seed)
