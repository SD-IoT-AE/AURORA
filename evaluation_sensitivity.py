
"""AURORA: Reproducible one-factor-at-a-time sensitivity analysis."""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from aurora.rfld.learning import FLConfig, AURORAFederatedTrainer


@dataclass(frozen=True)
class SensitivityConfig:
    seeds: tuple = (42, 43, 44, 45, 46)
    rounds: int = 50
    local_epochs: int = 5
    output_dir: str = "outputs/sensitivity"


PARAMETER_GRID = {
    "trust_threshold": (0.3, 0.4, 0.5, 0.6, 0.7),
    "dirichlet_alpha": (0.1, 0.3, 0.5, 1.0, 5.0),
    "malicious_fraction": (0.0, 0.1, 0.2, 0.3, 0.4),
    "poisoning_strength": (0.0, 0.1, 0.2, 0.3, 0.5),
    "clients_per_round": (10, 15, 20, 25, 30),
}


class AURORASensitivity:
    """Execute controlled sensitivity experiments."""

    def __init__(self, config=None):
        self.config = config or SensitivityConfig()
        self.output = Path(self.config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

    def run(self, data, parameter_grid=None, base_config=None):
        """Vary one parameter while keeping others fixed."""
        grid = parameter_grid or PARAMETER_GRID
        base = dict(base_config or {})
        records = []

        required = (
            "X_train", "y_train", "X_val", "y_val",
            "X_test", "y_test",
        )
        if any(key not in data for key in required):
            raise ValueError("Missing dataset arrays")

        for parameter, values in grid.items():
            for value in values:
                for seed in self.config.seeds:
                    settings = {
                        "rounds": self.config.rounds,
                        "local_epochs": self.config.local_epochs,
                        "seed": int(seed),
                        "aggregation_method": "aurora",
                        **base,
                        parameter: value,
                    }

                    run_dir = (
                        self.output / parameter
                        / f"value_{value}" / f"seed_{seed}"
                    )
                    settings["output_dir"] = str(run_dir)

                    trainer = AURORAFederatedTrainer(
                        FLConfig(**settings)
                    )

                    result = trainer.run(
                        train_features=data["X_train"],
                        train_labels=data["y_train"],
                        val_features=data["X_val"],
                        val_labels=data["y_val"],
                        test_features=data["X_test"],
                        test_labels=data["y_test"],
                    )

                    records.append({
                        "parameter": parameter,
                        "value": value,
                        "seed": seed,
                        "test_accuracy": result[
                            "final_test_accuracy"
                        ],
                        "test_loss": result[
                            "final_test_loss"
                        ],
                        "malicious_exclusion_rate": result[
                            "malicious_exclusion_rate"
                        ],
                        "trust_stability": result[
                            "trust_stability"
                        ],
                    })

        raw = pd.DataFrame(records)
        raw.to_csv(self.output / "raw_results.csv", index=False)

        metrics = [
            "test_accuracy", "test_loss",
            "malicious_exclusion_rate", "trust_stability",
        ]

        summary = raw.groupby(
            ["parameter", "value"], as_index=False
        )[metrics].agg(["mean", "std"])

        summary.columns = [
            "_".join(str(x) for x in col if x)
            if isinstance(col, tuple) else col
            for col in summary.columns
        ]

        summary.to_csv(
            self.output / "sensitivity_summary.csv",
            index=False,
        )

        return raw, summary

    def export_config(self, grid=None):
        """Save experiment configuration for reproducibility."""
        config = {
            "seeds": list(self.config.seeds),
            "rounds": self.config.rounds,
            "local_epochs": self.config.local_epochs,
            "parameter_grid": {
                k: list(v)
                for k, v in (grid or PARAMETER_GRID).items()
            },
        }

        path = self.output / "experiment_config.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return path


if __name__ == "__main__":
    dataset_path = Path("outputs/datasets/demo/dataset.npz")

    if not dataset_path.exists():
        raise SystemExit(
            "Prepare the dataset using dataset_mapping.py first."
        )

    with np.load(dataset_path) as archive:
        data = {key: archive[key] for key in archive.files}

    analyzer = AURORASensitivity(
        SensitivityConfig(
            seeds=(42, 43),
            rounds=3,
            local_epochs=1,
        )
    )

    # Small smoke test; full grid is available by default.
    grid = {
        "malicious_fraction": (0.1, 0.2, 0.3),
        "clients_per_round": (10, 20),
    }

    analyzer.export_config(grid)
    raw, summary = analyzer.run(data, grid)

    print("\nSensitivity analysis:")
    print(summary.to_string(index=False))
