
"""AURORA: Reproducible component ablation experiments."""

from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AblationConfig:
    """Component switches for a single experimental variant."""
    tosa: bool = True
    trust_weighting: bool = True
    poisoning_defense: bool = True
    acro: bool = True


ABLATIONS = {
    "Full AURORA": AblationConfig(),
    "Without TOSA": AblationConfig(tosa=False),
    "Without Trust Weighting": AblationConfig(
        trust_weighting=False
    ),
    "Without RFLD Defense": AblationConfig(
        poisoning_defense=False
    ),
    "Without ACRO": AblationConfig(acro=False),
}


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple = (42, 43, 44, 45, 46)
    rounds: int = 50
    output_dir: str = "outputs/ablation"


class AblationStudy:
    """Run matched ablation variants using an experiment callback."""

    def __init__(self, runner, config=None):
        """
        runner(components, seed, rounds) -> metric dictionary.

        The runner must apply component switches to the
        underlying TOSA/RFLD/ACRO experiment.
        """
        if not callable(runner):
            raise TypeError("runner must be callable")

        self.runner = runner
        self.config = config or ExperimentConfig()

    def run(self, variants=None):
        """Execute variants and export raw and aggregated results."""
        variants = variants or ABLATIONS
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        records = []

        for seed in self.config.seeds:
            for name, components in variants.items():
                metrics = self.runner(
                    components=components,
                    seed=int(seed),
                    rounds=self.config.rounds,
                )

                if not isinstance(metrics, dict):
                    raise TypeError("Runner must return a dictionary")

                record = {
                    "variant": name,
                    "seed": int(seed),
                    **metrics,
                }
                records.append(record)

        raw = pd.DataFrame(records)
        raw.to_csv(output / "raw_results.csv", index=False)

        metric_columns = [
            col for col in raw.select_dtypes(
                include=[np.number]
            ).columns
            if col != "seed"
        ]

        summary = raw.groupby(
            "variant", sort=False
        )[metric_columns].agg(["mean", "std"])

        summary.columns = [
            f"{metric}_{stat}"
            for metric, stat in summary.columns
        ]
        summary = summary.reset_index()

        summary.to_csv(
            output / "ablation_summary.csv", index=False
        )

        config_data = {
            "seeds": list(self.config.seeds),
            "rounds": self.config.rounds,
            "variants": {
                name: asdict(components)
                for name, components in variants.items()
            },
        }

        with (output / "configuration.json").open("w") as f:
            json.dump(config_data, f, indent=2)

        return raw, summary


def calculate_degradation(
    summary, metric="accuracy", higher_is_better=True
):
    """Compare each ablation against the full AURORA system."""
    column = f"{metric}_mean"
    values = summary.set_index("variant")[column]

    full = float(values["Full AURORA"])
    records = []

    for variant, value in values.items():
        if variant == "Full AURORA":
            continue

        value = float(value)
        degradation = (
            full - value if higher_is_better
            else value - full
        )

        records.append({
            "variant": variant,
            "full_aurora": full,
            "ablated": value,
            "degradation": degradation,
        })

    return pd.DataFrame(records)


def format_ablation_table(summary):
    """Format available metrics as mean ± standard deviation."""
    rows = []

    metrics = sorted({
        col[:-5] for col in summary.columns
        if col.endswith("_mean")
    })

    for _, item in summary.iterrows():
        row = {"Variant": item["variant"]}

        for metric in metrics:
            mean = item[f"{metric}_mean"]
            std = item[f"{metric}_std"]

            row[metric] = (
                f"{mean:.4f} ± {std:.4f}"
                if pd.notna(mean) and pd.notna(std)
                else "N/A"
            )

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    def demonstration_runner(components, seed, rounds):
        """Synthetic smoke test, not experimental evidence."""
        rng = np.random.default_rng(seed)

        # Illustrative output verifies the reporting pipeline.
        accuracy = 0.90 + rng.normal(0, 0.005)
        if not components.tosa:
            accuracy -= 0.03
        if not components.trust_weighting:
            accuracy -= 0.02
        if not components.poisoning_defense:
            accuracy -= 0.04

        return {
            "accuracy": float(accuracy),
            "mitigation_latency_ms": (
                150.0 if components.acro else np.nan
            ),
        }

    study = AblationStudy(
        demonstration_runner,
        ExperimentConfig(seeds=(42, 43, 44), rounds=3),
    )

    raw, summary = study.run()

    print("\nAblation comparison:")
    print(format_ablation_table(summary).to_string(index=False))

    print("\nAccuracy degradation:")
    print(calculate_degradation(summary).to_string(index=False))
