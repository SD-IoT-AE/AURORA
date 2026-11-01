
"""AURORA: Reproducible federated-learning baseline comparisons."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from aurora.rfld.learning import FLConfig, AURORAFederatedTrainer


METHODS = ("aurora", "fedavg", "krum", "trimmed_mean", "median")
METRICS = (
    "final_test_accuracy",
    "final_test_loss",
    "malicious_exclusion_rate",
    "trust_stability",
)


def compare_methods(data, seeds=(42, 43, 44, 45, 46),
                    rounds=50, output_dir="outputs/comparison",
                    **config_overrides):
    """Run each aggregation method under matched experiment settings."""
    required = ("X_train", "y_train", "X_val", "y_val",
                "X_test", "y_test")
    if any(key not in data for key in required):
        raise ValueError("Missing required dataset arrays")

    if not seeds:
        raise ValueError("At least one seed is required")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []

    for seed in seeds:
        for method in METHODS:
            settings = dict(
                rounds=rounds,
                seed=int(seed),
                aggregation_method=method,
                output_dir=str(output / method / f"seed_{seed}"),
            )
            settings.update(config_overrides)

            # Keep comparison dimensions controlled.
            settings["seed"] = int(seed)
            settings["aggregation_method"] = method
            settings["output_dir"] = str(
                output / method / f"seed_{seed}"
            )

            trainer = AURORAFederatedTrainer(FLConfig(**settings))
            result = trainer.run(
                train_features=data["X_train"],
                train_labels=data["y_train"],
                val_features=data["X_val"],
                val_labels=data["y_val"],
                test_features=data["X_test"],
                test_labels=data["y_test"],
            )

            history = result["history"]
            records.append({
                "method": method,
                "seed": seed,
                "final_test_accuracy": result["final_test_accuracy"],
                "final_test_loss": result["final_test_loss"],
                "malicious_exclusion_rate":
                    result["malicious_exclusion_rate"],
                "trust_stability": result["trust_stability"],
                "mean_aggregation_ms": float(np.mean([
                    r["aggregation_time_ms"] for r in history
                ])),
                "final_validation_accuracy":
                    history[-1]["validation_accuracy"],
            })

    raw = pd.DataFrame(records)
    raw.to_csv(output / "raw_results.csv", index=False)

    numeric = [
        "final_test_accuracy", "final_test_loss",
        "malicious_exclusion_rate", "trust_stability",
        "mean_aggregation_ms", "final_validation_accuracy",
    ]

    summary = raw.groupby("method", sort=False)[numeric].agg(
        ["mean", "std"]
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary = summary.reset_index()
    summary.to_csv(output / "summary.csv", index=False)

    with (output / "summary.json").open("w") as f:
        json.dump(
            summary.where(pd.notna(summary), None).to_dict(
                orient="records"
            ),
            f, indent=2,
        )

    return raw, summary


def relative_improvements(summary, metric="final_test_accuracy"):
    """Compute AURORA improvement relative to each baseline."""
    column = f"{metric}_mean"
    values = summary.set_index("method")[column]
    aurora = float(values["aurora"])

    records = []
    for method in METHODS[1:]:
        baseline = float(values[method])
        records.append({
            "baseline": method,
            "aurora": aurora,
            "baseline_value": baseline,
            "absolute_difference": aurora - baseline,
            "relative_improvement_percent": (
                100 * (aurora - baseline) / abs(baseline)
                if baseline != 0 else None
            ),
        })

    return pd.DataFrame(records)


def format_comparison_table(summary):
    """Produce a publication-friendly mean ± SD table."""
    columns = {
        "final_test_accuracy": "Accuracy (%)",
        "final_test_loss": "Test Loss",
        "malicious_exclusion_rate": "Malicious Exclusion (%)",
        "mean_aggregation_ms": "Aggregation Time (ms)",
    }

    rows = []
    for _, record in summary.iterrows():
        row = {"Method": record["method"].upper()}
        for metric, title in columns.items():
            mean = record.get(f"{metric}_mean")
            sd = record.get(f"{metric}_std")
            row[title] = (
                f"{mean:.3f} ± {sd:.3f}"
                if pd.notna(mean) and pd.notna(sd)
                else "N/A"
            )
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from aurora.experiments.dataset_mapping import DatasetMapper

    dataset_path = Path("outputs/datasets/demo/dataset.npz")
    if not dataset_path.exists():
        raise SystemExit(
            "Prepare the dataset first using dataset_mapping.py"
        )

    with np.load(dataset_path) as archive:
        data = {key: archive[key] for key in archive.files}

    raw, summary = compare_methods(
        data,
        seeds=(42, 43),
        rounds=3,
        local_epochs=1,
    )

    print("\nBaseline comparison:")
    print(format_comparison_table(summary).to_string(index=False))

    print("\nAccuracy improvements:")
    print(relative_improvements(summary).to_string(index=False))
