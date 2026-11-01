
"""AURORA: Statistical analysis of repeated experimental runs."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats


def _values(values):
    """Validate a one-dimensional array of independent run results."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite observations required")
    return x


def summarize(values, confidence=0.95):
    """Mean, sample SD, and Student-t confidence interval."""
    x = _values(values)
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")

    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    margin = stats.t.ppf(
        (1 + confidence) / 2, n - 1
    ) * sd / np.sqrt(n)

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_lower": float(mean - margin),
        "ci_upper": float(mean + margin),
    }


def paired_comparison(reference, candidate, confidence=0.95):
    """Paired t-test, improvement CI, and paired Cohen's dz.

    candidate - reference is the signed improvement.
    Arrays must correspond to identical random seeds.
    """
    a, b = _values(reference), _values(candidate)

    if len(a) != len(b):
        raise ValueError("Paired arrays must have equal lengths")

    diff = b - a
    result = summarize(diff, confidence)
    sd = result["sd"]

    if sd == 0:
        p_value = 1.0 if np.all(diff == 0) else 0.0
        effect = 0.0 if np.all(diff == 0) else None
    else:
        p_value = float(stats.ttest_rel(b, a).pvalue)
        effect = float(np.mean(diff) / sd)

    return {
        "mean_improvement": result["mean"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "p_value": p_value,
        "cohens_dz": effect,
        "significant_05": bool(p_value < 0.05),
    }


def compare_baselines(
    results, metric="final_test_accuracy",
    reference="aurora", confidence=0.95,
):
    """Compare methods using seed-aligned paired experiments.

    Expected columns: method, seed, and the selected metric.
    """
    required = {"method", "seed", metric}
    if not required.issubset(results.columns):
        raise ValueError(f"Missing columns: {required - set(results)}")

    if results.duplicated(["method", "seed"]).any():
        raise ValueError("Duplicate method/seed combinations")

    table = results.pivot(
        index="seed", columns="method", values=metric
    )

    if reference not in table:
        raise KeyError(f"Reference method not found: {reference}")

    records = []

    for method in table.columns:
        if method == reference:
            continue

        paired = table[[method, reference]].dropna()
        if len(paired) < 2:
            raise ValueError(f"Insufficient paired runs: {method}")

        # Positive difference means reference outperforms baseline.
        comparison = paired_comparison(
            paired[method].to_numpy(),
            paired[reference].to_numpy(),
            confidence,
        )

        records.append({
            "reference": reference,
            "baseline": method,
            "metric": metric,
            "n_pairs": len(paired),
            **comparison,
        })

    return pd.DataFrame(records)


def export_statistics(
    results, output_dir="outputs/statistics",
    metric="final_test_accuracy",
):
    """Export per-method summaries and paired comparisons."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    summaries = []
    for method, group in results.groupby("method"):
        summaries.append({
            "method": method,
            **summarize(group[metric].to_numpy()),
        })

    summary = pd.DataFrame(summaries)
    comparisons = compare_baselines(results, metric=metric)

    summary.to_csv(path / "descriptive_statistics.csv", index=False)
    comparisons.to_csv(path / "significance_tests.csv", index=False)

    report = {
        "metric": metric,
        "descriptive_statistics": summary.to_dict("records"),
        "paired_comparisons": comparisons.where(
            pd.notna(comparisons), None
        ).to_dict("records"),
    }

    with (path / "statistics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    return summary, comparisons


if __name__ == "__main__":
    # Synthetic demonstration; not AURORA benchmark results.
    rng = np.random.default_rng(42)
    records = []

    for seed in range(42, 52):
        shared = rng.normal(0, 0.5)
        for method, offset in {
            "aurora": 0.0,
            "fedavg": -2.0,
            "krum": -1.4,
            "trimmed_mean": -1.0,
            "median": -1.2,
        }.items():
            records.append({
                "method": method,
                "seed": seed,
                "final_test_accuracy": (
                    95 + offset + shared
                    + rng.normal(0, 0.3)
                ),
            })

    results = pd.DataFrame(records)
    summary, comparisons = export_statistics(results)

    print("\nDescriptive statistics:")
    print(summary.to_string(index=False))

    print("\nPaired significance tests:")
    print(comparisons.to_string(index=False))
