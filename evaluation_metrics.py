
"""AURORA: Detection, mitigation, trust, and FL evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)


def classification_metrics(y_true, y_pred):
    """Binary detection metrics; 0=benign, 1=malicious."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Invalid prediction arrays")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Ground truth must be binary")
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Predictions must be binary")

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_true, y_pred, zero_division=0
        )),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }


def poisoning_resistance_rate(malicious_updates, blocked_updates):
    """PRR: fraction of malicious updates rejected before aggregation."""
    malicious = set(malicious_updates)
    blocked = set(blocked_updates)

    if not blocked.issubset(malicious):
        raise ValueError("Blocked IDs must be malicious IDs")

    return (
        len(blocked) / len(malicious)
        if malicious else None
    )


def mitigation_latency(detection_times, enforcement_times):
    """Return mitigation latency statistics in milliseconds."""
    detection = np.asarray(detection_times, dtype=float)
    enforcement = np.asarray(enforcement_times, dtype=float)

    if detection.shape != enforcement.shape or detection.size == 0:
        raise ValueError("Invalid timestamp arrays")

    latency = (enforcement - detection) * 1000

    if not np.isfinite(latency).all() or np.any(latency < 0):
        raise ValueError("Invalid mitigation timestamps")

    return {
        "mean_ms": float(np.mean(latency)),
        "median_ms": float(np.median(latency)),
        "p95_ms": float(np.percentile(latency, 95)),
        "max_ms": float(np.max(latency)),
    }


def trust_stability(trust_history):
    """Mean absolute consecutive trust change; lower is more stable."""
    values = np.asarray(trust_history, dtype=float)

    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid trust history")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Trust values must be within [0, 1]")

    return (
        float(np.mean(np.abs(np.diff(values))))
        if len(values) >= 2 else None
    )


def communication_overhead(control_bytes, total_bytes):
    """Control-plane overhead as a fraction of transmitted bytes."""
    if not np.isfinite([control_bytes, total_bytes]).all():
        raise ValueError("Byte counts must be finite")
    if control_bytes < 0 or total_bytes <= 0:
        raise ValueError("Invalid byte counts")
    if control_bytes > total_bytes:
        raise ValueError("Control bytes exceed total bytes")

    return float(control_bytes / total_bytes)


def fl_convergence(accuracy_history, target_accuracy=0.95):
    """Find first round meeting target accuracy; accuracy in [0,1]."""
    values = np.asarray(accuracy_history, dtype=float)

    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid accuracy history")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Accuracy must be in [0, 1]")
    if not 0 <= target_accuracy <= 1:
        raise ValueError("Invalid target accuracy")

    reached = np.flatnonzero(values >= target_accuracy)

    return {
        "rounds_to_target": (
            int(reached[0] + 1) if len(reached) else None
        ),
        "final_accuracy": (
            float(values[-1]) if len(values) else None
        ),
        "best_accuracy": (
            float(np.max(values)) if len(values) else None
        ),
    }


def evaluate_aurora(
    y_true, y_pred,
    malicious_updates=None, blocked_updates=None,
    detection_times=None, enforcement_times=None,
    trust_history=None,
    control_bytes=None, total_bytes=None,
    accuracy_history=None,
):
    """Combine available AURORA performance metrics."""
    results = {
        "detection": classification_metrics(y_true, y_pred)
    }

    if malicious_updates is not None and blocked_updates is not None:
        results["prr"] = poisoning_resistance_rate(
            malicious_updates, blocked_updates
        )

    if detection_times is not None and enforcement_times is not None:
        results["mitigation"] = mitigation_latency(
            detection_times, enforcement_times
        )

    if trust_history is not None:
        results["trust_stability"] = trust_stability(trust_history)

    if control_bytes is not None and total_bytes is not None:
        results["communication_overhead"] = communication_overhead(
            control_bytes, total_bytes
        )

    if accuracy_history is not None:
        results["fl_convergence"] = fl_convergence(accuracy_history)

    return results


if __name__ == "__main__":
    results = evaluate_aurora(
        y_true=[0, 0, 1, 1, 1],
        y_pred=[0, 1, 1, 1, 0],
        malicious_updates=[2, 5, 8, 9],
        blocked_updates=[2, 5, 8],
        detection_times=[1.0, 2.0, 3.0],
        enforcement_times=[1.12, 2.15, 3.10],
        trust_history=[1.0, 0.85, 0.80, 0.75],
        control_bytes=15000,
        total_bytes=500000,
        accuracy_history=[0.75, 0.82, 0.89, 0.94, 0.96],
    )

    import json
    print(json.dumps(results, indent=2))
