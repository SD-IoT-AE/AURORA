
"""AURORA: Structured logging for experiments and system modules."""

import csv
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


def utc_timestamp():
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _serialize(value):
    """Convert common experimental values to JSON-safe objects."""
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"Unsupported value: {type(value).__name__}")


class AURORALogger:
    """Unified console, file, CSV, and JSONL experiment logger."""

    def __init__(self, output_dir="outputs/logs",
                 experiment="aurora", level=logging.INFO):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.lock = Lock()

        self.logger = logging.getLogger(
            f"aurora.{experiment}.{self.output_dir.resolve()}"
        )
        self.logger.setLevel(level)
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )

            console = logging.StreamHandler()
            console.setFormatter(formatter)

            file_handler = logging.FileHandler(
                self.output_dir / "experiment.log",
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)

            self.logger.addHandler(console)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def metric(self, module, name, value, step=None, **extra):
        """Append a metric record to CSV and JSONL."""
        record = {
            "timestamp": utc_timestamp(),
            "experiment": self.experiment,
            "module": module,
            "step": step,
            "metric": name,
            "value": value,
            "extra": extra,
        }

        self._append_jsonl("metrics.jsonl", record)

        # Fixed CSV schema prevents column inconsistencies.
        csv_record = {
            **{k: record[k] for k in (
                "timestamp", "experiment", "module",
                "step", "metric", "value"
            )},
            "extra": json.dumps(extra, default=_serialize),
        }
        self._append_csv("metrics.csv", csv_record)

    def event(self, module, event_type, **details):
        """Record structured TOSA, RFLD, or ACRO events."""
        record = {
            "timestamp": utc_timestamp(),
            "experiment": self.experiment,
            "module": module,
            "event_type": event_type,
            "details": details,
        }
        self._append_jsonl("events.jsonl", record)

    def _append_jsonl(self, filename, record):
        with self.lock:
            with (self.output_dir / filename).open(
                "a", encoding="utf-8"
            ) as file:
                file.write(
                    json.dumps(record, default=_serialize)
                    + "\n"
                )

    def _append_csv(self, filename, record):
        path = self.output_dir / filename

        with self.lock:
            write_header = (
                not path.exists() or path.stat().st_size == 0
            )
            with path.open(
                "a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=list(record)
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(record)

    def log_tosa(self, uav_id, probability, trust, risk):
        """Log a UAV threat assessment and trust update."""
        self.event(
            "TOSA", "threat_assessment",
            uav_id=uav_id,
            malicious_probability=float(probability),
            trust=float(trust),
            risk=float(risk),
        )

    def log_rfld(self, round_index, accuracy, loss,
                 participating_clients, aggregation_ms):
        """Log federated-learning round statistics."""
        self.metric("RFLD", "accuracy", accuracy, round_index)
        self.metric("RFLD", "loss", loss, round_index)
        self.metric(
            "RFLD", "aggregation_ms",
            aggregation_ms, round_index,
        )
        self.event(
            "RFLD", "round_completed",
            round=round_index,
            participating_clients=participating_clients,
        )

    def log_acro(self, uav_id, action, risk,
                 enforced=True, latency_ms=None):
        """Log an adaptive mitigation decision."""
        action_name = getattr(action, "value", str(action))
        self.event(
            "ACRO", "mitigation_decision",
            uav_id=uav_id,
            action=action_name,
            risk=float(risk),
            enforced=bool(enforced),
            latency_ms=latency_ms,
        )


if __name__ == "__main__":
    logger = AURORALogger(
        output_dir="outputs/logs/demo",
        experiment="S10_seed42",
    )

    logger.info("AURORA experiment started")

    logger.log_tosa(
        uav_id=7, probability=0.91,
        trust=0.35, risk=0.65,
    )

    logger.log_rfld(
        round_index=1,
        accuracy=94.2,
        loss=0.18,
        participating_clients=20,
        aggregation_ms=12.5,
    )

    logger.log_acro(
        uav_id=7, action="RATE_LIMIT",
        risk=0.65, enforced=True,
        latency_ms=145.0,
    )

    logger.info("AURORA experiment completed")
