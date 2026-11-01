
"""AURORA: Experiment integrity and reproducibility validation."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import re

import numpy as np

from aurora.config import AURORAConfig
from aurora.experiments.scenarios import get_scenario


@dataclass
class ValidationIssue:
    level: str
    component: str
    message: str


class AURORAValidator:
    """Collect validation errors and warnings without mutating inputs."""

    def __init__(self):
        self.issues = []

    def _issue(self, level, component, message):
        self.issues.append(
            ValidationIssue(level, component, str(message))
        )

    def check(self, condition, component, message, level="error"):
        if not condition:
            self._issue(level, component, message)

    def validate_config(self, config):
        """Validate scenario, trust, and federated settings."""
        try:
            config.validate()
        except (ValueError, TypeError) as exc:
            self._issue("error", "config", exc)

        c = config
        try:
            scenario = get_scenario(c.scenario.scenario_id)
            self.check(
                c.scenario.num_uavs == c.rfld.num_clients,
                "mapping", "UAV and FL client counts differ",
                "warning",
            )
            self.check(
                c.scenario.duration_s == scenario.duration_s,
                "scenario", "Configured duration differs from catalog",
                "warning",
            )
            self.check(
                c.scenario.num_uavs > 0,
                "scenario", "Number of UAVs must be positive",
            )
            self.check(
                0 < c.rfld.clients_per_round <= c.rfld.num_clients,
                "rfld", "Invalid clients-per-round setting",
            )
            self.check(
                c.rfld.rounds > 0 and c.rfld.local_epochs > 0,
                "rfld", "Rounds and local epochs must be positive",
            )
            self.check(
                c.rfld.dirichlet_alpha > 0,
                "rfld", "Dirichlet alpha must be positive",
            )
            self.check(
                0 <= c.rfld.malicious_fraction <= 1,
                "rfld", "Malicious fraction must be within [0,1]",
            )
            self.check(
                0 <= c.tosa.fl_trust_threshold <= 1,
                "tosa", "FL trust threshold must be within [0,1]",
            )
            self.check(
                c.rfld.aggregation_method in {
                    "aurora", "fedavg", "krum",
                    "trimmed_mean", "median",
                },
                "rfld", "Unsupported aggregation method",
            )
            if c.rfld.aggregation_method == "krum":
                f = int(
                    c.rfld.clients_per_round
                    * c.rfld.malicious_fraction
                )
                self.check(
                    c.rfld.clients_per_round >= 2 * f + 3,
                    "rfld", "Krum participation requirement violated",
                )
        except (AttributeError, KeyError, ValueError) as exc:
            self._issue("error", "config", exc)

    def validate_dataset(self, data):
        """Check array shapes, labels, finite values, and overlap."""
        required = ("X_train", "y_train", "X_test", "y_test")
        if not all(key in data for key in required):
            self._issue("error", "dataset", "Missing required arrays")
            return

        try:
            x, y, xt, yt = (
                np.asarray(data[key]) for key in required
            )
            self.check(
                x.ndim == xt.ndim == 2,
                "dataset", "Features must be two-dimensional",
            )
            self.check(
                y.ndim == yt.ndim == 1,
                "dataset", "Labels must be one-dimensional",
            )
            if x.ndim != 2 or xt.ndim != 2:
                return
            self.check(
                x.shape[1] == xt.shape[1],
                "dataset", "Train/test feature dimensions differ",
            )
            self.check(
                len(x) == len(y) and len(xt) == len(yt),
                "dataset", "Feature/label lengths differ",
            )
            self.check(
                len(x) > 0 and len(xt) > 0,
                "dataset", "Empty training or test partition",
            )
            self.check(
                np.isfinite(x.astype(float)).all()
                and np.isfinite(xt.astype(float)).all(),
                "dataset", "Non-finite or nonnumeric features",
            )
            self.check(
                set(np.unique(np.r_[y, yt])).issubset({0, 1}),
                "dataset", "Binary labels 0/1 required",
            )
            self.check(
                len(np.unique(y)) == 2,
                "dataset", "Training data lacks both classes",
                "warning",
            )
        except (TypeError, ValueError) as exc:
            self._issue("error", "dataset", exc)

    def validate_clients(self, eligible, num_clients, trust=None):
        """Check UAV identifiers, uniqueness, and trust coverage."""
        ids = []
        for uav in eligible:
            match = re.fullmatch(r"uav(\d+)", str(uav))
            if match is None:
                self._issue(
                    "error", "mapping", f"Invalid UAV ID: {uav}"
                )
                continue
            client = int(match.group(1))
            ids.append(client)
            self.check(
                0 <= client < num_clients,
                "mapping", f"Out-of-range client: {uav}",
            )

        self.check(
            len(ids) == len(set(ids)),
            "mapping", "Duplicate eligible client IDs",
        )

        if trust is not None:
            for client in ids:
                key = f"uav{client}"
                self.check(
                    key in trust,
                    "trust", f"Missing trust score: {key}",
                )
                if key in trust:
                    try:
                        value = float(trust[key])
                        valid = np.isfinite(value) and 0 <= value <= 1
                    except (TypeError, ValueError):
                        valid = False
                    self.check(
                        valid, "trust", f"Invalid trust: {key}"
                    )

    def validate_outputs(self, run_dir, require_fl=False):
        """Check expected experimental artifacts."""
        root = Path(run_dir)
        self.check(root.is_dir(), "outputs", "Run directory missing")
        if not root.is_dir():
            return

        required = ["config.json", "metadata.json"]
        if require_fl:
            required += [
                "federated/fl_results.json",
                "federated/round_metrics.csv",
            ]

        for relative in required:
            path = root / relative
            self.check(
                path.is_file(),
                "outputs", f"Missing artifact: {relative}",
            )

        for filename in ("config.json", "metadata.json"):
            path = root / filename
            if path.is_file():
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self._issue(
                        "error", "outputs",
                        f"Invalid JSON: {filename}"
                    )

    @staticmethod
    def fingerprint(data):
        """Hash array contents, shapes, and dtypes deterministically."""
        digest = hashlib.sha256()
        for key in sorted(data):
            arr = np.ascontiguousarray(np.asarray(data[key]))
            if arr.dtype.hasobject:
                raise ValueError("Object arrays cannot be fingerprinted")
            digest.update(key.encode())
            digest.update(str(arr.shape).encode())
            digest.update(str(arr.dtype).encode())
            digest.update(arr.tobytes())
        return digest.hexdigest()

    def report(self, output_dir=None):
        """Return and optionally save a machine-readable report."""
        errors = sum(i.level == "error" for i in self.issues)
        warnings = sum(i.level == "warning" for i in self.issues)
        result = {
            "passed": errors == 0,
            "errors": errors,
            "warnings": warnings,
            "issues": [asdict(i) for i in self.issues],
        }
        if output_dir is not None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "validation.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        return result


if __name__ == "__main__":
    config = AURORAConfig()
    validator = AURORAValidator()
    validator.validate_config(config)

    result = validator.report("outputs/validation")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
