
"""AURORA: Experiment reproducibility and metadata management."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys


def set_seed(seed=42, deterministic=True):
    """Seed Python, NumPy, and PyTorch when available."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(
                True, warn_only=True
            )
    except ImportError:
        pass

    return seed


def environment_info():
    """Collect software, platform, and hardware metadata."""
    packages = (
        "numpy", "pandas", "scipy", "scikit-learn",
        "torch", "xgboost", "lightgbm", "mininet-wifi",
    )

    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
    }

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        )
    except ImportError:
        info["cuda_available"] = False

    return info


def git_commit():
    """Return the current Git commit, when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def config_dict(config):
    """Convert a dataclass or dictionary to a JSON-safe mapping."""
    if is_dataclass(config):
        config = asdict(config)

    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dataclass or dict")

    return json.loads(json.dumps(config, sort_keys=True))


def config_hash(config):
    """Generate a stable configuration fingerprint."""
    payload = json.dumps(
        config_dict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:12]


def create_run_dir(config, root="outputs/experiments",
                   name="aurora", seed=42):
    """Create a deterministic, configuration-specific run directory."""
    fingerprint = config_hash(config)
    run_id = f"{name}_seed{seed}_{fingerprint}"

    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=True)

    return path


def save_metadata(run_dir, config, seed=42, extra=None):
    """Save configuration and execution metadata as JSON."""
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)

    configuration = config_dict(config)
    metadata = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "seed": seed,
        "config_hash": config_hash(configuration),
        "git_commit": git_commit(),
        "environment": environment_info(),
        "extra": extra or {},
    }

    for filename, content in (
        ("config.json", configuration),
        ("metadata.json", metadata),
    ):
        with (path / filename).open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(content, file, indent=2)

    return metadata


def initialize_experiment(config, name="aurora",
                          root="outputs/experiments"):
    """Initialize reproducibility and create experiment records."""
    seed = config.seed if hasattr(config, "seed") else config["seed"]

    if hasattr(config, "validate"):
        config.validate()

    set_seed(seed)

    run_dir = create_run_dir(
        config, root=root, name=name, seed=seed
    )

    metadata = save_metadata(
        run_dir, config, seed=seed
    )

    return run_dir, metadata


if __name__ == "__main__":
    from aurora.config import AURORAConfig

    config = AURORAConfig()
    run_dir, metadata = initialize_experiment(config)

    print("Run directory:", run_dir)
    print("Configuration hash:", metadata["config_hash"])
    print("Git commit:", metadata["git_commit"])
    print("Seed:", metadata["seed"])
