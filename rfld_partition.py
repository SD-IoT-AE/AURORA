
"""
AURORA - Reproducible Non-IID Dataset Partitioning
==================================================

Module:
    Risk-Aware Federated Learning Decision (RFLD)

Purpose:
    Distribute labeled training samples among UAV
    clients using class-wise Dirichlet partitioning.

Supported functionality:
    - Dirichlet non-IID label partitioning
    - IID partitioning for comparison
    - Reproducible seeded allocation
    - Minimum samples per client
    - Class-distribution statistics
    - Partition integrity validation
    - Client selection for FL rounds
    - Saving and loading partition indices

Default AURORA configuration:
    num_clients = 50
    dirichlet_alpha = 0.5
    min_samples_per_client = 10
    clients_per_round = 20
    seed = 42

Important:
    Partition only the training dataset.

    Validation and test datasets must remain
    separated from the federated training data.

    This module does not download or preprocess
    external benchmark datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)

import json

import numpy as np


@dataclass(frozen=True)
class PartitionConfig:
    """Configuration of the FL data partitioner."""

    num_clients: int = 50

    dirichlet_alpha: float = 0.5

    min_samples_per_client: int = 10

    clients_per_round: int = 20

    seed: int = 42

    max_attempts: int = 1000

    def __post_init__(self) -> None:

        if self.num_clients < 1:
            raise ValueError(
                "num_clients must be >= 1."
            )

        if (
            not np.isfinite(self.dirichlet_alpha)
            or self.dirichlet_alpha <= 0
        ):
            raise ValueError(
                "dirichlet_alpha must be positive "
                "and finite."
            )

        if self.min_samples_per_client < 0:
            raise ValueError(
                "min_samples_per_client must be >= 0."
            )

        if not (
            1 <= self.clients_per_round
            <= self.num_clients
        ):
            raise ValueError(
                "clients_per_round must be between "
                "1 and num_clients."
            )

        if self.max_attempts < 1:
            raise ValueError(
                "max_attempts must be >= 1."
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be nonnegative."
            )


@dataclass
class PartitionResult:
    """Container for client data partitions."""

    client_indices: Dict[int, np.ndarray]

    num_clients: int

    num_samples: int

    seed: int

    method: str

    alpha: Optional[float] = None

    def get_client_indices(
        self,
        client_id: int,
    ) -> np.ndarray:
        """Return sample indices assigned to a client."""

        if client_id not in self.client_indices:
            raise KeyError(
                f"Unknown client ID: {client_id}"
            )

        return self.client_indices[
            client_id
        ].copy()

    def get_client_data(
        self,
        client_id: int,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Retrieve one client's feature and label arrays.

        The caller supplies the same training arrays
        used to construct the partition.
        """

        features = np.asarray(features)
        labels = np.asarray(labels)

        if len(features) != self.num_samples:
            raise ValueError(
                "Feature count does not match "
                "the partitioned dataset."
            )

        if len(labels) != self.num_samples:
            raise ValueError(
                "Label count does not match "
                "the partitioned dataset."
            )

        indices = self.get_client_indices(
            client_id
        )

        return (
            features[indices],
            labels[indices],
        )

    def client_sample_counts(
        self,
    ) -> Dict[int, int]:
        """Return training sample count per client."""

        return {
            client_id: len(indices)
            for client_id, indices
            in self.client_indices.items()
        }


class FederatedDataPartitioner:
    """
    Reproducible dataset partitioning engine.

    Supports:
        - Dirichlet label-distribution non-IID
        - IID random partitioning

    All allocations are made using positional
    indices into the supplied training dataset.
    """

    def __init__(
        self,
        config: Optional[PartitionConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else PartitionConfig()
        )

    @staticmethod
    def _validate_labels(
        labels: Sequence,
    ) -> np.ndarray:
        """Validate one-dimensional class labels."""

        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError(
                "Labels must be one-dimensional."
            )

        if len(labels) == 0:
            raise ValueError(
                "Labels cannot be empty."
            )

        if np.issubdtype(
            labels.dtype,
            np.number,
        ):
            if not np.all(np.isfinite(labels)):
                raise ValueError(
                    "Numeric labels must be finite."
                )

        return labels

    def _check_feasibility(
        self,
        num_samples: int,
    ) -> None:
        """Check whether minimum client sizes are feasible."""

        required = (
            self.config.num_clients
            * self.config.min_samples_per_client
        )

        if num_samples < required:
            raise ValueError(
                "Insufficient training samples "
                "to satisfy minimum client sizes. "
                f"Required: {required}; "
                f"available: {num_samples}."
            )

    def dirichlet_partition(
        self,
        labels: Sequence,
    ) -> PartitionResult:
        """
        Partition labels using a class-wise
        Dirichlet distribution.

        For each class:
            1. Shuffle its sample indices.
            2. Draw client proportions from
               Dirichlet(alpha, ..., alpha).
            3. Convert proportions into counts
               using a multinomial allocation.
            4. Assign disjoint class indices.
            5. Repeat until minimum client
               sample counts are satisfied.

        The algorithm never duplicates or
        discards training samples.
        """

        labels = self._validate_labels(
            labels
        )

        n = len(labels)

        self._check_feasibility(n)

        num_clients = (
            self.config.num_clients
        )

        alpha = (
            self.config.dirichlet_alpha
        )

        classes = np.unique(labels)

        rng = np.random.default_rng(
            self.config.seed
        )

        for attempt in range(
            self.config.max_attempts
        ):

            partitions: Dict[int, List[int]] = {
                client_id: []
                for client_id in range(num_clients)
            }

            for class_label in classes:

                class_indices = np.flatnonzero(
                    labels == class_label
                )

                class_indices = (
                    rng.permutation(class_indices)
                )

                proportions = rng.dirichlet(
                    np.full(
                        num_clients,
                        alpha,
                        dtype=np.float64,
                    )
                )

                counts = rng.multinomial(
                    len(class_indices),
                    proportions,
                )

                boundaries = np.cumsum(
                    counts
                )[:-1]

                splits = np.split(
                    class_indices,
                    boundaries,
                )

                for client_id, split in enumerate(
                    splits
                ):
                    partitions[
                        client_id
                    ].extend(
                        split.tolist()
                    )

            sizes = [
                len(partitions[client_id])
                for client_id in range(
                    num_clients
                )
            ]

            if min(sizes) >= (
                self.config.min_samples_per_client
            ):

                client_indices = {}

                for client_id, indices in (
                    partitions.items()
                ):

                    shuffled = rng.permutation(
                        np.asarray(
                            indices,
                            dtype=np.int64,
                        )
                    )

                    client_indices[
                        client_id
                    ] = shuffled

                result = PartitionResult(
                    client_indices=client_indices,
                    num_clients=num_clients,
                    num_samples=n,
                    seed=self.config.seed,
                    method="dirichlet_non_iid",
                    alpha=alpha,
                )

                validate_partition(
                    result
                )

                return result

        raise RuntimeError(
            "Unable to generate a Dirichlet "
            "partition satisfying minimum "
            "client sizes after "
            f"{self.config.max_attempts} attempts. "
            "Consider reducing the minimum "
            "sample count, increasing the "
            "dataset size, or increasing alpha."
        )

    def iid_partition(
        self,
        labels: Sequence,
    ) -> PartitionResult:
        """
        Partition training samples randomly.

        This produces an IID-style random split.
        Exact class proportions are not guaranteed.
        """

        labels = self._validate_labels(
            labels
        )

        n = len(labels)

        self._check_feasibility(n)

        rng = np.random.default_rng(
            self.config.seed
        )

        shuffled_indices = rng.permutation(
            n
        )

        splits = np.array_split(
            shuffled_indices,
            self.config.num_clients,
        )

        client_indices = {
            client_id: np.asarray(
                indices,
                dtype=np.int64,
            )
            for client_id, indices in enumerate(
                splits
            )
        }

        result = PartitionResult(
            client_indices=client_indices,
            num_clients=self.config.num_clients,
            num_samples=n,
            seed=self.config.seed,
            method="iid",
            alpha=None,
        )

        validate_partition(
            result
        )

        return result

    def select_clients(
        self,
        round_index: int,
        eligible_clients: Optional[
            Sequence[int]
        ] = None,
    ) -> List[int]:
        """
        Select clients for one FL round.

        Selection is reproducible for each round.

        Different round indices produce independent
        deterministic selections.

        Eligible clients may be supplied by the
        TOSA trust manager.

        If fewer than clients_per_round clients
        are eligible, all eligible clients are
        selected.

        An empty eligible set raises an error.
        """

        if round_index < 0:
            raise ValueError(
                "round_index must be nonnegative."
            )

        if eligible_clients is None:

            candidates = list(
                range(
                    self.config.num_clients
                )
            )

        else:

            candidates = sorted(
                set(
                    int(client_id)
                    for client_id
                    in eligible_clients
                )
            )

        if not candidates:
            raise ValueError(
                "No eligible clients available "
                "for the FL round."
            )

        if any(
            client_id < 0
            or client_id >= self.config.num_clients
            for client_id in candidates
        ):
            raise ValueError(
                "Eligible client IDs must be "
                "within the configured range."
            )

        seed_sequence = np.random.SeedSequence(
            [
                self.config.seed,
                round_index,
            ]
        )

        rng = np.random.default_rng(
            seed_sequence
        )

        selection_count = min(
            self.config.clients_per_round,
            len(candidates),
        )

        selected = rng.choice(
            np.asarray(
                candidates,
                dtype=np.int64,
            ),
            size=selection_count,
            replace=False,
        )

        return sorted(
            int(client_id)
            for client_id in selected
        )


def validate_partition(
    result: PartitionResult,
) -> bool:
    """
    Validate partition integrity.

    Checks:
        - Correct number of clients
        - All client IDs present
        - Integer sample indices
        - Indices within dataset bounds
        - No duplicated sample assignments
        - No missing sample assignments
        - Total assigned count matches dataset size

    Returns True if all checks pass.
    """

    expected_clients = set(
        range(result.num_clients)
    )

    actual_clients = set(
        result.client_indices.keys()
    )

    if actual_clients != expected_clients:
        raise ValueError(
            "Partition client IDs do not match "
            "the configured client range."
        )

    arrays = []

    for client_id in sorted(
        result.client_indices
    ):

        indices = np.asarray(
            result.client_indices[
                client_id
            ]
        )

        if indices.ndim != 1:
            raise ValueError(
                f"Client {client_id} indices "
                "must be one-dimensional."
            )

        if not np.issubdtype(
            indices.dtype,
            np.integer,
        ):
            raise ValueError(
                f"Client {client_id} indices "
                "must be integers."
            )

        if np.any(indices < 0):
            raise ValueError(
                "Negative sample index detected."
            )

        if np.any(
            indices >= result.num_samples
        ):
            raise ValueError(
                "Out-of-range sample index detected."
            )

        arrays.append(
            indices.astype(
                np.int64,
                copy=False,
            )
        )

    combined = np.concatenate(
        arrays
    )

    if len(combined) != result.num_samples:
        raise ValueError(
            "Total assigned sample count "
            "does not match dataset size."
        )

    unique_indices = np.unique(
        combined
    )

    if len(unique_indices) != (
        result.num_samples
    ):
        raise ValueError(
            "Duplicate or missing sample "
            "assignments detected."
        )

    if not np.array_equal(
        unique_indices,
        np.arange(
            result.num_samples,
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            "Partition does not contain exactly "
            "the expected sample indices."
        )

    return True


def calculate_partition_statistics(
    result: PartitionResult,
    labels: Sequence,
) -> Dict:
    """
    Calculate client-level partition statistics.

    Reports:
        - Number of samples per client
        - Class counts per client
        - Class proportions per client
        - Minimum and maximum client sizes
        - Mean and standard deviation of sizes
        - Global class distribution
    """

    labels = np.asarray(
        labels
    )

    if len(labels) != result.num_samples:
        raise ValueError(
            "Label count does not match "
            "partition sample count."
        )

    validate_partition(
        result
    )

    classes = np.unique(
        labels
    )

    client_statistics = {}

    sample_counts = []

    for client_id, indices in sorted(
        result.client_indices.items()
    ):

        client_labels = labels[
            indices
        ]

        count = len(indices)

        sample_counts.append(
            count
        )

        class_counts = {
            str(class_label): int(
                np.sum(
                    client_labels == class_label
                )
            )
            for class_label in classes
        }

        class_proportions = {
            class_name: (
                class_count / count
                if count > 0
                else 0.0
            )
            for class_name, class_count
            in class_counts.items()
        }

        client_statistics[
            str(client_id)
        ] = {
            "num_samples": count,
            "class_counts": class_counts,
            "class_proportions": (
                class_proportions
            ),
        }

    global_counts = {
        str(class_label): int(
            np.sum(
                labels == class_label
            )
        )
        for class_label in classes
    }

    return {
        "method": result.method,
        "num_clients": result.num_clients,
        "num_samples": result.num_samples,
        "seed": result.seed,
        "dirichlet_alpha": result.alpha,
        "minimum_client_samples": int(
            np.min(sample_counts)
        ),
        "maximum_client_samples": int(
            np.max(sample_counts)
        ),
        "mean_client_samples": float(
            np.mean(sample_counts)
        ),
        "std_client_samples": float(
            np.std(sample_counts)
        ),
        "global_class_counts": global_counts,
        "clients": client_statistics,
    }


def save_partition(
    result: PartitionResult,
    output_path: str | Path,
) -> None:
    """
    Save partition indices as JSON.

    The output contains index assignments
    and metadata, not the underlying dataset.
    """

    validate_partition(
        result
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "method": result.method,
        "num_clients": result.num_clients,
        "num_samples": result.num_samples,
        "seed": result.seed,
        "alpha": result.alpha,
        "client_indices": {
            str(client_id): (
                indices.tolist()
            )
            for client_id, indices
            in result.client_indices.items()
        },
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )


def load_partition(
    input_path: str | Path,
) -> PartitionResult:
    """Load and validate a saved JSON partition."""

    input_path = Path(
        input_path
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )

    client_indices = {
        int(client_id): np.asarray(
            indices,
            dtype=np.int64,
        )
        for client_id, indices
        in payload["client_indices"].items()
    }

    result = PartitionResult(
        client_indices=client_indices,
        num_clients=int(
            payload["num_clients"]
        ),
        num_samples=int(
            payload["num_samples"]
        ),
        seed=int(
            payload["seed"]
        ),
        method=str(
            payload["method"]
        ),
        alpha=payload.get(
            "alpha"
        ),
    )

    validate_partition(
        result
    )

    return result


if __name__ == "__main__":

    print(
        "AURORA Non-IID Dataset Partitioning"
    )

    print("=" * 65)

    rng = np.random.default_rng(
        42
    )

    # Synthetic labels for functionality testing.
    # These are not benchmark dataset records.

    labels = rng.choice(
        [0, 1, 2, 3],
        size=10000,
        p=[0.50, 0.25, 0.15, 0.10],
    )

    config = PartitionConfig(
        num_clients=50,
        dirichlet_alpha=0.5,
        min_samples_per_client=10,
        clients_per_round=20,
        seed=42,
    )

    partitioner = FederatedDataPartitioner(
        config
    )

    result = partitioner.dirichlet_partition(
        labels
    )

    validate_partition(
        result
    )

    statistics = (
        calculate_partition_statistics(
            result,
            labels,
        )
    )

    print(
        "Partition method:",
        statistics["method"],
    )

    print(
        "Total samples:",
        statistics["num_samples"],
    )

    print(
        "Total UAV clients:",
        statistics["num_clients"],
    )

    print(
        "Dirichlet alpha:",
        statistics["dirichlet_alpha"],
    )

    print(
        "Minimum samples/client:",
        statistics["minimum_client_samples"],
    )

    print(
        "Maximum samples/client:",
        statistics["maximum_client_samples"],
    )

    print(
        "Mean samples/client:",
        round(
            statistics["mean_client_samples"],
            2,
        ),
    )

    selected_clients = (
        partitioner.select_clients(
            round_index=1
        )
    )

    print(
        "Selected clients for round 1:",
        selected_clients,
    )

    save_partition(
        result,
        "outputs/partitions/"
        "aurora_dirichlet_partition.json",
    )

    restored = load_partition(
        "outputs/partitions/"
        "aurora_dirichlet_partition.json"
    )

    assert validate_partition(
        restored
    )

    print(
        "Partition saved and reloaded successfully."
    )
