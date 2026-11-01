
"""
AURORA - Federated Learning Aggregation
=======================================

Module:
    Risk-Aware Federated Learning Decision (RFLD)

Implemented aggregation strategies:
    1. AURORA trust-weighted aggregation
    2. Federated Averaging (FedAvg)
    3. Krum
    4. Coordinate-wise Trimmed Mean
    5. Coordinate-wise Median

Features:
    - Support for multi-layer model parameters
    - Trust-based client eligibility
    - Configurable trust threshold
    - Dataset-size weighting for FedAvg
    - Byzantine-tolerant aggregation baselines
    - Input validation
    - Aggregation execution-time measurement
    - Optional aggregation metadata

AURORA defaults:
    trust_threshold = 0.50
    krum_byzantine_clients = 2
    trimmed_mean_fraction = 0.20

The implementation operates on NumPy arrays.

Client model updates must share the same parameter
structure, shape, and ordering.

All submitted parameters must be finite.

No experimental results are hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from time import perf_counter
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np


ModelParameters = Sequence[np.ndarray]


class AggregationMethod(str, Enum):
    """Supported aggregation strategies."""

    AURORA = "aurora"
    FEDAVG = "fedavg"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"


@dataclass(frozen=True)
class AggregationConfig:
    """Configuration for federated aggregation."""

    trust_threshold: float = 0.50

    krum_byzantine_clients: int = 2

    trimmed_mean_fraction: float = 0.20

    minimum_clients: int = 1

    def __post_init__(self) -> None:

        if not isfinite(self.trust_threshold):
            raise ValueError(
                "trust_threshold must be finite."
            )

        if not 0.0 <= self.trust_threshold <= 1.0:
            raise ValueError(
                "trust_threshold must be within [0, 1]."
            )

        if self.krum_byzantine_clients < 0:
            raise ValueError(
                "krum_byzantine_clients must be >= 0."
            )

        if not isfinite(self.trimmed_mean_fraction):
            raise ValueError(
                "trimmed_mean_fraction must be finite."
            )

        if not 0.0 <= self.trimmed_mean_fraction < 0.50:
            raise ValueError(
                "trimmed_mean_fraction must be in [0, 0.5)."
            )

        if self.minimum_clients < 1:
            raise ValueError(
                "minimum_clients must be >= 1."
            )


@dataclass
class ClientUpdate:
    """
    Model parameters submitted by one FL client.

    Parameters
    ----------
    client_id:
        Unique identifier of the UAV client.

    parameters:
        Sequence of NumPy arrays representing
        the locally trained model parameters.

    num_samples:
        Number of training examples held by
        the client.

    trust:
        Current normalized trust value in [0, 1].
    """

    client_id: int

    parameters: ModelParameters

    num_samples: int

    trust: float = 1.0

    def __post_init__(self) -> None:

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples must be positive."
            )

        if not isfinite(self.trust):
            raise ValueError(
                "Client trust must be finite."
            )

        if not 0.0 <= self.trust <= 1.0:
            raise ValueError(
                "Client trust must be within [0, 1]."
            )

        if not self.parameters:
            raise ValueError(
                "Client parameters cannot be empty."
            )

        converted = []

        for parameter in self.parameters:

            array = np.asarray(
                parameter,
                dtype=np.float64,
            )

            if array.size == 0:
                raise ValueError(
                    "Model parameter arrays cannot be empty."
                )

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    "Model parameters must be finite."
                )

            converted.append(array.copy())

        self.parameters = tuple(converted)


@dataclass
class AggregationResult:
    """Output of one federated aggregation operation."""

    method: AggregationMethod

    parameters: List[np.ndarray]

    participating_clients: List[int]

    excluded_clients: List[int]

    client_weights: Dict[int, float] = field(
        default_factory=dict
    )

    aggregation_time_ms: float = 0.0

    metadata: Dict[str, object] = field(
        default_factory=dict
    )


class FederatedAggregator:
    """
    Federated aggregation engine.

    Each aggregation method accepts a collection
    of client updates and returns global model
    parameters with diagnostic metadata.

    For fair baseline comparisons, the caller
    should supply the same local client updates
    and poisoning conditions to each method.

    AURORA applies trust-based filtering.

    Baseline methods operate on the complete
    supplied client-update set by default.
    """

    def __init__(
        self,
        config: Optional[AggregationConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else AggregationConfig()
        )

    @staticmethod
    def _validate_updates(
        updates: Sequence[ClientUpdate],
    ) -> List[ClientUpdate]:
        """Validate clients and parameter structures."""

        updates = list(updates)

        if not updates:
            raise ValueError(
                "At least one client update is required."
            )

        client_ids = [
            update.client_id
            for update in updates
        ]

        if len(client_ids) != len(set(client_ids)):
            raise ValueError(
                "Duplicate client identifiers detected."
            )

        reference_shapes = [
            parameter.shape
            for parameter in updates[0].parameters
        ]

        for update in updates:

            if len(update.parameters) != len(
                reference_shapes
            ):
                raise ValueError(
                    "All clients must have the same "
                    "number of parameter arrays."
                )

            for parameter, expected_shape in zip(
                update.parameters,
                reference_shapes,
            ):

                if parameter.shape != expected_shape:
                    raise ValueError(
                        "Inconsistent model parameter "
                        "shapes across clients."
                    )

        return updates

    @staticmethod
    def _weighted_average(
        updates: Sequence[ClientUpdate],
        weights: Sequence[float],
    ) -> List[np.ndarray]:
        """Compute weighted average of model parameters."""

        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        if weights.ndim != 1:
            raise ValueError(
                "Weights must be one-dimensional."
            )

        if len(weights) != len(updates):
            raise ValueError(
                "Number of weights must equal "
                "number of client updates."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "Weights must be finite."
            )

        if np.any(weights < 0):
            raise ValueError(
                "Weights cannot be negative."
            )

        total = float(np.sum(weights))

        if total <= 0:
            raise ValueError(
                "Sum of aggregation weights must "
                "be positive."
            )

        normalized = weights / total

        aggregated = []

        num_layers = len(updates[0].parameters)

        for layer_index in range(num_layers):

            layer_stack = np.stack(
                [
                    update.parameters[layer_index]
                    for update in updates
                ],
                axis=0,
            )

            result = np.tensordot(
                normalized,
                layer_stack,
                axes=(0, 0),
            )

            aggregated.append(
                np.asarray(result).copy()
            )

        return aggregated

    @staticmethod
    def _flatten_parameters(
        parameters: ModelParameters,
    ) -> np.ndarray:
        """Flatten model parameters into one vector."""

        return np.concatenate(
            [
                np.asarray(parameter).ravel()
                for parameter in parameters
            ]
        )

    @staticmethod
    def _restore_parameters(
        vector: np.ndarray,
        reference: ModelParameters,
    ) -> List[np.ndarray]:
        """Restore flattened parameters to original shapes."""

        result = []

        offset = 0

        for parameter in reference:

            size = parameter.size

            restored = vector[
                offset:offset + size
            ].reshape(parameter.shape)

            result.append(
                restored.copy()
            )

            offset += size

        if offset != vector.size:
            raise ValueError(
                "Flattened vector length does not "
                "match reference parameters."
            )

        return result

    def aurora_trust_weighted(
        self,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        """
        AURORA trust-weighted aggregation.

        Step 1:
            Exclude clients whose trust is below
            the configured threshold.

        Step 2:
            Normalize trust values of eligible
            clients.

        Step 3:
            Compute the weighted average.

        Trust weighting:
            w_i = T_i / sum(T_j)

        Dataset size is intentionally not included
        in the aggregation weight, consistent with
        the trust-only formulation in the manuscript.
        """

        start_time = perf_counter()

        updates = self._validate_updates(updates)

        eligible = [
            update
            for update in updates
            if (
                update.trust
                >= self.config.trust_threshold
            )
        ]

        excluded = [
            update.client_id
            for update in updates
            if (
                update.trust
                < self.config.trust_threshold
            )
        ]

        if len(eligible) < self.config.minimum_clients:
            raise ValueError(
                "Insufficient eligible clients "
                "for trust-weighted aggregation."
            )

        trust_values = np.asarray(
            [
                update.trust
                for update in eligible
            ],
            dtype=np.float64,
        )

        total_trust = float(
            np.sum(trust_values)
        )

        if total_trust <= 0:
            raise ValueError(
                "Eligible clients have zero "
                "total trust."
            )

        normalized_weights = (
            trust_values / total_trust
        )

        parameters = self._weighted_average(
            eligible,
            normalized_weights,
        )

        client_weights = {
            update.client_id: float(weight)
            for update, weight in zip(
                eligible,
                normalized_weights,
            )
        }

        elapsed_ms = (
            perf_counter() - start_time
        ) * 1000.0

        return AggregationResult(
            method=AggregationMethod.AURORA,
            parameters=parameters,
            participating_clients=[
                update.client_id
                for update in eligible
            ],
            excluded_clients=excluded,
            client_weights=client_weights,
            aggregation_time_ms=elapsed_ms,
            metadata={
                "trust_threshold": (
                    self.config.trust_threshold
                ),
                "weighting": "normalized_trust",
                "eligible_client_count": len(eligible),
            },
        )

    def fedavg(
        self,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        """
        Standard Federated Averaging.

        Weight of client i:
            w_i = n_i / sum(n_j)

        where n_i is the client's training
        sample count.
        """

        start_time = perf_counter()

        updates = self._validate_updates(updates)

        sample_counts = np.asarray(
            [
                update.num_samples
                for update in updates
            ],
            dtype=np.float64,
        )

        normalized_weights = (
            sample_counts / np.sum(sample_counts)
        )

        parameters = self._weighted_average(
            updates,
            normalized_weights,
        )

        elapsed_ms = (
            perf_counter() - start_time
        ) * 1000.0

        return AggregationResult(
            method=AggregationMethod.FEDAVG,
            parameters=parameters,
            participating_clients=[
                update.client_id
                for update in updates
            ],
            excluded_clients=[],
            client_weights={
                update.client_id: float(weight)
                for update, weight in zip(
                    updates,
                    normalized_weights,
                )
            },
            aggregation_time_ms=elapsed_ms,
            metadata={
                "weighting": "dataset_size",
            },
        )

    def krum(
        self,
        updates: Sequence[ClientUpdate],
        byzantine_clients: Optional[int] = None,
    ) -> AggregationResult:
        """
        Krum robust aggregation.

        For each client:
            1. Compute squared Euclidean distances
               to all other client updates.
            2. Select the n-f-2 nearest neighbors.
            3. Sum their squared distances.
            4. Return the update with minimum score.

        Requirements:
            n >= 2f + 3

        This is single-Krum, not Multi-Krum.
        """

        start_time = perf_counter()

        updates = self._validate_updates(updates)

        f = (
            self.config.krum_byzantine_clients
            if byzantine_clients is None
            else int(byzantine_clients)
        )

        if f < 0:
            raise ValueError(
                "Byzantine client count must "
                "be nonnegative."
            )

        n = len(updates)

        if n < 2 * f + 3:
            raise ValueError(
                "Krum requires n >= 2f + 3. "
                f"Received n={n}, f={f}."
            )

        vectors = np.stack(
            [
                self._flatten_parameters(
                    update.parameters
                )
                for update in updates
            ],
            axis=0,
        )

        differences = (
            vectors[:, None, :]
            - vectors[None, :, :]
        )

        distances = np.sum(
            differences ** 2,
            axis=2,
        )

        neighbor_count = n - f - 2

        scores = []

        for i in range(n):

            other_distances = np.delete(
                distances[i],
                i,
            )

            nearest = np.sort(
                other_distances
            )[:neighbor_count]

            scores.append(
                float(np.sum(nearest))
            )

        selected_index = int(
            np.argmin(scores)
        )

        selected_update = updates[
            selected_index
        ]

        elapsed_ms = (
            perf_counter() - start_time
        ) * 1000.0

        return AggregationResult(
            method=AggregationMethod.KRUM,
            parameters=[
                parameter.copy()
                for parameter
                in selected_update.parameters
            ],
            participating_clients=[
                selected_update.client_id
            ],
            excluded_clients=[
                update.client_id
                for update in updates
                if (
                    update.client_id
                    != selected_update.client_id
                )
            ],
            client_weights={
                selected_update.client_id: 1.0
            },
            aggregation_time_ms=elapsed_ms,
            metadata={
                "byzantine_clients": f,
                "selected_client": (
                    selected_update.client_id
                ),
                "krum_scores": {
                    update.client_id: score
                    for update, score in zip(
                        updates,
                        scores,
                    )
                },
                "input_client_count": n,
                "neighbor_count": neighbor_count,
            },
        )

    def trimmed_mean(
        self,
        updates: Sequence[ClientUpdate],
        trim_fraction: Optional[float] = None,
    ) -> AggregationResult:
        """
        Coordinate-wise Trimmed Mean.

        For each model coordinate:
            1. Sort client values.
            2. Remove the lowest f values.
            3. Remove the highest f values.
            4. Average the remaining values.

        f = floor(trim_fraction * n)

        All supplied clients contribute candidate
        updates. The trimming is coordinate-wise.
        """

        start_time = perf_counter()

        updates = self._validate_updates(updates)

        fraction = (
            self.config.trimmed_mean_fraction
            if trim_fraction is None
            else float(trim_fraction)
        )

        if (
            not isfinite(fraction)
            or not 0.0 <= fraction < 0.50
        ):
            raise ValueError(
                "trim_fraction must be in [0, 0.5)."
            )

        n = len(updates)

        trim_count = int(
            np.floor(fraction * n)
        )

        if 2 * trim_count >= n:
            raise ValueError(
                "Trimming removes all client values."
            )

        aggregated = []

        num_layers = len(
            updates[0].parameters
        )

        for layer_index in range(num_layers):

            layer_stack = np.stack(
                [
                    update.parameters[layer_index]
                    for update in updates
                ],
                axis=0,
            )

            sorted_values = np.sort(
                layer_stack,
                axis=0,
            )

            if trim_count > 0:
                retained = sorted_values[
                    trim_count:n - trim_count
                ]
            else:
                retained = sorted_values

            aggregated.append(
                np.mean(
                    retained,
                    axis=0,
                )
            )

        elapsed_ms = (
            perf_counter() - start_time
        ) * 1000.0

        return AggregationResult(
            method=AggregationMethod.TRIMMED_MEAN,
            parameters=aggregated,
            participating_clients=[
                update.client_id
                for update in updates
            ],
            excluded_clients=[],
            aggregation_time_ms=elapsed_ms,
            metadata={
                "trim_fraction": fraction,
                "trim_count_per_tail": trim_count,
                "retained_values_per_coordinate": (
                    n - 2 * trim_count
                ),
                "coordinate_wise": True,
            },
        )

    def coordinate_median(
        self,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        """
        Coordinate-wise Median aggregation.

        For each model coordinate, return
        the median of client-submitted values.
        """

        start_time = perf_counter()

        updates = self._validate_updates(updates)

        aggregated = []

        num_layers = len(
            updates[0].parameters
        )

        for layer_index in range(num_layers):

            layer_stack = np.stack(
                [
                    update.parameters[layer_index]
                    for update in updates
                ],
                axis=0,
            )

            aggregated.append(
                np.median(
                    layer_stack,
                    axis=0,
                )
            )

        elapsed_ms = (
            perf_counter() - start_time
        ) * 1000.0

        return AggregationResult(
            method=AggregationMethod.MEDIAN,
            parameters=aggregated,
            participating_clients=[
                update.client_id
                for update in updates
            ],
            excluded_clients=[],
            aggregation_time_ms=elapsed_ms,
            metadata={
                "coordinate_wise": True,
            },
        )

    def aggregate(
        self,
        updates: Sequence[ClientUpdate],
        method: str = "aurora",
        **kwargs,
    ) -> AggregationResult:
        """
        Unified aggregation interface.

        Supported methods:
            aurora
            fedavg
            krum
            trimmed_mean
            median
        """

        method = AggregationMethod(method)

        if method == AggregationMethod.AURORA:
            return self.aurora_trust_weighted(
                updates
            )

        if method == AggregationMethod.FEDAVG:
            return self.fedavg(
                updates
            )

        if method == AggregationMethod.KRUM:
            return self.krum(
                updates,
                byzantine_clients=kwargs.get(
                    "byzantine_clients"
                ),
            )

        if method == AggregationMethod.TRIMMED_MEAN:
            return self.trimmed_mean(
                updates,
                trim_fraction=kwargs.get(
                    "trim_fraction"
                ),
            )

        if method == AggregationMethod.MEDIAN:
            return self.coordinate_median(
                updates
            )

        raise ValueError(
            f"Unsupported aggregation method: {method}"
        )


def compare_aggregators(
    updates: Sequence[ClientUpdate],
    config: Optional[AggregationConfig] = None,
) -> Dict[str, AggregationResult]:
    """
    Run all aggregation methods on the same
    client-submitted model parameters.

    This function does not calculate FL accuracy
    or poisoning resistance. Those require
    independent evaluation against test data
    and known attack labels.
    """

    aggregator = FederatedAggregator(config)

    results = {}

    for method in AggregationMethod:

        results[method.value] = (
            aggregator.aggregate(
                updates,
                method=method.value,
            )
        )

    return results


if __name__ == "__main__":

    print(
        "AURORA Federated Aggregation Demonstration"
    )

    print("=" * 65)

    rng = np.random.default_rng(42)

    reference_parameters = [
        np.array(
            [
                [0.5, 0.2],
                [0.1, 0.8],
            ],
            dtype=np.float64,
        ),
        np.array(
            [0.1, -0.1],
            dtype=np.float64,
        ),
    ]

    updates = []

    for client_id in range(10):

        local_parameters = [
            parameter + rng.normal(
                0.0,
                0.02,
                size=parameter.shape,
            )
            for parameter in reference_parameters
        ]

        if client_id in (8, 9):

            local_parameters = [
                parameter + 5.0
                for parameter in local_parameters
            ]

            trust = 0.20

        else:
            trust = 0.90

        updates.append(
            ClientUpdate(
                client_id=client_id,
                parameters=local_parameters,
                num_samples=100 + 10 * client_id,
                trust=trust,
            )
        )

    config = AggregationConfig(
        trust_threshold=0.50,
        krum_byzantine_clients=2,
        trimmed_mean_fraction=0.20,
    )

    results = compare_aggregators(
        updates,
        config,
    )

    for method_name, result in results.items():

        print(
            f"\nMethod: {method_name}"
        )

        print(
            "Participating clients:",
            result.participating_clients,
        )

        print(
            "Excluded clients:",
            result.excluded_clients,
        )

        print(
            "Aggregation time:",
            f"{result.aggregation_time_ms:.4f} ms",
        )

        print(
            "First parameter matrix:"
        )

        print(
            result.parameters[0]
        )
