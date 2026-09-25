import torch
import numpy as np
from typing import List, Tuple

def aggregate_fedavg(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Standard Federated Averaging (Baseline)."""
    num_clients = len(weights_list)
    new_weights = []
    for layer_idx in range(len(weights_list[0])):
        layer_sum = np.sum([weights_list[c][layer_idx] for c in range(num_clients)], axis=0)
        new_weights.append(layer_sum / num_clients)
    return new_weights

def aggregate_coordinate_median(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Byzantine-Robust Coordinate-wise Median Aggregator."""
    num_layers = len(weights_list[0])
    aggregated = []
    for layer_idx in range(num_layers):
        layer_stack = np.stack([weights_list[c][layer_idx] for c in range(len(weights_list))], axis=0)
        median_layer = np.median(layer_stack, axis=0)
        aggregated.append(median_layer)
    return aggregated

def aggregate_trimmed_mean(weights_list: List[List[np.ndarray]], trim_ratio: float = 0.2) -> List[np.ndarray]:
    """Byzantine-Robust Trimmed Mean Aggregator."""
    num_clients = len(weights_list)
    num_layers = len(weights_list[0])
    k = int(num_clients * trim_ratio)
    aggregated = []
    
    for layer_idx in range(num_layers):
        layer_stack = np.stack([weights_list[c][layer_idx] for c in range(num_clients)], axis=0)
        if 2 * k >= num_clients:
            # Fallback to mean if client count is small
            aggregated.append(np.mean(layer_stack, axis=0))
        else:
            sorted_stack = np.sort(layer_stack, axis=0)
            trimmed = sorted_stack[k : num_clients - k]
            aggregated.append(np.mean(trimmed, axis=0))
    return aggregated

def aggregate_krum(weights_list: List[List[np.ndarray]], num_byzantine: int = 1) -> List[np.ndarray]:
    """Krum Aggregator: Selects update closest to the geometric center of trusted neighbors."""
    num_clients = len(weights_list)
    if num_clients <= 2:
        return aggregate_coordinate_median(weights_list)

    # Flatten client parameter vectors for distance calculation
    flat_vectors = []
    for w in weights_list:
        flat = np.concatenate([p.ravel() for p in w])
        flat_vectors.append(flat)

    scores = []
    for i in range(num_clients):
        dists = []
        for j in range(num_clients):
            if i != j:
                dists.append(np.linalg.norm(flat_vectors[i] - flat_vectors[j]) ** 2)
        dists.sort()
        # Sum distances to the closest n - f - 2 neighbors
        score = sum(dists[: max(1, num_clients - num_byzantine - 2)])
        scores.append(score)

    best_client_idx = int(np.argmin(scores))
    return weights_list[best_client_idx]
