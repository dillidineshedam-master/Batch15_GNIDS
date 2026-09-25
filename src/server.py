import os
import argparse
import numpy as np
import torch
import flwr as fl
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from model import SpatialTemporalGNNAutoencoder

class AdaptiveContributionScalingStrategy(FedAvg):
    def __init__(self, norm_threshold=5.0, min_fit_clients=2, *args, **kwargs):
        super().__init__(min_fit_clients=min_fit_clients, *args, **kwargs)
        self.norm_threshold = norm_threshold
        # Initialize model structure to hold ground truth global weights
        self.reference_model = SpatialTemporalGNNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64)
        self.global_weights = [val.cpu().numpy() for _, val in self.reference_model.state_dict().items()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n[+] === [ACS Strategy] Server Round {server_round} ({len(results)} updates received) ===")

        client_updates = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Calculate True Deltas: Delta_k = w_k - w_global
        deltas = []
        flat_deltas = []
        for client_w in client_updates:
            delta = [cw - gw for cw, gw in zip(client_w, self.global_weights)]
            deltas.append(delta)
            flat_deltas.append(np.concatenate([d.flatten() for d in delta]))

        flat_deltas = np.array(flat_deltas)

        # Coordinate-wise median consensus vector
        reference_vector = np.median(flat_deltas, axis=0)
        ref_norm = np.linalg.norm(reference_vector) + 1e-8

        # Calculate ACS weights
        raw_scores = []
        for idx, (f_delta, delta) in enumerate(zip(flat_deltas, deltas)):
            l2_norm = np.linalg.norm(f_delta)
            norm_factor = min(1.0, self.norm_threshold / (l2_norm + 1e-8))

            cosine_sim = np.dot(f_delta, reference_vector) / ((l2_norm * ref_norm) + 1e-8)
            alignment_factor = max(0.0, float(cosine_sim))

            alpha = alignment_factor * norm_factor
            raw_scores.append(alpha)
            status = "NEUTRALIZED" if alpha < 0.10 else "ACCEPTED"
            print(f"    -> Client {idx}: Update Norm={l2_norm:.2f} | Cosine={cosine_sim:+.3f} | ACS Weight={alpha:.4f} [{status}]")

        sum_scores = sum(raw_scores) + 1e-8
        normalized_weights = [s / sum_scores for s in raw_scores]

        # Weighted parameter aggregation
        aggregated_deltas = [np.zeros_like(w) for w in self.global_weights]
        for client_idx, weight in enumerate(normalized_weights):
            for layer_idx, layer_delta in enumerate(deltas[client_idx]):
                aggregated_deltas[layer_idx] += weight * layer_delta

        # Update global reference
        self.global_weights = [gw + ad for gw, ad in zip(self.global_weights, aggregated_deltas)]

        # Save checkpoint
        os.makedirs("data/processed", exist_ok=True)
        model = SpatialTemporalGNNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64)
        params_dict = zip(model.state_dict().keys(), [torch.tensor(w) for w in self.global_weights])
        model.load_state_dict({k: v for k, v in params_dict})
        torch.save(model.state_dict(), "data/processed/global_model.pt")

        return ndarrays_to_parameters(self.global_weights), {}

def run_server(rounds=5, port=8080, strategy_type="acs"):
    if strategy_type == "acs":
        strategy = AdaptiveContributionScalingStrategy(norm_threshold=5.0, min_fit_clients=2, min_available_clients=2)
        print(f"[*] Starting Server with ACS DEFENSE STRATEGY on port {port}...")
    else:
        strategy = FedAvg(min_fit_clients=2, min_available_clients=2)
        print(f"[*] Starting Server with STANDARD FEDAVG on port {port}...")

    fl.server.start_server(
        server_address=f"127.0.0.1:{port}",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--strategy", type=str, default="acs", choices=["acs", "fedavg"])
    args = parser.parse_args()

    run_server(rounds=args.rounds, port=args.port, strategy_type=args.strategy)