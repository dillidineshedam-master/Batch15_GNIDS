import os
import yaml
import torch
import numpy as np
import pandas as pd

from core.partitioner import get_non_iid_subnets
from models.autoencoder import SpatialTemporalAutoencoder
from models.threshold_engine import compute_anomaly_metrics
from federated.defense_engine import (
    aggregate_fedavg,
    aggregate_coordinate_median,
    aggregate_trimmed_mean,
    aggregate_krum
)
from config.logging_config import get_logger

logger = get_logger("AdversarialEval")
RESULTS_DIR = "results/metrics"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_one_epoch(model, snapshots, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    target = torch.stack([s.x for s in snapshots], dim=1)
    
    model.train()
    optimizer.zero_grad()
    recon = model(snapshots)
    loss = criterion(recon, target)
    loss.backward()
    optimizer.step()
    
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def simulate_poisoning_experiment():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    c1_snaps, c2_snaps, test_snaps = get_non_iid_subnets("config/config.yaml")

    strategies = {
        "Unprotected FedAvg (Under Attack)": aggregate_fedavg,
        "Coordinate-wise Median (Defended)": aggregate_coordinate_median,
        "Trimmed Mean (Defended)": aggregate_trimmed_mean,
        "Krum Aggregator (Defended)": aggregate_krum
    }

    results = []

    for name, agg_fn in strategies.items():
        logger.info(f"Evaluating Defense Strategy: {name}...")
        
        # Initialize Global & Client Models
        global_model = SpatialTemporalAutoencoder(in_features=16, hidden_dim=32, heads=2)
        c1_model = SpatialTemporalAutoencoder(in_features=16, hidden_dim=32, heads=2)
        c2_model = SpatialTemporalAutoencoder(in_features=16, hidden_dim=32, heads=2)

        # 3 Federated Training Rounds
        for rnd in range(3):
            # Client 1: Benign Volumetric Subnet
            c1_model.load_state_dict(global_model.state_dict())
            w1 = train_one_epoch(c1_model, c1_snaps)

            # Client 2: Benign Stealth Subnet
            c2_model.load_state_dict(global_model.state_dict())
            w2 = train_one_epoch(c2_model, c2_snaps)

            # Client 3 (Byzantine Attacker): Malicious Gradient Inversion & Noise Injection
            w3_malicious = [-10.0 * p + np.random.normal(0, 2.0, p.shape) for p in w1]

            # Aggregate across 2 benign + 1 malicious client
            client_updates = [w1, w2, w3_malicious]
            aggregated_weights = agg_fn(client_updates)

            # Update Global Parameters
            state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), aggregated_weights)}
            global_model.load_state_dict(state_dict, strict=True)

        # Evaluate Robust Global Model
        metrics = compute_anomaly_metrics(global_model, test_snaps[:8])
        results.append({
            "Aggregation Strategy": name,
            "Accuracy (%)": metrics["accuracy"] * 100,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
            "False Positive Rate": metrics["fpr"]
        })

    adv_df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "adversarial_byzantine_defense_matrix.csv")
    adv_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 85)
    print("🛡️ BYZANTINE MODEL POISONING DEFENSE BENCHMARK (1 Malicious Subnet Node)")
    print("=" * 85)
    print(adv_df.to_string(index=False))
    print("=" * 85)
    print(f"📁 Exported defense evaluation to: {out_csv}")

if __name__ == "__main__":
    simulate_poisoning_experiment()
