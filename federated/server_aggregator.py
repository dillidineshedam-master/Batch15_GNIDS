import os
import yaml
import torch
import numpy as np
import pandas as pd
import flwr as fl
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.autoencoder import SpatialTemporalAutoencoder
from models.threshold_engine import compute_anomaly_metrics
from core.partitioner import get_non_iid_subnets
from config.logging_config import get_logger

logger = get_logger("CentralServer")

class FederatedServerManager:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(self.cfg["system"]["device"])
        self.global_model = SpatialTemporalAutoencoder(
            in_features=self.cfg["model"]["in_dim"],
            hidden_dim=self.cfg["model"]["hidden_dim"],
            heads=self.cfg["model"]["heads"]
        ).to(self.device)

        _, _, self.test_snapshots = get_non_iid_subnets(config_path)
        self.metrics_history = []
        self.checkpoint_dir = "results/checkpoints"
        self.figures_dir = "results/figures"
        self.metrics_dir = "results/metrics"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def evaluate_fn(self, server_round: int, parameters, config):
        if server_round == 0:
            return 0.0, {}

        params_dict = zip(self.global_model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.global_model.load_state_dict(state_dict, strict=True)

        k_val = self.cfg["detection"]["threshold_sigma_multiplier"]
        metrics = compute_anomaly_metrics(self.global_model, self.test_snapshots[:8], k_multiplier=k_val)
        current_loss = float(np.mean(metrics["recon_error"]))

        print("\n" + "=" * 65)
        print(f"[*] [GLOBAL FEDERATED EVALUATION - ROUND {server_round}]")
        print(f"    Reconstruction Loss : {current_loss:.6f}")
        print(f"    Detection Accuracy  : {metrics['accuracy'] * 100:.2f}%")
        print(f"    Precision           : {metrics['precision']:.4f}")
        print(f"    Recall              : {metrics['recall']:.4f}")
        print(f"    F1-Score            : {metrics['f1_score']:.4f}")
        print(f"    ROC-AUC             : {metrics['roc_auc']:.4f}")
        print(f"    False Positive Rate : {metrics['fpr']:.4f}")
        print(f"    Dynamic Threshold   : {metrics['threshold']:.6f}")
        print("=" * 65 + "\n")

        ckpt_path = os.path.join(self.checkpoint_dir, "best_global_gnids.pt")
        torch.save(self.global_model.state_dict(), ckpt_path)

        self._plot_topology(self.test_snapshots[0], metrics["y_pred"], server_round)

        self.metrics_history.append({
            "round": server_round,
            "loss": current_loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "fpr": metrics["fpr"]
        })

        return current_loss, {"f1_score": metrics["f1_score"], "accuracy": metrics["accuracy"]}

    def _plot_topology(self, snapshot, y_pred, round_num):
        plt.figure(figsize=(9, 7), dpi=300)
        G = nx.DiGraph()
        G.add_nodes_from(range(snapshot.num_nodes))
        edges = snapshot.edge_index.t().cpu().numpy()
        G.add_edges_from(edges[:180])

        pos = nx.spring_layout(G, seed=42, k=0.3)
        node_colors = ['#2ecc71' if pred == 0 else '#e74c3c' for pred in y_pred]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=280, edgecolors='#2c3e50', linewidths=1.2)
        nx.draw_networkx_edges(G, pos, edge_color='#bdc3c7', alpha=0.35, arrows=True, arrowsize=10)
        labels = {i: f"H{i}" for i in range(snapshot.num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color='#2c3e50', font_weight='bold')

        plt.title(f"Dynamic Host Interaction Topology - Federated Round {round_num}\n(Green: Benign Host | Red: Detected Intrusion)", 
                  fontsize=11, fontweight='bold', pad=15)
        plt.axis('off')
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, f"topology_map_round_{round_num}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def export_results(self):
        if not self.metrics_history:
            return
        df = pd.DataFrame(self.metrics_history)
        csv_path = os.path.join(self.metrics_dir, "federated_training_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved full training history CSV to '{csv_path}'")
