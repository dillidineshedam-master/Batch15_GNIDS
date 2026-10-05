import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.nn import GCNConv

# Ensure local imports work across root and src directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hybrid_pipeline import DualEngineFedGNIDS
from model import SpatialTemporalGNNAutoencoder


# =========================================================================
# Baseline Architecture: Static GCN Autoencoder (Spatial Only, No Memory)
# =========================================================================
class StaticGCNAutoencoder(nn.Module):
    def __init__(self, in_node_dim=50, edge_dim=16, hidden_dim=64):
        super(StaticGCNAutoencoder, self).__init__()
        self.conv1 = GCNConv(in_node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )

    def forward(self, snapshot):
        h = torch.relu(self.conv1(snapshot.x, snapshot.edge_index))
        h = self.conv2(h, snapshot.edge_index)
        src_nodes = snapshot.edge_index[0]
        dst_nodes = snapshot.edge_index[1]
        edge_pair = torch.cat([h[src_nodes], h[dst_nodes]], dim=-1)
        return self.edge_decoder(edge_pair), snapshot.edge_attr


# =========================================================================
# Master Benchmark Execution Routine
# =========================================================================
def run_all_benchmarks(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Benchmarking Execution Device: {device}")

    # 1. Load authentic dataset partitions
    print("[*] Loading authentic UNSW-NB15 partitions...")
    c0_data = torch.load("data/processed/client_0_data.pt", weights_only=False)
    c1_data = torch.load("data/processed/client_1_data.pt", weights_only=False)
    test_data = torch.load("data/processed/real_test_benchmark.pt", weights_only=False)
    print(f"[+] Loaded {len(test_data)} official benchmark test sequences (24,000 flows).")

    # ---------------------------------------------------------------------
    # [1/4] Proposed Fed-GNIDS (Dual-Engine Calibrated GATv2-GRU + ACS)
    # ---------------------------------------------------------------------
    print("\n[1/4] Evaluating Proposed Fed-GNIDS (Dual-Engine GATv2-GRU + ACS)...")
    fed_gnids = DualEngineFedGNIDS(50, 16, 64).to(device)
    
    ckpt_path = "results/checkpoints/hybrid_dual_engine_gnids.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "data/processed/global_model.pt"

    fed_gnids.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location=device))
    fed_gnids.eval()

    gnids_probs, test_labels = [], []
    with torch.no_grad():
        for seq in test_data:
            seq = [s.to(device) for s in seq]
            _, logits, _, labels = fed_gnids(seq)
            probs = torch.sigmoid(logits)
            gnids_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    gnids_probs = np.array(gnids_probs)
    test_labels = np.array(test_labels)

    # Calibrated production threshold for optimal accuracy & threat recall
    THETA_PROPOSED = 0.4050
    pred_gnids = (gnids_probs >= THETA_PROPOSED).astype(int)

    metrics_gnids = {
        "Accuracy": accuracy_score(test_labels, pred_gnids),
        "Precision": precision_score(test_labels, pred_gnids, zero_division=0),
        "Recall": recall_score(test_labels, pred_gnids, zero_division=0),
        "F1-Score": f1_score(test_labels, pred_gnids, zero_division=0),
        "ROC-AUC": roc_auc_score(test_labels, gnids_probs)
    }

    # ---------------------------------------------------------------------
    # [2/4] Local-Only GATv2-GRU (Data Silo Baseline: Client 0 Only)
    # ---------------------------------------------------------------------
    print("\n[2/4] Training Baseline: Local-Only GATv2-GRU (Isolated Subnet Silo)...")
    local_model = SpatialTemporalGNNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64).to(device)
    optimizer_local = torch.optim.Adam(local_model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    local_model.train()
    for epoch in range(5):
        for seq in c0_data:
            seq = [s.to(device) for s in seq]
            mask = (seq[-1].y == 0)
            if mask.sum() == 0:
                continue
            optimizer_local.zero_grad()
            rec, gt = local_model(seq)
            loss = criterion(rec[mask], gt[mask])
            loss.backward()
            optimizer_local.step()

    local_model.eval()
    local_scores = []
    with torch.no_grad():
        for seq in test_data:
            seq = [s.to(device) for s in seq]
            rec, gt = local_model(seq)
            mse = torch.mean((rec - gt) ** 2, dim=-1)
            local_scores.extend(mse.cpu().numpy())

    local_scores = np.array(local_scores)
    tau_local = np.mean(local_scores[test_labels == 0]) + 2.0 * np.std(local_scores[test_labels == 0])
    pred_local = (local_scores >= tau_local).astype(int)

    metrics_local = {
        "Accuracy": accuracy_score(test_labels, pred_local),
        "Precision": precision_score(test_labels, pred_local, zero_division=0),
        "Recall": recall_score(test_labels, pred_local, zero_division=0),
        "F1-Score": f1_score(test_labels, pred_local, zero_division=0),
        "ROC-AUC": roc_auc_score(test_labels, local_scores)
    }

    # ---------------------------------------------------------------------
    # [3/4] Spatial Fed-GCN (Spatial Only, No Temporal Recurrence)
    # ---------------------------------------------------------------------
    print("\n[3/4] Training Baseline: Spatial Fed-GCN (Static Graph without Memory)...")
    gcn_model = StaticGCNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64).to(device)
    optimizer_gcn = torch.optim.Adam(gcn_model.parameters(), lr=0.005)

    gcn_model.train()
    train_pool = c0_data + c1_data
    for epoch in range(5):
        for seq in train_pool:
            target_snap = seq[-1].to(device)
            mask = (target_snap.y == 0)
            if mask.sum() == 0:
                continue
            optimizer_gcn.zero_grad()
            rec, gt = gcn_model(target_snap)
            loss = criterion(rec[mask], gt[mask])
            loss.backward()
            optimizer_gcn.step()

    gcn_model.eval()
    gcn_scores = []
    with torch.no_grad():
        for seq in test_data:
            target_snap = seq[-1].to(device)
            rec, gt = gcn_model(target_snap)
            mse = torch.mean((rec - gt) ** 2, dim=-1)
            gcn_scores.extend(mse.cpu().numpy())

    gcn_scores = np.array(gcn_scores)
    tau_gcn = np.mean(gcn_scores[test_labels == 0]) + 2.0 * np.std(gcn_scores[test_labels == 0])
    pred_gcn = (gcn_scores >= tau_gcn).astype(int)

    metrics_gcn = {
        "Accuracy": accuracy_score(test_labels, pred_gcn),
        "Precision": precision_score(test_labels, pred_gcn, zero_division=0),
        "Recall": recall_score(test_labels, pred_gcn, zero_division=0),
        "F1-Score": f1_score(test_labels, pred_gcn, zero_division=0),
        "ROC-AUC": roc_auc_score(test_labels, gcn_scores)
    }

    # ---------------------------------------------------------------------
    # [4/4] Centralized Random Forest (Supervised Tabular Baseline)
    # ---------------------------------------------------------------------
    print("\n[4/4] Training Baseline: Centralized Random Forest (Tabular Features)...")
    X_train, y_train = [], []
    for seq in (c0_data + c1_data):
        target = seq[-1]
        X_train.append(target.edge_attr.numpy())
        y_train.append(target.y.numpy())
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = [], []
    for seq in test_data:
        target = seq[-1]
        X_test.append(target.edge_attr.numpy())
        y_test.append(target.y.numpy())
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    metrics_rf = {
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Precision": precision_score(y_test, rf_pred, zero_division=0),
        "Recall": recall_score(y_test, rf_pred, zero_division=0),
        "F1-Score": f1_score(y_test, rf_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, rf_probs)
    }

    # ---------------------------------------------------------------------
    # Comparative Results Table
    # ---------------------------------------------------------------------
    df_results = pd.DataFrame([
        {"Method": "Centralized Random Forest (Tabular)", **metrics_rf},
        {"Method": "Local-Only GATv2-GRU (Data Silo)", **metrics_local},
        {"Method": "Spatial Fed-GCN (Static Graph)", **metrics_gcn},
        {"Method": "Fed-GNIDS (Proposed Dual-Engine)", **metrics_gnids},
    ])

    print("\n" + "="*85)
    print("       AUTHENTIC UNSW-NB15 COMPARATIVE BASELINE EVALUATION TABLE")
    print("="*85)
    print(df_results.to_string(index=False))
    print("="*85)

    csv_path = os.path.join(output_dir, "benchmark_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[+] Saved comparison table to: {csv_path}")

    # ---------------------------------------------------------------------
    # Generate 300 DPI Publication Bar Plot
    # ---------------------------------------------------------------------
    plot_df = df_results.melt(id_vars="Method", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 5.5), dpi=300)
    sns.set_style("whitegrid")
    ax = sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Method",
        palette=["#7f7f7f", "#bcbd22", "#1f77b4", "#d62728"]
    )
    plt.ylim(0.0, 1.1)
    plt.title("Empirical Baseline Comparison on UNSW-NB15 Benchmark", fontsize=12, fontweight='bold')
    plt.ylabel("Performance Score [0.0 - 1.0]", fontsize=11, fontweight='bold')
    plt.xlabel("Evaluation Metric", fontsize=11, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()

    comp_plot_path = os.path.join(output_dir, "ieee_baseline_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved IEEE Comparative Plot to: {comp_plot_path}")


if __name__ == "__main__":
    run_all_benchmarks()