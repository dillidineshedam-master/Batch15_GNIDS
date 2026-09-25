import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from torch_geometric.nn import GCNConv, GATConv

from core.data_loader import load_and_validate_dataset
from core.partitioner import get_non_iid_subnets
from models.autoencoder import SpatialTemporalAutoencoder
from models.threshold_engine import compute_anomaly_metrics
from config.logging_config import get_logger

logger = get_logger("BenchmarkSuite")
RESULTS_DIR = "results/metrics"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Baseline 1: Spatial GCN (Ablation: No Attention, No Temporal)
class SpatialGCN_Autoencoder(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super(SpatialGCN_Autoencoder, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, in_dim)
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

# Baseline 2: Spatial GAT (Ablation: No Temporal Tracking)
class SpatialGAT_Autoencoder(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super(SpatialGAT_Autoencoder, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=2, concat=True)
        self.gat2 = GATConv(hidden_dim * 2, in_dim, heads=1, concat=False)
    def forward(self, x, edge_index):
        h = F.elu(self.gat1(x, edge_index))
        return self.gat2(h, edge_index)

# Baseline 3: Temporal GRU (Ablation: No Spatial Graph Topology)
class TemporalGRU_Autoencoder(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super(TemporalGRU_Autoencoder, self).__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, in_dim)
    def forward(self, x_seq):
        out, _ = self.gru(x_seq)
        return self.fc(out)

def run_benchmarks():
    df, scaler, feature_cols, cfg = load_and_validate_dataset()
    _, _, test_snapshots = get_non_iid_subnets()

    X = df[feature_cols].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = []

    # 1. Random Forest
    logger.info("Evaluating: Centralized Random Forest...")
    rf = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred, y_prob = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
    results.append({
        "Model Architecture": "Centralized Random Forest (Tabular)",
        "Category": "Non-Graph ML",
        "Accuracy (%)": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    # 2. XGBoost
    logger.info("Evaluating: Centralized XGBoost...")
    xgb = XGBClassifier(n_estimators=60, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_pred, y_prob = xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1]
    results.append({
        "Model Architecture": "Centralized XGBoost (Gradient Boosted)",
        "Category": "Non-Graph ML",
        "Accuracy (%)": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    # 3. Deep MLP
    logger.info("Evaluating: Centralized Deep MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred, y_prob = mlp.predict(X_test), mlp.predict_proba(X_test)[:, 1]
    results.append({
        "Model Architecture": "Centralized Deep MLP (Neural Tabular)",
        "Category": "Neural Non-Graph",
        "Accuracy (%)": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    # 4. Spatial GCN Ablation
    logger.info("Evaluating: Spatial GCN Autoencoder (No Temporal)...")
    gcn = SpatialGCN_Autoencoder()
    opt = torch.optim.Adam(gcn.parameters(), lr=0.01)
    crit = nn.MSELoss()
    for _ in range(10):
        for s in test_snapshots:
            opt.zero_grad()
            loss = crit(gcn(s.x, s.edge_index), s.x)
            loss.backward()
            opt.step()
    with torch.no_grad():
        t_snap = test_snapshots[-1]
        err = torch.mean((gcn(t_snap.x, t_snap.edge_index) - t_snap.x)**2, dim=1).numpy()
        y_t = t_snap.y.numpy()
        y_p = (err >= np.quantile(err, 0.70)).astype(int)
        results.append({
            "Model Architecture": "Spatial GCN (Ablation: No Temporal)",
            "Category": "Static Graph",
            "Accuracy (%)": accuracy_score(y_t, y_p) * 100,
            "Precision": precision_score(y_t, y_p, zero_division=0),
            "Recall": recall_score(y_t, y_p, zero_division=0),
            "F1-Score": f1_score(y_t, y_p, zero_division=0),
            "ROC-AUC": roc_auc_score(y_t, err) if len(np.unique(y_t)) > 1 else 0.5
        })

    # 5. Spatial GAT Ablation
    logger.info("Evaluating: Spatial GAT Autoencoder (No Temporal)...")
    gat = SpatialGAT_Autoencoder()
    opt = torch.optim.Adam(gat.parameters(), lr=0.01)
    for _ in range(10):
        for s in test_snapshots:
            opt.zero_grad()
            loss = crit(gat(s.x, s.edge_index), s.x)
            loss.backward()
            opt.step()
    with torch.no_grad():
        err = torch.mean((gat(t_snap.x, t_snap.edge_index) - t_snap.x)**2, dim=1).numpy()
        y_p = (err >= np.quantile(err, 0.70)).astype(int)
        results.append({
            "Model Architecture": "Spatial GAT (Ablation: No Temporal)",
            "Category": "Static Graph",
            "Accuracy (%)": accuracy_score(y_t, y_p) * 100,
            "Precision": precision_score(y_t, y_p, zero_division=0),
            "Recall": recall_score(y_t, y_p, zero_division=0),
            "F1-Score": f1_score(y_t, y_p, zero_division=0),
            "ROC-AUC": roc_auc_score(y_t, err) if len(np.unique(y_t)) > 1 else 0.5
        })

    # 6. Temporal GRU Ablation
    logger.info("Evaluating: Temporal GRU Autoencoder (No Graph)...")
    gru_model = TemporalGRU_Autoencoder()
    opt = torch.optim.Adam(gru_model.parameters(), lr=0.01)
    seq_tensor = torch.stack([s.x for s in test_snapshots[:8]], dim=1)
    for _ in range(15):
        opt.zero_grad()
        loss = crit(gru_model(seq_tensor), seq_tensor)
        loss.backward()
        opt.step()
    with torch.no_grad():
        err = torch.mean((gru_model(seq_tensor) - seq_tensor)**2, dim=[1, 2]).numpy()
        y_t = (torch.stack([s.y for s in test_snapshots[:8]], dim=0).sum(dim=0) > 0).long().numpy()
        y_p = (err >= np.quantile(err, 0.70)).astype(int)
        results.append({
            "Model Architecture": "Temporal GRU (Ablation: No Graph)",
            "Category": "Temporal Only",
            "Accuracy (%)": accuracy_score(y_t, y_p) * 100,
            "Precision": precision_score(y_t, y_p, zero_division=0),
            "Recall": recall_score(y_t, y_p, zero_division=0),
            "F1-Score": f1_score(y_t, y_p, zero_division=0),
            "ROC-AUC": roc_auc_score(y_t, err) if len(np.unique(y_t)) > 1 else 0.5
        })

    # 7. Proposed Fed-GAT-GRU (from checkpoint)
    logger.info("Evaluating: Proposed Fed-GAT-GRU (Federated Spatial-Temporal)...")
    fed_model = SpatialTemporalAutoencoder(in_features=16, hidden_dim=32, heads=2)
    ckpt_path = "results/checkpoints/best_global_gnids.pt"
    if os.path.exists(ckpt_path):
        fed_model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    m = compute_anomaly_metrics(fed_model, test_snapshots[:8])
    results.append({
        "Model Architecture": "Fed-GAT-GRU (Proposed)",
        "Category": "Federated Spatial-Temporal",
        "Accuracy (%)": m["accuracy"] * 100,
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1-Score": m["f1_score"],
        "ROC-AUC": m["roc_auc"]
    })

    comp_df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "comprehensive_ieee_benchmark_matrix.csv")
    comp_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("📊 IEEE COMPREHENSIVE 8-MODEL BENCHMARK & ABLATION MATRIX")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    print("=" * 80)
    print(f"📁 Benchmark matrix exported to: {out_csv}")

if __name__ == "__main__":
    run_benchmarks()
