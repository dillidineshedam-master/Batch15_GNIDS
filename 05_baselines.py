import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from federated.server_aggregator import FederatedServerManager
from core.data_loader import load_and_validate_dataset
from core.partitioner import get_non_iid_subnets

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# Baseline 1: Centralized Tabular Random Forest
# -------------------------------------------------------------
def evaluate_rf_baseline():
    df, scaler, feature_cols, _ = load_and_validate_dataset()
    X = scaler.transform(df[feature_cols].values)
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "Model": "Centralized Random Forest (Tabular)",
        "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
    }

# -------------------------------------------------------------
# Baseline 2: Spatial-Only GCN Autoencoder (Non-Temporal)
# -------------------------------------------------------------
class SpatialGCN_Autoencoder(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super(SpatialGCN_Autoencoder, self).__init__()
        self.enc1 = GCNConv(in_dim, hidden_dim)
        self.enc2 = GCNConv(hidden_dim, hidden_dim)
        self.dec1 = nn.Linear(hidden_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.enc1(x, edge_index))
        h = F.relu(self.enc2(h, edge_index))
        z = F.relu(self.dec1(h))
        return self.dec2(z)

def evaluate_gcn_baseline(calib_snaps, test_snaps, k_multiplier=2.5):
    in_dim = calib_snaps[0].x.size(1)
    model = SpatialGCN_Autoencoder(in_dim=in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(25):
        for snap in calib_snaps:
            optimizer.zero_grad()
            recon = model(snap.x, snap.edge_index)
            loss = criterion(recon, snap.x)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        # Variance normalization on calibration baseline
        calib_feat_errs = [((model(s.x, s.edge_index) - s.x) ** 2).cpu().numpy() for s in calib_snaps]
        feat_mse_calib = np.mean(np.array(calib_feat_errs), axis=0) # (num_nodes, num_features)

        y_calib = (torch.stack([s.y for s in calib_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()
        benign_mask = (y_calib == 0)
        std_per_feat = np.std(feat_mse_calib[benign_mask], axis=0) + 1e-6

        node_mse_calib = np.mean(feat_mse_calib / std_per_feat, axis=1)
        mu_base = np.mean(node_mse_calib[benign_mask])
        sigma_base = np.std(node_mse_calib[benign_mask])
        tau = float(mu_base + k_multiplier * sigma_base)

        test_feat_errs = [((model(s.x, s.edge_index) - s.x) ** 2).cpu().numpy() for s in test_snaps]
        feat_mse_test = np.mean(np.array(test_feat_errs), axis=0)
        node_mse_test = np.mean(feat_mse_test / std_per_feat, axis=1)

        y_test = (torch.stack([s.y for s in test_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()
        y_pred = (node_mse_test >= tau).astype(int)

        return {
            "Model": "Spatial GCN (Non-Temporal)",
            "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, node_mse_test), 4)
        }

# -------------------------------------------------------------
# Baseline 3: Local-Only GAT-GRU (No Federated Sharing)
# -------------------------------------------------------------
def evaluate_local_only_baseline(mgr, calib_snaps, test_snaps):
    local_model = copy.deepcopy(mgr.global_model)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    c1_snaps, _, _ = get_non_iid_subnets("config/config.yaml")

    local_model.train()
    for _ in range(15):
        optimizer.zero_grad()
        recon = local_model(c1_snaps)
        target = torch.stack([s.x for s in c1_snaps], dim=1)
        loss = criterion(recon, target)
        loss.backward()
        optimizer.step()

    local_model.eval()
    with torch.no_grad():
        recon_calib = local_model(calib_snaps)
        target_calib = torch.stack([s.x for s in calib_snaps], dim=1)
        feat_mse_calib = torch.mean((recon_calib - target_calib) ** 2, dim=1).cpu().numpy()
        y_calib = (torch.stack([s.y for s in calib_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()

        benign_mask = (y_calib == 0)
        std_per_feat = np.std(feat_mse_calib[benign_mask], axis=0) + 1e-6
        node_mse_calib = np.mean(feat_mse_calib / std_per_feat, axis=1)

        mu_base = np.mean(node_mse_calib[benign_mask])
        sigma_base = np.std(node_mse_calib[benign_mask])
        tau = float(mu_base + 2.5 * sigma_base)

        recon_test = local_model(test_snaps)
        target_test = torch.stack([s.x for s in test_snaps], dim=1)
        feat_mse_test = torch.mean((recon_test - target_test) ** 2, dim=1).cpu().numpy()
        node_mse_test = np.mean(feat_mse_test / std_per_feat, axis=1)

        y_test = (torch.stack([s.y for s in test_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()
        y_pred = (node_mse_test >= tau).astype(int)

        return {
            "Model": "Local-Only GAT-GRU (No FL)",
            "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, node_mse_test), 4)
        }

# -------------------------------------------------------------
# Proposed Model: Fed-GAT-GRU (Physical Checkpoint)
# -------------------------------------------------------------
def evaluate_fed_gnids(mgr, calib_snaps, test_snaps, ckpt_path="results/checkpoints/best_global_gnids.pt"):
    mgr.global_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    mgr.global_model.eval()

    with torch.no_grad():
        recon_calib = mgr.global_model(calib_snaps)
        target_calib = torch.stack([s.x for s in calib_snaps], dim=1)
        feat_mse_calib = torch.mean((recon_calib - target_calib) ** 2, dim=1).cpu().numpy()
        y_calib = (torch.stack([s.y for s in calib_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()

        benign_mask = (y_calib == 0)
        std_per_feat = np.std(feat_mse_calib[benign_mask], axis=0) + 1e-6
        node_mse_calib = np.mean(feat_mse_calib / std_per_feat, axis=1)

        mu_baseline = np.mean(node_mse_calib[benign_mask])
        sigma_baseline = np.std(node_mse_calib[benign_mask])
        tau = float(mu_baseline + 2.5 * sigma_baseline)

        recon_test = mgr.global_model(test_snaps)
        target_test = torch.stack([s.x for s in test_snaps], dim=1)
        feat_mse_test = torch.mean((recon_test - target_test) ** 2, dim=1).cpu().numpy()
        node_mse_test = np.mean(feat_mse_test / std_per_feat, axis=1)

        y_test = (torch.stack([s.y for s in test_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()
        y_pred = (node_mse_test >= tau).astype(int)

        return {
            "Model": "Fed-GAT-GRU (Proposed)",
            "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, node_mse_test), 4)
        }

if __name__ == "__main__":
    print("=" * 75)
    print("PHASE 2: RUNNING RIGOROUS IEEE COMPARATIVE BENCHMARK")
    print("=" * 75)

    mgr = FederatedServerManager("config/config.yaml")
    all_snaps = mgr.test_snapshots
    calib_snaps = all_snaps[:5]
    test_snaps = all_snaps[5:10]

    print("\n[1/4] Evaluating Baseline: Centralized Tabular Random Forest...")
    rf_res = evaluate_rf_baseline()

    print("[2/4] Evaluating Baseline: Spatial GCN (Non-Temporal)...")
    gcn_res = evaluate_gcn_baseline(calib_snaps, test_snaps)

    print("[3/4] Evaluating Baseline: Local-Only GAT-GRU (No FL)...")
    local_res = evaluate_local_only_baseline(mgr, calib_snaps, test_snaps)

    print("[4/4] Evaluating Proposed: Fed-GAT-GRU (Multi-Node LAN)...")
    fed_res = evaluate_fed_gnids(mgr, calib_snaps, test_snaps)

    results = [rf_res, gcn_res, local_res, fed_res]
    comp_df = pd.DataFrame(results)

    comp_csv_path = os.path.join(RESULTS_DIR, "baseline_comparison_table.csv")
    comp_df.to_csv(comp_csv_path, index=False)

    print("\n" + "=" * 75)
    print("IEEE COMPARATIVE BENCHMARK TABLE")
    print("=" * 75)
    print(comp_df.to_string(index=False))
    print("=" * 75)
    print(f"Results saved to '{comp_csv_path}'\n")
