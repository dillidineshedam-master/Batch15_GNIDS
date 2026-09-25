import os
import copy
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from federated.server_aggregator import FederatedServerManager
from core.data_loader import load_and_validate_dataset
from core.partitioner import get_non_iid_subnets

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
SEEDS = [42, 101, 999]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_threshold_metrics(model, calib_snaps, test_snaps, k=2.5):
    model.eval()
    with torch.no_grad():
        recon_c = model(calib_snaps)
        target_c = torch.stack([s.x for s in calib_snaps], dim=1)
        feat_err_c = torch.mean((recon_c - target_c) ** 2, dim=1).cpu().numpy()
        y_c = (torch.stack([s.y for s in calib_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()

        benign = (y_c == 0)
        std_feat = np.std(feat_err_c[benign], axis=0) + 1e-6
        node_err_c = np.mean(feat_err_c / std_feat, axis=1)

        mu = np.mean(node_err_c[benign])
        sig = np.std(node_err_c[benign])
        tau = float(mu + k * sig)

        recon_t = model(test_snaps)
        target_t = torch.stack([s.x for s in test_snaps], dim=1)
        feat_err_t = torch.mean((recon_t - target_t) ** 2, dim=1).cpu().numpy()
        node_err_t = np.mean(feat_err_t / std_feat, axis=1)

        y_t = (torch.stack([s.y for s in test_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()
        y_pred = (node_err_t >= tau).astype(int)

        return {
            "Accuracy": accuracy_score(y_t, y_pred) * 100,
            "Precision": precision_score(y_t, y_pred, zero_division=0),
            "Recall": recall_score(y_t, y_pred, zero_division=0),
            "F1": f1_score(y_t, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_t, node_err_t) if len(np.unique(y_t)) > 1 else 0.5
        }

def run_seed_trial(seed, mgr, calib_snaps, test_snaps, df, feature_cols, scaler):
    set_seed(seed)
    print(f"\n--- Running Experimental Seed: {seed} ---")

    # 1. Centralized Tabular Random Forest
    X = scaler.transform(df[feature_cols].values)
    y = df['label'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    rf_pred = clf.predict(X_te)
    rf_prob = clf.predict_proba(X_te)[:, 1]
    rf_metrics = {
        "Accuracy": accuracy_score(y_te, rf_pred) * 100,
        "Precision": precision_score(y_te, rf_pred, zero_division=0),
        "Recall": recall_score(y_te, rf_pred, zero_division=0),
        "F1": f1_score(y_te, rf_pred, zero_division=0),
        "AUC": roc_auc_score(y_te, rf_prob)
    }

    # 2. Local-Only GAT-GRU (Client 1 Subnet Data Only)
    c1_snaps, c2_snaps, _ = get_non_iid_subnets("config/config.yaml")
    local_model = copy.deepcopy(mgr.global_model)
    opt_loc = torch.optim.Adam(local_model.parameters(), lr=0.005)
    crit = nn.MSELoss()
    
    local_model.train()
    for _ in range(25):
        opt_loc.zero_grad()
        recon = local_model(c1_snaps)
        loss = crit(recon, torch.stack([s.x for s in c1_snaps], dim=1))
        loss.backward()
        opt_loc.step()
    loc_metrics = evaluate_threshold_metrics(local_model, calib_snaps, test_snaps)

    # 3. Federated GAT-GRU (Convergence-Matched FedAvg across Subnets 1 & 2)
    fed_model = copy.deepcopy(mgr.global_model)
    local_epochs = 3  # Matches physical Flower client configuration

    for rnd in range(15):
        # Client 1 local update (Subnet 1)
        m1 = copy.deepcopy(fed_model)
        opt1 = torch.optim.Adam(m1.parameters(), lr=0.005)
        m1.train()
        for _ in range(local_epochs):
            opt1.zero_grad()
            loss1 = crit(m1(c1_snaps), torch.stack([s.x for s in c1_snaps], dim=1))
            loss1.backward()
            opt1.step()

        # Client 2 local update (Subnet 2)
        m2 = copy.deepcopy(fed_model)
        opt2 = torch.optim.Adam(m2.parameters(), lr=0.005)
        m2.train()
        for _ in range(local_epochs):
            opt2.zero_grad()
            loss2 = crit(m2(c2_snaps), torch.stack([s.x for s in c2_snaps], dim=1))
            loss2.backward()
            opt2.step()

        # Server FedAvg aggregation
        fed_dict = fed_model.state_dict()
        d1 = m1.state_dict()
        d2 = m2.state_dict()
        for k in fed_dict.keys():
            fed_dict[k] = (d1[k].float() + d2[k].float()) / 2.0
        fed_model.load_state_dict(fed_dict)

    fed_metrics = evaluate_threshold_metrics(fed_model, calib_snaps, test_snaps)

    return rf_metrics, loc_metrics, fed_metrics

if __name__ == "__main__":
    print("=" * 75)
    print("PHASE 3: STATISTICAL RIGOR - MULTI-SEED VALIDATION SWEEP")
    print("=" * 75)

    mgr = FederatedServerManager("config/config.yaml")
    df, scaler, feature_cols, _ = load_and_validate_dataset()
    calib_snaps = mgr.test_snapshots[:5]
    test_snaps = mgr.test_snapshots[5:10]

    rf_records, loc_records, fed_records = [], [], []

    for s in SEEDS:
        rf_m, loc_m, fed_m = run_seed_trial(s, mgr, calib_snaps, test_snaps, df, feature_cols, scaler)
        rf_records.append(rf_m)
        loc_records.append(loc_m)
        fed_records.append(fed_m)

    def summarize(name, records):
        df_rec = pd.DataFrame(records)
        return {
            "Model": name,
            "Accuracy (%)": f"{df_rec['Accuracy'].mean():.2f} +/- {df_rec['Accuracy'].std():.2f}",
            "Precision": f"{df_rec['Precision'].mean():.4f} +/- {df_rec['Precision'].std():.4f}",
            "Recall": f"{df_rec['Recall'].mean():.4f} +/- {df_rec['Recall'].std():.4f}",
            "F1-Score": f"{df_rec['F1'].mean():.4f} +/- {df_rec['F1'].std():.4f}",
            "ROC-AUC": f"{df_rec['AUC'].mean():.4f} +/- {df_rec['AUC'].std():.4f}"
        }

    summary_rows = [
        summarize("Centralized Random Forest", rf_records),
        summarize("Local-Only GAT-GRU (No FL)", loc_records),
        summarize("Fed-GAT-GRU (Proposed)", fed_records)
    ]

    out_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "multiseed_statistical_results.csv")
    out_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 75)
    print("IEEE STATISTICAL BENCHMARK TABLE (MEAN +/- STD)")
    print("=" * 75)
    print(out_df.to_string(index=False))
    print("=" * 75)
    print(f"Metrics saved to '{csv_path}'\n")