import torch
import numpy as np
from federated.server_aggregator import FederatedServerManager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix

mgr = FederatedServerManager("config/config.yaml")
ckpt_path = "results/checkpoints/best_global_gnids.pt"
mgr.global_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
mgr.global_model.eval()

all_snaps = mgr.test_snapshots

# Phase A: Validation Window (Snapshots 0 to 4)
val_snaps = all_snaps[:5]
with torch.no_grad():
    recon_val = mgr.global_model(val_snaps)
    target_val = torch.stack([s.x for s in val_snaps], dim=1)
    feat_mse_val = torch.mean((recon_val - target_val) ** 2, dim=1).cpu().numpy()
    y_val = (torch.stack([s.y for s in val_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()

    # Normalizer derived strictly on validation benign nodes
    benign_val = (y_val == 0)
    std_feat = np.std(feat_mse_val[benign_val], axis=0) + 1e-6
    node_mse_val = np.mean(feat_mse_val / std_feat, axis=1)

# Phase B: Test Window (Snapshots 5 to 9)
test_snaps = all_snaps[5:10]
with torch.no_grad():
    recon_test = mgr.global_model(test_snaps)
    target_test = torch.stack([s.x for s in test_snaps], dim=1)
    feat_mse_test = torch.mean((recon_test - target_test) ** 2, dim=1).cpu().numpy()
    y_test = (torch.stack([s.y for s in test_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()
    node_mse_test = np.mean(feat_mse_test / std_feat, axis=1)

# 1. Method A: Transfer optimal tau from Validation PR-Curve
prec_v, rec_v, thresh_v = precision_recall_curve(y_val, node_mse_val)
f1_v = (2 * prec_v[:-1] * rec_v[:-1]) / (prec_v[:-1] + rec_v[:-1] + 1e-8)
tau_val_opt = float(thresh_v[np.argmax(f1_v)])

# Evaluate tau_val_opt blindly on Test
y_pred_opt = (node_mse_test >= tau_val_opt).astype(int)
f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
prec_opt = precision_score(y_test, y_pred_opt, zero_division=0)
rec_opt = recall_score(y_test, y_pred_opt, zero_division=0)
cm_opt = confusion_matrix(y_test, y_pred_opt, labels=[0, 1]).ravel()

print("=" * 65)
print("METHOD 1: VALIDATION-CALIBRATED THRESHOLD TRANSFER")
print(f"Validation Optimal Threshold (tau) : {tau_val_opt:.4f}")
print(f"Test F1-Score                      : {f1_opt:.4f}")
print(f"Test Precision                     : {prec_opt:.4f} | Recall: {rec_opt:.4f}")
print(f"Confusion Matrix (TN, FP, FN, TP)  : {cm_opt}")
print("=" * 65)

# 2. Method B: Parametric k-sigma sweep on validation benign baseline
mu_b = np.mean(node_mse_val[benign_val])
sig_b = np.std(node_mse_val[benign_val])

print("METHOD 2: STATISTICAL SIGMA MULTIPLIER SWEEP")
print(f"{'k':<6} | {'Threshold':<12} | {'Test Prec':<12} | {'Test Rec':<12} | {'Test F1':<12} | {'FP Count':<10}")
print("-" * 65)
for k in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    t = mu_b + k * sig_b
    yp = (node_mse_test >= t).astype(int)
    p = precision_score(y_test, yp, zero_division=0)
    r = recall_score(y_test, yp, zero_division=0)
    f = f1_score(y_test, yp, zero_division=0)
    fp = np.sum((yp == 1) & (y_test == 0))
    print(f"{k:<6.1f} | {t:<12.4f} | {p:<12.4f} | {r:<12.4f} | {f:<12.4f} | {fp:<10}")
print("=" * 65)
