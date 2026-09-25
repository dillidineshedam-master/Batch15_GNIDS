import torch
import numpy as np
from federated.server_aggregator import FederatedServerManager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

mgr = FederatedServerManager("config/config.yaml")
ckpt_path = "results/checkpoints/best_global_gnids.pt"
mgr.global_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
mgr.global_model.eval()

# Total snapshots available on server: 10
all_snaps = mgr.test_snapshots

# Phase A: Calibration Window (Snapshots 0 to 4)
calib_snaps = all_snaps[:5]
with torch.no_grad():
    recon_calib = mgr.global_model(calib_snaps)
    target_calib = torch.stack([s.x for s in calib_snaps], dim=1)
    
    # Feature MSE: (num_nodes, num_features)
    feat_mse_calib = torch.mean((recon_calib - target_calib) ** 2, dim=1).cpu().numpy()
    y_calib = (torch.stack([s.y for s in calib_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()

    # Feature-variance normalizer computed strictly on benign nodes in calibration window
    benign_mask_calib = (y_calib == 0)
    std_per_feat = np.std(feat_mse_calib[benign_mask_calib], axis=0) + 1e-6
    node_mse_calib = np.mean(feat_mse_calib / std_per_feat, axis=1)

    # Statistical Extreme Threshold (Mean + 2.5 * Sigma on calibration baseline)
    mu_baseline = np.mean(node_mse_calib[benign_mask_calib])
    sigma_baseline = np.std(node_mse_calib[benign_mask_calib])
    tau = float(mu_baseline + 2.5 * sigma_baseline)

# Phase B: Blind Test Window (Snapshots 5 to 9)
test_snaps = all_snaps[5:10]
with torch.no_grad():
    recon_test = mgr.global_model(test_snaps)
    target_test = torch.stack([s.x for s in test_snaps], dim=1)
    
    feat_mse_test = torch.mean((recon_test - target_test) ** 2, dim=1).cpu().numpy()
    y_test = (torch.stack([s.y for s in test_snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()
    
    # Apply calibration normalizer and threshold blindly
    node_mse_test = np.mean(feat_mse_test / std_per_feat, axis=1)
    y_pred_test = (node_mse_test >= tau).astype(int)

# Out-of-Sample Metrics
acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)
auc = roc_auc_score(y_test, node_mse_test) if len(np.unique(y_test)) > 1 else 0.5
cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])

print("=" * 65)
print("OUT-OF-SAMPLE BENCHMARK (TEMPORAL SPLIT 5/5, NO LEAKAGE)")
print(f"Calibrated Decision Threshold (tau) : {tau:.6f}")
print("-" * 65)
print(f"  Detection Accuracy                 : {acc * 100:.2f}%")
print(f"  Precision                          : {prec:.4f}")
print(f"  Recall                             : {rec:.4f}")
print(f"  F1-Score                           : {f1:.4f}")
print(f"  ROC-AUC                            : {auc:.4f}")
print(f"  Confusion Matrix (TN, FP, FN, TP)  : {cm.ravel()}")
print("=" * 65)
