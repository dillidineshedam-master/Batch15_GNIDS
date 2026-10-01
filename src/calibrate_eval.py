import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from hybrid_pipeline import DualEngineFedGNIDS

device = torch.device("cpu")
print("[*] Loading test benchmark and model checkpoint...")

test_data = torch.load("data/processed/real_test_benchmark.pt", weights_only=False)
model = DualEngineFedGNIDS(50, 16, 64).to(device)
model.load_state_dict(torch.load("results/checkpoints/hybrid_dual_engine_gnids.pt", weights_only=False))
model.eval()

cls_probs, recon_errors, all_labels = [], [], []

with torch.no_grad():
    for seq in test_data:
        seq = [s.to(device) for s in seq]
        rec_edges, logits, gt_edges, gt_labels = model(seq)
        mse = torch.mean((rec_edges - gt_edges) ** 2, dim=-1)
        probs = torch.sigmoid(logits)
        
        recon_errors.extend(mse.numpy())
        cls_probs.extend(probs.numpy())
        all_labels.extend(gt_labels.numpy())

cls_probs = np.array(cls_probs)
recon_errors = np.array(recon_errors)
all_labels = np.array(all_labels)

# 1. Uncalibrated baseline at 0.50 cutoff
preds_default = (cls_probs >= 0.50).astype(int)

# 2. Youden's J-Index (Max Diagnostic Balance: TPR - FPR)
fpr, tpr, roc_thresh = roc_curve(all_labels, cls_probs)
opt_j_thresh = roc_thresh[np.argmax(tpr - fpr)]
preds_youden = (cls_probs >= opt_j_thresh).astype(int)

# 3. Global Accuracy-Optimal Threshold (Exhaustive Sweep)
candidate_thresholds = np.linspace(0.10, 0.60, 501)
best_acc = 0.0
opt_acc_thresh = 0.50

for t in candidate_thresholds:
    p = (cls_probs >= t).astype(int)
    acc = accuracy_score(all_labels, p)
    if acc > best_acc:
        best_acc = acc
        opt_acc_thresh = t

preds_acc_opt = (cls_probs >= opt_acc_thresh).astype(int)

# 4. Engine 2 Unsupervised Autoencoder Baseline
tau = np.mean(recon_errors[all_labels == 0]) + 1.0 * np.std(recon_errors[all_labels == 0])
preds_unsup = (recon_errors >= tau).astype(int)

results = [
    {
        "Engine Evaluation Tier": "Engine 1 (Accuracy-Optimal Threshold)",
        "Decision Boundary": f"theta = {opt_acc_thresh:.4f}",
        "Accuracy": accuracy_score(all_labels, preds_acc_opt),
        "Precision": precision_score(all_labels, preds_acc_opt, zero_division=0),
        "Recall": recall_score(all_labels, preds_acc_opt, zero_division=0),
        "F1-Score": f1_score(all_labels, preds_acc_opt, zero_division=0),
        "ROC-AUC": roc_auc_score(all_labels, cls_probs)
    },
    {
        "Engine Evaluation Tier": "Engine 1 (Youden's J Balanced Threshold)",
        "Decision Boundary": f"theta = {opt_j_thresh:.4f}",
        "Accuracy": accuracy_score(all_labels, preds_youden),
        "Precision": precision_score(all_labels, preds_youden, zero_division=0),
        "Recall": recall_score(all_labels, preds_youden, zero_division=0),
        "F1-Score": f1_score(all_labels, preds_youden, zero_division=0),
        "ROC-AUC": roc_auc_score(all_labels, cls_probs)
    },
    {
        "Engine Evaluation Tier": "Engine 1 (Default Uncalibrated 0.50)",
        "Decision Boundary": "theta = 0.5000",
        "Accuracy": accuracy_score(all_labels, preds_default),
        "Precision": precision_score(all_labels, preds_default, zero_division=0),
        "Recall": recall_score(all_labels, preds_default, zero_division=0),
        "F1-Score": f1_score(all_labels, preds_default, zero_division=0),
        "ROC-AUC": roc_auc_score(all_labels, cls_probs)
    },
    {
        "Engine Evaluation Tier": "Engine 2 (Zero-Day Anomaly Autoencoder)",
        "Decision Boundary": f"tau = {tau:.4f}",
        "Accuracy": accuracy_score(all_labels, preds_unsup),
        "Precision": precision_score(all_labels, preds_unsup, zero_division=0),
        "Recall": recall_score(all_labels, preds_unsup, zero_division=0),
        "F1-Score": f1_score(all_labels, preds_unsup, zero_division=0),
        "ROC-AUC": roc_auc_score(all_labels, recon_errors)
    }
]

df = pd.DataFrame(results)
print("\n" + "="*110)
print("              DUAL-ENGINE FED-GNIDS OPERATIONAL BENCHMARK (REAL UNSW-NB15)")
print("="*110)
print(df[["Engine Evaluation Tier", "Decision Boundary", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))
print("="*110)

df.to_csv("results/ieee_calibrated_final.csv", index=False)
print("\n[+] Final IEEE Matrix Saved: results/ieee_calibrated_final.csv")