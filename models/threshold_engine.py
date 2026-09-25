import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def compute_anomaly_metrics(model, snapshot_sequence, k_multiplier=2.5):
    model.eval()
    with torch.no_grad():
        reconstructed = model(snapshot_sequence)
        target = torch.stack([s.x for s in snapshot_sequence], dim=1)

        feature_mse = torch.mean((reconstructed - target) ** 2, dim=1).cpu().numpy()
        y_true = (torch.stack([s.y for s in snapshot_sequence], dim=0).sum(dim=0) > 0).long().cpu().numpy()

        benign_mask = (y_true == 0)
        if np.sum(benign_mask) > 0:
            std_per_feat = np.std(feature_mse[benign_mask], axis=0) + 1e-6
        else:
            std_per_feat = np.std(feature_mse, axis=0) + 1e-6

        node_mse = np.mean(feature_mse / std_per_feat, axis=1)

        if np.sum(benign_mask) > 0:
            mu_baseline = np.mean(node_mse[benign_mask])
            sigma_baseline = np.std(node_mse[benign_mask])
        else:
            mu_baseline = np.mean(node_mse)
            sigma_baseline = np.std(node_mse)

        threshold = float(mu_baseline + k_multiplier * sigma_baseline)
        y_pred = (node_mse >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, node_mse) if len(np.unique(y_true)) > 1 else 0.5

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn + 1e-6)

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "fpr": float(fpr),
            "y_true": y_true,
            "y_pred": y_pred,
            "recon_error": node_mse,
            "threshold": float(threshold)
        }
