import torch
import numpy as np
import yaml
from federated.server_aggregator import FederatedServerManager
from sklearn.metrics import precision_recall_curve

mgr = FederatedServerManager("config/config.yaml")
ckpt_path = "results/checkpoints/best_global_gnids.pt"
mgr.global_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
mgr.global_model.eval()

snaps = mgr.test_snapshots[:8]
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
feature_names = cfg["data"]["features"]

with torch.no_grad():
    reconstructed = mgr.global_model(snaps)
    target = torch.stack([s.x for s in snaps], dim=1)
    
    # Per-feature squared error across time sequence: shape (num_nodes, num_features)
    feature_mse = torch.mean((reconstructed - target) ** 2, dim=1).cpu().numpy()
    node_mse = np.mean(feature_mse, axis=1)
    y_true = (torch.stack([s.y for s in snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()

benign_mask = (y_true == 0)
attack_mask = (y_true == 1)

# Inspect false positives at the current 0.3858 threshold
fp_mask = benign_mask & (node_mse >= 0.3858)

print("=" * 65)
print(f"Total Hosts: {len(y_true)} | Benign False Positives: {np.sum(fp_mask)}")
print("=" * 65)
print(f"{'Feature Name':<15} | {'Benign FP Mean Error':<20} | {'True Attack Mean Error':<20}")
print("-" * 65)

fp_feat_mean = np.mean(feature_mse[fp_mask], axis=0) if np.sum(fp_mask) > 0 else np.zeros(feature_mse.shape[1])
attack_feat_mean = np.mean(feature_mse[attack_mask], axis=0)

for name, fp_err, atk_err in zip(feature_names, fp_feat_mean, attack_feat_mean):
    flag = " <-- DOMINANT" if fp_err > 1.0 else ""
    print(f"{name:<15} | {fp_err:<20.4f} | {atk_err:<20.4f}{flag}")

# Feature-variance normalized error to penalize outlier features evenly
std_per_feat = np.std(feature_mse[benign_mask], axis=0) + 1e-6
norm_mse = np.mean(feature_mse / std_per_feat, axis=1)

prec, rec, _ = precision_recall_curve(y_true, norm_mse)
f1_norm = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)

print("=" * 65)
print(f"Standard MSE Max F1    : 0.8333")
print(f"Normalized MSE Max F1  : {np.max(f1_norm):.4f}")
print("=" * 65)
