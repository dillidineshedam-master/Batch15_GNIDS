import torch
import numpy as np
from federated.server_aggregator import FederatedServerManager
from sklearn.metrics import precision_recall_curve, roc_auc_score

mgr = FederatedServerManager("config/config.yaml")
ckpt_path = "results/checkpoints/best_global_gnids.pt"
mgr.global_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
mgr.global_model.eval()

snaps = mgr.test_snapshots[:8]
with torch.no_grad():
    reconstructed = mgr.global_model(snaps)
    target = torch.stack([s.x for s in snaps], dim=1)
    node_mse = torch.mean((reconstructed - target) ** 2, dim=[1, 2]).cpu().numpy()
    y_true = (torch.stack([s.y for s in snaps], dim=0).sum(dim=0) > 0).long().cpu().numpy()

benign = node_mse[y_true == 0]
attack = node_mse[y_true == 1]

prec, rec, thresh = precision_recall_curve(y_true, node_mse)
f1_arr = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
best_i = int(np.argmax(f1_arr))

print("=" * 60)
print(f"Total Hosts: {len(y_true)} (Benign: {len(benign)}, Malicious: {len(attack)})")
print(f"Benign MSE  -> Min: {benign.min():.4f} | Median: {np.median(benign):.4f} | Max: {benign.max():.4f}")
print(f"Attack MSE  -> Min: {attack.min():.4f} | Median: {np.median(attack):.4f} | Max: {attack.max():.4f}")
print(f"Overlap     -> {np.sum(benign >= attack.min())} benign hosts have higher MSE than the lowest attack")
print(f"Ceiling F1  -> {f1_arr[best_i]:.4f} (Prec: {prec[best_i]:.4f}, Rec: {rec[best_i]:.4f})")
print("=" * 60)
