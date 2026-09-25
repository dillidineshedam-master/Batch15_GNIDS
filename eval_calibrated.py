import torch
import yaml
from federated.server_aggregator import FederatedServerManager
from models.threshold_engine import compute_anomaly_metrics

# Initialize server manager to load test split and baseline structures
mgr = FederatedServerManager("config/config.yaml")

# Load saved Round 5 model checkpoint
ckpt_path = "results/checkpoints/best_global_gnids.pt"
if not torch.os.path.exists(ckpt_path):
    ckpt_path = "checkpoints/best_global_gnids.pt"

print(f"Loading checkpoint from: {ckpt_path}")
state_dict = torch.load(ckpt_path, map_location="cpu")
mgr.global_model.load_state_dict(state_dict)

# Re-compute calibrated metrics
metrics = compute_anomaly_metrics(mgr.global_model, mgr.test_snapshots[:8])

print("=" * 60)
print(f"CALIBRATED IEEE EVALUATION RESULTS:")
print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  F1-Score  : {metrics['f1_score']:.4f}")
print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
print(f"  Threshold : {metrics['threshold']:.6f}")
print("=" * 60)
