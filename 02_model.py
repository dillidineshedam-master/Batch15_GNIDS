import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import importlib

class SpatialGATEncoder(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, heads=2):
        super(SpatialGATEncoder, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return x

class TemporalGRUDecoder(nn.Module):
    def __init__(self, hidden_dim=32, out_dim=16):
        super(TemporalGRUDecoder, self).__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.reconstruction_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, spatial_embeddings):
        out, _ = self.gru(spatial_embeddings)
        reconstruction = self.reconstruction_layer(out)
        return reconstruction

class GNIDS_Autoencoder(nn.Module):
    def __init__(self, in_features=16, hidden_dim=32):
        super(GNIDS_Autoencoder, self).__init__()
        self.encoder = SpatialGATEncoder(in_dim=in_features, hidden_dim=hidden_dim)
        self.decoder = TemporalGRUDecoder(hidden_dim=hidden_dim, out_dim=in_features)

    def forward(self, snapshot_sequence):
        embeddings = []
        for snapshot in snapshot_sequence:
            z = self.encoder(snapshot.x, snapshot.edge_index)
            embeddings.append(z)
            
        stacked = torch.stack(embeddings, dim=0)
        stacked = stacked.permute(1, 0, 2) # [Nodes, Time, Features]
        reconstructed = self.decoder(stacked)
        return reconstructed

def evaluate_intrusion_detection(model, snapshot_sequence):
    """Computes IEEE-standard intrusion metrics via reconstruction error distribution."""
    model.eval()
    with torch.no_grad():
        reconstructed = model(snapshot_sequence)
        target = torch.stack([s.x for s in snapshot_sequence], dim=1)
        
        # Node-level Mean Squared Error across all snapshots and features
        recon_error = torch.mean((reconstructed - target) ** 2, dim=[1, 2]).cpu().numpy()
        y_true = snapshot_sequence[0].y.cpu().numpy()
        
        # Adaptive Threshold: Intrusions reside in upper 80th percentile
        threshold = np.quantile(recon_error, 0.80)
        y_pred = (recon_error >= threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, recon_error)
        else:
            auc = 0.5
            
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
            "recon_error": recon_error
        }

if __name__ == "__main__":
    preprocess_lib = importlib.import_module("01_preprocess")
    snaps, _ = preprocess_lib.extract_dynamic_snapshots()
    model = GNIDS_Autoencoder()
    m = evaluate_intrusion_detection(model, snaps)
    print("=" * 60)
    print(f"📊 METRICS TEST: Acc: {m['accuracy']*100:.2f}% | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")
    print("=" * 60)