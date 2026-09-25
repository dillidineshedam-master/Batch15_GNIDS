import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class SpatialGATv2Encoder(nn.Module):
    """
    Spatial Attention Layer: Aggregates multi-hop host interactions
    conditioned on the 16-dimensional NetFlow edge attributes.
    """
    def __init__(self, in_node_dim=50, edge_dim=16, hidden_dim=64, heads=2):
        super(SpatialGATv2Encoder, self).__init__()
        self.conv1 = GATv2Conv(
            in_channels=in_node_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=True
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            edge_dim=edge_dim,
            concat=False
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.conv2(h, edge_index, edge_attr)
        return h

class TemporalGRUEncoder(nn.Module):
    """
    Temporal Recurrent Layer: Tracks sequential evolution of node
    embeddings across T consecutive graph snapshots.
    """
    def __init__(self, node_dim=64, gru_hidden_dim=64, num_layers=1):
        super(TemporalGRUEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=node_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, temporal_node_states):
        # Shape: [num_nodes, T, node_dim]
        out, h_n = self.gru(temporal_node_states)
        # Take the final temporal state as the summary representation
        return out[:, -1, :]  # Shape: [num_nodes, gru_hidden_dim]

class EdgeAttributeDecoder(nn.Module):
    """
    Reconstruction Decoder: Predicts original 16-dimensional NetFlow
    features for each directed link from source and target node states.
    """
    def __init__(self, node_dim=64, edge_dim=16):
        super(EdgeAttributeDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, edge_dim)
        )

    def forward(self, node_embeddings, edge_index):
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        src_features = node_embeddings[src_nodes]
        dst_features = node_embeddings[dst_nodes]
        
        edge_pair = torch.cat([src_features, dst_features], dim=-1)
        reconstructed_attrs = self.mlp(edge_pair)
        return reconstructed_attrs

class SpatialTemporalGNNAutoencoder(nn.Module):
    """
    Unified Fed-GNIDS Anomaly Extraction Engine.
    """
    def __init__(self, in_node_dim=50, edge_dim=16, hidden_dim=64):
        super(SpatialTemporalGNNAutoencoder, self).__init__()
        self.spatial_encoder = SpatialGATv2Encoder(in_node_dim, edge_dim, hidden_dim)
        self.temporal_encoder = TemporalGRUEncoder(hidden_dim, hidden_dim)
        self.edge_decoder = EdgeAttributeDecoder(hidden_dim, edge_dim)

    def forward(self, snapshot_sequence):
        spatial_states = []

        # 1. Spatial encoding across each snapshot
        for snapshot in snapshot_sequence:
            h_t = self.spatial_encoder(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            spatial_states.append(h_t)

        # Stack over time: [num_nodes, T, hidden_dim]
        temporal_tensor = torch.stack(spatial_states, dim=1)

        # 2. Temporal sequential aggregation
        final_node_embeddings = self.temporal_encoder(temporal_tensor)

        # 3. Decode & reconstruct edge attributes of the latest (T-th) snapshot
        target_snapshot = snapshot_sequence[-1]
        reconstructed_edges = self.edge_decoder(final_node_embeddings, target_snapshot.edge_index)

        return reconstructed_edges, target_snapshot.edge_attr

    def compute_anomaly_scores(self, snapshot_sequence):
        self.eval()
        with torch.no_grad():
            reconstructed, ground_truth = self.forward(snapshot_sequence)
            mse_per_edge = torch.mean((reconstructed - ground_truth) ** 2, dim=-1)
        return mse_per_edge


if __name__ == "__main__":
    print("[*] Running model verification test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    try:
        # weights_only=False allows unpickling custom PyG Data objects safely
        sample_data = torch.load("data/processed/client_0_data.pt", weights_only=False)
        test_sequence = sample_data[0]
        
        test_sequence = [snap.to(device) for snap in test_sequence]

        model = SpatialTemporalGNNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64).to(device)
        
        # Forward pass
        pred, target = model(test_sequence)
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        
        print(f"[+] Forward pass successful!")
        print(f"    - Target edge shape: {target.shape}")
        print(f"    - Predicted edge shape: {pred.shape}")
        print(f"    - Initial reconstruction loss: {loss.item():.4f}")
        
        # Anomaly scoring test
        scores = model.compute_anomaly_scores(test_sequence)
        print(f"[+] Computed anomaly scores for {scores.shape[0]} edges (Mean: {scores.mean().item():.4f})")
        print("[SUCCESS] Module 2 Model Architecture verified.")
    except Exception as e:
        print(f"[-] Verification error: {e}")