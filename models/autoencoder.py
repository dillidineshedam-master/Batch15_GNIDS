import torch
import torch.nn as nn
from models.spatial_gatv2 import DynamicSpatialGATv2Encoder
from models.temporal_decoder import TemporalGRUDecoder

class SpatialTemporalAutoencoder(nn.Module):
    """
    End-to-End GNIDS Spatial-Temporal Graph Attention Autoencoder.
    Couples GATv2 neighborhood spatial awareness with GRU multi-snapshot trajectory reconstruction.
    """
    def __init__(self, in_features=16, hidden_dim=32, heads=2, edge_dim=3):
        super(SpatialTemporalAutoencoder, self).__init__()
        self.spatial_encoder = DynamicSpatialGATv2Encoder(
            in_dim=in_features,
            hidden_dim=hidden_dim,
            heads=heads,
            edge_dim=edge_dim
        )
        self.temporal_decoder = TemporalGRUDecoder(
            hidden_dim=hidden_dim,
            out_dim=in_features
        )

    def forward(self, snapshot_sequence):
        # 1. Spatial encoding across each snapshot in temporal sequence
        spatial_embeddings = []
        for snap in snapshot_sequence:
            z_t = self.spatial_encoder(snap.x, snap.edge_index, snap.edge_attr)
            spatial_embeddings.append(z_t)

        # 2. Stack into temporal sequence [Num_Nodes, Time_Steps, Hidden_Dim]
        stacked = torch.stack(spatial_embeddings, dim=0) # [Time_Steps, Num_Nodes, Hidden_Dim]
        stacked = stacked.permute(1, 0, 2)              # [Num_Nodes, Time_Steps, Hidden_Dim]

        # 3. Temporal decoding and feature reconstruction
        reconstructed = self.temporal_decoder(stacked)   # [Num_Nodes, Time_Steps, In_Features]
        return reconstructed

if __name__ == "__main__":
    from core.partitioner import get_non_iid_subnets
    _, _, test_snaps = get_non_iid_subnets()
    model = SpatialTemporalAutoencoder()
    reconstructed = model(test_snaps[:8])
    print(f"? Full Autoencoder Forward Pass: Reconstructed Shape = {reconstructed.shape}")
