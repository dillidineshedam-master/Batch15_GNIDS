import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class DynamicSpatialGATv2Encoder(nn.Module):
    """
    Edge-aware dynamic Multi-Head Graph Attention Network (GATv2).
    Computes dynamic attention coefficients between host nodes conditioned on flow attributes.
    """
    def __init__(self, in_dim=16, hidden_dim=32, heads=2, edge_dim=3, dropout=0.1):
        super(DynamicSpatialGATv2Encoder, self).__init__()
        self.gat1 = GATv2Conv(
            in_channels=in_dim, 
            out_channels=hidden_dim, 
            heads=heads, 
            edge_dim=edge_dim, 
            concat=True,
            dropout=dropout
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads, 
            out_channels=hidden_dim, 
            heads=1, 
            edge_dim=edge_dim, 
            concat=False,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        h = F.elu(self.gat1(x, edge_index, edge_attr))
        h = F.elu(self.gat2(h, edge_index, edge_attr))
        h = self.layer_norm(h)
        return h

if __name__ == "__main__":
    from core.partitioner import get_non_iid_subnets
    _, _, test_snaps = get_non_iid_subnets()
    encoder = DynamicSpatialGATv2Encoder()
    snap = test_snaps[0]
    out = encoder(snap.x, snap.edge_index, snap.edge_attr)
    print(f"? GATv2 Spatial Encoder Test Passed: Output Tensor = {out.shape}")
