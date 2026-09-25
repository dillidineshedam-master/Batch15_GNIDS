import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGRUDecoder(nn.Module):
    """
    Temporal Recurrent Decoder with Layer Normalization and Residual Reconstruction Head.
    Decodes spatial embedding sequences across snapshot time-windows.
    """
    def __init__(self, hidden_dim=32, out_dim=16, num_layers=2, dropout=0.1):
        super(TemporalGRUDecoder, self).__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.reconstruct = nn.Linear(hidden_dim, out_dim)

    def forward(self, temporal_embeddings):
        # temporal_embeddings: [Num_Nodes, Sequence_Length, Hidden_Dim]
        gru_out, _ = self.gru(temporal_embeddings)
        normed = self.norm(gru_out)
        h = F.relu(self.fc1(normed))
        reconstruction = self.reconstruct(h)
        return reconstruction

if __name__ == "__main__":
    decoder = TemporalGRUDecoder()
    dummy_input = torch.randn(50, 8, 32) # 50 nodes, 8 time-steps, 32 dims
    out = decoder(dummy_input)
    print(f"? Temporal GRU Decoder Test Passed: Reconstructed Tensor = {out.shape}")
