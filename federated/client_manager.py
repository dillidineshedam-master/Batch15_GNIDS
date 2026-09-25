import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
import yaml
from models.autoencoder import SpatialTemporalAutoencoder
from config.logging_config import get_logger

logger = get_logger("FederatedClient")

class GNIDSEdgeClient(fl.client.NumPyClient):
    """
    Decentralized Edge Client Node with Differential Privacy (DP-SGD).
    Trains local GATv2-GRU autoencoders strictly on edge subnet partitions.
    """
    def __init__(self, snapshots, client_id, config_path="config/config.yaml"):
        self.snapshots = snapshots
        self.client_id = client_id
        
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(self.cfg["system"]["device"])
        self.model = SpatialTemporalAutoencoder(
            in_features=self.cfg["model"]["in_dim"],
            hidden_dim=self.cfg["model"]["hidden_dim"],
            heads=self.cfg["model"]["heads"]
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.cfg["model"]["learning_rate"], 
            weight_decay=self.cfg["model"]["weight_decay"]
        )
        self.criterion = nn.MSELoss()
        self.dp_enabled = self.cfg["federated"]["dp_enabled"]
        self.dp_clip = self.cfg["federated"]["dp_clip_norm"]
        self.dp_noise = self.cfg["federated"]["dp_noise_multiplier"]

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        target = torch.stack([s.x for s in self.snapshots], dim=1).to(self.device)
        epochs = self.cfg["model"]["local_epochs"]
        epoch_loss = 0.0

        for ep in range(epochs):
            self.optimizer.zero_grad()
            reconstructed = self.model(self.snapshots)
            loss = self.criterion(reconstructed, target)
            loss.backward()

            # Differential Privacy: Gradient Clipping and Gaussian Perturbation
            if self.dp_enabled:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.dp_clip)
                for p in self.model.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * (self.dp_noise * self.dp_clip)
                        p.grad.add_(noise)

            self.optimizer.step()
            epoch_loss = float(loss.item())

        logger.info(f"Client #{self.client_id} finished local training. Loss: {epoch_loss:.6f}")
        return self.get_parameters(config={}), len(self.snapshots), {"loss": epoch_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        target = torch.stack([s.x for s in self.snapshots], dim=1).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(self.snapshots)
            loss = self.criterion(reconstructed, target)
        return float(loss.item()), len(self.snapshots), {"val_loss": float(loss.item())}
