import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import importlib
import sys

preprocess_mod = importlib.import_module("01_preprocess")
model_mod = importlib.import_module("02_model")

extract_dynamic_snapshots = preprocess_mod.extract_dynamic_snapshots
GNIDS_Autoencoder = model_mod.GNIDS_Autoencoder

class GNIDSClient(fl.client.NumPyClient):
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.model = GNIDS_Autoencoder()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.008, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        target = torch.stack([s.x for s in self.snapshots], dim=1)
        epoch_loss = 0.0
        
        for _ in range(3): # Local Epochs
            self.optimizer.zero_grad()
            reconstructed = self.model(self.snapshots)
            loss = self.criterion(reconstructed, target)
            loss.backward()
            self.optimizer.step()
            epoch_loss = float(loss.item())

        return self.get_parameters(config={}), len(self.snapshots), {"loss": epoch_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        target = torch.stack([s.x for s in self.snapshots], dim=1)
        with torch.no_grad():
            reconstructed = self.model(self.snapshots)
            loss = self.criterion(reconstructed, target)
        return float(loss.item()), len(self.snapshots), {"val_loss": float(loss.item())}

if __name__ == "__main__":
    print("=" * 60)
    print("💻 STARTING FEDERATED EDGE CLIENT NODE")
    print("=" * 60)
    snapshots, _ = extract_dynamic_snapshots()
    client = GNIDSClient(snapshots)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=client
    )