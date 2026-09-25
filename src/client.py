import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict
from model import SpatialTemporalGNNAutoencoder

class EdgeFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_path, epochs=3, lr=0.005, dp_clip=1.0, attack="none"):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.lr = lr
        self.dp_clip = dp_clip
        self.attack = attack

        # Load local real dataset partition
        self.dataset = torch.load(data_path, weights_only=False)
        self.model = SpatialTemporalGNNAutoencoder(in_node_dim=50, edge_dim=16, hidden_dim=64).to(self.device)
        self.criterion = nn.MSELoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        epoch_losses = []
        for epoch in range(self.epochs):
            batch_loss = 0.0
            trained_count = 0
            for seq in self.dataset:
                seq = [snap.to(self.device) for snap in seq]
                target_snap = seq[-1]
                
                # Unsupervised baseline learning: Train on normal traffic (label == 0)
                benign_mask = (target_snap.y == 0)
                if benign_mask.sum() == 0:
                    continue  # Skip if snapshot contains zero benign references

                optimizer.zero_grad()
                pred, target = self.model(seq)
                
                loss = self.criterion(pred[benign_mask], target[benign_mask])
                loss.backward()

                # Differential Privacy gradient norm clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.dp_clip)
                optimizer.step()
                
                batch_loss += loss.item()
                trained_count += 1

            avg_loss = batch_loss / max(1, trained_count)
            epoch_losses.append(avg_loss)

        # Extract weights
        updated_params = self.get_parameters(config={})

        # --- Adversarial Attack Simulation ---
        if self.attack == "sign_flip":
            print(f"[!] [ADVERSARY Client {self.client_id}] Executing Byzantine SIGN-FLIP Attack...")
            updated_params = [-1.0 * (p - initial_p) + initial_p for p, initial_p in zip(updated_params, parameters)]
        elif self.attack == "scale_poison":
            print(f"[!] [ADVERSARY Client {self.client_id}] Executing Byzantine SCALE-POISONING Attack (20x)...")
            updated_params = [20.0 * (p - initial_p) + initial_p for p, initial_p in zip(updated_params, parameters)]
        else:
            print(f"[Client {self.client_id}] Honest training complete. Loss: {epoch_losses[-1]:.4f}")

        return updated_params, len(self.dataset), {"loss": float(epoch_losses[-1])}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for seq in self.dataset:
                seq = [snap.to(self.device) for snap in seq]
                pred, target = self.model(seq)
                loss = self.criterion(pred, target)
                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(self.dataset))
        return float(avg_loss), len(self.dataset), {"eval_loss": float(avg_loss)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--attack", type=str, default="none", choices=["none", "sign_flip", "scale_poison"])
    args = parser.parse_args()

    data_file = f"data/processed/client_{args.client_id}_data.pt"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"[-] Data for client {args.client_id} not found at {data_file}")

    client = EdgeFlowerClient(
        client_id=args.client_id,
        data_path=data_file,
        epochs=args.epochs,
        lr=args.lr,
        attack=args.attack
    )
    
    print(f"[*] Launching Client {args.client_id} (Attack Mode: {args.attack.upper()}) on port {args.port}...")
    fl.client.start_client(
        server_address=f"127.0.0.1:{args.port}",
        client=client.to_client(),
    )

if __name__ == "__main__":
    main()