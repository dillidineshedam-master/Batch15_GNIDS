import argparse
import flwr as fl
import yaml
from core.partitioner import get_non_iid_subnets
from federated.client_manager import GNIDSEdgeClient
from config.logging_config import get_logger

logger = get_logger("ClientRunner")

def main():
    parser = argparse.ArgumentParser(description="GNIDS Federated Client Runner")
    parser.add_argument("--client_id", type=int, default=1, help="Client Subnet Partition ID (1 or 2)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Target Server IP:Port")
    args = parser.parse_args()

    c1_snaps, c2_snaps, _ = get_non_iid_subnets("config/config.yaml")
    local_snapshots = c1_snaps if args.client_id == 1 else c2_snaps

    print("=" * 65)
    print(f"?? STARTING GNIDS EDGE CLIENT #{args.client_id}")
    print(f"?? Target Aggregator : {args.server}")
    print(f"?? Local Subnet Data : {len(local_snapshots)} Temporal Snapshots")
    print("=" * 65)

    client = GNIDSEdgeClient(local_snapshots, client_id=args.client_id, config_path="config/config.yaml")
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()
