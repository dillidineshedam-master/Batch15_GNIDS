import flwr as fl
import yaml
from federated.server_aggregator import FederatedServerManager
from federated.network_utils import get_local_ip
from config.logging_config import get_logger

logger = get_logger("ServerRunner")

def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local_ip = get_local_ip()
    port = cfg["federated"]["server_address"].split(":")[-1]
    bind_addr = f"0.0.0.0:{port}"

    server_mgr = FederatedServerManager("config/config.yaml")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg["federated"]["fraction_fit"],
        fraction_evaluate=1.0,
        min_fit_clients=cfg["federated"]["min_fit_clients"],
        min_evaluate_clients=cfg["federated"]["min_eval_clients"],
        min_available_clients=cfg["federated"]["min_fit_clients"],
        evaluate_fn=server_mgr.evaluate_fn
    )

    print("=" * 65)
    print("LAUNCHING GNIDS CENTRAL FEDERATED AGGREGATOR")
    print(f"Network Interface : {local_ip}:{port}")
    print(f"Rounds Configured : {cfg['federated']['num_rounds']}")
    print(f"DP Mechanism      : {'ENABLED' if cfg['federated']['dp_enabled'] else 'DISABLED'}")
    print("=" * 65)

    fl.server.start_server(
        server_address=bind_addr,
        config=fl.server.ServerConfig(num_rounds=cfg["federated"]["num_rounds"]),
        strategy=strategy
    )

    server_mgr.export_results()

if __name__ == "__main__":
    main()
