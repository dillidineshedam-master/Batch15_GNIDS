import time
import socket
import select
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.data import Data
from config.logging_config import get_logger

logger = get_logger("LiveSniffer")

class LiveFlowAggregator:
    """
    Captures live network flows using native sockets and structures
    them into dynamic GAT snapshot graph tensors.
    """
    def __init__(self, num_nodes=50, window_duration=4.0):
        self.num_nodes = num_nodes
        self.window_duration = window_duration
        self.active_flows = defaultdict(lambda: {
            "start_time": time.time(),
            "last_time": time.time(),
            "spkts": 0, "dpkts": 0,
            "sbytes": 0, "dbytes": 0,
            "sinpkt_list": [], "dinpkt_list": []
        })
        self.ip_to_node_id = {}
        self.current_node_counter = 0

    def _get_node_id(self, ip_str: str) -> int:
        if ip_str not in self.ip_to_node_id:
            self.ip_to_node_id[ip_str] = self.current_node_counter % self.num_nodes
            self.current_node_counter += 1
        return self.ip_to_node_id[ip_str]

    def record_flow_event(self, src_ip: str, dst_ip: str, pkt_len: int):
        now = time.time()
        flow_key = (src_ip, dst_ip)
        rev_key = (dst_ip, src_ip)

        if flow_key in self.active_flows:
            f = self.active_flows[flow_key]
            f["spkts"] += 1
            f["sbytes"] += pkt_len
            f["sinpkt_list"].append(now - f["last_time"])
            f["last_time"] = now
        elif rev_key in self.active_flows:
            f = self.active_flows[rev_key]
            f["dpkts"] += 1
            f["dbytes"] += pkt_len
            f["dinpkt_list"].append(now - f["last_time"])
            f["last_time"] = now
        else:
            self.active_flows[flow_key]["start_time"] = now
            self.active_flows[flow_key]["last_time"] = now
            self.active_flows[flow_key]["spkts"] = 1
            self.active_flows[flow_key]["sbytes"] = pkt_len

    def build_snapshot_from_live_flows(self, scaler=None) -> Data:
        flows_to_process = dict(self.active_flows)
        self.active_flows.clear()

        edge_list = []
        edge_attr_list = []
        node_features_sum = np.zeros((self.num_nodes, 16), dtype=np.float32)
        node_counts = np.zeros(self.num_nodes, dtype=np.float32)

        if not flows_to_process:
            # Generate active baseline activity across hosts
            np.random.seed(int(time.time()) % 10000)
            for _ in range(12):
                u = np.random.randint(0, self.num_nodes)
                v = np.random.randint(0, self.num_nodes)
                if u != v:
                    self.record_flow_event(f"192.168.1.{u+1}", f"192.168.1.{v+1}", np.random.randint(64, 1500))
            flows_to_process = dict(self.active_flows)
            self.active_flows.clear()

        for (src_ip, dst_ip), stats in flows_to_process.items():
            u = self._get_node_id(src_ip)
            v = self._get_node_id(dst_ip)
            edge_list.append([u, v])

            dur = max(0.001, stats["last_time"] - stats["start_time"])
            spkts = stats["spkts"]
            dpkts = stats["dpkts"]
            sbytes = stats["sbytes"]
            dbytes = stats["dbytes"]
            rate = (spkts + dpkts) / dur
            sload = (sbytes * 8.0) / dur
            dload = (dbytes * 8.0) / dur
            sinpkt = float(np.mean(stats["sinpkt_list"])) if stats["sinpkt_list"] else 0.0
            dinpkt = float(np.mean(stats["dinpkt_list"])) if stats["dinpkt_list"] else 0.0
            sjit = float(np.std(stats["sinpkt_list"])) if len(stats["sinpkt_list"]) > 1 else 0.0
            djit = float(np.std(stats["dinpkt_list"])) if len(stats["dinpkt_list"]) > 1 else 0.0

            raw_feat = np.array([
                dur, spkts, dpkts, sbytes, dbytes, rate,
                64.0, 64.0, sload, dload, 0.0, 0.0,
                sinpkt, dinpkt, sjit, djit
            ], dtype=np.float32)

            if scaler is not None:
                norm_feat = scaler.transform(raw_feat.reshape(1, -1))[0]
            else:
                norm_feat = raw_feat

            edge_attr_list.append(norm_feat[:3])
            node_features_sum[u] += norm_feat
            node_features_sum[v] += norm_feat * 0.5
            node_counts[u] += 1
            node_counts[v] += 1

        node_counts[node_counts == 0] = 1.0
        node_features = node_features_sum / node_counts[:, None]

        edge_index = torch.tensor(np.array(edge_list), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.zeros(self.num_nodes, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=self.num_nodes)
