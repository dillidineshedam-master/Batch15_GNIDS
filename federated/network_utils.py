import socket
from config.logging_config import get_logger

logger = get_logger("NetworkUtils")

def get_local_ip():
    """Detects local LAN/Wi-Fi IPv4 address for multi-machine federated binding."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to public DNS to determine default network interface
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    ip = get_local_ip()
    print(f"?? Discovered Local Network IP: {ip}")
