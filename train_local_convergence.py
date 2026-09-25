import subprocess
import time
import sys

print("=" * 60)
print("STARTING 15-ROUND CONVERGENCE RUN (LOCAL TESTBED)")
print("=" * 60)

# 1. Start Server
server_proc = subprocess.Popen([sys.executable, "run_server.py"])
time.sleep(3)

# 2. Start Client 1
c1_proc = subprocess.Popen([sys.executable, "run_client.py", "--client_id", "1", "--server", "127.0.0.1:8080"])
time.sleep(1)

# 3. Start Client 2
c2_proc = subprocess.Popen([sys.executable, "run_client.py", "--client_id", "2", "--server", "127.0.0.1:8080"])

# Wait for server to finish 15 rounds
server_proc.wait()
c1_proc.wait()
c2_proc.wait()

print("\n" + "=" * 60)
print("TRAINING COMPLETE. EVALUATING CONVERGED CHECKPOINT...")
print("=" * 60)
