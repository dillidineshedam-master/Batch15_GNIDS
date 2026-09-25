import os
import sys
import time
import subprocess

def run_command(command, desc):
    print(f"\n{'='*75}")
    print(f"[*] RUNNING: {desc}")
    print(f"[*] Command: {command}")
    print(f"{'='*75}")
    t0 = time.time()
    result = subprocess.run(command, shell=True)
    t1 = time.time()
    if result.returncode != 0:
        print(f"[-] ERROR: Step failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[+] COMPLETED: {desc} in {t1 - t0:.2f} seconds.")

def run_federated_training(rounds=8):
    print(f"\n{'='*75}")
    print(f"[*] LAUNCHING FEDERATED TRAINING (Flower Server + 2 Clients, {rounds} Rounds)")
    print(f"{'='*75}")
    
    server_cmd = [sys.executable, "src/server.py", "--rounds", str(rounds), "--strategy", "acs"]
    c0_cmd = [sys.executable, "src/client.py", "--client_id", "0"]
    c1_cmd = [sys.executable, "src/client.py", "--client_id", "1"]

    # Launch server
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(2)  # Allow gRPC port 8080 to bind

    # Launch clients concurrently
    c0_proc = subprocess.Popen(c0_cmd)
    c1_proc = subprocess.Popen(c1_cmd)

    # Await completion
    server_proc.wait()
    c0_proc.wait()
    c1_proc.wait()

    if server_proc.returncode != 0:
        print("[-] Federated training failed.")
        sys.exit(1)
    print(f"[+] Federated training completed successfully.")

def main():
    print("="*75)
    print("      FED-GNIDS END-TO-END AUTOMATED RESEARCH PIPELINE")
    print("="*75)
    start_time = time.time()

    # Step 1: Preprocess raw data if processed files do not exist
    if not os.path.exists("data/processed/real_test_benchmark.pt"):
        run_command(f"{sys.executable} src/preprocess.py", "Data Preprocessing & Non-IID Graph Slicing")
    else:
        print("[+] Preprocessed partitions verified. Skipping preprocessing.")

    # Step 2: Federated Training with ACS
    run_federated_training(rounds=8)

    # Step 3: Pure Evaluation on Official Test Benchmark
    run_command(f"{sys.executable} src/evaluate.py", "Benchmark Evaluation & Metric Extraction")

    # Step 4: Baseline Benchmarks (RF, Local Silo, Static GCN)
    run_command(f"{sys.executable} src/baselines.py", "Comparative Baselines Execution")

    # Step 5: Byzantine Adversarial Robustness Simulation
    run_command(f"{sys.executable} src/simulate_attacks.py", "Byzantine Poisoning Simulation (K=4, f=25%)")

    # Step 6: Architectural Ablation Study
    run_command(f"{sys.executable} src/ablation.py", "Modular Ablation Study")

    # Step 7: Computational Profiling
    run_command(f"{sys.executable} src/profiler.py", "Hardware & Network Profiling")

    elapsed = time.time() - start_time
    print("\n" + "="*75)
    print(f"[SUCCESS] Complete research pipeline executed in {elapsed / 60:.2f} minutes.")
    print("[+] All empirical figures and CSV tables are updated in results/")
    print("="*75)

if __name__ == "__main__":
    main()