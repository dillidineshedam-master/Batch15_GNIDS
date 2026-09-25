import os
import urllib.request
import ssl
import pandas as pd
import sys

print("=" * 60)
print("📥 DOWNLOADING REAL IEEE BENCHMARK DATASET: UNSW-NB15")
print("=" * 60)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "unsw_nb15_benchmark.csv")

# Active academic and Hugging Face mirror endpoints
MIRRORS = [
    "https://huggingface.co/datasets/AI-R-and-D/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
    "https://raw.githubusercontent.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/master/data/UNSW-NB15/UNSW_NB15_training-set.csv",
    "https://huggingface.co/datasets/Hrel/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv"
]

# Configure SSL context to prevent handshake blocks on Windows
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

download_success = False

for idx, url in enumerate(MIRRORS, 1):
    print(f"⏳ Connecting to Mirror {idx}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=45) as response, open(DATA_FILE, 'wb') as out_file:
            total_size = 0
            while True:
                chunk = response.read(1024 * 1024) # 1MB buffer
                if not chunk:
                    break
                out_file.write(chunk)
                total_size += len(chunk)
                print(f"\r   Downloaded: {total_size / (1024 * 1024):.2f} MB", end="")
        
        print()
        if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 500000:
            print(f"✅ Download completed successfully from Mirror {idx}!")
            download_success = True
            break
        else:
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
    except Exception as e:
        print(f"\n⚠️ Mirror {idx} error: {e}")
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)

if not download_success:
    print("\n❌ Automated download was blocked by network firewall.")
    print("👉 Manual Alternative: Download 'UNSW_NB15_training-set.csv' from Kaggle or Hugging Face and place it inside the 'data/' folder as 'unsw_nb15_benchmark.csv'.")
    sys.exit(1)

# Verify dataset structure and flow labels
df = pd.read_csv(DATA_FILE)
print("\n" + "=" * 60)
print(f"📊 BENCHMARK DATASET LOADED: {df.shape[0]:,} Rows, {df.shape[1]} Feature Columns")
print(f"🔍 Class Distribution:")
print(f"   - Normal Traffic  (Class 0): {(df['label'] == 0).sum():,} flows")
print(f"   - Intrusion Flows (Class 1): {(df['label'] == 1).sum():,} flows")
if "attack_cat" in df.columns:
    print(f"\n🛡️ Real Attack Categories Present:\n{df['attack_cat'].value_counts().to_string()}")
print("=" * 60)