import os
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    print(f"[*] Downloading dataset from: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print(f"[+] Download complete: {output_path}")

def main():
    target_dir = "data/raw"
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "UNSW_NB15_training-set.csv")

    # Primary reliable mirror for the official 175,341-flow benchmark split
    dataset_url = "https://huggingface.co/datasets/alireza-k/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv"

    if os.path.exists(target_file) and os.path.getsize(target_file) > 10 * 1024 * 1024:
        print(f"[!] Real dataset already exists ({os.path.getsize(target_file) / (1024*1024):.2f} MB). Skipping download.")
        return

    try:
        download_file(dataset_url, target_file)
    except Exception as e:
        print(f"[-] Automated download failed: {e}")
        print("\n[MANUAL ALTERNATIVE]")
        print("1. Download 'UNSW_NB15_training-set.csv' directly from:")
        print("   https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
        print(f"2. Place the downloaded CSV directly into: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    main()