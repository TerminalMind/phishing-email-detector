"""
Download the FERB benchmark used for the main training dataset.
"""

import os
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(BASE, "data", "emails_large.csv")
URL = "https://zenodo.org/records/20055767/files/ferb.csv?download=1"

def main():
    print("Downloading FERB dataset...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to: {DEST}")

if __name__ == "__main__":
    main()
