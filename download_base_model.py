#!/usr/bin/env python3
"""Download all files from hexgrad/Kokoro-82M to base_model directory."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get HF token from environment (optional for public models)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    hf_token = hf_token.strip('"').strip("'")
    print(f"Using HF token: {hf_token[:10]}...")
else:
    print("No HF_TOKEN found, attempting public download...")

from huggingface_hub import snapshot_download

print("Downloading Kokoro-82M base model files...")

repo_id = "hexgrad/Kokoro-82M"
local_dir = Path("base_model")
local_dir.mkdir(exist_ok=True)

print(f"Downloading to: {local_dir.absolute()}")
print(f"Repository: {repo_id}")

try:
    files = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        token=hf_token,
        ignore_patterns=["*.md", "*.txt"],  # Skip markdown files
    )
except Exception as e:
    if "401" in str(e) and hf_token:
        print(f"Authentication failed with provided token. Error: {e}")
        print("Try running: huggingface-cli login")
        raise
    elif "401" in str(e):
        print(f"Model requires authentication. Error: {e}")
        print("Please set HF_TOKEN environment variable or run: huggingface-cli login")
        raise
    else:
        raise

print("\nDownload complete!")
print(f"\nFiles downloaded to: {local_dir.absolute()}")
print("\nContents:")
for item in sorted(local_dir.iterdir()):
    size = item.stat().st_size / (1024 * 1024)  # Size in MB
    print(f"  - {item.name} ({size:.2f} MB)")
