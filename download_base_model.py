#!/usr/bin/env python3
"""Download all files from hexgrad/Kokoro-82M to base_model directory."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get HF token from environment
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is required to download models")

# Remove quotes if present
hf_token = hf_token.strip('"').strip("'")

from huggingface_hub import snapshot_download

print("Downloading Kokoro-82M base model files...")
print(f"Token: {hf_token[:10]}...")

repo_id = "hexgrad/Kokoro-82M"
local_dir = Path("base_model")
local_dir.mkdir(exist_ok=True)

print(f"Downloading to: {local_dir.absolute()}")
print(f"Repository: {repo_id}")

files = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=str(local_dir),
    token=hf_token,
    ignore_patterns=["*.md", "*.txt"],  # Skip markdown files
)

print("\nDownload complete!")
print(f"\nFiles downloaded to: {local_dir.absolute()}")
print("\nContents:")
for item in sorted(local_dir.iterdir()):
    size = item.stat().st_size / (1024 * 1024)  # Size in MB
    print(f"  - {item.name} ({size:.2f} MB)")
