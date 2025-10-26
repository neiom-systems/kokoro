#!/usr/bin/env python3

"""
Check alignment between CSV metadata and audio files in a dataset.

Verifies:
- All audio files referenced in metadata exist
- Audio files are readable and valid
- Optional: Check audio duration and other properties
- Reports statistics and any mismatches
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARNING] soundfile not available. Audio validation will be limited.", file=sys.stderr)


def check_audio_file(audio_path: Path, check_content: bool = True) -> dict[str, Any]:
    """Check if an audio file exists and is valid."""
    result = {
        "exists": False,
        "readable": False,
        "duration": None,
        "sample_rate": None,
        "channels": None,
        "error": None,
    }
    
    if not audio_path.exists():
        result["error"] = "File does not exist"
        return result
    
    result["exists"] = True
    
    if not audio_path.is_file():
        result["error"] = "Path is not a file"
        return result
    
    if not check_content or not AUDIO_AVAILABLE:
        result["readable"] = True
        return result
    
    try:
        info = sf.info(audio_path)
        result["readable"] = True
        result["duration"] = info.duration
        result["sample_rate"] = info.samplerate
        result["channels"] = info.channels
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_split(corpus_dir: Path, split_name: str, check_content: bool = True) -> dict[str, Any]:
    """Check alignment for a single split (train/test)."""
    split_dir = corpus_dir / split_name
    metadata_path = split_dir / "metadata.csv"
    audio_dir = split_dir / "audio"
    
    if not split_dir.exists():
        return {"error": f"Split directory {split_name} does not exist"}
    
    if not metadata_path.exists():
        return {"error": f"Metadata file {metadata_path} does not exist"}
    
    if not audio_dir.exists():
        return {"error": f"Audio directory {audio_dir} does not exist"}
    
    # Read metadata
    samples = []
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    
    # Check each sample
    missing_files = []
    broken_files = []
    stats = {
        "total_samples": len(samples),
        "valid_samples": 0,
        "missing_audio": 0,
        "broken_audio": 0,
        "audio_files": 0,
        "total_duration": 0.0,
        "min_duration": float("inf"),
        "max_duration": 0.0,
    }
    
    for idx, sample in enumerate(samples, 1):
        audio_path = split_dir / sample["path"]
        check_result = check_audio_file(audio_path, check_content=check_content)
        
        if not check_result["exists"]:
            missing_files.append((idx, sample["path"], sample.get("text", "")[:50]))
            stats["missing_audio"] += 1
        elif check_result["error"]:
            broken_files.append((idx, sample["path"], check_result["error"]))
            stats["broken_audio"] += 1
        else:
            stats["valid_samples"] += 1
            if check_result["duration"] is not None:
                stats["audio_files"] += 1
                stats["total_duration"] += check_result["duration"]
                stats["min_duration"] = min(stats["min_duration"], check_result["duration"])
                stats["max_duration"] = max(stats["max_duration"], check_result["duration"])
    
    # Find audio files not referenced in metadata
    metadata_files = {sample["path"] for sample in samples}
    audio_files_on_disk = set(f"audio/{f.name}" for f in audio_dir.glob("*.wav"))
    unreferenced_files = sorted(audio_files_on_disk - metadata_files)
    
    return {
        "split": split_name,
        "stats": stats,
        "missing_files": missing_files,
        "broken_files": broken_files,
        "unreferenced_files": unreferenced_files,
    }


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds == float("inf") or seconds == 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.2f}s"


def print_report(results: list[dict[str, Any]]) -> None:
    """Print a detailed report of the alignment check."""
    print("\n" + "=" * 80)
    print("DATASET ALIGNMENT REPORT")
    print("=" * 80)
    
    for result in results:
        if "error" in result:
            print(f"\n[ERROR] {result['error']}")
            continue
        
        split = result["split"]
        stats = result["stats"]
        
        print(f"\n{split.upper()} Split:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Valid samples: {stats['valid_samples']}")
        print(f"  Missing audio: {stats['missing_audio']}")
        print(f"  Broken audio: {stats['broken_audio']}")
        
        if stats["audio_files"] > 0:
            avg_duration = stats["total_duration"] / stats["audio_files"]
            print(f"\n  Audio Statistics:")
            print(f"    Files checked: {stats['audio_files']}")
            print(f"    Total duration: {format_duration(stats['total_duration'])}")
            print(f"    Average duration: {format_duration(avg_duration)}")
            print(f"    Min duration: {format_duration(stats['min_duration'])}")
            print(f"    Max duration: {format_duration(stats['max_duration'])}")
        
        # Report missing files
        if result["missing_files"]:
            print(f"\n  Missing Files ({len(result['missing_files'])}):")
            for idx, path, text in result["missing_files"][:10]:
                print(f"    Row {idx}: {path}")
                print(f"      Text: {text}...")
            if len(result["missing_files"]) > 10:
                print(f"    ... and {len(result['missing_files']) - 10} more")
        
        # Report broken files
        if result["broken_files"]:
            print(f"\n  Broken Files ({len(result['broken_files'])}):")
            for idx, path, error in result["broken_files"][:10]:
                print(f"    Row {idx}: {path}")
                print(f"      Error: {error}")
            if len(result["broken_files"]) > 10:
                print(f"    ... and {len(result['broken_files']) - 10} more")
        
        # Report unreferenced files
        if result["unreferenced_files"]:
            print(f"\n  Unreferenced Audio Files ({len(result['unreferenced_files'])}):")
            for path in result["unreferenced_files"][:10]:
                print(f"    {path}")
            if len(result["unreferenced_files"]) > 10:
                print(f"    ... and {len(result['unreferenced_files']) - 10} more")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_samples = sum(r["stats"]["total_samples"] for r in results if "stats" in r)
    total_valid = sum(r["stats"]["valid_samples"] for r in results if "stats" in r)
    total_missing = sum(r["stats"]["missing_audio"] for r in results if "stats" in r)
    total_broken = sum(r["stats"]["broken_audio"] for r in results if "stats" in r)
    
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {total_valid}")
    print(f"Missing audio: {total_missing}")
    print(f"Broken audio: {total_broken}")
    
    if total_samples > 0:
        success_rate = (total_valid / total_samples) * 100
        print(f"\nSuccess rate: {success_rate:.2f}%")
    
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check alignment between CSV metadata and audio files."
    )
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Path to the corpus directory containing train/ and test/ splits",
    )
    parser.add_argument(
        "--no-content-check",
        action="store_true",
        help="Skip audio content validation (faster, only checks file existence)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Which splits to check (default: train test)",
    )
    
    args = parser.parse_args()
    
    corpus_dir = args.corpus_dir
    if not corpus_dir.exists():
        print(f"[ERROR] Corpus directory does not exist: {corpus_dir}", file=sys.stderr)
        return 1
    
    results = []
    for split in args.splits:
        print(f"Checking {split} split...", file=sys.stderr)
        result = check_split(corpus_dir, split, check_content=not args.no_content_check)
        results.append(result)
    
    print_report(results)
    
    # Return error code if there are issues
    total_issues = sum(
        r["stats"]["missing_audio"] + r["stats"]["broken_audio"]
        for r in results
        if "stats" in r
    )
    
    return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

