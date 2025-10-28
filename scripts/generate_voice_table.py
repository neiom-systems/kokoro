#!/usr/bin/env python3
"""Generate a Luxembourgish voice table tensor for Kokoro fine-tuning."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List

from kokoro.training import (
    TableGenerationConfig,
    TrainingConfig,
    VoiceTableArtifacts,
    generate_voice_table,
    save_voice_table,
)


def load_audio_paths(metadata_csv: Path, audio_dir: Path, limit: int) -> List[Path]:
    paths: List[Path] = []
    with metadata_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = row.get("path")
            if not rel:
                continue
            audio_path = audio_dir / rel
            if audio_path.exists():
                paths.append(audio_path.resolve())
            if len(paths) >= limit:
                break
    if not paths:
        raise RuntimeError(f"No audio files found in {audio_dir}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a voice table tensor for Kokoro")
    parser.add_argument("--config", type=Path, required=True, help="Path to training TOML config")
    parser.add_argument(
        "--num-clips",
        type=int,
        default=64,
        help="Number of training clips to average for the speaker embedding",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for voice table output path",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    cfg = TrainingConfig.from_toml(args.config)
    cfg.paths.ensure_directories()

    metadata_csv = cfg.paths.train_csv
    audio_dir = metadata_csv.parent
    audio_paths = load_audio_paths(metadata_csv, audio_dir, args.num_clips)
    logging.info("Collected %d audio clips for voice embedding", len(audio_paths))

    artifacts: VoiceTableArtifacts = generate_voice_table(audio_paths, TableGenerationConfig())

    output_path = args.output or cfg.paths.voice_export_path
    save_voice_table(artifacts, output_path)
    logging.info("Voice table saved to %s", output_path)


if __name__ == "__main__":
    main()
