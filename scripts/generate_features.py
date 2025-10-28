#!/usr/bin/env python3
"""Generate cached acoustic features for Kokoro fine-tuning."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from kokoro.training import (
    FeatureExtractionConfig,
    FeatureExtractor,
    TrainingConfig,
    run_split_extraction,
)


def configure_feature_extractor(cfg: TrainingConfig, require_alignments: bool) -> FeatureExtractor:
    feature_cfg = FeatureExtractionConfig(require_alignments=require_alignments)
    feature_cfg.max_phoneme_tokens = cfg.data.max_input_len
    feature_cfg.mel.sample_rate = cfg.data.sample_rate
    feature_cfg.mel.hop_length = cfg.data.hop_length
    feature_cfg.vocab_path = cfg.paths.config_json

    try:
        with open(cfg.paths.config_json, "r", encoding="utf-8") as handle:
            model_cfg = json.load(handle)
        istft_cfg = model_cfg.get("istftnet", {})
        feature_cfg.mel.n_fft = istft_cfg.get("gen_istft_n_fft", feature_cfg.mel.n_fft)
        feature_cfg.mel.win_length = istft_cfg.get("gen_istft_n_fft", feature_cfg.mel.win_length)
        feature_cfg.mel.mel_fmin = model_cfg.get("mel_fmin", feature_cfg.mel.mel_fmin)
        feature_cfg.mel.mel_fmax = model_cfg.get("mel_fmax", feature_cfg.mel.mel_fmax)
        feature_cfg.mel.n_mels = model_cfg.get("n_mels", feature_cfg.mel.n_mels)
    except FileNotFoundError:
        logging.warning("config.json not found at %s; using default mel settings", cfg.paths.config_json)

    return FeatureExtractor(feature_cfg)


def determine_alignment_root(base: Path | None, split: str) -> Path | None:
    if base is None:
        return None
    if (base / split).exists():
        return base / split
    if base.exists():
        return base
    logging.warning("Alignment directory %s does not exist; skipping alignments for %s", base, split)
    return None


def extract_for_split(
    *,
    split: str,
    cfg: TrainingConfig,
    extractor: FeatureExtractor,
    alignment_root: Path | None,
    force: bool,
) -> None:
    if split == "train":
        metadata_csv = cfg.paths.train_csv
    elif split == "test":
        metadata_csv = cfg.paths.test_csv
    else:
        raise ValueError(f"Unknown split '{split}'")

    audio_root = metadata_csv.parent
    output_root = cfg.paths.feature_root / split
    summary = run_split_extraction(
        metadata_csv=metadata_csv,
        audio_root=audio_root,
        alignment_root=alignment_root,
        output_root=output_root,
        extractor=extractor,
        skip_existing=not force,
    )
    logging.info(
        "Completed feature extraction for %s: processed=%d, skipped=%d",
        split,
        len(summary.processed),
        len(summary.skipped),
    )
    if summary.skipped:
        logging.warning("Failed items (%s): %s", split, summary.skipped[:5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached features for Kokoro fine-tuning")
    parser.add_argument("--config", type=Path, required=True, help="Training configuration TOML file")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "test"),
        help="Dataset splits to process (default: train test)",
    )
    parser.add_argument(
        "--alignment-root",
        type=Path,
        default=None,
        help="Optional directory containing TextGrid alignments (per split or flat)",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if cache files exist")
    parser.add_argument(
        "--require-alignments",
        action="store_true",
        help="Fail if alignments are missing instead of using uniform durations",
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

    require_alignments = args.require_alignments or args.alignment_root is not None
    extractor = configure_feature_extractor(cfg, require_alignments=require_alignments)

    for split in args.splits:
        alignment_root = determine_alignment_root(args.alignment_root, split)
        extract_for_split(
            split=split,
            cfg=cfg,
            extractor=extractor,
            alignment_root=alignment_root,
            force=args.force,
        )


if __name__ == "__main__":
    main()
