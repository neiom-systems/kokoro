#!/usr/bin/env python3
"""Generate Montreal Forced Aligner TextGrid alignments for the Luxembourgish corpus."""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from kokoro.training import TrainingConfig
from kokoro.training.extractors import (
    LuxembourgishPhonemizer,
    normalize_phoneme_tokens,
    read_metadata,
)


LOGGER = logging.getLogger("kokoro.scripts.generate_alignments")


def sanitize_word(word: str) -> Optional[str]:
    word = word.strip()
    if not word:
        return None
    word = word.replace("’", "'")
    match = re.match(r"[A-Za-zÀ-ÿÄÖÜäöüß'\\-]+$", word)
    if match is None:
        word = re.sub(r"[^A-Za-zÀ-ÿÄÖÜäöüß'\\-]", "", word)
    if not word:
        return None
    return word.lower()


def build_dictionary(
    entries: Iterable[Tuple[str, str, str]],
    phonemizer: LuxembourgishPhonemizer,
) -> OrderedDict[str, Sequence[str]]:
    lexicon: OrderedDict[str, Sequence[str]] = OrderedDict()
    for _, text, _ in entries:
        words = text.split()
        for raw_word in words:
            word = sanitize_word(raw_word)
            if not word or word in lexicon:
                continue
            phoneme_tokens = phonemizer(word)
            phonemes = normalize_phoneme_tokens(phoneme_tokens)
            if not phonemes:
                continue
            lexicon[word] = phonemes
    return lexicon


def write_dictionary(path: Path, lexicon: OrderedDict[str, Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for word, phonemes in lexicon.items():
            handle.write(f"{word} {' '.join(phonemes)}\n")


def prepare_corpus_split(
    entries: Iterable[Tuple[str, str, str]],
    audio_root: Path,
    corpus_root: Path,
    split_name: str,
) -> None:
    speaker_dir = corpus_root / split_name
    speaker_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, text, _ in entries:
        src_audio = audio_root / relative_path
        if not src_audio.exists():
            LOGGER.warning("Audio file missing for alignment: %s", src_audio)
            continue
        utt_id = Path(relative_path).stem
        dst_audio = speaker_dir / f"{utt_id}.wav"
        if not dst_audio.exists():
            try:
                os.symlink(src_audio.resolve(), dst_audio)
            except OSError:
                shutil.copy2(src_audio, dst_audio)
        lab_path = speaker_dir / f"{utt_id}.lab"
        lab_path.write_text(text.strip() + "\n", encoding="utf-8")


def run_mfa_align(
    mfa_executable: str,
    corpus_dir: Path,
    dictionary_path: Path,
    acoustic_model: str,
    output_dir: Path,
    num_jobs: int,
) -> None:
    cmd = [
        mfa_executable,
        "align",
        str(corpus_dir),
        str(dictionary_path),
        acoustic_model,
        str(output_dir),
        "--clean",
        "--overwrite",
        "--output_format",
        "long_textgrid",
    ]
    if num_jobs > 0:
        cmd.extend(["--num_jobs", str(num_jobs)])
    LOGGER.info("Running MFA command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MFA alignments for the Luxembourgish corpus")
    parser.add_argument("--config", type=Path, required=True, help="Training configuration TOML file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/luxembourgish_male_corpus/alignments"),
        help="Directory to write TextGrid files",
    )
    parser.add_argument(
        "--dictionary-path",
        type=Path,
        default=None,
        help="Optional path to save the generated pronunciation dictionary",
    )
    parser.add_argument(
        "--acoustic-model",
        default="german_mfa",
        help="Name or path of the MFA acoustic model to use",
    )
    parser.add_argument(
        "--mfa-executable",
        default=os.environ.get("MFA_BIN", "mfa"),
        help="Path to the MFA executable (default: mfa)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of MFA parallel jobs",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying audio into the corpus directory",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Optional path for the temporary MFA corpus (defaults to <output-dir>/corpus)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if shutil.which(args.mfa_executable) is None:
        LOGGER.error("MFA executable '%s' not found. Install Montreal Forced Aligner first.", args.mfa_executable)
        sys.exit(1)

    cfg = TrainingConfig.from_toml(args.config)
    cfg.paths.ensure_directories()

    train_rows = read_metadata(cfg.paths.train_csv)
    test_rows = read_metadata(cfg.paths.test_csv)
    all_rows = train_rows + test_rows

    phonemizer = LuxembourgishPhonemizer()
    lexicon = build_dictionary(all_rows, phonemizer)
    dictionary_path = args.dictionary_path or (args.output_dir / "dictionary.txt")
    write_dictionary(dictionary_path, lexicon)
    LOGGER.info("Wrote dictionary with %d entries to %s", len(lexicon), dictionary_path)

    corpus_dir = args.corpus_dir or (args.output_dir / "corpus")
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    prepare_corpus_split(train_rows, cfg.paths.train_csv.parent, corpus_dir, "train")
    prepare_corpus_split(test_rows, cfg.paths.test_csv.parent, corpus_dir, "test")

    output_dir = args.output_dir / "TextGrid"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_mfa_align(
            args.mfa_executable,
            corpus_dir,
            dictionary_path,
            args.acoustic_model,
            output_dir,
            args.num_workers,
        )
    except subprocess.CalledProcessError as exc:
        LOGGER.error("MFA alignment failed: %s", exc)
        sys.exit(exc.returncode or 1)

    LOGGER.info("TextGrids written to %s", output_dir)


if __name__ == "__main__":
    main()
