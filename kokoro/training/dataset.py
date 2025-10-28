"""Luxembourgish training dataset and collation utilities.

This module implements the design notes documented in this directory.  It
expects that phonemisation and acoustic features have been pre-computed and
cached on disk, but it can optionally fall back to a user-supplied extraction
callable for misses.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - numpy is required for .npz caches
    raise RuntimeError("numpy is required for loading cached Luxembourgish features") from exc

from .config import DataConfig

logger = logging.getLogger(__name__)

Pathish = Union[str, Path]

FEATURE_EXTENSION_CANDIDATES: Tuple[str, ...] = (".pt", ".pth", ".npz")
EXPECTED_MEL_BINS = 80


@dataclass(frozen=True)
class MetadataEntry:
    """Light-weight representation of a single utterance."""

    utt_id: str
    audio_path: Path
    text: str
    source: str


@dataclass
class CachedSample:
    """Container for cached training targets."""

    input_ids: torch.LongTensor
    durations: torch.LongTensor
    mel: torch.FloatTensor  # [n_mels, T]
    f0: torch.FloatTensor  # [T]
    uv: torch.FloatTensor  # [T] in {0, 1}


FeatureExtractor = Callable[[MetadataEntry], CachedSample]


def _derive_utt_id(path_value: str) -> str:
    path = Path(path_value)
    stem = path.stem
    if not stem:
        raise ValueError(f"Cannot derive utt_id from path '{path_value}'")
    return stem


def _build_cache_path(feature_root: Path, split: str, utt_id: str, extension: str) -> Path:
    return feature_root / split / f"{utt_id}{extension}"


def _load_from_npz(path: Path) -> CachedSample:
    data = np.load(path, allow_pickle=False)
    required = {"input_ids", "durations", "mel", "f0"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{path}: missing keys {sorted(missing)}")
    uv = data["uv"] if "uv" in data.files else None
    return CachedSample(
        input_ids=torch.as_tensor(data["input_ids"], dtype=torch.long),
        durations=torch.as_tensor(data["durations"], dtype=torch.long),
        mel=torch.as_tensor(data["mel"], dtype=torch.float32),
        f0=torch.as_tensor(data["f0"], dtype=torch.float32),
        uv=torch.as_tensor(data["uv"], dtype=torch.float32) if uv is not None else torch.zeros_like(torch.as_tensor(data["f0"], dtype=torch.float32)),
    )


def _load_from_pt(path: Path) -> CachedSample:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: expected mapping payload, found {type(payload).__name__}")
    for key in ("input_ids", "durations", "mel", "f0"):
        if key not in payload:
            raise KeyError(f"{path}: missing key '{key}'")
    uv = payload.get("uv")
    f0_tensor = torch.as_tensor(payload["f0"], dtype=torch.float32)
    uv_tensor = torch.as_tensor(uv, dtype=torch.float32) if uv is not None else (f0_tensor > 0).float()
    return CachedSample(
        input_ids=torch.as_tensor(payload["input_ids"], dtype=torch.long),
        durations=torch.as_tensor(payload["durations"], dtype=torch.long),
        mel=torch.as_tensor(payload["mel"], dtype=torch.float32),
        f0=f0_tensor,
        uv=uv_tensor,
    )


def load_cached_sample(path: Path) -> CachedSample:
    """Load a cached feature pack from disk."""

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return _load_from_pt(path)
    if suffix == ".npz":
        return _load_from_npz(path)
    raise ValueError(f"Unsupported cache extension '{suffix}' for {path}")


class LuxembourgishDataset(torch.utils.data.Dataset):
    """Torch dataset that serves Luxembourgish fine-tuning samples."""

    def __init__(
        self,
        metadata_csv: Pathish,
        feature_root: Pathish,
        split: str,
        data_config: DataConfig,
        *,
        feature_extractor: Optional[FeatureExtractor] = None,
        write_cache: bool = False,
        strict: bool = True,
    ) -> None:
        """
        Args:
            metadata_csv: Path to the metadata CSV file.
            feature_root: Root directory containing cached features.
            split: Sub-directory under feature_root (e.g., 'train', 'test').
            data_config: Data configuration parameters.
            feature_extractor: Optional callable to compute features on cache miss.
            write_cache: Whether to persist newly-computed features to disk.
            strict: If True, missing caches raise immediately when feature_extractor is None.
        """

        self.metadata_csv = Path(metadata_csv)
        self.feature_root = Path(feature_root)
        self.split = split
        self.data_config = data_config
        self.feature_extractor = feature_extractor
        self.write_cache = write_cache
        self.strict = strict

        if not self.metadata_csv.is_file():
            raise FileNotFoundError(f"metadata CSV not found: {self.metadata_csv}")

        self.entries: List[MetadataEntry] = self._load_metadata()
        if not self.entries:
            raise ValueError(f"No rows found in metadata CSV: {self.metadata_csv}")
        logger.info(
            "Loaded %d metadata entries for split '%s' from %s",
            len(self.entries),
            self.split,
            self.metadata_csv,
        )

        self.feature_dir = self.feature_root / split
        self.feature_dir.mkdir(parents=True, exist_ok=True)

        self._validate_cache_availability()

    def _load_metadata(self) -> List[MetadataEntry]:
        entries: List[MetadataEntry] = []
        with self.metadata_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                path_value = row.get("path")
                text = row.get("text")
                source = row.get("source", "")
                if path_value is None or text is None:
                    raise KeyError(f"Metadata row missing 'path' or 'text': {row}")
                utt_id = _derive_utt_id(path_value)
                audio_path = (self.metadata_csv.parent / "audio" / path_value).resolve()
                entries.append(MetadataEntry(utt_id=utt_id, audio_path=audio_path, text=text.strip(), source=source.strip()))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Metadata sample preview for '%s': %s",
                self.split,
                [e.utt_id for e in entries[:5]],
            )
        return entries

    def _validate_cache_availability(self) -> None:
        if self.feature_extractor is not None:
            return  # allow lazy generation
        missing: List[Path] = []
        for entry in self.entries:
            if not self._resolve_cache_path(entry).exists():
                missing.append(self._resolve_cache_path(entry))
        if missing and self.strict:
            sample = "\n  ".join(str(path) for path in missing[:5])
            more = "" if len(missing) <= 5 else f"\n  ... +{len(missing) - 5} more"
            raise FileNotFoundError(
                "Cached features not found for one or more utterances. "
                "Run the preprocessing pipeline first or provide a feature_extractor.\n  "
                f"{sample}{more}"
            )

    def _resolve_cache_path(self, entry: MetadataEntry) -> Path:
        for extension in FEATURE_EXTENSION_CANDIDATES:
            candidate = _build_cache_path(self.feature_root, self.split, entry.utt_id, extension)
            if candidate.exists():
                return candidate
        # Default to .pt for new caches
        return _build_cache_path(self.feature_root, self.split, entry.utt_id, ".pt")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        cache_path = self._resolve_cache_path(entry)
        if cache_path.exists():
            sample = load_cached_sample(cache_path)
            logger.debug("Loaded cached sample %s from %s", entry.utt_id, cache_path)
        else:
            if self.feature_extractor is None:
                raise FileNotFoundError(
                    f"Missing cached features for {entry.utt_id} ({cache_path}). "
                    "Provide feature_extractor or pre-compute caches."
                )
            sample = self.feature_extractor(entry)
            if self.write_cache:
                self._write_cache(cache_path, sample)

        self._validate_sample(entry, sample)
        seq_len_without_special = sample.input_ids.numel() - 2
        voice_row = max(0, min(seq_len_without_special - 1, 509))

        mel = sample.mel
        if mel.dim() != 2:
            raise ValueError(f"{entry.utt_id}: mel tensor must be 2D (n_mels, frames), got shape {tuple(mel.shape)}")
        # Ensure mel orientation [n_mels, frames]
        if mel.shape[0] != EXPECTED_MEL_BINS and mel.shape[1] == EXPECTED_MEL_BINS:
            mel = mel.transpose(0, 1).contiguous()

        mel_frames = mel.shape[1]
        durations = sample.durations
        if durations.dim() != 1:
            raise ValueError(f"{entry.utt_id}: durations must be 1D, got {durations.shape}")

        f0 = sample.f0
        uv = sample.uv
        if uv.shape != f0.shape:
            uv = (f0 > 0).float()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Prepared dataset item %s | text='%s' | phoneme_len=%d | frames=%d | voice_row=%d",
                entry.utt_id,
                entry.text,
                seq_len_without_special,
                mel_frames,
                voice_row,
            )
            logger.debug(
                "input_ids (first 12): %s",
                sample.input_ids[:12].tolist(),
            )

        return {
            "utt_id": entry.utt_id,
            "text": entry.text,
            "source": entry.source,
            "audio_path": entry.audio_path,
            "input_ids": sample.input_ids,
            "phoneme_len_no_special": seq_len_without_special,
            "durations": durations,
            "mel": mel,
            "mel_frames": mel_frames,
            "f0": f0,
            "uv": uv,
            "voice_row": voice_row,
        }

    def _write_cache(self, cache_path: Path, sample: CachedSample) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_ids": sample.input_ids.clone(),
            "durations": sample.durations.clone(),
            "mel": sample.mel.clone(),
            "f0": sample.f0.clone(),
            "uv": sample.uv.clone(),
        }
        torch.save(payload, cache_path)

    def _validate_sample(self, entry: MetadataEntry, sample: CachedSample) -> None:
        seq_len = sample.input_ids.numel()
        if seq_len < 3:
            raise ValueError(f"{entry.utt_id}: token sequence too short ({seq_len}); expected BOS+tokens+EOS")
        max_allowed = self.data_config.max_input_len + 2  # include BOS/EOS
        if seq_len > max_allowed:
            raise ValueError(
                f"{entry.utt_id}: sequence length {seq_len} exceeds configured maximum {max_allowed}. "
                "Chunk the text during preprocessing."
            )
        mel = sample.mel
        if mel.shape[0] != EXPECTED_MEL_BINS:
            if mel.shape[1] == EXPECTED_MEL_BINS:
                mel = mel.transpose(0, 1)
                sample.mel = mel
            else:
                raise ValueError(
                    f"{entry.utt_id}: expected {EXPECTED_MEL_BINS} mel bins, got {mel.shape[0]} (shape={tuple(mel.shape)})"
                )
        total_duration = int(sample.durations.sum().item())
        mel_frames = mel.shape[1]
        if total_duration != mel_frames:
            raise ValueError(
                f"{entry.utt_id}: duration sum ({total_duration}) does not equal mel frames ({mel_frames}). "
                "Re-run alignment or inspect cached features."
            )
        if sample.f0.numel() != mel_frames:
            raise ValueError(
                f"{entry.utt_id}: f0 length ({sample.f0.numel()}) mismatch with mel frames ({mel_frames})."
            )
        if sample.uv.numel() != mel_frames:
            sample.uv = (sample.f0 > 0).float()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Validated sample %s: phonemes=%d frames=%d duration_sum=%d voice_row=%d",
                entry.utt_id,
                seq_len - 2,
                mel_frames,
                total_duration,
                seq_len_without_special - 1,
            )

    def shuffled_indices(self, *, generator: Optional[torch.Generator] = None) -> List[int]:
        """Return deterministically shuffled indices using the provided generator."""

        indices = list(range(len(self.entries)))
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(0)
        perm = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in perm]


def luxembourgish_collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate function that pads and stacks Luxembourgish samples."""

    if not batch:
        raise ValueError("Cannot collate empty batch")

    batch_size = len(batch)
    max_seq_len = max(item["input_ids"].shape[0] for item in batch)
    max_dur_len = max(item["durations"].shape[0] for item in batch)
    max_frames = max(int(item["mel_frames"]) for item in batch)

    input_ids = torch.full((batch_size, max_seq_len), fill_value=0, dtype=torch.long)
    phoneme_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    durations = torch.zeros((batch_size, max_dur_len), dtype=torch.long)
    duration_mask = torch.zeros((batch_size, max_dur_len), dtype=torch.bool)
    mels = torch.zeros((batch_size, EXPECTED_MEL_BINS, max_frames), dtype=torch.float32)
    mel_mask = torch.zeros((batch_size, max_frames), dtype=torch.bool)
    f0 = torch.zeros((batch_size, max_frames), dtype=torch.float32)
    uv = torch.zeros((batch_size, max_frames), dtype=torch.float32)
    voice_rows = torch.zeros(batch_size, dtype=torch.long)
    phoneme_lengths = torch.zeros(batch_size, dtype=torch.long)

    texts: List[str] = []
    utt_ids: List[str] = []
    sources: List[str] = []
    audio_paths: List[Path] = []

    for idx, item in enumerate(batch):
        seq = item["input_ids"]
        seq_len = seq.shape[0]
        input_ids[idx, :seq_len] = seq
        phoneme_mask[idx, :seq_len] = True
        phoneme_lengths[idx] = seq_len

        dur = item["durations"]
        dur_len = dur.shape[0]
        durations[idx, :dur_len] = dur
        duration_mask[idx, :dur_len] = True

        mel = item["mel"]
        frames = mel.shape[1]
        mels[idx, :, :frames] = mel
        mel_mask[idx, :frames] = True

        f0_vec = item["f0"]
        uv_vec = item["uv"]
        f0[idx, :frames] = f0_vec
        uv[idx, :frames] = uv_vec

        voice_rows[idx] = item["voice_row"]

        texts.append(item["text"])
        utt_ids.append(item["utt_id"])
        sources.append(item["source"])
        audio_paths.append(item["audio_path"])

    return {
        "input_ids": input_ids,
        "phoneme_mask": phoneme_mask,
        "phoneme_lengths": phoneme_lengths,
        "durations": durations,
        "duration_mask": duration_mask,
        "mel": mels,
        "mel_mask": mel_mask,
        "f0": f0,
        "uv": uv,
        "voice_rows": voice_rows,
        "texts": texts,
        "utt_ids": utt_ids,
        "sources": sources,
        "audio_paths": audio_paths,
    }


__all__ = [
    "LuxembourgishDataset",
    "luxembourgish_collate",
    "load_cached_sample",
    "CachedSample",
    "MetadataEntry",
]
