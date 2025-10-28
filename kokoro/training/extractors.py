"""Offline feature extraction utilities for Luxembourgish fine-tuning.

This module implements the design notes recorded in this directory.  It focuses on
idempotent preprocessing so that expensive steps (phonemisation, alignment, F0 and
mel extraction) can be cached once and consumed by the training pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:  # Optional dependencies used when available
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - soundfile required for audio loading
    raise RuntimeError("soundfile is required for feature extraction") from exc

try:
    import librosa
except ImportError as exc:  # pragma: no cover - librosa required for mel/F0 fallbacks
    raise RuntimeError("librosa is required for feature extraction") from exc

try:
    import pyworld  # type: ignore
except ImportError:  # pragma: no cover - allow fallback to librosa.pyin
    pyworld = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

logger = logging.getLogger(__name__)

PHONEME_SUBSTITUTIONS: Dict[str, Sequence[str]] = {
    "ʀ": ("r",),
    "ɐ": ("ə",),
    "❓": tuple(),
}

_TORCHAUDIO_TRANSFORMS: Dict[Tuple[int, int, int, int, int, float, float, str], "torchaudio.transforms.MelSpectrogram"] = {}


def _map_symbol(symbol: str) -> Iterable[str]:
    mapping = PHONEME_SUBSTITUTIONS.get(symbol)
    if mapping is None:
        return (symbol,)
    return mapping


def normalize_phoneme_tokens(tokens: Sequence[str]) -> List[str]:
    symbols: List[str] = []
    for token in tokens:
        if not token:
            continue
        normalized = unicodedata.normalize("NFD", token)
        for ch in normalized:
            if unicodedata.category(ch) == "Mn":
                continue
            symbols.append(ch)
    phonemes: List[str] = []
    for symbol in symbols:
        for mapped in _map_symbol(symbol):
            if mapped:
                phonemes.append(mapped)
    return phonemes


@dataclass(slots=True)
class MelConfig:
    """STFT/mel parameters pulled from the base Kokoro configuration."""

    sample_rate: int = 24_000
    n_fft: int = 1024
    hop_length: int = 240
    win_length: int = 1024
    n_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 12_000.0
    preemphasis: Optional[float] = 0.97


@dataclass(slots=True)
class F0Config:
    method: str = "pyworld"
    min_f0: float = 60.0
    max_f0: float = 700.0
    voicing_threshold: float = 0.6


@dataclass(slots=True)
class AlignmentConfig:
    max_duration_frames: int = 50
    min_duration_frames: int = 1


@dataclass(slots=True)
class FeatureExtractionConfig:
    """Top-level configuration for preprocessing."""

    mel: MelConfig = field(default_factory=MelConfig)
    f0: F0Config = field(default_factory=F0Config)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    allow_frame_mismatch: bool = False
    max_phoneme_tokens: int = 510
    silence_trim_db: Optional[float] = None
    voice_table_rows: int = 510
    vocab_path: Path = Path("base_model/config.json")
    require_alignments: bool = True
    mel_device: Optional[str] = None
    mel_device: Optional[str] = None


@dataclass(slots=True)
class ExtractionResult:
    """Container used to persist per-utterance features."""

    mel: torch.FloatTensor
    durations: torch.LongTensor
    f0: torch.FloatTensor
    uv: torch.FloatTensor
    num_frames: int
    tokens: List[str]
    phoneme_ids: torch.LongTensor
    phoneme_len_no_special: int
    metadata: Dict[str, float] = field(default_factory=dict)
    noise: Optional[torch.FloatTensor] = None


class PhonemeTokenizer:
    """Utility that maps phoneme strings to Kokoro token ids."""

    # Mapping for Luxembourgish phonemes not in Kokoro vocab to similar phonemes
    LUXEMBOURGISH_PHONEME_MAPPING = {
        'ʑ': 'ʒ',  # voiced postalveolar fricative -> voiced postalveolar fricative (English)
        # Add more mappings as needed
    }

    def __init__(self, config_json: Path) -> None:
        with open(config_json, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        vocab: Dict[str, int] = config["vocab"]
        self.vocab = vocab
        self.pad_id = config.get("pad_token_id", 0)
        self.bos_id = 0
        self.eos_id = 0

    def encode(self, phonemes: Sequence[str]) -> torch.LongTensor:
        ids = [self.bos_id]
        for phoneme in phonemes:
            # Try original phoneme first
            idx = self.vocab.get(phoneme)
            
            # If not found, try Luxembourgish mapping
            if idx is None and phoneme in self.LUXEMBOURGISH_PHONEME_MAPPING:
                mapped_phoneme = self.LUXEMBOURGISH_PHONEME_MAPPING[phoneme]
                idx = self.vocab.get(mapped_phoneme)
                if idx is not None:
                    logging.debug(f"Mapped Luxembourgish phoneme '{phoneme}' -> '{mapped_phoneme}'")
            
            if idx is None:
                raise KeyError(f"Unknown phoneme '{phoneme}' in Kokoro vocab")
            ids.append(idx)
        ids.append(self.eos_id)
        return torch.tensor(ids, dtype=torch.long)


class LuxembourgishPhonemizer:
    """Wrapper around the custom Misaki Luxembourgish G2P."""

    def __init__(self) -> None:
        try:
            from misaki import lb
        except ImportError as exc:  # pragma: no cover - dependency required for phonemisation
            raise RuntimeError(
                "Install the custom Misaki fork with Luxembourgish support: https://github.com/neiom-systems/misaki"
            ) from exc
        self.g2p = lb.LBG2P()

    def __call__(self, text: str) -> List[str]:
        phoneme_string, _ = self.g2p(text)
        phoneme_string = phoneme_string.strip()
        if not phoneme_string:
            return []
        return phoneme_string.split()


def load_audio(path: Path, target_sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
    return audio.astype(np.float32)


def apply_preemphasis(waveform: np.ndarray, coeff: Optional[float]) -> np.ndarray:
    if coeff is None or coeff <= 0.0:
        return waveform
    return np.append(waveform[0], waveform[1:] - coeff * waveform[:-1])


def compute_mel_spectrogram(audio: np.ndarray, cfg: MelConfig, device: Optional[torch.device] = None) -> torch.FloatTensor:
    audio = apply_preemphasis(audio, cfg.preemphasis)
    if torchaudio is not None:
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        key = (
            cfg.sample_rate,
            cfg.n_fft,
            cfg.hop_length,
            cfg.win_length,
            cfg.n_mels,
            cfg.mel_fmin,
            cfg.mel_fmax,
            device.type,
        )
        if key not in _TORCHAUDIO_TRANSFORMS:
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
                n_mels=cfg.n_mels,
                f_min=cfg.mel_fmin,
                f_max=cfg.mel_fmax,
                power=1.0,
            )
            _TORCHAUDIO_TRANSFORMS[key] = transform.to(device).eval()
        transform = _TORCHAUDIO_TRANSFORMS[key]
        tensor = torch.from_numpy(audio).to(device=device, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            mel = transform(tensor)
        mel = torch.clamp(mel, min=1e-5).log()
        return mel.squeeze(0).cpu().float()
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.mel_fmin,
        fmax=cfg.mel_fmax,
        center=True,
        power=1.0,
    )
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return torch.from_numpy(mel).float()


def extract_f0_pyworld(audio: np.ndarray, mel_cfg: MelConfig, f0_cfg: F0Config, frame_count: int) -> Tuple[np.ndarray, np.ndarray]:
    if pyworld is None:
        raise RuntimeError("pyworld is not installed; install it or switch f0.method to 'pyin'")
    hop_time_ms = mel_cfg.hop_length / mel_cfg.sample_rate * 1000.0
    _f0, time_axis = pyworld.harvest(
        audio.astype(np.float64),
        mel_cfg.sample_rate,
        f0_floor=f0_cfg.min_f0,
        f0_ceil=f0_cfg.max_f0,
        frame_period=hop_time_ms,
    )
    f0 = pyworld.stonemask(audio.astype(np.float64), _f0, time_axis, mel_cfg.sample_rate)
    f0 = f0.astype(np.float32)
    if abs(len(f0) - frame_count) > 1:
        f0 = librosa.util.fix_length(f0, size=frame_count)
    uv = (f0 > 0).astype(np.float32)
    return f0, uv


def extract_f0_pyint(audio: np.ndarray, mel_cfg: MelConfig, f0_cfg: F0Config, frame_count: int) -> Tuple[np.ndarray, np.ndarray]:
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=f0_cfg.min_f0,
        fmax=f0_cfg.max_f0,
        sr=mel_cfg.sample_rate,
        hop_length=mel_cfg.hop_length,
    )
    f0 = np.nan_to_num(f0).astype(np.float32)
    if abs(len(f0) - frame_count) > 1:
        f0 = librosa.util.fix_length(f0, size=frame_count)
        voiced_flag = librosa.util.fix_length(voiced_flag.astype(np.float32), size=frame_count)
    uv = voiced_flag.astype(np.float32)
    return f0, uv


def extract_f0(audio: np.ndarray, mel_cfg: MelConfig, f0_cfg: F0Config, frame_count: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    if f0_cfg.method.lower() == "pyworld":
        f0, uv = extract_f0_pyworld(audio, mel_cfg, f0_cfg, frame_count)
    elif f0_cfg.method.lower() == "pyin":
        f0, uv = extract_f0_pyint(audio, mel_cfg, f0_cfg, frame_count)
    else:
        raise ValueError(f"Unsupported f0 extraction method '{f0_cfg.method}'")
    return torch.from_numpy(f0).float(), torch.from_numpy(uv).float()


def trim_silence(audio: np.ndarray, threshold_db: Optional[float]) -> np.ndarray:
    if threshold_db is None:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=threshold_db)
    return trimmed


def load_textgrid(path: Path) -> List[Tuple[str, float, float]]:
    try:
        from textgrid import TextGrid  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency required for alignments
        raise RuntimeError("Install 'textgrid' to parse MFA alignments") from exc
    tg = TextGrid.fromFile(str(path))
    tier = None
    for candidate in tg.tiers:
        name = (candidate.name or "").lower()
        if name in {"phones", "phonemes", "phoneme", "segments"}:
            tier = candidate
            break
    if tier is None:
        raise ValueError(f"No phoneme tier found in TextGrid: {path}")
    entries: List[Tuple[str, float, float]] = []
    for interval in tier:
        label = interval.mark.strip()
        entries.append((label, float(interval.minTime), float(interval.maxTime)))
    return entries


def durations_from_alignment(
    phonemes: Sequence[str],
    alignment_entries: Sequence[Tuple[str, float, float]],
    mel_cfg: MelConfig,
    align_cfg: AlignmentConfig,
) -> torch.LongTensor:
    durations: List[int] = []
    phoneme_iter = iter(phonemes)
    current = next(phoneme_iter, None)
    for label, start, end in alignment_entries:
        if not label:
            continue
        if current is None:
            break
        label = label.strip()
        if label != current:
            continue
        duration_frames = max(
            align_cfg.min_duration_frames,
            int(round((end - start) * mel_cfg.sample_rate / mel_cfg.hop_length)),
        )
        duration_frames = min(duration_frames, align_cfg.max_duration_frames)
        durations.append(duration_frames)
        current = next(phoneme_iter, None)
    if len(durations) != len(phonemes):
        raise ValueError(
            f"Alignment length mismatch: expected {len(phonemes)} phonemes, got {len(durations)}"
        )
    return torch.tensor(durations, dtype=torch.long)


def estimate_uniform_durations(num_phonemes: int, total_frames: int) -> torch.LongTensor:
    if num_phonemes <= 0:
        raise ValueError("Cannot assign durations without phonemes")
    if total_frames <= 0:
        return torch.ones(num_phonemes, dtype=torch.long)
    if total_frames < num_phonemes:
        durations = torch.zeros(num_phonemes, dtype=torch.long)
        durations[:total_frames] = 1
        return durations
    base = max(1, total_frames // num_phonemes)
    durations = torch.full((num_phonemes,), base, dtype=torch.long)
    remaining = total_frames - durations.sum().item()
    idx = 0
    while remaining > 0:
        durations[idx % num_phonemes] += 1
        remaining -= 1
        idx += 1
    return durations


joint_statistics_keys = ("mel_min", "mel_max", "mel_mean", "mel_std")


class FeatureExtractor:
    """Coordinator that phonemises, aligns, and extracts acoustic features."""

    def __init__(
        self,
        cfg: FeatureExtractionConfig,
        *,
        tokenizer: Optional[PhonemeTokenizer] = None,
        phonemizer: Optional[LuxembourgishPhonemizer] = None,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer or PhonemeTokenizer(cfg.vocab_path)
        self.phonemizer = phonemizer or LuxembourgishPhonemizer()
        device_str = cfg.mel_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        device_str = cfg.mel_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

    def process(
        self,
        *,
        text: str,
        audio_path: Path,
        alignment_path: Optional[Path],
        output_path: Path,
        skip_existing: bool = True,
    ) -> Optional[ExtractionResult]:
        if skip_existing and output_path.exists():
            logger.debug("Skipping %s (already exists)", output_path)
            return None

        logger.info("Processing utterance %s", audio_path.stem)
        logger.debug("Raw text: %s", text)

        phoneme_tokens = self.phonemizer(text)
        if not phoneme_tokens:
            raise ValueError("Empty phoneme sequence")
        phonemes = normalize_phoneme_tokens(phoneme_tokens)
        if not phonemes:
            raise ValueError("Phoneme normalization produced an empty sequence")
        if len(phonemes) > self.cfg.max_phoneme_tokens:
            raise ValueError(
                f"Phoneme sequence exceeds max length ({len(phonemes)} > {self.cfg.max_phoneme_tokens})"
            )
        logger.debug("Phonemes (%d): %s", len(phonemes), " ".join(phonemes))

        audio = load_audio(audio_path, self.cfg.mel.sample_rate)
        audio = trim_silence(audio, self.cfg.silence_trim_db)

        mel = compute_mel_spectrogram(audio, self.cfg.mel, self.device)
        frame_count = mel.shape[1]
        logger.debug("Mel shape for %s: %s", audio_path.stem, tuple(mel.shape))

        durations: Optional[torch.LongTensor] = None
        if alignment_path and alignment_path.exists():
            alignment_entries = load_textgrid(alignment_path)
            durations = durations_from_alignment(phonemes, alignment_entries, self.cfg.mel, self.cfg.alignment)
            logger.debug("Duration sum=%d frames=%d", int(durations.sum().item()), frame_count)
        else:
            if self.cfg.require_alignments:
                raise FileNotFoundError(f"Alignment file not found for {audio_path.stem}")
            durations = estimate_uniform_durations(len(phonemes), frame_count)
            logger.warning(
                "No alignment for %s; distributing %d frames uniformly across %d phonemes",
                audio_path.stem,
                frame_count,
                len(phonemes),
            )

        if not self.cfg.allow_frame_mismatch:
            if int(durations.sum().item()) != frame_count:
                raise ValueError(
                    f"Duration sum {int(durations.sum().item())} != mel frames {frame_count}"
                )

        f0, uv = extract_f0(audio, self.cfg.mel, self.cfg.f0, frame_count)
        logger.debug("Voiced frames: %d / %d", int(uv.sum().item()), frame_count)
        noise = None
        if frame_count > 0:
            frame_energy = torch.from_numpy(
                librosa.feature.rms(
                    y=audio,
                    frame_length=self.cfg.mel.win_length,
                    hop_length=self.cfg.mel.hop_length,
                )
            ).squeeze(0)
            noise = torch.log(frame_energy.clamp(min=1e-6))
            noise = F.pad(noise, (0, frame_count - noise.shape[0]))[:frame_count]

        phoneme_ids = self.tokenizer.encode(phonemes)
        result = ExtractionResult(
            mel=mel,
            durations=durations,
            f0=f0,
            uv=uv,
            num_frames=frame_count,
            tokens=list(phonemes),
            phoneme_ids=phoneme_ids,
            phoneme_len_no_special=len(phonemes),
            metadata=self._compute_statistics(mel),
            noise=noise,
        )

        self._write_result(output_path, result)
        logger.info("Wrote features for %s → %s", audio_path.stem, output_path)
        return result

    def _write_result(self, path: Path, result: ExtractionResult) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mel": result.mel,
            "durations": result.durations,
            "f0": result.f0,
            "uv": result.uv,
            "num_frames": torch.tensor(result.num_frames, dtype=torch.long),
            "tokens": result.tokens,
            "phoneme_ids": result.phoneme_ids,
            "phoneme_len_no_special": torch.tensor(result.phoneme_len_no_special, dtype=torch.long),
            "metadata": result.metadata,
        }
        if result.noise is not None:
            payload["noise"] = result.noise
        torch.save(payload, path)

    @staticmethod
    def _compute_statistics(mel: torch.Tensor) -> Dict[str, float]:
        mel_np = mel.numpy()
        return {
            "mel_min": float(mel_np.min()),
            "mel_max": float(mel_np.max()),
            "mel_mean": float(mel_np.mean()),
            "mel_std": float(mel_np.std()),
        }


@dataclass(slots=True)
class ExtractionSummary:
    processed: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    def add_success(self, utt_id: str) -> None:
        self.processed.append(utt_id)

    def add_failure(self, utt_id: str, reason: str) -> None:
        self.skipped.append((utt_id, reason))

    def to_json(self) -> Dict[str, List]:
        return {
            "processed": self.processed,
            "skipped": [{"utt_id": utt, "reason": reason} for utt, reason in self.skipped],
        }


def read_metadata(metadata_csv: Path) -> List[Tuple[str, str, str]]:
    import csv

    rows: List[Tuple[str, str, str]] = []
    with metadata_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path = row.get("path")
            text = row.get("text")
            source = row.get("source", "")
            if not path or text is None:
                continue
            rows.append((path, text.strip(), source.strip()))
    return rows


def _process_entry(
    extractor: FeatureExtractor,
    relative_path: str,
    text: str,
    audio_root: Path,
    alignment_root: Optional[Path],
    output_root: Path,
    skip_existing: bool,
) -> Tuple[bool, str, Optional[str]]:
    utt_id = Path(relative_path).stem
    audio_path = audio_root / relative_path
    alignment_path = None if alignment_root is None else alignment_root / f"{utt_id}.TextGrid"
    output_path = output_root / f"{utt_id}.pt"
    try:
        extractor.process(
            text=text,
            audio_path=audio_path,
            alignment_path=alignment_path,
            output_path=output_path,
            skip_existing=skip_existing,
        )
        return True, utt_id, None
    except Exception as exc:  # pragma: no cover - logging for user awareness
        return False, utt_id, str(exc)


def run_split_extraction(
    *,
    metadata_csv: Path,
    audio_root: Path,
    alignment_root: Optional[Path],
    output_root: Path,
    extractor: FeatureExtractor,
    skip_existing: bool = True,
    num_workers: int = 1,
) -> ExtractionSummary:
    summary = ExtractionSummary()
    rows = read_metadata(metadata_csv)
    logger.info("Starting feature extraction for %d items from %s", len(rows), metadata_csv)
    num_workers = max(1, num_workers)

    if num_workers == 1:
        for relative_path, text, _ in rows:
            success, utt_id, err = _process_entry(
                extractor,
                relative_path,
                text,
                audio_root,
                alignment_root,
                output_root,
                skip_existing,
            )
            if success:
                summary.add_success(utt_id)
            else:
                summary.add_failure(utt_id, err or "")
                logger.warning("Failed to process %s: %s", utt_id, err)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_entry,
                    extractor,
                    relative_path,
                    text,
                    audio_root,
                    alignment_root,
                    output_root,
                    skip_existing,
                )
                for relative_path, text, _ in rows
            ]
            for future in as_completed(futures):
                success, utt_id, err = future.result()
                if success:
                    summary.add_success(utt_id)
                else:
                    summary.add_failure(utt_id, err or "")
                    logger.warning("Failed to process %s: %s", utt_id, err)
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_json(), handle, indent=2, ensure_ascii=False)
    return summary


__all__ = [
    "FeatureExtractionConfig",
    "FeatureExtractor",
    "ExtractionSummary",
    "normalize_phoneme_tokens",
    "read_metadata",
    "run_split_extraction",
]
