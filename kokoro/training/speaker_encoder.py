"""Luxembourgish voice-table generation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover - torchaudio is required for audio I/O
    raise RuntimeError("torchaudio is required for speaker embedding extraction") from exc

try:
    from transformers import AutoFeatureExtractor, AutoModel
except ImportError as exc:  # pragma: no cover - huggingface transformers required for WavLM/HuBERT
    raise RuntimeError("Install 'transformers' to use the speaker encoder utilities") from exc

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SpeakerEncoderConfig:
    model_name: str = "microsoft/wavlm-large"
    layer: Optional[int] = None  # None => use last hidden state
    normalize: bool = True
    sample_rate: int = 16_000
    chunk_seconds: float = 6.0
    device: Optional[str] = None


@dataclass(slots=True)
class ProjectionConfig:
    latent_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1
    init_scale: float = 0.05


@dataclass(slots=True)
class VoiceTableConfig:
    rows: int = 510
    latent_dim: int = 256
    positional_dim: int = 64
    sinusoidal_scale: float = 1.0
    target_mean: float = 0.0
    target_std: float = 0.15


@dataclass(slots=True)
class TableGenerationConfig:
    encoder: SpeakerEncoderConfig = field(default_factory=SpeakerEncoderConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    table: VoiceTableConfig = field(default_factory=VoiceTableConfig)
    batch_size: int = 1
    cache_embeddings: Optional[Path] = None


def load_audio_waveform(path: Path, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
    logger.debug("Loaded audio %s (orig_sr=%d → %d, samples=%d)", path, sr, target_sr, waveform.numel())
    return waveform.squeeze(0)


class AverageSpeakerEmbedding:
    """Compute an averaged speaker embedding using a pretrained encoder."""

    def __init__(self, cfg: SpeakerEncoderConfig) -> None:
        self.cfg = cfg
        self.processor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(
            cfg.model_name,
            use_safetensors=True,
        )
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.model_dtype = next(self.model.parameters()).dtype
        logger.info("Loaded speaker encoder %s on %s", cfg.model_name, self.device)

    @torch.no_grad()
    def __call__(self, audio_paths: Sequence[Path]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        for path in audio_paths:
            waveform = load_audio_waveform(path, self.cfg.sample_rate).to(self.device)
            inputs = self.processor(
                waveform,
                sampling_rate=self.cfg.sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Fix attention mask type mismatch for WavLM
            # The issue is that WavLM expects both attention_mask and key_padding_mask to be the same type
            for key, value in inputs.items():
                if key == "attention_mask":
                    # Convert attention_mask to boolean to match internal key_padding_mask
                    inputs[key] = value.bool()
                elif torch.is_floating_point(value):
                    inputs[key] = value.to(self.model_dtype)
            
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            if self.cfg.layer is not None and hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[self.cfg.layer]
            embedding = hidden_states.mean(dim=1)
            if self.cfg.normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)
            embeddings.append(embedding.squeeze(0).cpu())
            logger.debug("Generated embedding for %s with norm %.4f", path.name, embeddings[-1].norm().item())
        stacked = torch.stack(embeddings, dim=0)
        mean_embedding = stacked.mean(dim=0)
        if self.cfg.normalize:
            mean_embedding = F.normalize(mean_embedding, p=2, dim=-1)
        logger.info("Averaged speaker embedding across %d files (dim=%d)", len(audio_paths), mean_embedding.shape[-1])
        return mean_embedding


class ProjectionHead(nn.Module):
    """Small MLP that maps speaker-encoder embeddings to the 256-d latent."""

    def __init__(self, in_dim: int, cfg: ProjectionConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, cfg.hidden_dim)
        self.linear2 = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.xavier_uniform_(self.linear1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=cfg.init_scale)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


def sinusoidal_embeddings(num_positions: int, dim: int, scale: float = 1.0) -> torch.Tensor:
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    embeddings = torch.zeros(num_positions, dim)
    embeddings[:, 0::2] = torch.sin(position * div_term) * scale
    embeddings[:, 1::2] = torch.cos(position * div_term) * scale
    return embeddings


class PositionalExpansion(nn.Module):
    """Expand base latent vector into 510 position-specific rows."""

    def __init__(self, cfg: VoiceTableConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.index_embedding = nn.Embedding(cfg.rows, cfg.positional_dim)
        self.linear = nn.Linear(cfg.latent_dim + cfg.positional_dim, cfg.latent_dim)
        nn.init.normal_(self.index_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, base_latent: torch.Tensor) -> torch.Tensor:
        if base_latent.dim() == 1:
            base_latent = base_latent.unsqueeze(0)
        rows = self.cfg.rows
        device = base_latent.device
        positional = self.index_embedding.weight[:rows].to(device)
        positional = positional.unsqueeze(0).expand(base_latent.shape[0], -1, -1)
        base = base_latent.unsqueeze(1).expand(-1, rows, -1)
        combined = torch.cat([base, positional], dim=-1)
        table = self.linear(combined)
        return table.squeeze(0)


def match_statistics(table: torch.Tensor, target_mean: float, target_std: float) -> torch.Tensor:
    mean = table.mean()
    std = table.std(unbiased=False)
    if std < 1e-6:
        return table
    normalized = (table - mean) / std
    return normalized * target_std + target_mean


@dataclass(slots=True)
class VoiceTableArtifacts:
    base_embedding: torch.Tensor
    projection_output: torch.Tensor
    table: torch.Tensor


def generate_voice_table(
    audio_paths: Sequence[Path],
    cfg: TableGenerationConfig,
    *,
    cache_path: Optional[Path] = None,
) -> VoiceTableArtifacts:
    logger.info("Generating Luxembourgish voice table from %d audio clips", len(audio_paths))
    encoder = AverageSpeakerEmbedding(cfg.encoder)
    base_embedding = encoder(audio_paths)
    projection_input_dim = base_embedding.shape[-1]
    projection_head = ProjectionHead(projection_input_dim, cfg.projection)
    projection_output = projection_head(base_embedding.unsqueeze(0)).squeeze(0)
    logger.debug("Projection output stats: mean %.4f std %.4f", projection_output.mean().item(), projection_output.std(unbiased=False).item())

    expansion = PositionalExpansion(cfg.table)
    table = expansion(projection_output)
    table = match_statistics(table, cfg.table.target_mean, cfg.table.target_std)
    table = table.unsqueeze(1)  # [510, 1, 256]
    logger.info("Generated voice table with shape %s", tuple(table.shape))

    artifacts = VoiceTableArtifacts(
        base_embedding=base_embedding,
        projection_output=projection_output,
        table=table,
    )

    if cache_path is not None:
        save_voice_table(artifacts, cache_path)
    return artifacts


def save_voice_table(artifacts: VoiceTableArtifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "table": artifacts.table,
        "base_embedding": artifacts.base_embedding,
        "projection_output": artifacts.projection_output,
    }
    torch.save(payload, path)
    logger.info("Saved voice table artifacts to %s", path)


def load_voice_table(path: Path) -> VoiceTableArtifacts:
    payload = torch.load(path, map_location="cpu")
    if "table" not in payload:
        raise KeyError(f"Voice table file missing 'table' tensor: {path}")
    logger.info("Loaded voice table from %s", path)
    return VoiceTableArtifacts(
        table=payload["table"],
        base_embedding=payload.get("base_embedding", torch.empty(0)),
        projection_output=payload.get("projection_output", torch.empty(0)),
    )


def get_voice_rows(table: torch.Tensor, phoneme_lengths: torch.Tensor) -> torch.Tensor:
    rows = phoneme_lengths.clamp(min=1, max=table.shape[0]) - 1
    selected = table.squeeze(1).index_select(0, rows)
    return selected.unsqueeze(1)


__all__ = [
    "TableGenerationConfig",
    "VoiceTableArtifacts",
    "generate_voice_table",
    "save_voice_table",
    "load_voice_table",
    "get_voice_rows",
]
