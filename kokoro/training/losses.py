"""Loss functions for Luxembourgish fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn as nn

try:  # Optional torchaudio dependency for mel losses
    import torchaudio
except ImportError:  # pragma: no cover - torchaudio required when mel loss enabled
    torchaudio = None

from .config import LossConfig
from .model import TrainableKModelOutput
from ..custom_stft import CustomSTFT


@dataclass(frozen=True)
class STFTSpec:
    """Configuration tuple for a single STFT resolution."""

    fft_size: int
    hop_length: int
    win_length: Optional[int] = None


class MultiResolutionSTFTLoss(nn.Module):
    """Compute spectral convergence and magnitude losses across multiple STFTs."""

    def __init__(self, specs: Sequence[STFTSpec]) -> None:
        super().__init__()
        if not specs:
            raise ValueError("At least one STFTSpec is required")
        self.specs = specs
        self.stfts = nn.ModuleList(
            [
                CustomSTFT(
                    filter_length=spec.fft_size,
                    hop_length=spec.hop_length,
                    win_length=spec.win_length or spec.fft_size,
                    window="hann",
                    center=True,
                    pad_mode="replicate",
                )
                for spec in specs
            ]
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if prediction.dtype != torch.float32:
            prediction = prediction.to(torch.float32)
        if target.dtype != torch.float32:
            target = target.to(torch.float32)
        if prediction.dim() == 3:
            prediction = prediction.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        if prediction.shape != target.shape:
            min_len = min(prediction.shape[-1], target.shape[-1])
            prediction = prediction[..., :min_len]
            target = target[..., :min_len]

        sc_loss = prediction.new_tensor(0.0)
        mag_loss = prediction.new_tensor(0.0)
        for stft_module in self.stfts:
            y_hat_mag, _ = stft_module.transform(prediction)
            y_mag, _ = stft_module.transform(target)
            sc = ((y_mag - y_hat_mag).norm(p="fro") / (y_mag.norm(p="fro") + 1e-7))
            mag = F.l1_loss(y_hat_mag, y_mag)
            sc_loss = sc_loss + sc
            mag_loss = mag_loss + mag

        count = len(self.specs)
        sc_loss = sc_loss / count
        mag_loss = mag_loss / count
        return sc_loss, mag_loss


class MelSpectrogramLoss(nn.Module):
    """Compute log-mel L1 loss between prediction and ground-truth."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int],
        n_mels: int,
        mel_fmin: float,
        mel_fmax: float,
    ) -> None:
        super().__init__()
        if torchaudio is None:
            raise RuntimeError("torchaudio is required for MelSpectrogramLoss")
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length or n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=mel_fmin,
            f_max=mel_fmax,
            power=1.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )

    def forward(self, prediction: torch.Tensor, target_mel: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if prediction.dtype != torch.float32:
            prediction = prediction.to(torch.float32)
        if target_mel.dtype != torch.float32:
            target_mel = target_mel.to(torch.float32)
        if prediction.dim() == 2:
            prediction = prediction.unsqueeze(1)
        mel_pred = self.transform(prediction)
        if not torch.isfinite(mel_pred).all():
            raise RuntimeError("MelSpectrogramLoss received non-finite prediction; check model outputs.")
        mel_pred = torch.log(torch.clamp(mel_pred, min=1e-5))
        mel_tgt = torch.log(torch.clamp(target_mel, min=1e-5))
        if not torch.isfinite(mel_tgt).all():
            raise RuntimeError("MelSpectrogramLoss received non-finite target mel features; regenerate caches.")
        mel_pred = mel_pred.transpose(-2, -1)  # [batch, frames, n_mels]
        mel_tgt = mel_tgt.transpose(-2, -1) if mel_tgt.shape[1] == mel_pred.shape[-1] else mel_tgt.transpose(-2, -1)
        if mel_pred.shape[1] != mel_tgt.shape[1]:
            min_len = min(mel_pred.shape[1], mel_tgt.shape[1])
            mel_pred = mel_pred[:, :min_len]
            mel_tgt = mel_tgt[:, :min_len]
            if mask is not None:
                mask = mask[..., :min_len]
        if mask is not None:
            valid = mask.float().unsqueeze(-1)
            loss = (valid * (mel_pred - mel_tgt).abs()).sum() / valid.sum().clamp(min=1.0)
        else:
            loss = F.l1_loss(mel_pred, mel_tgt)
        return loss


def build_duration_targets(durations: torch.Tensor, max_dur: int) -> torch.Tensor:
    """Expand per-token duration integers into cumulative bins for BCE loss."""

    bins = torch.arange(max_dur, device=durations.device).view(1, 1, -1)
    expanded = durations.unsqueeze(-1)
    targets = (bins < expanded).float()
    targets = targets * (expanded > 0).float()
    return targets


class LossComputer:
    """Central loss aggregation utility used by the training loop."""

    def __init__(
        self,
        config: LossConfig,
        *,
        sample_rate: int,
        stft_specs: Optional[Sequence[STFTSpec]] = None,
        mel_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = config
        self.sample_rate = sample_rate
        self.stft_loss = None
        if config.lambda_stft > 0.0:
            specs = stft_specs or (
                STFTSpec(fft_size=1024, hop_length=256, win_length=1024),
                STFTSpec(fft_size=512, hop_length=128, win_length=512),
                STFTSpec(fft_size=2048, hop_length=512, win_length=2048),
            )
            self.stft_loss = MultiResolutionSTFTLoss(specs)
        self.mel_loss = None
        if config.lambda_mel > 0.0:
            if mel_kwargs is None:
                raise ValueError("mel_kwargs must be provided when lambda_mel > 0")
            self.mel_loss = MelSpectrogramLoss(**mel_kwargs)

    def to(self, device: torch.device) -> "LossComputer":
        if self.stft_loss is not None:
            self.stft_loss = self.stft_loss.to(device)
        if self.mel_loss is not None:
            self.mel_loss = self.mel_loss.to(device)
        return self

    def _duration_targets_from_batch(
        self,
        output_duration_shape: torch.Size,
        batch: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        seq_len = output_duration_shape[1]
        durations = batch.get("durations")
        if durations is None:
            raise KeyError("Batch missing 'durations' needed for duration loss")
        durations = durations.to(device)
        target = torch.zeros(output_duration_shape, device=device, dtype=torch.long)
        lengths = batch.get("phoneme_lengths")
        if lengths is None:
            raise KeyError("Batch missing 'phoneme_lengths' for duration alignment")
        lengths = lengths.to(device).long()
        core_len = durations.shape[1]
        for b in range(durations.size(0)):
            length = int(lengths[b].item())
            usable = min(core_len, max(0, length - 2))
            if usable > 0:
                target[b, 1 : 1 + usable] = durations[b, :usable]
        return target

    def duration_loss(
        self,
        logits: torch.Tensor,
        output: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "duration_teacher" in output and output["duration_teacher"] is not None:
            durations = output["duration_teacher"].to(device)
        else:
            durations = self._duration_targets_from_batch(logits.shape[:2], batch, device=device)
        targets = build_duration_targets(durations, logits.shape[-1])
        mask = (durations > 0).unsqueeze(-1).float()
        denom = mask.sum().clamp(min=1.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=mask, reduction="sum") / denom
        frame_pred = output["duration_frames"].to(device).float()
        frame_l1 = (mask.squeeze(-1) * (frame_pred - durations.float()).abs()).sum() / mask.squeeze(-1).sum().clamp(min=1.0)
        return bce, frame_l1

    def f0_loss(
        self,
        f0_pred: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        f0_target = batch["f0"].to(device).float()
        uv = batch.get("uv")
        if uv is None:
            uv = (f0_target > 0).float()
        else:
            uv = uv.to(device).float()
        min_len = min(f0_pred.shape[-1], f0_target.shape[-1])
        f0_pred = f0_pred[..., :min_len]
        f0_target = f0_target[..., :min_len]
        uv = uv[..., :min_len]
        if uv.sum() == 0:
            return f0_pred.new_tensor(0.0)
        log_pred = torch.log(torch.clamp(f0_pred, min=1e-5))
        log_target = torch.log(torch.clamp(f0_target, min=1e-5))
        loss = (uv * (log_pred - log_target).abs()).sum() / uv.sum()
        return loss

    def noise_loss(
        self,
        noise_pred: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if "noise" not in batch:
            return None
        target = batch["noise"].to(device).float()
        min_len = min(noise_pred.shape[-1], target.shape[-1])
        noise_pred = noise_pred[..., :min_len]
        target = target[..., :min_len]
        return F.l1_loss(noise_pred, target)

    def __call__(
        self,
        output: Union[TrainableKModelOutput, Mapping[str, torch.Tensor]],
        batch: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output_map: Dict[str, torch.Tensor]
        if is_dataclass(output):
            output_map = {field.name: getattr(output, field.name) for field in fields(output)}
        else:
            output_map = dict(output)
        device = next(value.device for value in output_map.values() if isinstance(value, torch.Tensor))
        components: Dict[str, torch.Tensor] = {}

        duration_logits = output_map["duration_logits"].to(device)
        duration_bce, duration_l1 = self.duration_loss(duration_logits, output_map, batch, device=device)
        components["duration_bce"] = duration_bce
        components["duration_l1"] = duration_l1

        if self.cfg.lambda_f0 > 0.0:
            components["f0"] = self.f0_loss(output_map["f0_pred"], batch, device=device)
        else:
            components["f0"] = torch.zeros((), device=device)

        noise_component = self.noise_loss(output_map["noise_pred"], batch, device=device)
        components["noise"] = noise_component if noise_component is not None else torch.zeros((), device=device)

        if self.cfg.lambda_mel > 0.0 and self.mel_loss is not None:
            target_mel = batch["mel"].to(device)
            mel_mask = batch.get("mel_mask")
            if mel_mask is not None:
                mel_mask = mel_mask.to(device)
            components["mel"] = self.mel_loss(output_map["audio"], target_mel, mel_mask)
        else:
            components["mel"] = torch.zeros((), device=device)

        if self.cfg.lambda_stft > 0.0 and self.stft_loss is not None:
            if "audio" not in batch:
                raise KeyError("Batch missing 'audio' required for STFT loss")
            target_audio = batch["audio"].to(device)
            sc, mag = self.stft_loss(output_map["audio"], target_audio)
            components["stft_sc"] = sc
            components["stft_mag"] = mag
        else:
            components["stft_sc"] = torch.zeros((), device=device)
            components["stft_mag"] = torch.zeros((), device=device)

        total = (
            self.cfg.lambda_dur * (components["duration_bce"] + components["duration_l1"])
            + self.cfg.lambda_f0 * components["f0"]
            + self.cfg.lambda_noise * components["noise"]
            + self.cfg.lambda_mel * components["mel"]
            + self.cfg.lambda_stft * (components["stft_sc"] + components["stft_mag"])
        )
        components["total"] = total
        if "mel_mask" in batch and isinstance(batch["mel_mask"], torch.Tensor):
            frame_count = batch["mel_mask"].to(device).float().sum()
        else:
            mel_tensor = batch.get("mel")
            if mel_tensor is not None and isinstance(mel_tensor, torch.Tensor):
                frame_count = torch.tensor(mel_tensor.shape[0] * mel_tensor.shape[-1], device=device, dtype=torch.float32)
            else:
                frame_count = torch.tensor(batch["f0"].numel(), device=device, dtype=torch.float32)
        components["total_normalized"] = total / frame_count.clamp(min=1.0)
        return components


__all__ = [
    "LossComputer",
    "STFTSpec",
]
