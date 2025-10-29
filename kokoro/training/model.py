"""Trainable wrapper around the inference-only `kokoro.model.KModel`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..model import KModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainableKModelOutput:
    """Aggregated tensors returned by the fine-tuning forward pass."""

    audio: torch.FloatTensor
    duration_logits: torch.FloatTensor
    duration_frames: torch.LongTensor
    duration_prob: torch.FloatTensor
    f0_pred: torch.FloatTensor
    noise_pred: torch.FloatTensor
    alignment: torch.FloatTensor
    alignment_mask: torch.BoolTensor
    bert_hidden: torch.FloatTensor
    text_hidden: torch.FloatTensor
    style: torch.FloatTensor
    voice_embedding: torch.FloatTensor
    alignment_pred: torch.FloatTensor
    alignment_pred_mask: torch.BoolTensor
    duration_teacher: Optional[torch.LongTensor] = None
    alignment_teacher: Optional[torch.FloatTensor] = None
    alignment_teacher_mask: Optional[torch.BoolTensor] = None


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove an optional `module.` prefix that appears in DDP checkpoints."""

    cleaned: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


class TrainableKModel(nn.Module):
    """Gradient-enabled Kokoro model suitable for Luxembourgish fine-tuning."""

    VOICE_STATE_KEY = "voices/lb_max"

    def __init__(
        self,
        *,
        base_model: Optional[KModel] = None,
        repo_id: Optional[str] = None,
        config: Optional[Any] = None,
        checkpoint: Optional[str] = None,
        voice_table: torch.FloatTensor,
        train_voice: bool = True,
        disable_complex_decoder: bool = False,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        """
        Args:
            base_model: Optional pre-loaded inference model to wrap.
            repo_id: HuggingFace repo to load if ``base_model`` is absent.
            config: Path or dict with Kokoro configuration (passed to ``KModel``).
            checkpoint: Path to pretrained weights (passed to ``KModel``).
            voice_table: Tensor with shape ``[510, 1, 256]`` or ``[510, 256]``.
            train_voice: Whether the voice table should receive gradients.
            disable_complex_decoder: Forwarded to ``KModel`` for ISTFTNet init.
        """

        super().__init__()
        if base_model is None:
            base_model = KModel(
                repo_id=repo_id,
                config=config,
                model=checkpoint,
                disable_complex=disable_complex_decoder,
            )
        self.core = base_model
        self.core.train()

        voice_tensor = voice_table.detach().clone().to(dtype=torch.float32)
        if voice_tensor.dim() == 3 and voice_tensor.shape[1] == 1:
            voice_tensor = voice_tensor.squeeze(1)
        if voice_tensor.dim() != 2:
            raise ValueError(
                f"voice_table must be 2D after optional squeeze, got shape {tuple(voice_tensor.shape)}"
            )
        if voice_tensor.shape[0] != 510:
            raise ValueError(
                f"voice_table must have 510 rows, got {voice_tensor.shape[0]}"
            )
        if train_voice:
            self.voice_table = nn.Parameter(voice_tensor)
        else:
            self.register_buffer("voice_table", voice_tensor, persistent=True)
        self._train_voice = train_voice
        self._use_gradient_checkpointing = use_gradient_checkpointing

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        use_teacher_durations: bool = False,
        detach_voice: bool = False,
    ) -> TrainableKModelOutput:
        """
        Args:
            batch: Dictionary emitted by ``LuxembourgishDataset``/``luxembourgish_collate``.
            use_teacher_durations: Build alignments from ground-truth durations when available.
            detach_voice: Stop gradients from flowing into the voice table.
        """

        device = self.device
        input_ids = batch["input_ids"].to(device)
        batch_size, seq_len = input_ids.shape

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "TrainableKModel.forward batch_size=%d seq_len=%d teacher=%s detach_voice=%s",
                batch_size,
                seq_len,
                use_teacher_durations,
                detach_voice,
            )

        voice_rows = batch["voice_rows"].to(device)
        voice = self.select_voice_rows(voice_rows, detach=detach_voice)
        style = voice[:, 128:]
        decoder_style = voice[:, :128]

        phoneme_mask = batch.get("phoneme_mask")
        if phoneme_mask is not None:
            phoneme_mask = phoneme_mask.to(device).bool()
            input_lengths = phoneme_mask.sum(dim=1).to(torch.long)
            text_mask = ~phoneme_mask
        else:
            input_lengths = (input_ids != 0).sum(dim=1).to(torch.long)
            ranges = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            text_mask = ranges >= input_lengths.unsqueeze(1)

        mel_mask = batch.get("mel_mask")
        if mel_mask is not None:
            mel_mask = mel_mask.to(device).bool()
            frame_lengths = mel_mask.sum(dim=1).to(torch.long)
            max_frames = mel_mask.shape[1]
        else:
            mel = batch["mel"].to(device)
            max_frames = mel.shape[-1]
            frame_lengths = torch.full(
                (batch_size,),
                fill_value=max_frames,
                dtype=torch.long,
                device=device,
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Input lengths stats → min=%d max=%d | frame_lengths min=%d max=%d",
                int(input_lengths.min().item()),
                int(input_lengths.max().item()),
                int(frame_lengths.min().item()),
                int(frame_lengths.max().item()),
            )

        bert_hidden = self.core.bert(
            input_ids,
            attention_mask=(~text_mask).int(),
        )
        bert_proj = self.core.bert_encoder(bert_hidden).transpose(-1, -2)

        d_temporal = self.core.predictor.text_encoder(
            bert_proj,
            style,
            input_lengths,
            text_mask,
        )
        lstm_out, _ = self.core.predictor.lstm(d_temporal)
        duration_logits = self.core.predictor.duration_proj(lstm_out)
        duration_prob = torch.sigmoid(duration_logits)
        duration_frames = torch.round(duration_prob.sum(dim=-1)).clamp_(min=1).long()
        duration_frames = duration_frames.masked_fill(text_mask, 0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Predicted duration frames stats → mean=%.2f std=%.2f",
                duration_frames.float().mean().item(),
                duration_frames.float().std(unbiased=False).item(),
            )

        # Remove BOS/EOS from predicted durations.
        for idx in range(batch_size):
            length = int(input_lengths[idx].item())
            if length > 0:
                duration_frames[idx, 0] = 0
            if length > 1:
                duration_frames[idx, length - 1] = 0
            if length < seq_len:
                duration_frames[idx, length:] = 0

        alignment_pred, alignment_pred_mask = self.build_alignment(
            duration_frames,
            input_lengths,
            frame_lengths,
            max_frames=max_frames,
        )

        duration_teacher_full: Optional[torch.LongTensor] = None
        alignment_teacher: Optional[torch.FloatTensor] = None
        alignment_teacher_mask: Optional[torch.BoolTensor] = None

        if use_teacher_durations and "durations" in batch:
            durations_gt = batch["durations"].to(device)
            duration_mask = batch.get("duration_mask")
            if duration_mask is not None:
                duration_mask = duration_mask.to(device).bool()
                core_lengths = duration_mask.sum(dim=1).to(torch.long)
            else:
                core_lengths = (durations_gt > 0).sum(dim=1).to(torch.long)
            duration_teacher_full = torch.zeros_like(duration_frames)
            for idx in range(batch_size):
                length = int(input_lengths[idx].item())
                core_len = int(core_lengths[idx].item())
                core_len = min(core_len, max(0, length - 2))
                if core_len > 0:
                    duration_teacher_full[idx, 1 : 1 + core_len] = durations_gt[idx, :core_len]
                if length > 0:
                    duration_teacher_full[idx, 0] = 0
                if length > 1:
                    duration_teacher_full[idx, length - 1] = 0
                if length < seq_len:
                    duration_teacher_full[idx, length:] = 0
            alignment_teacher, alignment_teacher_mask = self.build_alignment(
                duration_teacher_full,
                input_lengths,
                frame_lengths,
                max_frames=max_frames,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Teacher durations applied for %d sequences", duration_teacher_full.size(0))

        alignment = alignment_teacher if alignment_teacher is not None else alignment_pred
        alignment_mask = (
            alignment_teacher_mask if alignment_teacher_mask is not None else alignment_pred_mask
        )

        en = d_temporal.transpose(-1, -2) @ alignment
        f0_pred, noise_pred = self.core.predictor.F0Ntrain(en, style)

        text_hidden = self.core.text_encoder(
            input_ids,
            input_lengths,
            text_mask,
        )
        asr = text_hidden @ alignment

        autocast_active = torch.is_autocast_enabled()
        def _run_decoder(asr_t, f0_t, noise_t, style_t):
            decoder_inputs = (
                asr_t.float(),
                f0_t.float(),
                noise_t.float(),
                style_t.float(),
            )
            if autocast_active and torch.cuda.is_available():
                with torch.amp.autocast("cuda", enabled=False):
                    return self.core.decoder(*decoder_inputs)
            return self.core.decoder(*decoder_inputs)

        if self._use_gradient_checkpointing and self.training:
            audio = checkpoint(
                _run_decoder,
                asr,
                f0_pred,
                noise_pred,
                decoder_style,
                use_reentrant=False,
            )
        else:
            audio = _run_decoder(asr, f0_pred, noise_pred, decoder_style)
        audio = audio.float()
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Forward output audio_len=%s f0_mean=%.4f",
                tuple(audio.shape),
                f0_pred.mean().item(),
            )

        return TrainableKModelOutput(
            audio=audio,
            duration_logits=duration_logits,
            duration_frames=duration_frames,
            duration_prob=duration_prob,
            f0_pred=f0_pred,
            noise_pred=noise_pred,
            alignment=alignment,
            alignment_mask=alignment_mask,
            bert_hidden=bert_hidden,
            text_hidden=text_hidden,
            style=style,
            voice_embedding=voice,
            alignment_pred=alignment_pred,
            alignment_pred_mask=alignment_pred_mask,
            duration_teacher=duration_teacher_full,
            alignment_teacher=alignment_teacher,
            alignment_teacher_mask=alignment_teacher_mask,
        )

    def select_voice_rows(
        self,
        indices: torch.LongTensor,
        *,
        detach: bool = False,
    ) -> torch.FloatTensor:
        table = self.voice_table
        clamped = torch.clamp(indices, 0, table.shape[0] - 1)
        voice = table.index_select(0, clamped)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Selected voice rows: %s", clamped.detach().cpu().tolist())
        return voice.detach() if detach else voice

    @staticmethod
    def build_alignment(
        durations: torch.LongTensor,
        input_lengths: torch.LongTensor,
        frame_lengths: torch.LongTensor,
        *,
        max_frames: Optional[int] = None,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """Convert per-token frame counts into a batched alignment matrix."""

        batch_size, seq_len = durations.shape
        if max_frames is None:
            max_frames = int(frame_lengths.max().item())
        device = durations.device
        alignment = torch.zeros(
            batch_size,
            seq_len,
            max_frames,
            device=device,
            dtype=torch.float32,
        )
        mask = torch.zeros(batch_size, max_frames, device=device, dtype=torch.bool)
        for b in range(batch_size):
            length = int(input_lengths[b].item())
            frame_limit = int(frame_lengths[b].item())
            cursor = 0
            for t in range(length):
                dur = int(durations[b, t].item())
                if dur <= 0:
                    continue
                end = min(cursor + dur, max_frames)
                if cursor >= max_frames:
                    break
                alignment[b, t, cursor:end] = 1.0
                cursor = end
            mask_len = min(frame_limit, max_frames)
            if mask_len > 0:
                mask[b, :mask_len] = True
        return alignment, mask

    def freeze_submodules(
        self,
        *,
        bert: bool = False,
        text_encoder: bool = False,
        predictor: bool = False,
        decoder: bool = False,
    ) -> None:
        if bert:
            self._set_requires_grad(self.core.bert, False)
            self._set_requires_grad(self.core.bert_encoder, False)
        if text_encoder:
            self._set_requires_grad(self.core.text_encoder, False)
        if predictor:
            self._set_requires_grad(self.core.predictor, False)
        if decoder:
            self._set_requires_grad(self.core.decoder, False)

    def unfreeze_submodules(
        self,
        *,
        bert: bool = False,
        text_encoder: bool = False,
        predictor: bool = False,
        decoder: bool = False,
    ) -> None:
        if bert:
            self._set_requires_grad(self.core.bert, True)
            self._set_requires_grad(self.core.bert_encoder, True)
        if text_encoder:
            self._set_requires_grad(self.core.text_encoder, True)
        if predictor:
            self._set_requires_grad(self.core.predictor, True)
        if decoder:
            self._set_requires_grad(self.core.decoder, True)

    def set_voice_trainable(self, enable: bool) -> None:
        """Toggle whether gradients flow into the Luxembourgish voice table."""

        self._train_voice = enable
        if isinstance(self.voice_table, nn.Parameter):
            self.voice_table.requires_grad_(enable)

    def parameter_groups(
        self,
        *,
        base_lr: float,
        voice_lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        base_params: List[torch.nn.Parameter] = []
        voice_params: List[torch.nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name == "voice_table":
                voice_params.append(param)
            else:
                base_params.append(param)
        groups: List[Dict[str, Any]] = []
        if base_params:
            group = {"params": base_params, "lr": base_lr}
            if weight_decay is not None:
                group["weight_decay"] = weight_decay
            groups.append(group)
        if voice_params:
            lr = voice_lr if voice_lr is not None else base_lr
            group = {"params": voice_params, "lr": lr}
            if weight_decay is not None:
                group["weight_decay"] = weight_decay
            groups.append(group)
        return groups

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        cleaned = _strip_module_prefix(state_dict)
        remapped: Dict[str, Any] = {}
        for key, value in cleaned.items():
            if key in {self.VOICE_STATE_KEY, self.VOICE_STATE_KEY.replace("/", ".")}:
                remapped["voice_table"] = value
            else:
                remapped[key] = value
        return super().load_state_dict(remapped, strict=strict)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        state = super().state_dict(*args, **kwargs)
        voice_key = "voice_table"
        if voice_key in state:
            state[self.VOICE_STATE_KEY] = state.pop(voice_key)
        return state

    def export_voice_table(self) -> torch.Tensor:
        """Return the current voice table in `[510, 1, 256]` format for saving."""

        table = self.voice_table.detach().clone()
        return table.unsqueeze(1)

    @staticmethod
    def _set_requires_grad(module: nn.Module, flag: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(flag)


__all__ = [
    "TrainableKModel",
    "TrainableKModelOutput",
]
