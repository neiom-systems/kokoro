"""End-to-end fine-tuning loop for the Luxembourgish Kokoro model."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

try:  # Optional tensorboard logging
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None  # type: ignore

import torchaudio

from .config import TrainingConfig
from .dataset import LuxembourgishDataset, luxembourgish_collate
from .losses import LossComputer, STFTSpec
from .model import TrainableKModel, TrainableKModelOutput

LOG_LEVEL = os.environ.get("KOKORO_LOG_LEVEL", "INFO").upper()

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d", seed)


def load_json(path: Path) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_voice_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, Mapping):
        if "table" in payload:
            tensor = payload["table"]
        else:
            raise KeyError(f"Voice file {path} missing 'table' tensor")
    else:
        raise TypeError(f"Unsupported voice file payload type: {type(payload)!r}")
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)
    logger.info("Loaded voice tensor from %s with shape %s", path, tuple(tensor.shape))
    return tensor.float()


def prepare_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    train_dataset = LuxembourgishDataset(
        metadata_csv=cfg.paths.train_csv,
        feature_root=cfg.paths.feature_root,
        split="train",
        data_config=cfg.data,
        strict=True,
    )
    val_dataset = LuxembourgishDataset(
        metadata_csv=cfg.paths.test_csv,
        feature_root=cfg.paths.feature_root,
        split="test",
        data_config=cfg.data,
        strict=True,
    )

    generator = torch.Generator().manual_seed(cfg.runtime.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.runtime.batch_size,
        shuffle=True,
        num_workers=cfg.runtime.num_workers,
        pin_memory=True,
        collate_fn=luxembourgish_collate,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.runtime.batch_size,
        shuffle=False,
        num_workers=cfg.runtime.num_workers,
        pin_memory=True,
        collate_fn=luxembourgish_collate,
    )
    logger.info(
        "Created dataloaders (train_batches=%d, val_batches=%d, batch_size=%d)",
        len(train_loader),
        len(val_loader),
        cfg.runtime.batch_size,
    )
    return train_loader, val_loader


def create_model(cfg: TrainingConfig, device: torch.device) -> TrainableKModel:
    voice_path = cfg.paths.voice_init or cfg.paths.voice_export_path
    if voice_path is None or not Path(voice_path).exists():
        raise FileNotFoundError(
            "A Luxembourgish voice pack is required. Set paths.voice_init to an existing file."
        )
    voice_tensor = load_voice_tensor(Path(voice_path))
    train_voice = cfg.model.train_voice_pack
    model = TrainableKModel(
        config=cfg.paths.config_json,
        checkpoint=str(cfg.paths.base_ckpt),
        voice_table=voice_tensor.squeeze(1),
        train_voice=train_voice,
        disable_complex_decoder=cfg.model.disable_complex_decoder,
    )
    model.to(device)
    if cfg.model.freeze_bert_epochs > 0:
        model.freeze_submodules(bert=True)
    if cfg.model.freeze_text_encoder_epochs > 0:
        model.freeze_submodules(text_encoder=True)
    return model


def create_optimizer(cfg: TrainingConfig, model: TrainableKModel) -> Optimizer:
    param_groups = model.parameter_groups(
        base_lr=cfg.optim.lr,
        voice_lr=cfg.model.voice_pack_lr if cfg.model.train_voice_pack else cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    if cfg.optim.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(param_groups, betas=cfg.optim.betas, eps=cfg.optim.eps)
    else:
        optimizer = torch.optim.Adam(param_groups, betas=cfg.optim.betas, eps=cfg.optim.eps)
    return optimizer


def create_scheduler(cfg: TrainingConfig, optimizer: Optimizer, total_steps: int) -> Optional[_LRScheduler]:
    scheduler_type = cfg.optim.scheduler.lower()
    if scheduler_type == "none":
        return None

    def lr_lambda(current_step: int) -> float:
        if current_step < cfg.optim.warmup_steps:
            return float(current_step) / float(max(1, cfg.optim.warmup_steps))
        progress = (current_step - cfg.optim.warmup_steps) / float(max(1, total_steps - cfg.optim.warmup_steps))
        progress = max(0.0, min(1.0, progress))
        if scheduler_type == "cosine":
            return max(cfg.optim.min_lr / cfg.optim.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        if scheduler_type == "exponential":
            return max(cfg.optim.min_lr / cfg.optim.lr, math.exp(-5 * progress))
        raise ValueError(f"Unsupported scheduler type '{scheduler_type}'")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_audio_batch(paths: Sequence[Path], sample_rate: int, device: torch.device) -> torch.Tensor:
    waveforms: List[torch.Tensor] = []
    max_length = 0
    for path in paths:
        waveform, sr = torchaudio.load(str(path))
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        waveform = waveform.squeeze(0)
        max_length = max(max_length, waveform.numel())
        waveforms.append(waveform)
    batch = torch.zeros(len(waveforms), max_length, dtype=torch.float32, device=device)
    for idx, waveform in enumerate(waveforms):
        batch[idx, : waveform.numel()] = waveform.to(device)
    return batch


def move_batch_to_device(batch: MutableMapping[str, object], device: torch.device) -> None:
    tensor_keys = {
        "input_ids",
        "phoneme_mask",
        "phoneme_lengths",
        "durations",
        "duration_mask",
        "mel",
        "mel_mask",
        "f0",
        "uv",
        "voice_rows",
    }
    for key in tensor_keys:
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)


def compute_loss(
    loss_fn: LossComputer,
    model_output: TrainableKModelOutput,
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return loss_fn(model_output, batch)


@dataclass
class Checkpoint:
    epoch: int
    global_step: int
    best_score: float


def save_checkpoint(
    cfg: TrainingConfig,
    model: TrainableKModel,
    optimizer: Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    checkpoint_state: Checkpoint,
    *,
    is_best: bool,
    scheduler: Optional[_LRScheduler],
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "checkpoint": checkpoint_state.__dict__,
        "config": cfg.to_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    path = cfg.paths.checkpoint_dir / "latest.pt"
    torch.save(state, path)
    if is_best:
        torch.save(state, cfg.paths.checkpoint_dir / "best.pt")


def export_voice_table(model: TrainableKModel, path: Path) -> None:
    table = model.export_voice_table()
    payload = {
        "table": table.cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_one_epoch(
    *,
    cfg: TrainingConfig,
    model: TrainableKModel,
    loss_fn: LossComputer,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    writer: Optional[SummaryWriter],
    scheduler: Optional[_LRScheduler],
) -> Tuple[int, float]:
    model.train()
    total_loss = 0.0
    grad_accum = cfg.optim.grad_accum_steps
    log_interval = cfg.runtime.log_interval
    batches_processed = 0
    for batch_idx, batch in enumerate(train_loader, start=1):
        batch_start = time.time()
        if batch_idx == 1:
            logger.info("Starting first training batch (may take a moment while kernels warm up)")
        batch = dict(batch)
        move_batch_to_device(batch, device)
        audio_paths = batch.get("audio_paths", [])
        if audio_paths:
            batch["audio_target"] = load_audio_batch(audio_paths, cfg.data.sample_rate, device)

        use_teacher = epoch < cfg.model.teacher_force_epochs
        autocast_enabled = cfg.optim.use_amp and device.type == "cuda"
        forward_start = time.time()
        with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
            output = model(
                batch,
                use_teacher_durations=use_teacher,
                detach_voice=not cfg.model.train_voice_pack,
            )
            loss_components = compute_loss(loss_fn, output, batch)
            loss = loss_components["total"] / grad_accum
        forward_time = time.time() - forward_start
        if batch_idx == 1:
            logger.info(
                "First forward pass complete (%.2fs) | total=%.4f",
                forward_time,
                loss_components["total"].item(),
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Train batch %d | total=%.4f dur=%.4f f0=%.4f stft=%.4f",
                batch_idx,
                loss_components["total"].item(),
                (loss_components.get("duration_bce", torch.tensor(0.0)) + loss_components.get("duration_l1", torch.tensor(0.0))).item(),
                loss_components.get("f0", torch.tensor(0.0)).item() if "f0" in loss_components else 0.0,
                (loss_components.get("stft_sc", torch.tensor(0.0)) + loss_components.get("stft_mag", torch.tensor(0.0))).item(),
            )

        backward_start = time.time()
        scaler.scale(loss).backward()
        if batch_idx == 1:
            logger.info("First backward pass complete (%.2fs)", time.time() - backward_start)

        if batch_idx % grad_accum == 0:
            opt_start = time.time()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            step_time = time.time() - opt_start
            if batch_idx == grad_accum:
                logger.info(
                    "Completed first optimizer step (batch %.2fs, step %.2fs)",
                    time.time() - batch_start,
                    step_time,
                )
            if global_step % log_interval == 0:
                dur_loss = (
                    loss_components.get("duration_bce", torch.tensor(0.0))
                    + loss_components.get("duration_l1", torch.tensor(0.0))
                ).item()
                stft_loss = (
                    loss_components.get("stft_sc", torch.tensor(0.0))
                    + loss_components.get("stft_mag", torch.tensor(0.0))
                ).item()
                f0_loss = loss_components.get("f0", torch.tensor(0.0)).item()
                logger.info(
                    "Step %d | total=%.4f dur=%.4f f0=%.4f stft=%.4f",
                    global_step,
                    loss_components["total"].item(),
                    dur_loss,
                    f0_loss,
                    stft_loss,
                )

                if writer is not None:
                    for key, value in loss_components.items():
                        writer.add_scalar(f"train/{key}", value.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        total_loss += loss_components["total"].item()
        batches_processed += 1

    if batches_processed and batches_processed % grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        global_step += 1

    avg_loss = total_loss / len(train_loader)
    return global_step, avg_loss


def evaluate(
    *,
    cfg: TrainingConfig,
    model: TrainableKModel,
    loss_fn: LossComputer,
    val_loader: DataLoader,
    device: torch.device,
    global_step: int,
    writer: Optional[SummaryWriter],
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_stft = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = dict(batch)
            move_batch_to_device(batch, device)
            audio_paths = batch.get("audio_paths", [])
            if audio_paths:
                batch["audio_target"] = load_audio_batch(audio_paths, cfg.data.sample_rate, device)

            output = model(
                batch,
                use_teacher_durations=False,
                detach_voice=True,
            )
            components = compute_loss(loss_fn, output, batch)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Validation batch | total=%.4f stft=%.4f",
                    components["total"].item(),
                    (components.get("stft_sc", torch.tensor(0.0)) + components.get("stft_mag", torch.tensor(0.0))).item(),
                )
            total_loss += components["total"].item()
            total_stft += (components["stft_sc"] + components["stft_mag"]).item()

    mean_loss = total_loss / len(val_loader)
    mean_stft = total_stft / len(val_loader)

    if writer is not None:
        writer.add_scalar("val/total", mean_loss, global_step)
        writer.add_scalar("val/stft", mean_stft, global_step)
    return mean_loss, mean_stft


def setup_logging(log_dir: Path) -> Optional[SummaryWriter]:
    log_dir.mkdir(parents=True, exist_ok=True)
    if SummaryWriter is None:
        logger.warning("TensorBoard not available; proceeding without summary writer")
        return None
    return SummaryWriter(log_dir=str(log_dir))


def train(config_path: Path, *, resume: Optional[Path] = None) -> None:
    cfg = TrainingConfig.from_toml(config_path)
    cfg.paths.ensure_directories()
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Loaded training configuration from %s", config_path)
    logger.debug("Resolved paths: %s", cfg.paths)

    set_seed(cfg.runtime.seed)

    base_config = load_json(cfg.paths.config_json)
    mel_kwargs = {
        "sample_rate": cfg.data.sample_rate,
        "n_fft": base_config["istftnet"].get("gen_istft_n_fft", 1024),
        "hop_length": base_config["istftnet"].get("gen_istft_hop_size", cfg.data.hop_length),
        "win_length": base_config["istftnet"].get("gen_istft_n_fft", 1024),
        "n_mels": base_config.get("n_mels", 80),
        "mel_fmin": base_config.get("mel_fmin", 0.0),
        "mel_fmax": base_config.get("mel_fmax", cfg.data.sample_rate / 2),
    }
    stft_specs = (
        STFTSpec(fft_size=1024, hop_length=256, win_length=1024),
        STFTSpec(fft_size=2048, hop_length=512, win_length=2048),
    )

    device_str = cfg.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model = create_model(cfg, device)
    optimizer = create_optimizer(cfg, model)
    train_loader, val_loader = prepare_dataloaders(cfg)
    total_steps = cfg.runtime.epochs * math.ceil(len(train_loader) / max(1, cfg.optim.grad_accum_steps))
    logger.info("Total optimisation steps across training: %d", total_steps)
    scheduler = create_scheduler(cfg, optimizer, total_steps)

    loss_fn = LossComputer(
        cfg.losses,
        sample_rate=cfg.data.sample_rate,
        stft_specs=stft_specs,
        mel_kwargs=mel_kwargs,
    ).to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.optim.use_amp)
    writer = setup_logging(cfg.paths.log_dir)

    start_epoch = 0
    global_step = 0
    best_score = float("inf")

    if resume is not None and resume.exists():
        logger.info("Resuming from checkpoint %s", resume)
        state = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        checkpoint_state = state.get("checkpoint", {})
        start_epoch = checkpoint_state.get("epoch", 0) + 1
        global_step = checkpoint_state.get("global_step", 0)
        best_score = checkpoint_state.get("best_score", float("inf"))
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        logger.info("Resumed training from %s (epoch=%d, global_step=%d)", resume, start_epoch, global_step)

    for epoch in range(start_epoch, cfg.runtime.epochs):
        if epoch == cfg.model.freeze_bert_epochs:
            model.unfreeze_submodules(bert=True)
        if epoch == cfg.model.freeze_text_encoder_epochs:
            model.unfreeze_submodules(text_encoder=True)

        global_step, train_loss = train_one_epoch(
            cfg=cfg,
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            writer=writer,
            scheduler=scheduler,
        )

        val_loss, val_stft = evaluate(
            cfg=cfg,
            model=model,
            loss_fn=loss_fn,
            val_loader=val_loader,
            device=device,
            global_step=global_step,
            writer=writer,
        )

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_stft=%.4f",
            epoch + 1,
            cfg.runtime.epochs,
            train_loss,
            val_loss,
            val_stft,
        )

        checkpoint_state = Checkpoint(epoch=epoch, global_step=global_step, best_score=best_score)
        is_best = val_stft < best_score
        if is_best:
            best_score = val_stft
            checkpoint_state.best_score = best_score

        save_checkpoint(
            cfg,
            model,
            optimizer,
            scaler,
            checkpoint_state,
            is_best=is_best,
            scheduler=scheduler,
        )

    export_voice_table(model, cfg.paths.voice_export_path)
    if writer is not None:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the Kokoro model on Luxembourgish data")
    parser.add_argument("--config", type=Path, required=True, help="Path to training TOML configuration file")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
