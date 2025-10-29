"""Configuration objects for Luxembourgish fine-tuning.

This module defines a validated configuration surface for the Kokoro training
pipeline.  It mirrors the design notes in this directory and is intentionally
explicit about every knob so downstream code can rely on a single source of
truth.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, List, Mapping, MutableMapping, Optional, Tuple, Union, get_args, get_origin, get_type_hints

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    tomllib = None  # type: ignore

Pathish = Union[str, Path]


def _coerce_path(value: Optional[Pathish]) -> Optional[Path]:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def _ensure_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value})")


def _ensure_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0 (got {value})")


@dataclass(slots=True)
class PathsConfig:
    """Filesystem layout for fine-tuning artefacts."""

    WORKSPACE_ENV_VARS: ClassVar[Tuple[str, ...]] = ("KOKORO_WORKSPACE", "RUNPOD_WORKSPACE", "WORKSPACE")

    base_ckpt: Path = Path("base_model/kokoro-v1_0.pth")
    config_json: Path = Path("base_model/config.json")
    train_csv: Path = Path("data/luxembourgish_male_corpus/train/metadata.csv")
    test_csv: Path = Path("data/luxembourgish_male_corpus/test/metadata.csv")
    feature_root: Path = Path("data/luxembourgish_male_corpus/features")
    checkpoint_dir: Path = Path("experiments/lb/checkpoints")
    log_dir: Path = Path("experiments/lb/logs")
    voice_init: Optional[Path] = None
    voice_export_path: Path = Path("experiments/lb/voices/lb_max.pt")
    aligner_acoustic_model: Optional[Path] = None
    aligner_dictionary: Optional[Path] = None

    def __post_init__(self) -> None:
        for name in (
            "base_ckpt",
            "config_json",
            "train_csv",
            "test_csv",
            "feature_root",
            "checkpoint_dir",
            "log_dir",
            "voice_init",
            "voice_export_path",
            "aligner_acoustic_model",
            "aligner_dictionary",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _coerce_path(value))

    def ensure_directories(self) -> None:
        """Create directories that must exist before training."""

        for directory in {
            self.feature_root,
            self.checkpoint_dir,
            self.log_dir,
            self.voice_export_path.parent,
        }:
            directory.mkdir(parents=True, exist_ok=True)

    def resolve(self, base_dir: Optional[Path] = None) -> None:
        """Convert all configured paths to absolute paths.

        The search order is: provided ``base_dir`` (typically the config file
        location), any recognised workspace environment variables, and finally
        the current working directory.
        """

        candidate_roots: List[Path] = []
        if base_dir is not None:
            candidate_roots.append(base_dir)
        for env_var in self.WORKSPACE_ENV_VARS:
            value = os.environ.get(env_var)
            if value:
                candidate_roots.append(Path(value))
        candidate_roots.append(Path.cwd())

        def resolve_path(path_value: Optional[Path]) -> Optional[Path]:
            if path_value is None:
                return None
            if path_value.is_absolute():
                return path_value
            for root in candidate_roots:
                candidate = (root / path_value).resolve()
                if candidate:
                    return candidate
            return path_value.resolve()

        for name in (
            "base_ckpt",
            "config_json",
            "train_csv",
            "test_csv",
            "feature_root",
            "checkpoint_dir",
            "log_dir",
            "voice_init",
            "voice_export_path",
            "aligner_acoustic_model",
            "aligner_dictionary",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, resolve_path(value))


@dataclass(slots=True)
class DataConfig:
    """Data preprocessing and caching parameters."""

    sample_rate: int = 24_000
    hop_length: int = 240
    max_input_len: int = 510
    recompute_features: bool = False
    use_cache: bool = True
    forced_aligner: str = "mfa"
    phoneme_inventory_path: Optional[Path] = None
    g2p_cache: Optional[Path] = None
    max_mel_frames: Optional[int] = None

    def __post_init__(self) -> None:
        _ensure_positive("sample_rate", self.sample_rate)
        _ensure_positive("hop_length", self.hop_length)
        _ensure_positive("max_input_len", self.max_input_len)
        if self.sample_rate % self.hop_length != 0:
            raise ValueError(
                f"sample_rate ({self.sample_rate}) must be divisible by hop_length ({self.hop_length})"
            )
        self.phoneme_inventory_path = _coerce_path(self.phoneme_inventory_path)
        self.g2p_cache = _coerce_path(self.g2p_cache)
        if not (self.use_cache or self.recompute_features):
            raise ValueError("At least one of use_cache or recompute_features must be True")
        if self.max_mel_frames is not None and self.max_mel_frames <= 0:
            raise ValueError("max_mel_frames must be > 0 when set")


@dataclass(slots=True)
class ModelConfig:
    """Model-specific toggles and training aids."""

    freeze_bert_epochs: int = 10
    freeze_text_encoder_epochs: int = 5
    teacher_force_epochs: int = 3
    train_voice_pack: bool = True
    voice_pack_lr: float = 5e-4
    dropout_override: Optional[float] = None
    disable_complex_decoder: bool = False
    max_duration_frames: int = 50
    use_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        for name in ("freeze_bert_epochs", "freeze_text_encoder_epochs", "teacher_force_epochs", "max_duration_frames"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0 (got {value})")
        if self.dropout_override is not None and not (0.0 <= self.dropout_override < 1.0):
            raise ValueError("dropout_override must be within [0, 1) when provided")
        _ensure_non_negative("voice_pack_lr", self.voice_pack_lr)
        if not self.train_voice_pack and self.voice_pack_lr > 0.0:
            raise ValueError("Set voice_pack_lr to 0 when train_voice_pack is False")
        if not isinstance(self.use_gradient_checkpointing, bool):
            raise TypeError("use_gradient_checkpointing must be a boolean")


@dataclass(slots=True)
class OptimConfig:
    """Optimisation hyperparameters."""

    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    scheduler: str = "cosine"
    warmup_steps: int = 1_000
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    use_amp: bool = True

    def __post_init__(self) -> None:
        if self.optimizer.lower() not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        _ensure_non_negative("learning rate", self.lr)
        _ensure_non_negative("weight_decay", self.weight_decay)
        if not (0 < self.betas[0] < 1 and 0 < self.betas[1] < 1):
            raise ValueError("betas must be in (0, 1)")
        _ensure_positive("warmup_steps", self.warmup_steps)
        _ensure_non_negative("min_lr", self.min_lr)
        _ensure_positive("grad_clip_norm", self.grad_clip_norm)
        _ensure_positive("grad_accum_steps", self.grad_accum_steps)
        _ensure_non_negative("eps", self.eps)
        if self.scheduler.lower() not in {"cosine", "none", "exponential"}:
            raise ValueError("scheduler must be one of: 'cosine', 'none', 'exponential'")


@dataclass(slots=True)
class LossConfig:
    """Weights for individual training losses."""

    lambda_dur: float = 1.0
    lambda_f0: float = 1.0
    lambda_noise: float = 0.0
    lambda_stft: float = 1.0
    lambda_mel: float = 0.5
    voice_l2: float = 1e-4

    def __post_init__(self) -> None:
        for name in ("lambda_dur", "lambda_f0", "lambda_noise", "lambda_stft", "lambda_mel", "voice_l2"):
            _ensure_non_negative(name, getattr(self, name))
        if self.lambda_stft == 0.0 and self.lambda_mel == 0.0:
            raise ValueError("At least one of lambda_stft or lambda_mel must be > 0")


@dataclass(slots=True)
class RuntimeConfig:
    """Loop control and reproducibility settings."""

    epochs: int = 50
    batch_size: int = 4
    steps_per_val: int = 500
    num_workers: int = 4
    seed: int = 1337
    deterministic: bool = False
    device: Optional[str] = None
    log_interval: int = 50
    checkpoint_interval_steps: int = 10_000
    prefetch_factor: int = 2
    persistent_workers: bool = True
    enable_compile: bool = False
    max_steps: Optional[int] = None
    keep_step_checkpoints: int = 5
    force_checkpoint_steps: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _ensure_positive("epochs", self.epochs)
        _ensure_positive("batch_size", self.batch_size)
        _ensure_positive("steps_per_val", self.steps_per_val)
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0 (got {self.num_workers})")
        _ensure_positive("log_interval", self.log_interval)
        _ensure_positive("checkpoint_interval_steps", self.checkpoint_interval_steps)
        if self.num_workers > 0 and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be > 0 when num_workers > 0")
        if not isinstance(self.enable_compile, bool):
            raise TypeError("enable_compile must be a boolean")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be > 0 when set")
        if self.keep_step_checkpoints <= 0:
            raise ValueError("keep_step_checkpoints must be > 0")
        self.force_checkpoint_steps = tuple(self.force_checkpoint_steps)


@dataclass(slots=True)
class F0Config:
    """F0 extraction configuration."""
    method: str = "fast"
    min_f0: float = 60.0
    max_f0: float = 700.0
    voicing_threshold: float = 0.6


@dataclass(slots=True)
class TrainingConfig:
    """Top-level container for all fine-tuning configuration blocks."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    f0: F0Config = field(default_factory=F0Config)

    def __post_init__(self) -> None:
        self._validate_cross_dependencies()

    def resolve_paths(self, base_dir: Optional[Path] = None) -> None:
        self.paths.resolve(base_dir)

    def _validate_cross_dependencies(self) -> None:
        if self.model.teacher_force_epochs > self.runtime.epochs:
            raise ValueError(
                "teacher_force_epochs cannot exceed total epochs "
                f"({self.model.teacher_force_epochs} > {self.runtime.epochs})"
            )
        if self.model.freeze_text_encoder_epochs > self.runtime.epochs:
            raise ValueError("freeze_text_encoder_epochs cannot exceed total epochs")
        if self.model.freeze_bert_epochs > self.runtime.epochs:
            raise ValueError("freeze_bert_epochs cannot exceed total epochs")
        if self.model.max_duration_frames <= 0:
            raise ValueError("max_duration_frames must be > 0")
        if self.model.max_duration_frames > 512:
            raise ValueError("max_duration_frames exceeds reasonable bound (512)")
        text_context = self.data.max_input_len + 2  # +BOS/EOS
        if text_context > 512:
            raise ValueError(
                f"max_input_len ({self.data.max_input_len}) breaches Kokoro context window (512 tokens)"
            )
        if self.losses.voice_l2 > 0.0 and not self.model.train_voice_pack:
            raise ValueError("voice_l2 regulariser requires train_voice_pack=True")

    def assert_with_model_config(self, model_cfg: Mapping[str, Any]) -> None:
        """Validate against a loaded config.json dictionary."""

        required = {
            "n_mels",
            "istftnet",
            "max_dur",
            "n_token",
            "hidden_dim",
        }
        missing = required.difference(model_cfg.keys())
        if missing:
            raise KeyError(f"Model config missing required keys: {sorted(missing)}")
        max_dur = int(model_cfg["max_dur"])
        if max_dur != self.model.max_duration_frames:
            raise ValueError(
                f"max_dur mismatch: model={max_dur}, config={self.model.max_duration_frames}"
            )
        hop = int(model_cfg["istftnet"].get("gen_istft_hop_size", 0))
        if hop and hop != self.data.hop_length:
            raise ValueError(
                f"hop_length mismatch: model={hop}, config={self.data.hop_length}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the configuration into a plain dictionary."""

        return {
            "paths": self._dataclass_to_dict(self.paths),
            "data": self._dataclass_to_dict(self.data),
            "model": self._dataclass_to_dict(self.model),
            "optim": self._dataclass_to_dict(self.optim),
            "losses": self._dataclass_to_dict(self.losses),
            "runtime": self._dataclass_to_dict(self.runtime),
        }

    def with_updates(self, **kwargs: Any) -> "TrainingConfig":
        """Return a new config with updated top-level sections."""

        current = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in kwargs.items():
            if key not in current:
                raise KeyError(f"Unknown configuration section '{key}'")
            section = current[key]
            if dataclass_is_instance(section) and isinstance(value, Mapping):
                current[key] = replace(section, **value)
            else:
                current[key] = value
        return TrainingConfig(**current)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrainingConfig":
        """Instantiate a configuration from a nested dictionary."""

        kwargs: MutableMapping[str, Any] = {}
        # Use get_type_hints to resolve forward references
        type_hints = get_type_hints(cls)
        
        for field_info in fields(cls):
            name = field_info.name
            # Get the resolved type from type hints
            section_cls = type_hints.get(name, field_info.type)
            section_data = raw.get(name, {})
            
            if not dataclass_is_type(section_cls):
                raise TypeError(f"Unsupported section type for '{name}': {section_cls!r}")
            kwargs[name] = build_dataclass(section_cls, section_data)
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, path: Pathish) -> "TrainingConfig":
        """Load configuration from a TOML file."""

        if tomllib is None:
            raise RuntimeError("tomllib is unavailable; upgrade to Python 3.11+")
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        cfg = cls.from_dict(raw)
        cfg.resolve_paths(base_dir=Path(path).resolve().parent)
        return cfg

    @staticmethod
    def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field_info in fields(obj):
            value = getattr(obj, field_info.name)
            if isinstance(value, Path):
                result[field_info.name] = str(value)
            elif dataclass_is_instance(value):
                result[field_info.name] = TrainingConfig._dataclass_to_dict(value)
            else:
                result[field_info.name] = value
        return result


def dataclass_is_type(obj: Any) -> bool:
    return isinstance(obj, type) and is_dataclass(obj)


def dataclass_is_instance(obj: Any) -> bool:
    return is_dataclass(obj) and not isinstance(obj, type)


def field_accepts_path(field_type: Any) -> bool:
    if field_type is Path:
        return True
    origin = get_origin(field_type)
    if origin is Union:
        return any(field_accepts_path(arg) for arg in get_args(field_type) if arg is not type(None))
    return False


def build_dataclass(cls: type, data: Optional[Mapping[str, Any]]) -> Any:
    if data is None:
        return cls()  # type: ignore[call-arg]
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        name = field_info.name
        if name not in data:
            continue
        value = data[name]
        if isinstance(value, (str, Path)) and field_accepts_path(field_info.type):
            kwargs[name] = _coerce_path(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "TrainingConfig",
    "PathsConfig",
    "DataConfig",
    "ModelConfig",
    "OptimConfig",
    "LossConfig",
    "RuntimeConfig",
]
