"""
Kokoro Training Module for Luxembourgish Fine-Tuning

This module contains all components needed to fine-tune the pretrained Kokoro-82M model
on single-speaker Luxembourgish data (32k samples from data/luxembourgish_male_corpus).

Key insight: Voice embeddings are [510, 1, 256] per position, not a single [1, 256] vector.
The model uses position-specific speaker conditioning throughout the 510-token sequence.
"""

from .dataset import LuxembourgishDataset
from .speaker_encoder import SpeakerEncoder
from .model import TrainableKModel
from .losses import KokoroTrainingLoss
from .config import TrainingConfig

__all__ = [
    'LuxembourgishDataset',
    'SpeakerEncoder',
    'TrainableKModel',
    'KokoroTrainingLoss',
    'TrainingConfig',
]

