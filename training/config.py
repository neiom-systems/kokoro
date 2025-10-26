"""
Training Configuration for Luxembourgish Fine-Tuning

Defines all hyperparameters, paths, and settings for training.
Based on Kokoro-82M architecture: 82M params, 512 hidden dim, 178 phoneme vocab.
"""

# Configuration for fine-tuning Kokoro on single-speaker Luxembourgish data.
# Handles paths to pretrained weights (base_model/kokoro-v1_0.pth with 'module.' prefix),
# dataset locations, training hyperparameters, and loss weights.
#
# Key settings:
# - Batch size: Limited by [510, 1, 256] speaker embeddings per sample
# - Learning rate: Lower than pretraining for fine-tuning stability
# - Loss weights: Balance duration, F0, and reconstruction losses
# - Freezing strategy: Option to freeze BERT (12 layers, 768-dim) vs full fine-tune

