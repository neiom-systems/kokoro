"""
Training Loss Functions for Kokoro Fine-Tuning

Implements loss functions for supervised fine-tuning on Luxembourgish data.

Based on Kokoro architecture components:
- Predictor: Outputs duration and F0 predictions (lines 105-109 in model.py)
- Decoder: Generates audio waveform (line 118 in model.py)

Loss Components:
1. Duration Loss:
   - MSE between predicted phoneme durations and ground truth alignments
   - Predicted: torch.sigmoid(duration_proj(x)).sum(axis=-1) from model forward
   - Ground truth: Per-phoneme frame counts from forced alignment
   - Shape: [batch, phoneme_len], values 1-50 frames (max_dur=50 from config)

2. F0 Loss:
   - MSE between predicted F0 contour and ground truth pitch
   - Predicted: F0_pred from predictor.F0Ntrain() (line 115 in model.py)
   - Ground truth: F0 extracted from audio using pyworld/CREPE
   - Shape: [batch, T_audio]

3. Reconstruction Loss:
   - L1 or MSE between generated audio and ground truth audio
   - Predicted: Audio waveform from decoder (line 118 in model.py)
   - Ground truth: Original Luxembourgish audio @ 24kHz
   - Shape: [batch, T_audio], sampled at 24000 Hz

4. Total Weighted Loss:
   - Weighted sum: w1*loss_dur + w2*loss_f0 + w3*loss_recon
   - Weights configurable in config.py (typically: dur=1.0, f0=0.5, recon=1.0)
   - Returns single scalar for backpropagation

Optional: Spectral loss (multi-scale STFT) for better audio quality.
"""

