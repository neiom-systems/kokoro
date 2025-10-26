"""
Audio Feature Extraction Utilities

Extracts ground truth acoustic features from audio for training supervision.

Responsibilities:
1. F0 Extraction:
   - Extract fundamental frequency (pitch) contour from audio
   - Use pyworld, CREPE, or similar F0 tracker
   - Output: [T_audio] float tensor matching audio length at 24kHz
   - Used for F0 prediction loss (model predicts F0, we compare to ground truth)

2. Duration Extraction:
   - Compute per-phoneme frame durations via forced alignment
   - Use Montreal Forced Aligner (MFA) or pretrained aligner
   - Output: [phoneme_len] int tensor, values 1-50 frames (max_dur from config)
   - Maps each phoneme to its duration in acoustic frames

3. Mel Spectrogram Extraction:
   - Convert audio to mel-spectrogram (80 mel bins per config.json)
   - Output: [80, T_audio] float tensor
   - Optional: Used for spectral reconstruction loss

4. Duration Alignment:
   - Align Luxembourgish phonemes (from lb.LBG2P) to audio frames
   - Handle phoneme sequence length vs audio length mismatch
   - Ensure phoneme durations sum to audio length in frames

All extractors operate on 24kHz audio from the Luxembourgish corpus.
"""

