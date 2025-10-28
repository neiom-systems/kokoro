"""
Loss design for Luxembourgish fine-tuning.

The predictor/decoder architecture dictates what supervision signals we can apply. We
need to document the target tensors, appropriate loss functions, and masking rules
before committing to code.

Duration loss
-------------
- Predictor outputs `duration_logits` with shape `[batch, seq_len, max_dur]`.
- Kokoro converts logits into frame counts via `torch.sigmoid(duration_logits).sum(-1)`.
- Training options:
  1. Reproduce the inference path, then apply L1 between predicted frame counts and
     ground-truth durations (in frames). This is simple but ignores the per-bin logits.
  2. Construct a soft target vector per phoneme where bins `< target_duration` are ones
     and the rest zeros, then use BCEWithLogitsLoss on the raw logits (closer to how the
     sigmoid-sum behaves). Favoured approach—document how to build the target mask and
     apply attention masks for padded phonemes.

Pitch (F0) loss
---------------
- `predictor.F0Ntrain` returns `F0_pred` with shape `[batch, T]`.
- Use an L1 or smooth L1 loss on log-F0, masked by voiced frames. Consider additional
  penalties to encourage continuity (delta loss).
- Optionally include an energy/noise loss if we supervise `N_pred` (see extractors).

Spectral / reconstruction losses
--------------------------------
- Direct waveform MSE is brittle. Instead, follow common TTS practice:
  - Multi-resolution STFT loss (sum of magnitude L1 + spectral convergence across
    several FFT sizes).
  - Optional log-mel loss comparing generated vs. ground-truth mels.
- If we keep teacher forcing (decoder consumes ground-truth durations/F0 early in
  training), compute the spectral losses on the teacher-forced audio; otherwise use
  the model’s own predictions.

Auxiliary terms
---------------
- Duration prior (encourage minimal deviation from teacher-forced durations).
- Voice table regulariser (L2 towards initial embedding to prevent drift).
- KL or consistency loss if we later add diffusion/noise scheduling.

Implementation checklist
------------------------
- Accept masks for both phoneme and frame dimensions to ignore padding.
- Return a struct/dict with every component plus the weighted total so the trainer can
  log them separately.
- Pull loss weights from `TrainingConfig` (see `config.py`).
"""
