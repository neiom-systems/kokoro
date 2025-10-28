"""
Offline feature extraction plan.

Everything here should run as preprocessing jobs before training starts. Keep the code
idempotent so we can re-run failed chunks without recomputing the entire corpus.

Mel spectrograms
----------------
- Use the exact STFT settings from `config.json`:
  - `sample_rate = 24000`
  - `n_fft`, `hop_length`, `win_length`, `n_mels`, `mel_fmin`, `mel_fmax`
  - Log-mel scaling (clip to avoid `-inf`)
- Store as float32 tensors (PyTorch `.pt` or NumPy `.npy`). Keep track of the number of
  frames `T`; this must equal the sum of phoneme durations for the sample.

Durations / alignments
----------------------
- Run Montreal Forced Aligner (MFA) or a comparable tool with a Luxembourgish lexicon.
  Reuse the phoneme inventory that `misaki.lb.LBG2P` emits so IDs stay consistent.
- Convert alignments to frame counts in units of the mel hop length (e.g. 240 samples
  per frame ⇒ 100 fps). Enforce `1 ≤ duration ≤ max_dur (50)`; clip or drop samples that
  exceed this bound.
- Persist alignments as integer tensors per utterance. Include metadata for skipped
  segments (for auditing).

Pitch (F0) extraction
---------------------
- Tools: pyworld (fast) or CREPE (more robust). Start with pyworld; fall back to CREPE
  on voiced segments with large gaps.
- Output both `f0` and a voiced/unvoiced mask so the loss can ignore unvoiced frames.
- Ensure the frame rate matches the mel hop (interpolate if necessary).

Noise / energy (optional)
-------------------------
- Kokoro predictor also emits `N_pred` (noise envelope). Decide whether to supervise it:
  - Option A: treat as unsupervised and drop the loss (simplest).
  - Option B: approximate via log-energy or aperiodicity from pyworld.

Caching layout
--------------
```
feature_root/
  train/
    <utt_id>.pt  # contains dict {mel, durations, f0, uv, num_frames}
  test/
    ...
```

Include a manifest file summarising how many utterances were successfully processed,
which ones were skipped, and reasons (too long, alignment failure, etc.).
"""
