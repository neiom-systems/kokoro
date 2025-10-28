"""
Training configuration contract.

This file should eventually expose a dataclass (or Hydra/OmegaConf style config) that
captures every knob we need for Luxembourgish fine-tuning. Before writing code, spell
out the fields so we agree on the experiment surface area.

Suggested structure:

1. **Paths**
   - `base_ckpt`: path to the pretrained Kokoro checkpoint (default: `base_model/kokoro-v1_0.pth`).
   - `config_json`: path to the downloaded `config.json` (needed to mirror STFT + vocab).
   - `train_csv` / `test_csv`: metadata files under `data/luxembourgish_male_corpus`.
   - `feature_root`: root directory that stores cached mels/F0/durations/voice packs.
   - `checkpoint_dir` / `log_dir`: where to write checkpoints and tensorboard logs.

2. **Data / preprocessing**
   - `sample_rate`: 24_000 (fixed by base model).
   - `hop_length`: value pulled from `config.json['istftnet']['hop_length']` (confirm).
   - `max_input_len`: 510 phoneme tokens (context length minus BOS/EOS).
   - Flags for `recompute_features` vs. `use_cache`.
   - Choice of forced aligner + paths to MFA dictionary/acoustic model.

3. **Model toggles**
   - Freeze schedule for ALBERT/text encoder (e.g. `freeze_bert_epochs=10`).
   - Whether the Luxembourgish voice pack is trainable (`train_voice_pack=True`) and its LR.
   - Dropout overrides, possibility to disable complex STFT in decoder for memory savings.

4. **Optimisation**
   - Optimiser type (AdamW), base LR, weight decay.
   - Separate LR for newly introduced parameters (voice pack / projection head).
   - Scheduler choice (cosine with warmup, exponential decay, etc.).
   - Gradient clipping norm and gradient accumulation steps.
   - AMP flag (`use_amp=True`).

5. **Loss weights**
   - Duration regression weight (`lambda_dur`).
   - Pitch/F0 loss weight (`lambda_f0`).
   - Noise/energy loss weight (`lambda_noise` if we decide to supervise it).
  - Multi-scale STFT / mel loss weights (`lambda_stft`, `lambda_mel`).

6. **Runtime**
   - `epochs`, `steps_per_val`, `batch_size`, `num_workers`.
   - Seed, deterministic flags, device override (`cuda`, `cpu`, `mps`).

Document any coupling between fields (e.g. hop length must agree with duration frame unit)
and default values that mirror the base Kokoro config. Once agreed, implement a dataclass
with validation that raises on inconsistent setups.
"""
