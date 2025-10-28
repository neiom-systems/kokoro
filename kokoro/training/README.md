# Training Design Notes for Luxembourgish Fine-Tuning

These notes capture what we actually need in order to fine-tune Kokoro‑82M on the Luxembourgish corpus that lives in `kokoro/data/luxembourgish_male_corpus`. Every file in this directory is currently documentation only; the intent is to turn each into a concrete component once the plan is solid.

## Quick Reality Check

- **Voice packs are lookup tables.** The structure analysis shows that each speaker pack is a tensor with shape `[510, 1, 256]`. Kokoro does *not* consume the entire tensor at once: for a sequence of length `L`, inference takes row `L-1` (shape `[1, 256]`) and splits it into style (`[:128]`) and decoder conditioning (`[:128]`). For training we will
  1. keep the whole 510-row table around,
  2. expose row selection in the dataset/loader, and
  3. generate those 510 rows via a learned projection from a modern speaker encoder (no direct cloning of existing voices).
- **`forward_with_tokens` is wrapped in `@torch.no_grad()`.** The stock `KModel` is inference-only. A trainable wrapper has to remove the decorator, optionally add teacher-forcing hooks, and return intermediate tensors (durations, F0, noise) needed for losses.
- **Predictor outputs shape decisions.** Duration logits have length `max_dur=50` per phoneme and are squashed with `sigmoid().sum(-1)` to produce frame counts. F0/Noise predictions are 1D sequences aligned with expanded phoneme durations. The losses must respect this behaviour; a vanilla MSE on raw logits will not work.
- **Dataset scale.** The metadata files confirm 28 800 train and 3 200 test utterances (single male speaker, 24 kHz audio). This same corpus also feeds the Luxembourgish voice pack generation, ensuring the `lb_max` embedding matches the fine-tuned weights exactly. We need a streaming loader with phonemization + cached acoustic features because recomputing on the fly will be too expensive.

## Components and Open Questions

| File | Role | Key decisions to finalise |
| ---- | ---- | ------------------------ |
| `config.py` | Central place for experiment hyperparameters and file-system layout | GPU budget, mixed precision, whether to freeze ALBERT, optimiser schedule, gradient clipping, path conventions for cached features |
| `dataset.py` | Builds batches of tokenised text, cached mels/F0/duration, and the correct voice row index | How to run/call Luxembourgish G2P (`misaki.lb.LBG2P`), where to persist forced-alignment artefacts, padding strategy (phoneme length ≤ 510, mel frames align with duration sum) |
| `speaker_encoder.py` | Produces the `[510, 1, 256]` Luxembourgish voice table via a speaker encoder | Define the external encoder (e.g. WavLM/HuBERT), the projection to 256 dims, positional conditioning to expand to 510 rows, caching/export format |
| `extractors.py` | Offline feature extraction scripts (mel spectrogram, F0, energy, alignment) | Target frame hop (likely 240 samples = 10 ms to match Kokoro), tool choices (MFA, Montreal Forced Aligner + Luxembourgish lexicon), failure handling and caching |
| `losses.py` | Aggregates losses for predictor + decoder | Duration regression loss matching sigmoid-sum output, pitch loss with voiced/unvoiced masking, multi-scale STFT loss for waveform reconstruction, optional feature matching |
| `model.py` | Gradient-enabled wrapper around `KModel` | Provide `forward_train(tokens, voice_row, *, teacher_dur=None, teacher_f0=None, teacher_noise=None)`, keep compatibility with pretrained weights (`module.` prefix stripping), expose modules for optimiser parameter groups |
| `train.py` | Orchestration script tying everything together | AMP + gradient accumulation strategy, checkpointing, evaluation hooks (MOS-like samples), logging (tensorboard/comet), early stopping on validation STFT loss |

## High-Level Training Plan

1. **Offline preprocessing**
   - Run the Luxembourgish G2P on all transcripts; save phoneme strings and input_ids to disk.
   - Generate Montréal Forced Aligner (or equivalent) alignments to obtain phoneme durations (in frames that match Kokoro’s hop length).
   - Extract F0 (pyworld/crepe) and 80-bin mel spectrograms with the same STFT settings as `config.json`.
   - Train or precompute the voice-table generator: feed Luxembourgish reference audio through the chosen speaker encoder, map to a 256-dim latent, expand to the 510-position table, and cache the result for fast loading.

2. **Model preparation**
   - Clone the pretrained weights, strip the `module.` prefix, and load into a writable wrapper.
   - Add a training forward method that can optionally consume ground-truth durations/F0 during early epochs (teacher-forcing) to stabilise gradients.
   - Determine which submodules remain frozen (baseline: freeze ALBERT + early text encoder layers for the first few epochs, unfreeze later).

3. **Training loop**
   - Use AdamW with a lower LR for pretrained weights and a higher LR for the new Luxembourgish voice pack.
   - Apply gradient clipping (`clip_grad_norm_=1.0`) and mixed precision (`torch.cuda.amp`) to fit reasonable batch sizes (likely 2–4 on 24 GB GPUs).
   - Track losses separately (duration, F0, STFT) and checkpoint the best validation STFT score.
   - Periodically synthesise validation sentences via `KPipeline(lang_code='l')` using the current voice pack for qualitative checks.

4. **Evaluation & packaging**
   - Export the fine-tuned weights plus the learned voice pack (`voices/lb_max.pt`) so inference can request `voice="lb_max"`.
   - Update the Luxembourgish pipeline configuration to point at the new voice.
   - Capture metrics: loss curves, pitch RMSE, mel reconstruction loss, plus audio samples for listening tests.

## Deliverables Expected from This Directory

- Well-scoped TODO blocks or stub implementations for each module listed above.
- Clear instructions for running offline preprocessing (probably a separate script/notebook referenced from here).
- Rationale for any architectural tweaks (e.g. whether to fine-tune the decoder fully).
- Validation checklist before we declare the Luxembourgish fine-tune “done”.

Once these decisions are locked in, we can start turning each file from prose into code.
