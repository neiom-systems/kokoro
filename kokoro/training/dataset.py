"""
Luxembourgish dataset loader design.

Goal: produce batches that match Kokoro’s expectations during training.

Data source
-----------
- Metadata CSVs: `data/luxembourgish_male_corpus/{train,test}/metadata.csv`
  with columns `path,text,source`.
- Audio: 24 kHz mono wav files under the corresponding `audio/` directory.

Outputs per sample
------------------
1. `phoneme_ids`: LongTensor shaped `[seq_len]` including BOS/EOS, where `seq_len ≤ 512`.
   - Tokenisation uses `KModel.vocab` (178 phoneme symbols from `config.json`).
   - We must preserve `seq_len_without_special` to select the correct voice row.
2. `mel`: FloatTensor `[80, T]` (or `[T, 80]` depending on convention) computed with the
   same STFT params as Kokoro (`n_fft`, `hop_length`, `win_length`).
3. `durations`: LongTensor `[seq_len_without_special]` giving frame counts per phoneme.
   Sum equals `T` when measured in hop units.
4. `f0`: FloatTensor `[T]` (with voiced/unvoiced mask).
5. `voice_row`: Integer index `seq_len_without_special - 1` selecting the correct `[1, 256]`
   slice from the 510-row Luxembourgish voice table.

For batching, pad phoneme sequences to the max length in batch (<=510 without BOS/EOS),
pad mels/F0 to the max frame length, and create masks (`phoneme_mask`, `mel_mask`).

Caching strategy
----------------
- Run phonemisation + feature extraction offline and store results under
  `feature_root/{split}/{utt_id}.{npz,pt}` to avoid recomputation.
- The dataset class should detect missing caches and either compute them on the fly
  (slow path) or raise with a helpful message.

Integration hooks
-----------------
- Provide a `collate_fn` that stacks tensors, generates masks, and packages the voice
  indices so the dataloader can fetch `voice_table[voice_row]` lazily.
- Support deterministic shuffling via seeded `Generator`.
- Optionally expose an iterable API for streaming (if we ever need to train on-the-fly).

Open questions
--------------
- Where to validate that durations sum to mel frames (reject / auto-fix samples)?
- How to handle utterances that exceed the 510 phoneme limit (skip or split before cache)?
- Should we store energy/noise targets alongside F0 for supervising `predictor.N_proj`?
"""
