"""
Luxembourgish voice-table strategy.

Structure analysis recap:
- Every stock voice file is a tensor shaped `[510, 1, 256]`.
- During inference, Kokoro selects row `(sequence_length - 1)` and feeds the resulting
  `[1, 256]` vector into the predictor/decoder. The first 128 dims condition the decoder,
  the remaining 128 feed the predictor (`ref_s[:, :128]` vs `ref_s[:, 128:]`).

Tasks for this module
---------------------
1. Decide how to bootstrap a Luxembourgish voice table:
   - **Option A (recommended to start):** clone an existing neutral male voice (e.g.
     `af_narrator.pt`), make it trainable, and let fine-tuning adapt it. This matches the
     original scale/variance and avoids reverse-engineering the extraction process.
   - **Option B:** learn a projection from an external speaker encoder (WavLM/HuBERT) to
     256 dims, then map it to 510 positions (e.g. via positional conditioning). This is
     higher effort and should only be attempted if Option A fails.
2. Provide utilities to load/save the trainable table (`torch.load`/`torch.save` in the
   same format as `base_model/voices/*.pt`) so inference can pick it up seamlessly.
3. During training, expose both the parameter tensor and helper functions:
   - `get_row(batch_seq_lengths)` → returns the `[batch, 1, 256]` tensor matching each
     sample’s phoneme length.
   - Optional temperature or smoothing logic in case we want interpolation between rows.
4. Handle gradient flow: the training wrapper should register the tensor as an
   `nn.Parameter`, optionally with per-row dropout/regularisation to prevent collapse.

Design notes
------------
- Because the Luxembourgish corpus has a single speaker, we only need one table. If we
  ever add multiple speakers, this module must support multiple named tables.
- Keep the numeric range close to the pretrained voices (mean ≈ 0, std ≈ 0.15). When
  cloning an existing voice, store the original copy so we can reset/restart experiments.
- Document how to export the final table to `kokoro/base_model/voices/luxembourgish_male.pt`
  for deployment.
"""
