"""
Luxembourgish voice-table generation via a speaker encoder.

Structure analysis recap:
- Each Kokoro voice pack is a tensor shaped `[510, 1, 256]`.
- Inference selects row `(sequence_length - 1)` to obtain the `[1, 256]` vector that
  conditions both the decoder (`[:128]`) and the prosody predictor (`[128:]`).

Chosen approach
---------------
- Feed curated Luxembourgish reference audio through a robust speaker encoder
  (e.g. WavLM-large or HuBERT-large) to obtain a fixed-length speaker embedding.
- Map that embedding to a 256-dim latent via a projection head (small MLP or linear stack).
- Expand the latent into 510 position-specific rows by injecting positional information
  (sinusoidal embeddings, learned index embeddings, or a lightweight transformer).
- Register the resulting `[510, 1, 256]` table as trainable so gradients can refine it
  jointly with Kokoro during fine-tuning.

Key responsibilities
--------------------
1. **Preprocessing**  
   - Normalise audio to match the upstream encoder requirements (16 kHz vs. 24 kHz, volume).
   - Optionally average embeddings over multiple reference clips to reduce noise.
2. **Projection head**  
   - Define the architecture and initialisation scheme that maps the speaker-encoder
     embedding to the base 256-dim latent.
   - Provide hooks for regularisation (e.g. L2 towards initial weights).
3. **Positional expansion**  
   - Produce 510 distinct rows using positional encodings or a learned index-conditional
     network; maintain the same scale (mean ≈ 0, std ≈ 0.15) observed in stock voices.
4. **Caching & I/O**  
   - Implement `generate_table(audio_paths, *, cache_path=None)` that returns the tensor and
     optionally saves it via `torch.save` for reuse.
   - Implement `load_table(path)` / `save_table(tensor, path)` for consistency with existing
     voice packs (stored under `base_model/voices/*.pt`).
5. **Training hooks**  
   - Expose helper functions such as `get_rows(table, phoneme_lengths)` to select the correct
     `[batch, 1, 256]` slices inside the dataloader or model wrapper.
   - Optionally surface a regulariser that nudges the table towards the projection output if
     we later allow the table to drift far during fine-tuning.

Design notes
------------
- Because the corpus is single-speaker, one table suffices; keep the API open for future
  multi-speaker extensions (e.g. dictionary of tables or conditioning on speaker IDs).
- Store both the raw speaker-encoder embedding(s) and the generated table for debugging and
  reproducibility.
- After training, export the refined table to `kokoro/base_model/voices/luxembourgish_male.pt`
  (or another agreed filename) so inference can consume it directly.
"""
