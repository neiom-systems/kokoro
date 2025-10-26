"""
Luxembourgish Single-Speaker Dataset Loader

Loads data from data/luxembourgish_male_corpus/{train,test}/metadata.csv.

PRE-PROCESSING REQUIRED: Pre-compute phonemes using misaki.lb.LBG2P() and add 
'phonemes' column to metadata.csv files (run once before training).

Responsibilities:
1. Load audio files (WAV @ 24kHz) from audio/ subdirectory
2. Read pre-computed phonemes from metadata.csv (use misaki.lb.LBG2P() if not present)
3. Convert phonemes to input_ids using Kokoro's 178-token vocab from config.json
4. Return speaker embeddings ([510, 1, 256] per sample - same embedding for all positions)
5. Extract/load ground truth: F0 curves, per-phoneme durations, mel spectrograms
6. Handle padding and batching for variable-length sequences (max 510 phonemes)

Note: All 32k samples use the SAME speaker embedding since data is single-speaker male voice.
"""

