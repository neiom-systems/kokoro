"""
Luxembourgish Single-Speaker Dataset Loader

Loads data from data/luxembourgish_male_corpus/{train,test}/metadata.csv.
Each sample contains: audio path, Luxembourgish text, source label.

Responsibilities:
1. Load audio files (WAV @ 24kHz) from audio/ subdirectory
2. Phonemize Luxembourgish text using misaki.lb.LBG2P() (33,786 word dictionary)
3. Convert phonemes to input_ids using Kokoro's 178-token vocab from config.json
4. Return speaker embeddings ([510, 1, 256] per sample - same embedding for all positions)
5. Extract/load ground truth: F0 curves, per-phoneme durations, mel spectrograms
6. Handle padding and batching for variable-length sequences (max 510 phonemes)

Note: All 32k samples use the SAME speaker embedding since data is single-speaker male voice.
"""

