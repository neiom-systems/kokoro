# Kokoro Training Module for Luxembourgish Fine-Tuning

This directory contains all components needed to fine-tune the pretrained Kokoro-82M model on single-speaker Luxembourgish data.

## File Structure

```
training/
├── __init__.py           # Module exports and initialization
├── config.py             # Training hyperparameters and paths
├── dataset.py            # Luxembourgish dataset loader
├── speaker_encoder.py    # Speaker embedding extraction [510, 1, 256]
├── extractors.py         # F0, duration, mel spectrogram extraction
├── losses.py             # Duration, F0, and reconstruction losses
├── model.py              # Trainable Kokoro wrapper (removes @torch.no_grad)
└── train.py              # Main training loop
```

## Key Architecture Insights (from BASE_MODEL_ANALYSIS.md)

1. **Speaker Embeddings**: `[510, 1, 256]` shape
   - NOT a single `[1, 256]` vector
   - 510 position-specific 256-dim embeddings
   - One per token position in the sequence

2. **Model Components**: 5 main parts
   - `bert`: BERT/ALBERT encoder (12 layers, 768-dim)
   - `bert_encoder`: Linear projection to 512-dim
   - `decoder`: ISTFTNet generator
   - `predictor`: Duration and F0 prediction
   - `text_encoder`: Phoneme sequence encoder

3. **State Dict**: All weights have `'module.'` prefix (DataParallel)

4. **Dataset**: 32,000 Luxembourgish samples
   - Train: 28,800 samples
   - Test: 3,200 samples
   - Single male speaker

## Implementation Order

1. ✅ `config.py` — Define all hyperparameters
2. ✅ `speaker_encoder.py` — Extract one `[510, 1, 256]` embedding
3. ✅ `extractors.py` — F0 and duration extraction utilities
4. ✅ `dataset.py` — Load and preprocess Luxembourgish data
5. ✅ `model.py` — Trainable model wrapper
6. ✅ `losses.py` — Loss function implementations
7. ✅ `train.py` — Main training script

## Next Steps

Each file contains detailed comments explaining:
- What the component does
- Input/output shapes and formats
- Integration points with Kokoro architecture
- Usage in the training pipeline

Start implementing from top to bottom, testing each component before moving to the next.

## Usage (Future)

```python
from training import TrainingConfig, train

config = TrainingConfig(
    base_model_path='base_model/kokoro-v1_0.pth',
    train_data_dir='data/luxembourgish_male_corpus/train',
    test_data_dir='data/luxembourgish_male_corpus/test',
    batch_size=4,
    learning_rate=1e-4,
    num_epochs=50
)

train(config)
```

## Output

Fine-tuned single-speaker Luxembourgish TTS model compatible with:
```python
from kokoro import KPipeline

pipeline = KPipeline(lang_code='l')  # Luxembourgish
audio = pipeline("Moien alleguer, wéi geet et?", voice='luxembourgish_male')
```

