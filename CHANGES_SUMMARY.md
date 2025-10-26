# Summary of Changes for Luxembourgish Support

## Files Modified

### 1. `kokoro/pipeline.py`
**Purpose**: Core pipeline file that handles language-specific G2P initialization

**Changes**:
- Added `'lb': 'l'` and `'luxembourgish': 'l'` to `ALIASES` (lines 21-22)
- Added `'Luxembourgish'` to `LANG_CODES` with key `'l'` (lines 43-44)
- Added Luxembourgish initialization in `__init__` method (lines 146-154)
  - Imports `from misaki import lb`
  - Creates `lb.LBG2P()` instance
  - Provides helpful error messages

**Integration Point**: 
- Uses the same G2P pattern as Japanese and Chinese (lines 130-145, 146-154)
- Returns `(phonemes, None)` which is compatible with existing non-English processing (line 448)

### 2. `kokoro/__main__.py`
**Purpose**: Command-line interface for Kokoro

**Changes**:
- Added `"l"` (Luxembourgish) to `languages` list (line 33)

**Impact**: Users can now use `-l l` or `--language l` in CLI

### 3. `README.md`
**Purpose**: Documentation for users

**Changes**:
- Added Luxembourgish to language list with installation instructions:
  ```python
  # 🇱🇺 'l' => Luxembourgish lb: pip install git+https://github.com/neiom-systems/misaki.git
  ```

## How to Use

### Python API
```python
from kokoro import KPipeline

pipeline = KPipeline(lang_code='l')  # Luxembourgish
results = list(pipeline("Moien alleguer, wéi geet et?", voice='ll_your_voice'))
```

### CLI
```bash
python -m kokoro -l l --text "Moien" -o output.wav --voice ll_your_voice
```

## Installation
```bash
pip install git+https://github.com/neiom-systems/misaki.git
```

## How It Works

1. **Language Detection**: When `lang_code='l'` is passed, the pipeline initializes Luxembourgish G2P
2. **G2P Module**: Uses `misaki.lb.LBG2P()` from the custom fork
3. **Dictionary**: Contains 33,786 word-to-phoneme mappings from LOD
4. **Output Format**: Returns IPA phonemes that are compatible with Kokoro's model
5. **Processing**: Same chunking and processing logic as other non-English languages

## Compatibility

The Luxembourgish G2P (`lb.LBG2P.__call__()`) returns `(phonemes: str, None)` which is exactly what the existing pipeline expects at line 448:
```python
ps, _ = self.g2p(chunk)  # Matches LBG2P return signature
```

This means no changes were needed to the processing logic - only the initialization.

## Testing

Run the test script to verify integration:
```bash
python test_luxembourgish_integration.py
```

This will test:
1. Import and initialization
2. G2P conversion
3. Text chunking
4. Phoneme formatting

