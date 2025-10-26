# Luxembourgish Language Support for Kokoro

## Summary

Kokoro has been successfully modified to support the Luxembourgish (Lëtzebuergesch) language using a custom Misaki G2P implementation from [neiom-systems/misaki](https://github.com/neiom-systems/misaki).

## Changes Made

### 1. `kokoro/pipeline.py`

#### Added Luxembourgish Language Mapping
- **Lines 21-22**: Added `'lb': 'l'` and `'luxembourgish': 'l'` to `ALIASES` dictionary
- **Lines 43-44**: Added `'Luxembourgish'` to `LANG_CODES` dictionary with key `'l'`

#### Added Luxembourgish G2P Initialization
- **Lines 146-154**: Added Luxembourgish initialization branch in `KPipeline.__init__()`
  - Imports `from misaki import lb`
  - Creates `lb.LBG2P()` instance
  - Provides helpful error message with installation instructions
  - Logs successful initialization

### 2. `kokoro/__main__.py`

#### Added Luxembourgish to CLI Languages
- **Line 33**: Added `"l",  # Luxembourgish` to the `languages` list for CLI support

### 3. `README.md`

#### Updated Language Documentation
- **Line 47**: Added Luxembourgish to the language list with installation instructions:
  ```python
  # 🇱🇺 'l' => Luxembourgish lb: pip install git+https://github.com/neiom-systems/misaki.git
  ```

## How It Works

### Language Code
- **Short code**: `'l'`
- **Aliases**: `'lb'`, `'luxembourgish'`

### G2P Implementation
The custom Luxembourgish G2P (`lb.LBG2P`) provides:
- **Dictionary**: 33,786 word-to-phoneme mappings from LOD (Lëtzebuerger Onlinedict)
- **Phoneme format**: IPA (International Phonetic Alphabet)
- **Unknown word handling**: Returns special symbol `❓` for unknown words
- **Punctuation**: Preserves punctuation in output

### Processing Pipeline
1. **Text Input**: Luxembourgish text is passed to the pipeline
2. **Chunking**: Text is split into ~400 character chunks (sentence-aware when possible)
3. **G2P**: Each chunk is converted to phonemes using `lb.LBG2P()`
4. **Model Inference**: Phonemes are fed to the Kokoro model for audio generation
5. **Output**: Generated audio is returned

## Usage

### Basic Usage
```python
from kokoro import KPipeline

# Initialize Luxembourgish pipeline
pipeline = KPipeline(lang_code='l')

# Generate audio
text = "Moien alleguer, wéi geet et?"
results = list(pipeline(text, voice='ll_your_voice'))
```

### CLI Usage
```bash
# Using language flag
python -m kokoro -l l --text "Moien alleguer" -o output.wav --voice ll_your_voice

# Using the alias
python -m kokoro -l lb --text "Moien alleguer" -o output.wav
```

## Installation Requirements

To use Luxembourgish support, install the custom Misaki fork:

```bash
pip install git+https://github.com/neiom-systems/misaki.git
```

This installs the custom G2P module with the Luxembourgish dictionary containing 33,786 word-to-phoneme mappings.

## Architecture

### Module Structure
```
misaki/
├── __init__.py
├── lb.py              # Luxembourgish G2P module
│   └── LBG2P          # Main G2P class
└── data/
    └── lb_gold.json   # Luxembourgish pronunciation dictionary (33,786 entries)
```

### Processing Flow
```
User Text → KPipeline(lang_code='l')
                ↓
         Text Chunking (400 chars)
                ↓
         lb.LBG2P() → Phonemes
                ↓
         Model Forward Pass
                ↓
         Audio Output (24kHz)
```

## Features

1. **Dictionary-Based G2P**: Uses comprehensive LOD dictionary for accurate pronunciation
2. **Unknown Word Handling**: Gracefully handles out-of-vocabulary words
3. **Chunking**: Automatically handles long texts via sentence-aware chunking
4. **Phoneme Limiting**: Respects 510-character phoneme limit per chunk
5. **Error Handling**: Clear error messages and logging

## Integration Points

### In `KPipeline.__init__()`
Luxembourgish is initialized with the same pattern as Japanese and Chinese:
```python
elif lang_code == 'l':
    from misaki import lb
    self.g2p = lb.LBG2P()
```

### In `KPipeline.__call__()`
Luxembourgish text follows the non-English processing path:
- Uses chunking logic (line 398-442)
- Calls `ps, _ = self.g2p(chunk)` (line 448)
- Returns `Result(graphemes=chunk, phonemes=ps, output=output)` (line 442)

## Testing

The Luxembourgish integration follows the same pattern as other non-English languages. The `LBG2P` class returns `(phoneme_string, None)` which is compatible with the existing pipeline code.

## Future Enhancements

Potential improvements:
1. Add fallback mechanism for unknown words (e.g., using espeak)
2. Support for orthographic variants (e.g., accent-insensitive lookup)
3. Add training pipeline for Luxembourgish-specific voices
4. Expand dictionary coverage for domain-specific vocabulary

## Notes

- The phonemizer uses the `kokoro/train/phenomenize_lb.py` module only for **training** purposes
- The **inference** pipeline uses the custom Misaki fork's `lb.LBG2P()` class
- Both use the same LOD dictionary but serve different purposes (training vs. inference)

## References

- [Kokoro Model](https://huggingface.co/hexgrad/Kokoro-82M)
- [Custom Misaki Fork](https://github.com/neiom-systems/misaki)
- [LOD - Lëtzebuerger Online Dictionnaire](https://lod.lu/)

