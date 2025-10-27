# Kokoro-82M Model Structure Analysis Results

Generated: 1761520299.0449662

## Voice File Analysis (Sample: af_heart.pt)

- **File Path**: base_model/voices/af_heart.pt
- **File Size**: 0.50 MB
- **Tensor Shape**: (510, 1, 256)
- **Data Type**: torch.float32
- **Total Elements**: 130,560

### Statistical Properties

| Metric | Value |
|--------|-------|
| Min | -1.509776 |
| Max | 1.756575 |
| Mean | -0.004483 |
| Std Dev | 0.152997 |
| Median | -0.012223 |

### Dimension Interpretation

- **Sequence Length**: 510 (max phoneme positions)
- **Batch Size**: 1
- **Embedding Dimension**: 256
- **Purpose**: Pre-computed speaker embeddings for each timestep

## All Voice Files Analysis

- **Total Voices**: 54

### Shape Distribution

- Shape `(510, 1, 256)`: 54 voices

## Model Weights Analysis

- **Total Parameters**: 81,763,410
- **Total Top-level Keys**: 5

### Component Breakdown

| Component | Keys | Parameters |
|-----------|------|------------|
| BERT | 25 | 6,292,480 |
| BERT_ENCODER | 2 | 393,728 |
| TEXT_ENCODER | 54 | 11,520,000 |
| DECODER | 375 | 53,276,190 |
| PREDICTOR | 92 | 10,281,012 |

### Detailed Component Structure

#### BERT

- **Number of Keys**: 25
- **Total Parameters**: 6,292,480
- **Sample Keys**:

```
  bert.module.embeddings.word_embeddings.weight
  bert.module.embeddings.position_embeddings.weight
  bert.module.embeddings.token_type_embeddings.weight
  bert.module.embeddings.LayerNorm.weight
  bert.module.embeddings.LayerNorm.bias
  ... and 20 more keys
```

#### BERT_ENCODER

- **Number of Keys**: 2
- **Total Parameters**: 393,728
- **Sample Keys**:

```
  bert_encoder.module.weight
  bert_encoder.module.bias
```

#### TEXT_ENCODER

- **Number of Keys**: 54
- **Total Parameters**: 11,520,000
- **Sample Keys**:

```
  predictor.module.text_encoder.lstms.0.weight_ih_l0
  predictor.module.text_encoder.lstms.0.weight_hh_l0
  predictor.module.text_encoder.lstms.0.bias_ih_l0
  predictor.module.text_encoder.lstms.0.bias_hh_l0
  predictor.module.text_encoder.lstms.0.weight_ih_l0_reverse
  ... and 49 more keys
```

#### DECODER

- **Number of Keys**: 375
- **Total Parameters**: 53,276,190
- **Sample Keys**:

```
  decoder.module.decode.0.conv1.bias
  decoder.module.decode.0.conv1.weight_g
  decoder.module.decode.0.conv1.weight_v
  decoder.module.decode.0.conv2.bias
  decoder.module.decode.0.conv2.weight_g
  ... and 370 more keys
```

#### PREDICTOR

- **Number of Keys**: 92
- **Total Parameters**: 10,281,012
- **Sample Keys**:

```
  predictor.module.lstm.weight_ih_l0
  predictor.module.lstm.weight_hh_l0
  predictor.module.lstm.bias_ih_l0
  predictor.module.lstm.bias_hh_l0
  predictor.module.lstm.weight_ih_l0_reverse
  ... and 87 more keys
```

## Model Configuration

### ISTFTNET

```json
{
  "upsample_kernel_sizes": [
    20,
    12
  ],
  "upsample_rates": [
    10,
    6
  ],
  "gen_istft_hop_size": 5,
  "gen_istft_n_fft": 20,
  "resblock_dilation_sizes": [
    [
      1,
      3,
      5
    ],
    [
      1,
      3,
      5
    ],
    [
      1,
      3,
      5
    ]
  ],
  "resblock_kernel_sizes": [
    3,
    7,
    11
  ],
  "upsample_initial_channel": 512
}
```

- **dim_in**: 64
- **dropout**: 0.2
- **hidden_dim**: 512
- **max_conv_dim**: 512
- **max_dur**: 50
- **multispeaker**: True
- **n_layer**: 3
- **n_mels**: 80
- **n_token**: 178
- **style_dim**: 128
- **text_encoder_kernel_size**: 5
### PLBERT

```json
{
  "hidden_size": 768,
  "num_attention_heads": 12,
  "intermediate_size": 2048,
  "max_position_embeddings": 512,
  "num_hidden_layers": 12,
  "dropout": 0.1
}
```

### VOCAB

```json
{
  ";": 1,
  ":": 2,
  ",": 3,
  ".": 4,
  "!": 5,
  "?": 6,
  "\u2014": 9,
  "\u2026": 10,
  "\"": 11,
  "(": 12,
  ")": 13,
  "\u201c": 14,
  "\u201d": 15,
  " ": 16,
  "\u0303": 17,
  "\u02a3": 18,
  "\u02a5": 19,
  "\u02a6": 20,
  "\u02a8": 21,
  "\u1d5d": 22,
  "\uab67": 23,
  "A": 24,
  "I": 25,
  "O": 31,
  "Q": 33,
  "S": 35,
  "T": 36,
  "W": 39,
  "Y": 41,
  "\u1d4a": 42,
  "a": 43,
  "b": 44,
  "c": 45,
  "d": 46,
  "e": 47,
  "f": 48,
  "h": 50,
  "i": 51,
  "j": 52,
  "k": 53,
  "l": 54,
  "m": 55,
  "n": 56,
  "o": 57,
  "p": 58,
  "q": 59,
  "r": 60,
  "s": 61,
  "t": 62,
  "u": 63,
  "v": 64,
  "w": 65,
  "x": 66,
  "y": 67,
  "z": 68,
  "\u0251": 69,
  "\u0250": 70,
  "\u0252": 71,
  "\u00e6": 72,
  "\u03b2": 75,
  "\u0254": 76,
  "\u0255": 77,
  "\u00e7": 78,
  "\u0256": 80,
  "\u00f0": 81,
  "\u02a4": 82,
  "\u0259": 83,
  "\u025a": 85,
  "\u025b": 86,
  "\u025c": 87,
  "\u025f": 90,
  "\u0261": 92,
  "\u0265": 99,
  "\u0268": 101,
  "\u026a": 102,
  "\u029d": 103,
  "\u026f": 110,
  "\u0270": 111,
  "\u014b": 112,
  "\u0273": 113,
  "\u0272": 114,
  "\u0274": 115,
  "\u00f8": 116,
  "\u0278": 118,
  "\u03b8": 119,
  "\u0153": 120,
  "\u0279": 123,
  "\u027e": 125,
  "\u027b": 126,
  "\u0281": 128,
  "\u027d": 129,
  "\u0282": 130,
  "\u0283": 131,
  "\u0288": 132,
  "\u02a7": 133,
  "\u028a": 135,
  "\u028b": 136,
  "\u028c": 138,
  "\u0263": 139,
  "\u0264": 140,
  "\u03c7": 142,
  "\u028e": 143,
  "\u0292": 147,
  "\u0294": 148,
  "\u02c8": 156,
  "\u02cc": 157,
  "\u02d0": 158,
  "\u02b0": 162,
  "\u02b2": 164,
  "\u2193": 169,
  "\u2192": 171,
  "\u2197": 172,
  "\u2198": 173,
  "\u1d7b": 177
}
```


## Summary

### Voice Embeddings
- **Shape**: [510, 1, 256]
  - 510 sequence positions (max phoneme tokens)
  - 1 batch size
  - 256-dimensional speaker embeddings
- **Purpose**: Pre-computed speaker characteristics at each timestep for conditioning the model
- **Value Range**: [-1.51, 1.76] with mean ≈ 0

### Model Architecture (81.7M parameters)
1. **BERT** (6.3M params) - Linguistic encoder with 12 layers, 768 hidden dim
2. **BERT Encoder** (394K params) - Maps BERT output to model hidden dimension (512)
3. **Text Encoder** (11.5M params) - Additional text processing with LSTMs
4. **Decoder** (53.3M params) - ISTFTNet vocoder for audio reconstruction
5. **Predictor** (10.3M params) - Duration, F0, and prosody prediction

### Key Specifications
- **Max Sequence Length**: 512 tokens
- **Mel Spectrogram Dimension**: 80
- **Phoneme Vocabulary**: 178 tokens
- **Speaker Embedding Size**: 128 (+ 256-dim vectors in voice files)
- **Multispeaker Support**: Yes
- **Supported Languages**: English, Spanish, French, Hindi, Italian, Japanese, Portuguese, Chinese

