# Base Model Analysis - Kokoro-82M

## Downloaded Files

From Hugging Face: `hexgrad/Kokoro-82M`

### Files Structure
```
base_model/
├── config.json          # Model configuration (2.3 KB)
├── kokoro-v1_0.pth      # Model weights (312 MB)
├── voices/              # Speaker embeddings (56 voice files, ~511 KB each)
│   ├── af_*.pt         # American English voices
│   ├── am_*.pt         # American male voices
│   ├── ef_*.pt         # Spanish voices
│   ├── ff_*.pt         # French voices
│   ├── em_*.pt         # Spanish male voices
│   ├── hm_*.pt         # Hindi voices
│   ├── hf_*.pt         # Hindi female voices
│   ├── if_*.pt         # Italian voices
│   ├── im_*.pt         # Italian male voices
│   ├── jf_*.pt         # Japanese voices
│   ├── jm_*.pt         # Japanese male voices
│   ├── pf_*.pt         # Portuguese voices
│   ├── pm_*.pt         # Portuguese male voices
│   ├── zf_*.pt         # Chinese female voices
│   └── zm_*.pt         # Chinese male voices
├── samples/             # Example audio samples
└── eval/                # Evaluation results
```

## Model Configuration

Key parameters from `config.json`:

### Architecture
- **Hidden Dimensions**: 512
- **Style Dimensions**: 128 (speaker embedding size)
- **Max Duration**: 50 frames per phoneme
- **Dropout**: 0.2

### BERT/PLBERT Configuration
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

### Decoder (ISTFTNet) Configuration
```json
{
  "upsample_rates": [10, 6],
  "upsample_kernel_sizes": [20, 12],
  "gen_istft_n_fft": 20,
  "gen_istft_hop_size": 5,
  "upsample_initial_channel": 512,
  "resblock_kernel_sizes": [3, 7, 11],
  "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
}
```

### Text Processing
- **Vocabulary Size**: 178 tokens (phonemes + punctuation)
- **Number of Layers**: 3
- **Text Encoder Kernel Size**: 5
- **Mel Spectrogram Dimensions**: 80

### Multispeaker Support
- **Multispeaker**: true (supports speaker embeddings)

## Training Implications

### What We Need to Implement

1. **Model Architecture** (already available in `kokoro/model.py`)
   - BERT encoder: 768-dim, 12 layers, 512 max position embeddings
   - Text encoder: 512-dim hidden, 5 kernel, 3 layers
   - Prosody predictor: 512-dim hidden, 3 layers
   - Decoder: ISTFTNet with upsample rates [10, 6]

2. **Speaker Embeddings**
   - Size: [1, 256] (not 128 as in config - note config shows 128 but speaker files are 256)
   - Each voice file (.pt) contains a speaker embedding tensor
   - Need to extract from audio using WavLM/HuBERT

3. **Training Data Format**
   - Need: `input_ids`, `speaker_embedding`, `gt_duration`, `gt_f0`, `gt_audio`
   - Duration per phoneme (max 50 frames)

4. **Loss Functions**
   - Duration loss (MSE between predicted and ground truth duration)
   - F0 loss (MSE between predicted and ground truth F0)
   - Reconstruction loss (L1/MSE between predicted and ground truth audio)

### Next Steps

1. **Analyze Voice Files**
   ```python
   # Check speaker embedding dimensions
   voice = torch.load('base_model/voices/af_heart.pt', weights_only=True)
   print(voice.shape)  # Should be [1, 256] or similar
   ```

2. **Study Model Weights Structure**
   ```python
   # Check model structure from state_dict
   model_weights = torch.load('base_model/kokoro-v1_0.pth', weights_only=True, map_location='cpu')
   for key in model_weights.keys():
       print(key, model_weights[key].shape)
   ```

3. **Design Training Loop**
   - Remove `@torch.no_grad()` decorators
   - Implement loss functions
   - Set up data loading for Luxembourgish corpus
   - Implement speaker embedding extraction
   - Implement F0 extraction from audio

4. **Training Strategy**
   - Start with pretrained weights as initialization
   - Fine-tune on Luxembourgish data
   - Consider freezing BERT vs fine-tuning
   - Monitor reconstruction, duration, and F0 losses

## Key Insights

1. **Model Size**: 312 MB (82M parameters)
2. **Context Length**: 512 positions (model can handle up to 510 phoneme tokens)
3. **Speaker Style**: 128-dim (plus additional speaker embedding dimensions)
4. **Phoneme Vocabulary**: 178 tokens
5. **Multispeaker**: Fully supported with speaker embeddings

## Luxembourgish Fine-Tuning Strategy

1. Load pretrained model from `base_model/kokoro-v1_0.pth`
2. Extract speaker embeddings from Luxembourgish audio
3. Use your custom `lb.LBG2P()` for phonemization
4. Train on Luxembourgish corpus (32k samples)
5. Monitor validation loss and generate test audio


## Key Discoveries

### Voice File Structure
- **Shape**: `[510, 1, 256]` - 510 speaker embedding vectors of 256 dimensions each
- **Purpose**: Pre-computed speaker embeddings for each position in the sequence  
- **Values**: Min=-1.51, Max=1.76, Mean≈0
- **Usage**: Used to condition the model on speaker characteristics at each timestep

### Model Component Structure
The model.pth file contains 5 main components:
1. **bert**: BERT/ALBERT encoder weights (~27 keys with 'module.' prefix)
2. **bert_encoder**: Linear layer to map BERT output to model hidden dimension
3. **decoder**: Generator/ISTFTNet decoder with upsample layers, resblocks, noise processing
4. **predictor**: Prosody predictor for duration, F0, and noise prediction
5. **text_encoder**: Text encoder with embedding, CNN layers, and LSTM

All weights use the 'module.' prefix pattern, suggesting they were saved with DataParallel.
