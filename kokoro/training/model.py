"""
Trainable Kokoro Model Wrapper

Wraps kokoro.model.KModel to enable gradient-based training.

Key modifications from inference model:
1. Remove @torch.no_grad() decorator from forward_with_tokens (line 86 in kokoro/model.py)
2. Enable gradient computation for backpropagation
3. Handle 'module.' prefix in state_dict (model saved with DataParallel)
4. Return intermediate outputs (duration, F0) for loss computation

Responsibilities:
1. Load pretrained weights from base_model/kokoro-v1_0.pth
   - Strip 'module.' prefix if present (DataParallel artifact)
   - Initialize 5 components: bert, bert_encoder, decoder, predictor, text_encoder

2. Forward pass with gradient tracking:
   - Input: phoneme input_ids [batch, seq_len], speaker embeddings [batch, 510, 1, 256]
   - BERT encodes phonemes → 768-dim (line 102)
   - bert_encoder maps to 512-dim hidden (line 103)
   - Predictor computes duration + F0 (lines 105-115)
   - Text encoder processes phonemes (line 116)
   - Decoder generates audio waveform (line 118)
   - Output: audio [batch, T_audio], duration [batch, seq_len], F0 [batch, T_audio]

3. Training mode support:
   - Set model.train() to enable dropout (0.2 per config)
   - Freeze BERT option: model.bert.eval() + requires_grad=False

4. Optimizer compatibility:
   - Return all trainable parameters via .parameters()
   - Support AdamW optimizer with weight decay

Note: Speaker embeddings are [510, 1, 256] per sample, not single [1, 256].
"""

