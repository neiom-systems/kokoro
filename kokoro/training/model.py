"""
Trainable wrapper around `kokoro.model.KModel`.

What needs to change compared to the inference module:

1. **Gradient flow**
   - Remove the `@torch.no_grad()` decorator from `forward_with_tokens`.
   - Ensure every submodule is registered under `nn.Module` so `.train()` / `.eval()` work.
   - Decide which parts are frozen initially (ALBERT encoder, early text encoder layers).

2. **Forward API for training**
   - Accept already-tokenised inputs: `input_ids` `[batch, seq_len]`, `input_lengths`,
     `voice_rows`, plus ground-truth features for teacher forcing (`durations`, `f0`, `noise`).
   - Return a rich object/dict with:
       * `audio_pred`: decoder output waveform.
       * `duration_logits`: raw logits before sigmoid.
       * `duration_pred`: frame counts after sigmoid-sum.
       * `f0_pred`, `noise_pred`.
       * Any intermediate encodings that losses might need (e.g. `alignment` matrices).
   - Provide toggles:
       * `use_teacher_durations`: if true, build alignment matrices from ground-truth
         durations instead of predicted ones (helpful for the first N epochs).
       * `detach_voice`: optional to stop gradients flowing into the voice table when
         experimenting with frozen embeddings.

3. **State dict management**
   - Loading: handle checkpoints where keys are prefixed with `module.` (DataParallel).
     Strip the prefix before calling `.load_state_dict`.
   - Saving: return a dict with the same key layout as `KModel` so inference can reuse it.
   - Voice table: if we treat it as part of the model, include it in the checkpoint under
     `voices/lb_max`.

4. **Utility methods**
   - `freeze_submodules`, `unfreeze_submodules` helpers for staged fine-tuning.
   - Parameter group builder for the optimiser (different LR for voice table vs. rest).
   - Optional hooks to dump intermediate tensors for debugging (e.g. attention maps).

Implementation outline
----------------------
```python
class TrainableKModel(nn.Module):
    def __init__(self, base_ckpt: str, voice_table: torch.Tensor, *, freeze_bert: bool = True):
        ...

    def forward(self, batch, *, use_teacher_durations: bool = False) -> dict[str, torch.Tensor]:
        ...

    def build_alignment(self, durations, lengths):
        # convert durations to alignment matrix compatible with decoder
```

Keep this file focused on model mechanics; CLI/training orchestration belongs in `train.py`.
"""
