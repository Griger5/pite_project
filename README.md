# pite_project

Compact, config-driven audio anti-spoofing pipeline.

## Inference (Python)

```python
from AI_dio.inference import predict_file

result = predict_file(
    checkpoint="checkpoints/your_checkpoint.pt",
    wav="path/to/audio.wav",
)
print(result.score, result.label)
```

For in-memory tensors:

```python
import torch
from AI_dio.inference import predict_audio

audio = torch.randn(1, 16000)
result = predict_audio(
    checkpoint="checkpoints/your_checkpoint.pt",
    audio=audio,
    sample_rate=16000,
)
print(result.score, result.label)
```

Output fields:
- `score`: mean spoof probability across windows (0.0–1.0)
- `scores`: per-window probabilities in order
- `label`: `"fake"` if `score >= threshold`, else `"real"`
- `threshold`: decision threshold used
- `window_sec`/`stride_sec`: windowing parameters used
- `wav`: resolved path when using `predict_file`
