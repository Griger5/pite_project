from pathlib import Path

import torch
import torchaudio


def crop_or_pad(audio_tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    _, num_samples = audio_tensor.size()
    if num_samples > target_length:
        return audio_tensor[:, :target_length]
    if num_samples < target_length:
        return torch.nn.functional.pad(audio_tensor, (0, target_length - num_samples))
    return audio_tensor


def resample(
    audio_tensor: torch.Tensor, original_sr: int, target_sr: int
) -> torch.Tensor:
    if original_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, original_sr, target_sr
        )
    return audio_tensor


def to_mono(audio_tensor: torch.Tensor) -> torch.Tensor:
    channels, _ = audio_tensor.size()
    if channels == 1:
        return audio_tensor
    return audio_tensor.mean(dim=0, keepdim=True)


def load_audio_mono_resampled(
    filepath: str | Path, target_sr: int, target_length: int
) -> torch.Tensor:
    audio_tensor, sr = torchaudio.load(str(filepath), channels_first=True)  # [C, N]
    audio_tensor = to_mono(audio_tensor)  # [1, N]
    audio_tensor = resample(audio_tensor, sr, target_sr)  # [1, N']
    audio_tensor = crop_or_pad(audio_tensor, target_length)  # [1, target_length]
    return audio_tensor
