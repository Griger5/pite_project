import math
from pathlib import Path
from typing import Optional

import torch
import torchaudio

try:
    import soundfile as sf

    _HAS_SF = True
except Exception:
    sf = None
    _HAS_SF = False

_WARNED_AUDIO: set[str] = set()

try:
    backends = torchaudio.list_audio_backends()
    if "soundfile" in backends:
        torchaudio.set_audio_backend("soundfile")
except Exception:
    pass


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


def _warn_once(filepath: str, exc: Exception) -> None:
    if filepath in _WARNED_AUDIO:
        return
    _WARNED_AUDIO.add(filepath)
    print(f"[warn] audio load failed for {filepath}: {exc}")


def _load_audio_raw(filepath: str | Path) -> tuple[torch.Tensor, int]:
    if _HAS_SF:
        data, sr = sf.read(
            str(filepath),
            dtype="float32",
            always_2d=True,
        )
        audio_tensor = torch.from_numpy(data.T)
        return audio_tensor, int(sr)
    return torchaudio.load(str(filepath), channels_first=True)


def load_audio_mono_resampled(
    filepath: str | Path, target_sr: int, target_length: Optional[int] = None
) -> torch.Tensor:
    try:
        audio_tensor, sr = _load_audio_raw(filepath)  # [C, N]
    except Exception as exc:
        _warn_once(str(filepath), exc)
        length = int(target_length or 0)
        return torch.zeros((1, length), dtype=torch.float32)
    audio_tensor = to_mono(audio_tensor)  # [1, N]
    audio_tensor = resample(audio_tensor, sr, target_sr)  # [1, N']
    if target_length is not None:
        audio_tensor = crop_or_pad(audio_tensor, target_length)  # [1, target_length]
    return audio_tensor


def load_audio_segment_mono_resampled(
    filepath: str | Path, target_sr: int, target_length: int, random_start: bool = True
) -> torch.Tensor:
    try:
        if _HAS_SF:
            with sf.SoundFile(str(filepath)) as f:
                sr = int(f.samplerate)
                total_frames = int(len(f))
                frames_needed = int(
                    math.ceil(target_length * float(sr) / float(target_sr))
                )
                max_start = max(total_frames - frames_needed, 0)
                start = 0
                if random_start and max_start > 0:
                    start = int(torch.randint(0, max_start + 1, (1,)).item())
                f.seek(start)
                data = f.read(frames=frames_needed, dtype="float32", always_2d=True)
            audio_tensor = torch.from_numpy(data.T)
        else:
            info = torchaudio.info(str(filepath))
            sr = int(info.sample_rate)
            frames_needed = int(math.ceil(target_length * float(sr) / float(target_sr)))
            start = 0
            max_start = max(int(info.num_frames) - frames_needed, 0)
            if random_start and max_start > 0:
                start = int(torch.randint(0, max_start + 1, (1,)).item())
            audio_tensor, _ = torchaudio.load(
                str(filepath),
                channels_first=True,
                frame_offset=start,
                num_frames=frames_needed,
            )
    except Exception as exc:
        _warn_once(str(filepath), exc)
        return torch.zeros((1, target_length), dtype=torch.float32)

    audio_tensor = to_mono(audio_tensor)  # [1, N]
    audio_tensor = resample(audio_tensor, sr, target_sr)  # [1, N']
    audio_tensor = crop_or_pad(audio_tensor, target_length)  # [1, target_length]
    return audio_tensor
