from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from AI_dio.data_preprocessing.audio_utils import (
    load_audio_mono_resampled,
    load_audio_segment_mono_resampled,
)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def _collect_audio_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in AUDIO_EXTS:
            files.extend(root.rglob(f"*{ext}"))
    return files


def _load_audio(path: Path, target_sr: int) -> torch.Tensor:
    return load_audio_mono_resampled(path, target_sr, target_length=None)


def _load_audio_segment(path: Path, target_sr: int, length: int) -> torch.Tensor:
    return load_audio_segment_mono_resampled(
        path, target_sr=target_sr, target_length=length, random_start=True
    )


def _rms(audio: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(audio**2) + 1e-8)


def _mix_with_snr(
    clean: torch.Tensor, noise: torch.Tensor, snr_db: float
) -> torch.Tensor:
    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    if float(noise_rms) <= 0.0:
        return clean
    scale = clean_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
    mixed = clean + noise * scale
    peak = float(mixed.abs().max())
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed


def _apply_rir(audio: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    if rir.numel() == 0:
        return audio
    rir = rir / (rir.abs().max() + 1e-6)
    rir = torch.flip(rir, dims=(1,))
    rir_kernel = rir.unsqueeze(0)
    audio_batch = audio.unsqueeze(0)
    convolved = torch.nn.functional.conv1d(
        audio_batch, rir_kernel, padding=rir.size(1) - 1
    )
    convolved = convolved[:, :, : audio.size(1)]
    convolved = convolved.squeeze(0)
    peak = float(convolved.abs().max())
    if peak > 0.99:
        convolved = convolved * (0.99 / peak)
    return convolved


class AudioAugmenter:
    def __init__(
        self,
        *,
        augment_root: Path,
        target_sr: int,
        chunk_length: int,
        p_noise: float = 0.6,
        p_music: float = 0.3,
        p_rir: float = 0.3,
        snr_db_min: float = -5.0,
        snr_db_max: float = 15.0,
        music_snr_db_min: float = -10.0,
        music_snr_db_max: float = 10.0,
        gain_db_min: float = -6.0,
        gain_db_max: float = 6.0,
        allow_music_and_noise: bool = False,
        preload: bool = False,
        preload_max_files: int = 256,
        preload_segments_per_file: int = 1,
    ) -> None:
        self._target_sr = target_sr
        self._chunk_length = chunk_length

        self._p_noise = p_noise
        self._p_music = p_music
        self._p_rir = p_rir
        self._snr_db_min = snr_db_min
        self._snr_db_max = snr_db_max
        self._music_snr_db_min = music_snr_db_min
        self._music_snr_db_max = music_snr_db_max
        self._gain_db_min = gain_db_min
        self._gain_db_max = gain_db_max
        self._allow_music_and_noise = allow_music_and_noise

        noise_dirs = [
            augment_root / "RIRS_NOISES",
            augment_root / "DKITCHEN",
            augment_root / "OOFFICE",
            augment_root / "PCAFETER",
            augment_root / "STRAFFIC",
            augment_root / "TMETRO",
        ]
        music_dirs = [augment_root / "fma_small"]
        rir_dirs = [augment_root / "simulated_rirs_16k"]

        self._noise_files = _collect_audio_files(noise_dirs)
        self._music_files = _collect_audio_files(music_dirs)
        self._rir_files = _collect_audio_files(rir_dirs)
        self._noise_segments: list[torch.Tensor] = []
        self._music_segments: list[torch.Tensor] = []
        self._rir_segments: list[torch.Tensor] = []

        if preload:
            max_files = max(int(preload_max_files), 0)
            segments_per_file = max(int(preload_segments_per_file), 1)
            if max_files > 0:
                self._noise_segments = self._preload_segments(
                    self._noise_files, max_files, segments_per_file
                )
                self._music_segments = self._preload_segments(
                    self._music_files, max_files, segments_per_file
                )
                self._rir_segments = self._preload_rirs(self._rir_files, max_files)
                print(
                    "[info] preloaded augment audio segments: "
                    f"noise={len(self._noise_segments)}, "
                    f"music={len(self._music_segments)}, "
                    f"rir={len(self._rir_segments)}"
                )

    def _rand_uniform(self, low: float, high: float) -> float:
        if high <= low:
            return float(low)
        return float((high - low) * torch.rand(1).item() + low)

    def _maybe_gain(self, audio: torch.Tensor) -> torch.Tensor:
        gain_db = self._rand_uniform(self._gain_db_min, self._gain_db_max)
        gain = 10.0 ** (gain_db / 20.0)
        return audio * gain

    def _maybe_apply_rir(self, audio: torch.Tensor) -> torch.Tensor:
        if self._p_rir <= 0.0 or (not self._rir_files and not self._rir_segments):
            return audio
        if torch.rand(1).item() > self._p_rir:
            return audio
        if self._rir_segments:
            rir = self._pick_preloaded(self._rir_segments)
        else:
            rir_path = self._rir_files[
                int(torch.randint(0, len(self._rir_files), (1,)))
            ]
            rir = _load_audio(rir_path, self._target_sr)
        return _apply_rir(audio, rir)

    def _maybe_add_background(self, audio: torch.Tensor) -> torch.Tensor:
        use_noise = self._p_noise > 0.0 and (self._noise_files or self._noise_segments)
        use_music = self._p_music > 0.0 and (self._music_files or self._music_segments)
        if not use_noise and not use_music:
            return audio

        pick_noise = use_noise and torch.rand(1).item() < self._p_noise
        pick_music = use_music and torch.rand(1).item() < self._p_music

        if not self._allow_music_and_noise and pick_noise and pick_music:
            if torch.rand(1).item() < 0.5:
                pick_music = False
            else:
                pick_noise = False

        if pick_noise:
            if self._noise_segments:
                noise = self._pick_preloaded(self._noise_segments)
            else:
                noise_path = self._noise_files[
                    int(torch.randint(0, len(self._noise_files), (1,)))
                ]
                noise = _load_audio_segment(
                    noise_path, self._target_sr, self._chunk_length
                )
            snr_db = self._rand_uniform(self._snr_db_min, self._snr_db_max)
            audio = _mix_with_snr(audio, noise, snr_db)

        if pick_music:
            if self._music_segments:
                music = self._pick_preloaded(self._music_segments)
            else:
                music_path = self._music_files[
                    int(torch.randint(0, len(self._music_files), (1,)))
                ]
                music = _load_audio_segment(
                    music_path, self._target_sr, self._chunk_length
                )
            snr_db = self._rand_uniform(self._music_snr_db_min, self._music_snr_db_max)
            audio = _mix_with_snr(audio, music, snr_db)

        return audio

    def _pick_preloaded(self, items: list[torch.Tensor]) -> torch.Tensor:
        return items[int(torch.randint(0, len(items), (1,)).item())]

    def _preload_segments(
        self, files: list[Path], max_files: int, segments_per_file: int
    ) -> list[torch.Tensor]:
        if not files:
            return []
        count = min(len(files), max_files)
        order = torch.randperm(len(files))[:count].tolist()
        segments: list[torch.Tensor] = []
        for idx in order:
            path = files[idx]
            for _ in range(segments_per_file):
                segments.append(
                    _load_audio_segment(path, self._target_sr, self._chunk_length)
                )
        return segments

    def _preload_rirs(self, files: list[Path], max_files: int) -> list[torch.Tensor]:
        if not files:
            return []
        count = min(len(files), max_files)
        order = torch.randperm(len(files))[:count].tolist()
        return [_load_audio(files[idx], self._target_sr) for idx in order]

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.clone()
        audio = self._maybe_gain(audio)
        audio = self._maybe_apply_rir(audio)
        audio = self._maybe_add_background(audio)
        return audio
