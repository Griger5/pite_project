import numpy as np
import pytest

from src.AI_dio.audio.audio_file_reader import (
    compute_log_mel_spectrogram,
    get_sound_parameters,
)


def test_log_mel_spectrogram_shape():
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    mel = compute_log_mel_spectrogram(
        audio_data=audio,
        sample_rate=sample_rate,
        mel_band_num=128,
        sample_frame_length=2048,
        hop_length=512,
    )
    params = get_sound_parameters(audio, sample_rate)
    assert pytest.approx(params["avg_volume"], 0.1) == 0.35
    assert pytest.approx(params["peak_amplitude"], 0.01) == 0.5
    assert mel.shape[0] == 128
    assert mel.ndim == 2


def test_sound_parameters_values():
    sample_rate = 44100
    audio = np.ones(sample_rate) * 0.5
    params = get_sound_parameters(audio, sample_rate)

    assert params["sample_rate"] == 44100
    assert pytest.approx(params["duration_sec"], 0.01) == 1.0
    assert pytest.approx(params["avg_volume"], 0.01) == 0.5
    assert pytest.approx(params["peak_amplitude"], 0.01) == 0.5
    assert pytest.approx(params["loudness_db"], 0.01) == -6.02
