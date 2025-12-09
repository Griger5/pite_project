import logging

import matplotlib.pyplot as plt
import numpy as np

import librosa
import librosa.display

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def plot_waveform(audio_channel, output_path="waveform.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(audio_channel)
    plt.title("Audio signal waveform")
    plt.savefig(output_path)
    plt.close()

def plot_melspectrogram(
        log_mel, sample_rate, 
        hop_length=512, output_path="spectrogram.png"
):
    plt.figure(figsize=(10,4))
    librosa.display.specshow(
        log_mel, sr=sample_rate, hop_length=hop_length,
        x_axis='time', y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram')
    plt.savefig(output_path)
    plt.close()

def compute_log_mel_spectrogram(
    audio_data, sample_rate, 
    mel_band_num=128, sample_frame_length=2048, hop_length=512
):
    S = librosa.feature.melspectrogram(
        y=audio_data, sr=sample_rate, n_fft=sample_frame_length,
        hop_length=hop_length, n_mels=mel_band_num
    )
    log_mel = librosa.power_to_db(S, ref=np.max)
    return log_mel

def get_sound_parameters(data, sr):
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))
    epsilon = 1e-10
    loudness_db = 20 * np.log10(rms + epsilon)
    duration = len(data) / sr
    logging.info(f"Sample rate: {sr} Hz")
    logging.info(f"Duration: {duration:.2f} sec")
    logging.info(f"RMS (volume): {rms:.4f}")
    logging.info(f"Peak amplitude: {peak:.4f}")
    logging.info(f"Loudness (dB): {loudness_db:.2f} dB")
    return {
        "sample_rate": sr,
        "duration_sec": duration,
        "avg_volume": rms,
        "peak_amplitude": peak,
        "loudness_db": loudness_db
    }

def read_sound(file = "test_samples/sample-3s.wav"):
    audio_data,sample_rate = librosa.load(file,sr=None,mono=True)
    logging.info(f"Number of samples: {len(audio_data)}")
    logging.info(f"Sampling rate: {sample_rate}")
    log_mel = compute_log_mel_spectrogram(
        audio_data, sample_rate,
    mel_band_num=128, sample_frame_length=2048, hop_length=512
    )
    plot_waveform(audio_data)
    plot_melspectrogram(log_mel, sample_rate)
    get_sound_parameters(audio_data, sample_rate)
    return log_mel

if __name__ == "__main__":
    read_sound()
