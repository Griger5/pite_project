import csv

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape

    time_axis = torch.arange(num_frames, dtype=torch.float32) / float(sr)

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)

    ax.plot(time_axis.numpy(), waveform[0], linewidth=1)
    ax.grid(True)

    max_x = float(time_axis[-1].item())
    if max_x <= 0.0:
        return

    ax.set_xlim(0.0, max_x)
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    power_to_db = T.AmplitudeToDB("power", 80.0)
    ax.imshow(
        power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest"
    )


class AIDetectDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        split: str,
        chunk_duration=3.0,
        target_sr=16000,
        win_ms=25.0,
        hop_ms=10.0,
        n_mels=80,
    ):
        self._target_sr = target_sr
        self._chunk_length = int(chunk_duration * target_sr)

        win_length = int(round(win_ms / 1000.0 * self._target_sr))
        hop_length = int(round(hop_ms / 1000.0 * self._target_sr))
        self._n_fft = 1
        while self._n_fft < win_length:
            self._n_fft *= 2

        self._mel = T.MelSpectrogram(
            sample_rate=self._target_sr,
            n_fft=self._n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            center=True,
        )
        self._to_db = T.AmplitudeToDB(stype="power")

        with open(manifest_csv) as f:
            reader = csv.DictReader(f.readlines(), delimiter=",")
            self.rows = list(filter(lambda r: r["split"] == split, reader))

    def _crop_or_pad(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        _, N = audio_tensor.size()
        if N > self._chunk_length:
            return audio_tensor[:, : self._chunk_length]
        if N < self._chunk_length:
            return torch.nn.functional.pad(audio_tensor, (0, self._chunk_length - N))
        return audio_tensor

    def _resample(self, audio_tensor: torch.Tensor, original_sr: int) -> torch.Tensor:
        if original_sr != self._target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, original_sr, self._target_sr
            )
        return audio_tensor

    def _to_mono(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        C, _ = audio_tensor.size()
        if C == 1:
            return audio_tensor
        return audio_tensor.mean(dim=1, keepdim=True)

    def _load_from_filepath(self, fp: str) -> torch.Tensor:
        audio_tensor, sr = torchaudio.load(fp, channels_first=True)  # [C, N]
        audio_tensor = self._to_mono(audio_tensor)  # [1, N], N = sr * duration
        audio_tensor = self._resample(audio_tensor, sr)  # [1, N'], N' = 16k * duration
        audio_tensor = self._crop_or_pad(audio_tensor)  # [1, 16k * 3]
        return audio_tensor

    def __getitem__(self, index) -> tuple:
        r = self.rows[index]
        path = r["path"]
        y = int(r["label"])
        audio_tensor = self._load_from_filepath(path)
        mel: torch.Tensor = self._mel(audio_tensor)  # (1, 80, T)
        mel_db: torch.Tensor = self._to_db(mel)  # (1, 80, T)
        tokens = mel_db.squeeze(0).transpose(0, 1)  # (T, 80)

        return tokens, y


if __name__ == "__main__":
    # TODO: to be removed after demo
    ds = AIDetectDataset(manifest_csv="manifest.csv", split="train")
    wav = ds._load_from_filepath(
        "./data/raw/for-norm/validation/real/file4.wav_16k.wav_norm.wav_mono.wav_silence.wav"
    )

    spectrogram = T.Spectrogram(n_fft=ds._n_fft)

    spec = spectrogram(wav)
    fig, axs = plt.subplots(2, 1)
    plot_waveform(wav, 16000, title="Original waveform", ax=axs[0])
    plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
    fig.tight_layout()
    fig.savefig("out.png")
    tokens, y = ds[0]
    print(tokens.size(), y)
