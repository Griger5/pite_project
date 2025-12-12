import logging
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
ROOT: Path = Path(__file__).resolve().parents[3]


def microphone_input(
    filename: Path = ROOT / "audio_output_files/file.wav",
    record_sec: int = 5,
    channels: int = 1,
    rate: int = 44100,
) -> None:
    logging.info("Recording...")
    audio: np.ndarray = sd.rec(
        int(record_sec * rate), samplerate=rate, channels=channels, dtype="int16"
    )
    sd.wait()
    logging.info("Finished recording.")
    sf.write(filename, audio, rate)


if __name__ == "__main__":
    microphone_input()
