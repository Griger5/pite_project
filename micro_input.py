import sounddevice as sd
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def microphone_input(
    filename = "file.wav",
    record_sec = 5,
    channels = 1,
    rate = 44100
):
    logging.info("Recording...")
    audio = sd.rec(int(record_sec * rate), samplerate=rate, channels=channels, dtype='int16')
    sd.wait()
    logging.info("Finished recording.")
    sf.write(filename, audio, rate)

if __name__ == "__main__":
    microphone_input()