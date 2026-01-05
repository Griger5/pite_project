from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from src.AI_dio.audio.audio_file_reader import get_sound_parameters, read_sound
from src.AI_dio.audio.microphone_input import microphone_input


class WorkerAudio(QObject):
    signal_status = Signal(str)
    signal_audio_info = Signal(object)
    signal_update_plots = Signal()
    signal_reset = Signal()
    signal_finished = Signal()

    def __init__(self, is_microphone_used, file_path):
        super().__init__()
        self.is_microphone_used = is_microphone_used
        self.file_path = file_path
        self.is_recording = True

    def run_analysis(self):
        try:
            if self.is_microphone_used:
                full_audio = None
                while self.is_recording:
                    self.signal_status.emit("Recording...")
                    audio, rate = microphone_input(3)
                    if full_audio is None:
                        full_audio = audio
                    else:
                        full_audio = np.concatenate((full_audio, audio))
                    sound_params = get_sound_parameters(full_audio, rate)
                    self.signal_audio_info.emit(sound_params)

                self.signal_status.emit("Recording ended")
            elif self.file_path:
                self.signal_status.emit("Analyzing...")
                _, sound_params = read_sound(Path(self.file_path))
                self.signal_audio_info.emit(sound_params)
                self.signal_update_plots.emit()
                self.signal_status.emit("Analysis ended")
            else:
                self.signal_status.emit("Source not selected")
        except FileNotFoundError:
            self.signal_reset.emit()
            self.signal_status.emit("Audio file not found")
        finally:
            self.signal_finished.emit()
