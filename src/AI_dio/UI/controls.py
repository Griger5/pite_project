import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.AI_dio.audio.audio_file_reader import get_sound_parameters, read_sound
from src.AI_dio.audio.microphone_input import microphone_input


class Controls(QWidget):
    signal_file_path = Signal(str)
    signal_status = Signal(str)
    signal_reset = Signal(bool)
    signal_audio_info = Signal(object)
    signal_update_plots = Signal(bool)

    def __init__(self):
        super().__init__()

        self.file_path = None
        self.is_microphone_used = None

        controls_box = QGroupBox("Controls")
        box_layout = QVBoxLayout()
        sound_source_layout = QHBoxLayout()
        app_controls_layout = QHBoxLayout()
        main_layout = QVBoxLayout(self)

        button_load_file = QPushButton("Load File")
        button_use_microphone = QPushButton("Use microphone")
        button_start = QPushButton("Start")
        button_reset = QPushButton("Reset")

        button_load_file.clicked.connect(self.show_load_dialog)
        button_use_microphone.clicked.connect(self.microphone_in_use)
        button_start.clicked.connect(self.start_analysis)
        button_reset.clicked.connect(lambda: self.signal_reset.emit(True))

        sound_source_layout.addWidget(button_load_file)
        sound_source_layout.addWidget(button_use_microphone)
        app_controls_layout.addWidget(button_start)
        app_controls_layout.addWidget(button_reset)

        box_layout.addLayout(sound_source_layout)
        box_layout.addLayout(app_controls_layout)

        controls_box.setLayout(box_layout)
        main_layout.addWidget(controls_box)

    def show_load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio files (*.wav *.mp3 *.flax *.ogg *.m4a *.aiff)",
        )

        if path:
            self.signal_reset.emit(True)
            self.is_microphone_used = False
            self.file_path = path
            self.signal_file_path.emit(os.path.basename(self.file_path))
            self.signal_status.emit("File Loaded")

    def microphone_in_use(self):
        self.signal_reset.emit(True)
        self.is_microphone_used = True
        self.signal_status.emit("Microphone in use")

    def start_analysis(self):
        if self.is_microphone_used:
            audio, rate = microphone_input()
            sound_params = get_sound_parameters(audio, rate)
            self.signal_audio_info.emit(sound_params)
            self.signal_status.emit("Analysis ended")
        elif self.file_path:
            try:
                _, sound_params = read_sound(Path(self.file_path))
                self.signal_audio_info.emit(sound_params)
                self.signal_status.emit("Analysis ended")
                self.signal_update_plots.emit(True)
            except FileNotFoundError:
                self.signal_reset.emit(True)
                self.signal_status.emit("Audio file not found")
        else:
            self.signal_status.emit("Source not selected")
