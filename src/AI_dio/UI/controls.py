import os

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.AI_dio.UI.worker_audio import WorkerAudio


class Controls(QWidget):
    signal_file_path = Signal(str)
    signal_status = Signal(str)
    signal_reset = Signal()
    signal_audio_info = Signal(object)
    signal_update_plots = Signal()

    def __init__(self):
        super().__init__()

        self.file_path = None
        self.is_microphone_used = None
        self.thread = None
        self.worker_audio = None

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
        button_reset.clicked.connect(lambda: self.signal_reset.emit())

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
            self.signal_reset.emit()
            self.is_microphone_used = False
            self.file_path = path
            self.signal_file_path.emit(os.path.basename(self.file_path))
            self.signal_status.emit("File Loaded")

    def microphone_in_use(self):
        self.signal_reset.emit()
        self.is_microphone_used = True
        self.signal_status.emit("Microphone in use")

    def start_analysis(self):
        self.thread = QThread()
        self.worker_audio = WorkerAudio(self.is_microphone_used, self.file_path)
        self.worker_audio.moveToThread(self.thread)

        self.thread.started.connect(self.worker_audio.run)

        self.worker_audio.signal_status.connect(self.signal_status)
        self.worker_audio.signal_audio_info.connect(self.signal_audio_info)
        self.worker_audio.signal_update_plots.connect(self.signal_update_plots)
        self.worker_audio.signal_reset.connect(self.signal_reset)

        self.worker_audio.signal_finished.connect(self.thread.quit)
        self.worker_audio.signal_finished.connect(self.worker_audio.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
