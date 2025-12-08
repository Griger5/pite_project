# pip install PySide6
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget
from PySide6.QtWidgets import QVBoxLayout, QGroupBox, QLabel, QHBoxLayout, QPushButton, QFileDialog
from PySide6.QtGui import QAction
import os
import sys

class SoundApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('The Best Sound App!')
        self.setFixedSize(600, 400)

        self.file_label = QLabel("---")

        self.build_menu()

        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)
        self.build_main_interface()


    def build_menu(self):

        exit_action, about_action = self.build_menu_action()

        menu_bar = self.menuBar()

        menu_file = menu_bar.addMenu('&File')
        menu_help = menu_bar.addMenu('&Help')

        menu_file.addAction(exit_action)
        menu_help.addAction(about_action)

        self.statusBar()

    def build_menu_action(self):
        exit_action = QAction('&Exit', self)
        about_action = QAction('&About', self)

        exit_action.setStatusTip('Exit app')
        about_action.setStatusTip('About app')

        exit_action.triggered.connect(self.close)
        about_action.triggered.connect(self.show_about)

        return exit_action, about_action

    def build_main_interface(self):
        self.build_basic_info()
        self.build_controls()
        self.build_area()
        self.build_audio_info()

        self.setCentralWidget(self.main_widget)

    def build_basic_info(self):
        box = QGroupBox("Basic Info")
        box.setFixedHeight(64)
        box_layout = QHBoxLayout()

        file_name_label = QLabel("File: ")
        file_name_label.setFixedWidth(26)

        box_layout.addWidget(file_name_label)
        box_layout.addWidget(self.file_label)

        box.setLayout(box_layout)
        self.layout.addWidget(box)

    def build_controls(self):
        controls_box = QGroupBox("Controls")
        controls_box.setFixedHeight(64)
        controls_box_layout = QHBoxLayout()

        button_load = QPushButton("Load File")
        button_reset = QPushButton("Reset file")

        button_load.setStatusTip("Load file")
        button_reset.setStatusTip("Reset file")

        button_load.clicked.connect(self.show_load_dialog)
        button_reset.clicked.connect(self.reset_file)

        controls_box_layout.addWidget(button_load)
        controls_box_layout.addWidget(button_reset)

        controls_box.setLayout(controls_box_layout)
        self.layout.addWidget(controls_box)

    def build_area(self):
        plot_box = QGroupBox("Plot Area")
        plot_box_layout = QVBoxLayout()

        plot_label = QLabel("There will be some plots")
        plot_box_layout.addWidget(plot_label)

        plot_box.setLayout(plot_box_layout)
        self.layout.addWidget(plot_box)

    def build_audio_info(self):
        info_box = QGroupBox("Audio Info")
        info_box_layout = QVBoxLayout()

        info_label = QLabel("There will be Audio/Microphone Info")
        info_box_layout.addWidget(info_label)

        info_box.setLayout(info_box_layout)
        self.layout.addWidget(info_box)

    def show_load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio files (*.wav)")

        if path:
            self.file_label.setText(os.path.basename(path))
            self.file_label.setStatusTip(path)

    def reset_file(self):
        self.file_label.setText("---")
        self.file_label.setStatusTip("")

    @staticmethod
    def show_about():
        msg = QMessageBox()
        msg.setWindowTitle('About')
        msg.setText(
            '<p style="margin-left: 20px; font-weight: bold">Authors:</p>'
            '- <i>Gracjan Adamus</i><br/>'
            '- <i>Kacper Wojtowicz</i><br/>'
            '- <i>Jakub Ledwoń</i><br/>'
            '- <i>Hubert Regec</i><br/>'
        )
        msg.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SoundApp()
    window.show()
    app.exec()