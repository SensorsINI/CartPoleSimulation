# Necessary only for debugging in Visual Studio Code IDE
try:
    import ptvsd
except:
    pass

import numpy as np

# Import functions from PyQt5 module (creating GUI)
from PyQt5.QtWidgets import (
    QMainWindow,
    QRadioButton,
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QCheckBox,
    QSlider,
    QLineEdit,
    QMessageBox,
    QComboBox,
    QButtonGroup,
)
from PyQt5.QtCore import QThreadPool, QTimer, Qt

import CartPole.noise_adder as noise_settings

NOISE_STD_MAX_RANGE = 1
SLIDER_MAX_RANGE_INT = 1000

class NoiseOptionsWindow(QWidget):
    def __init__(self):
        super(NoiseOptionsWindow, self).__init__()

        self.sigma_angle = noise_settings.sigma_angle
        self.sigma_position = noise_settings.sigma_position

        self.sigma_angleD = noise_settings.sigma_angleD
        self.sigma_positionD = noise_settings.sigma_positionD

        layout = QVBoxLayout()

        self.setLayout(layout)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setGeometry(0, 0, 400, 50)

        sigmas_layout = QVBoxLayout()

        def make_slider(MAX_RANGE, INIT_VALUE):
            label = QLabel("")
            label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

            slider = QSlider(orientation=Qt.Horizontal)
            slider.setRange(0, SLIDER_MAX_RANGE_INT)
            slider.setValue(int(SLIDER_MAX_RANGE_INT*(INIT_VALUE/MAX_RANGE)))
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(int(SLIDER_MAX_RANGE_INT*0.1))
            slider.setSingleStep(1)

            return slider, label

        self.slider_angle, self.sigma_angle_label = make_slider(MAX_RANGE=NOISE_STD_MAX_RANGE,
                                                     INIT_VALUE=self.sigma_angle)
        sigmas_layout.addWidget(self.sigma_angle_label)
        sigmas_layout.addWidget(self.slider_angle)
        self.slider_angle.valueChanged.connect(self.sigma_angle_changed)

        self.slider_position, self.sigma_position_label = make_slider(MAX_RANGE=NOISE_STD_MAX_RANGE,
                                                     INIT_VALUE=self.sigma_position)
        sigmas_layout.addWidget(self.sigma_position_label)
        sigmas_layout.addWidget(self.slider_position)
        self.slider_position.valueChanged.connect(self.sigma_position_changed)



        self.slider_angleD, self.sigma_angleD_label = make_slider(MAX_RANGE=NOISE_STD_MAX_RANGE,
                                                     INIT_VALUE=self.sigma_angleD)
        sigmas_layout.addWidget(self.sigma_angleD_label)
        sigmas_layout.addWidget(self.slider_angleD)
        self.slider_angleD.valueChanged.connect(self.sigma_angleD_changed)

        self.slider_positionD, self.sigma_positionD_label = make_slider(MAX_RANGE=NOISE_STD_MAX_RANGE,
                                                        INIT_VALUE=self.sigma_positionD)
        sigmas_layout.addWidget(self.sigma_positionD_label)
        sigmas_layout.addWidget(self.slider_positionD)
        self.slider_positionD.valueChanged.connect(self.sigma_positionD_changed)



        self.update_slider_labels()
        layout.addLayout(sigmas_layout)

        self.show()
        self.setWindowTitle("Noise Options")

    def sigma_angle_changed(self, val: int):
        self.sigma_angle = float(val) * NOISE_STD_MAX_RANGE / SLIDER_MAX_RANGE_INT
        noise_settings.sigma_angle = self.sigma_angle
        self.update_slider_labels()

    def sigma_position_changed(self, val: int):
        self.sigma_position = float(val) * NOISE_STD_MAX_RANGE / SLIDER_MAX_RANGE_INT
        noise_settings.sigma_position = self.sigma_position
        self.update_slider_labels()

    def sigma_angleD_changed(self, val: int):
        self.sigma_angleD = float(val)  * NOISE_STD_MAX_RANGE / SLIDER_MAX_RANGE_INT
        noise_settings.sigma_angleD = self.sigma_angleD
        self.update_slider_labels()

    def sigma_positionD_changed(self, val: int):
        self.sigma_positionD = float(val)  * NOISE_STD_MAX_RANGE / SLIDER_MAX_RANGE_INT
        noise_settings.sigma_positionD = self.sigma_positionD
        self.update_slider_labels()

    def update_slider_labels(self):

        self.sigma_angle_label.setText(
            f"Angle Noise Std: {round(self.sigma_angle, 3)}"
        )

        self.sigma_position_label.setText(
            f"Position Noise Std: {round(self.sigma_position, 3)}"
        )

        self.sigma_angleD_label.setText(
            f"Angular Velocity Noise Std: {round(self.sigma_angleD, 3)}"
        )

        self.sigma_positionD_label.setText(
            f"Velocity Noise Std: {round(self.sigma_positionD, 3)}"
        )