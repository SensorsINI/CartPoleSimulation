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
from numpy.core.numeric import roll

import Controllers.controller_mppi as controller_mppi

import CartPole.noise_adder as noise_settings

class NoiseOptionsWindow(QWidget):
    def __init__(self):
        super(NoiseOptionsWindow, self).__init__()

        self.sigma_angle = noise_settings.sigma_angle
        self.sigma_position = noise_settings.sigma_position

        self.sigma_angleD = noise_settings.sigma_angleD
        self.sigma_positionD = noise_settings.sigma_positionD

        self.angle_smoothing = noise_settings.angle_smoothing
        self.position_smoothing = noise_settings.position_smoothing

        layout = QVBoxLayout()

        self.setLayout(layout)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setGeometry(0, 0, 400, 50)

        self.show()
        self.setWindowTitle("Noise Options")