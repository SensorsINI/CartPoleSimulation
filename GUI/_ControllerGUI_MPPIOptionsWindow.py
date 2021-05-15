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

import Controllers.controller_mppi as controller_mppi


class MPPIOptionsWindow(QWidget):
    def __init__(self):
        super(MPPIOptionsWindow, self).__init__()

        self.horizon_steps = 50
        self.dd_scale = controller_mppi.dd_scale
        self.ep_scale = controller_mppi.ep_scale
        self.ekp_scale = controller_mppi.ekp_scale * 1.0e3
        self.ekc_scale = controller_mppi.ekc_scale * 1.0e1

        layout = QVBoxLayout()

        # Section: Set Horizon Length
        horizon_options_layout = QVBoxLayout()

        self.horizon_label = QLabel("")
        self.horizon_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        horizon_options_layout.addWidget(self.horizon_label)

        slider = QSlider(orientation=Qt.Horizontal)
        slider.setRange(10, 300)
        slider.setValue(self.horizon_steps)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        slider.setSingleStep(10)
        horizon_options_layout.addWidget(slider)

        slider.valueChanged.connect(self.horizon_length_changed)

        # Section: Set Cost Weights
        cost_weight_layout = QVBoxLayout()
        
        self.dd_scale_label = QLabel("")
        self.dd_scale_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        cost_weight_layout.addWidget(self.dd_scale_label)
        slider = QSlider(orientation=Qt.Horizontal)
        slider.setRange(0, 990)
        slider.setValue(self.dd_scale)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        slider.setSingleStep(10)
        cost_weight_layout.addWidget(slider)
        slider.valueChanged.connect(self.dd_scale_changed)

        self.ep_scale_label = QLabel("")
        self.ep_scale_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        cost_weight_layout.addWidget(self.ep_scale_label)
        slider = QSlider(orientation=Qt.Horizontal)
        slider.setRange(0, 1e5-1e3)
        slider.setValue(self.ep_scale)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1e3)
        slider.setSingleStep(1e3)
        cost_weight_layout.addWidget(slider)
        slider.valueChanged.connect(self.ep_scale_changed)

        self.ekp_scale_label = QLabel("")
        self.ekp_scale_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        cost_weight_layout.addWidget(self.ekp_scale_label)
        slider = QSlider(orientation=Qt.Horizontal)
        slider.setRange(0, 99)
        slider.setValue(self.ekp_scale)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.setSingleStep(1)
        cost_weight_layout.addWidget(slider)
        slider.valueChanged.connect(self.ekp_scale_changed)

        self.ekc_scale_label = QLabel("")
        self.ekc_scale_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        cost_weight_layout.addWidget(self.ekc_scale_label)
        slider = QSlider(orientation=Qt.Horizontal)
        slider.setRange(0, 99)
        slider.setValue(self.ekc_scale)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.setSingleStep(1)
        cost_weight_layout.addWidget(slider)
        slider.valueChanged.connect(self.ekc_scale_changed)

        # Put together layout
        self.update_labels()
        layout.addLayout(horizon_options_layout)
        layout.addLayout(cost_weight_layout)

        self.setLayout(layout)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setGeometry(0, 0, 800, 100)

        self.show()
        self.setWindowTitle("MPPI Options")

        controller_mppi.LOGGING = False

    def horizon_length_changed(self, length: int):
        self.horizon_steps = length
        # TODO: Replace by setter method
        controller_mppi.mpc_samples = self.horizon_steps
        self.update_labels()

    def dd_scale_changed(self, length: int):
        self.dd_scale = length
        # TODO: Replace by setter method
        controller_mppi.dd_scale = self.dd_scale * 1.0
        self.update_labels()
    
    def ep_scale_changed(self, length: int):
        self.ep_scale = length
        # TODO: Replace by setter method
        controller_mppi.ep_scale = self.ep_scale * 1.0
        self.update_labels()
    
    def ekp_scale_changed(self, length: int):
        self.ekp_scale = length
        # TODO: Replace by setter method
        controller_mppi.ekp_scale = self.ekp_scale * 1.0e-3
        self.update_labels()
    
    def ekc_scale_changed(self, length: int):
        self.ekc_scale = length
        # TODO: Replace by setter method
        controller_mppi.ekc_scale = self.ekc_scale * 1.0e-1
        self.update_labels()
    
    def update_labels(self):
        self.horizon_label.setText(
            f"Horizon: {self.horizon_steps} steps = {round(self.horizon_steps * controller_mppi.dt, 2)} s"
        )
        self.dd_scale_label.setText(
            f"Distance Difference cost scale: {round(self.dd_scale, 2)}"
        )
        self.ep_scale_label.setText(
            f"Potential Energy cost scale: {round(self.ep_scale, 2)}"
        )
        self.ekp_scale_label.setText(
            f"Pole Kinetic Energy cost scale: {round(self.ekp_scale * 1.0e-3, 4)}"
        )
        self.ekc_scale_label.setText(
            f"Cart Kinetic Energy cost scale: {round(self.ekc_scale * 1.0e-1, 3)}"
        )
