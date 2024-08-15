import numpy as np

import os

# Import functions from PyQt6 module (creating GUI)
from PyQt6.QtWidgets import QRadioButton, QSlider, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QCheckBox, \
    QLineEdit, QMessageBox, QComboBox, QButtonGroup, \
    QSpacerItem, QSizePolicy
from PyQt6.QtCore import QThreadPool, QTimer, Qt
from PyQt6.QtGui import QFontMetrics
# The main drawing functionalities are implemented in CartPole Class
# Some more functions needed for interaction of matplotlib with PyQt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from CartPole.cartpole_parameters import TrackHalfLength


class CartPole_GuiLayout:
    def __init__(self):
        self.GuiActions = None
        self.layout = None

        # region Introducing multithreading
        # To ensure smooth functioning of the app,
        # the calculations and redrawing of the figures have to be done in a different thread
        # than the one capturing the mouse position and running the animation
        self.threadpool = QThreadPool()

    def create_layout(self, main_window, GuiActions, quit_callback):
        self.GuiActions = GuiActions
        # region Create GUI Layout

        # region - Create container for top level layout
        layout = QVBoxLayout()
        # endregion

        self.canvas = FigureCanvas(self.GuiActions.fig)

        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.canvas)

        # endregion

        # region - Radio buttons selecting current controller
        self.rbs_controllers = []
        for controller_name in self.GuiActions.controller_names:
            self.rbs_controllers.append(QRadioButton(controller_name))
        self.rbs_optimizers = []
        for optimizer_name in self.GuiActions.optimizer_names:
            self.rbs_optimizers.append(QRadioButton(optimizer_name))
        if self.GuiActions.controller is not None and hasattr(self.GuiActions.controller, 'has_optimizer'):
            self.GuiActions.update_rbs_optimizers_status(visible=self.GuiActions.controller.has_optimizer)

        # Ensures that radio buttons are exclusive
        self.controllers_buttons_group = QButtonGroup()
        for button in self.rbs_controllers:
            self.controllers_buttons_group.addButton(button)

        self.optimizers_buttons_group = QButtonGroup()
        for button in self.rbs_optimizers:
            self.optimizers_buttons_group.addButton(button)

        lr_c = QVBoxLayout()
        lr_c.addStretch(1)
        lr_c.addWidget(QLabel("Controller"))
        for rb in self.rbs_controllers:
            rb.clicked.connect(self.GuiActions.RadioButtons_controller_selection)
            lr_c.addWidget(rb)
        lr_c.addStretch(1)
        lr_c.addWidget(QLabel("MPC Optimizer"))
        for rb in self.rbs_optimizers:
            rb.clicked.connect(self.GuiActions.RadioButtons_optimizer_selection)
            lr_c.addWidget(rb)
        lr_c.addStretch(1)

        self.rbs_controllers[self.GuiActions.controller_idx].setChecked(True)
        if self.GuiActions.optimizer_idx is not None:
            self.rbs_optimizers[self.GuiActions.optimizer_idx].setChecked(True)

        # endregion

        # region - Create central part of the layout for figures and radio buttons and add it to the whole layout
        lc = QHBoxLayout()
        lc.addLayout(lf)
        lc.addLayout(lr_c)
        layout.addLayout(lc)

        # endregion

        # region - Gauges displaying current values of various states and parameters (time, velocity, angle,...)

        # First row
        ld = QHBoxLayout()
        # User time
        self.labTime = QLabel("User's time (s): ")
        self.timer = QTimer()
        self.timer.setInterval(100)  # Tick every 1/10 of the second
        self.timer.timeout.connect(self.GuiActions.set_user_time_label)
        self.timer.start()
        ld.addWidget(self.labTime)
        # Speed, angle, motor power (Q)
        self.labSpeed = QLabel('Speed (m/s):')
        self.labAngle = QLabel('Angle (deg):')
        self.labMotor = QLabel('')
        self.labTargetPosition = QLabel('')
        ld.addWidget(self.labSpeed)
        ld.addWidget(self.labAngle)
        ld.addWidget(self.labMotor)
        ld.addWidget(self.labTargetPosition)
        layout.addLayout(ld)

        # Second row of labels
        # Simulation time, Measured (real) speed-up, slider-value
        ld2 = QHBoxLayout()
        self.labTimeSim = QLabel('Simulation Time (s):')
        ld2.addWidget(self.labTimeSim)
        self.labSpeedUp = QLabel('Speed-up (measured):')
        ld2.addWidget(self.labSpeedUp)
        self.labSliderInstant = QLabel('')
        ld2.addWidget(self.labSliderInstant)
        layout.addLayout(ld2)

        # endregion

        # region - Buttons "START!" / "STOP!", "PAUSE", "QUIT"
        self.bss = QPushButton("START!")
        self.bss.pressed.connect(self.GuiActions.start_stop_button)
        self.bp = QPushButton("PAUSE")
        self.bp.pressed.connect(self.GuiActions.pause_unpause_button)
        bq = QPushButton("QUIT")
        bq.pressed.connect(quit_callback)
        lspb = QHBoxLayout()  # Sub-Layout for Start/Stop and Pause Buttons
        lspb.addWidget(self.bss)
        lspb.addWidget(self.bp)

        lb = QVBoxLayout()  # Layout for buttons
        lb.addLayout(lspb)
        lb.addWidget(bq)

        # endregion

        # region - up/down equilibrium switch
        lud = QHBoxLayout()

        # Left side - Tag
        lud_left = QVBoxLayout()
        lud_left.addStretch(1)
        lud_left.addWidget(QLabel('Target\nequilibrium:'))
        lud_left.addStretch(1)

        # Right side - buttons
        self.available_equilibria = ['Up', 'Down']
        self.rbs_equilibrium = []
        for equilibrium_name in self.available_equilibria:
            self.rbs_equilibrium.append(QRadioButton(equilibrium_name))

        # Ensures that radio buttons are exclusive
        self.equilibria_buttons_group = QButtonGroup()
        for button in self.rbs_equilibrium:
            self.equilibria_buttons_group.addButton(button)

        lud_right = QVBoxLayout()
        lud_right.addStretch(1)

        for rb in self.rbs_equilibrium:
            rb.clicked.connect(self.GuiActions.RadioButtons_equilibrium)
            lud_right.addWidget(rb)
        lud_right.addStretch(1)

        if self.GuiActions.target_equilibrium == 1.0:
            initial_target_equilibrium = 'Up'
        else:
            initial_target_equilibrium = 'Down'

        self.rbs_equilibrium[self.available_equilibria.index(initial_target_equilibrium)].setChecked(True)

        lud.addLayout(lud_left)
        lud.addLayout(lud_right)

        l_main_buttons_and_equilibria = QHBoxLayout()
        l_main_buttons_and_equilibria.addLayout(lb, stretch=1)
        l_main_buttons_and_equilibria.addLayout(lud)
        layout.addLayout(l_main_buttons_and_equilibria)

        # region - Sliders setting initial state and buttons for kicking the pole

        # Replace addStretch with a spacer item that has a specific size
        font_metrics = QFontMetrics(main_window.font())
        character_width = font_metrics.horizontalAdvance('W')  # Using 'W' as it's typically the widest
        spacer_with = 4 * character_width
        spacer = QSpacerItem(spacer_with, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        # Sliders setting initial position and angle
        ip = QHBoxLayout()  # Layout for initial position sliders
        self.initial_position_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.initial_position_slider.setRange(-int(float(1000 * TrackHalfLength)), int(float(1000 * TrackHalfLength)))
        self.initial_position_slider.setValue(0)
        self.initial_position_slider.setSingleStep(1)
        self.initial_position_slider.valueChanged.connect(self.GuiActions.update_initial_position)
        self.initial_angle_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.initial_angle_slider.setRange(-int(float(100 * np.pi)), int(float(100 * np.pi)))
        self.initial_angle_slider.setValue(0)
        self.initial_angle_slider.setSingleStep(1)
        self.initial_angle_slider.valueChanged.connect(self.GuiActions.update_initial_angle)
        ip.addWidget(QLabel("Initial position:"))
        ip.addWidget(self.initial_position_slider)
        ip.addWidget(QLabel("Initial angle:"))
        ip.addWidget(self.initial_angle_slider)
        ip.addSpacerItem(spacer)

        # Slider setting latency
        self.LATENCY_SLIDER_RANGE_INT = 1000
        self.latency_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.latency_slider.setRange(0, self.LATENCY_SLIDER_RANGE_INT)
        self.latency_slider.setValue(
            int(self.GuiActions.latency * self.LATENCY_SLIDER_RANGE_INT / self.GuiActions.max_latency))
        self.latency_slider.setSingleStep(1)
        self.latency_slider.valueChanged.connect(self.GuiActions.update_latency)
        ip.addWidget(QLabel("Latency:"))
        ip.addWidget(self.latency_slider)
        self.labLatency = QLabel(
            'Latency (ms): {:.1f}'.format(self.GuiActions.latency * 1000))
        ip.addWidget(self.labLatency)

        # Buttons activating noise
        self.rbs_noise = []
        for mode_name in ['ON', 'OFF']:
            self.rbs_noise.append(QRadioButton(mode_name))

        # Ensures that radio buttons are exclusive
        self.noise_buttons_group = QButtonGroup()
        for button in self.rbs_noise:
            self.noise_buttons_group.addButton(button)

        lr_n = QHBoxLayout()
        lr_n.addWidget(QLabel('Noise:'))
        for rb in self.rbs_noise:
            rb.clicked.connect(self.GuiActions.RadioButtons_noise_on_off)
            lr_n.addWidget(rb)

        self.rbs_noise[1].setChecked(True)

        ip.addSpacerItem(spacer)
        ip.addLayout(lr_n)
        ip.addSpacerItem(spacer)

        self.textbox_zero_angle_shift, l_zero_angle_shift = zero_angle_layout()
        self.GuiActions.zero_angle_shift_handler.connect_textbox(self.textbox_zero_angle_shift)
        ip.addLayout(l_zero_angle_shift)

        ip.addSpacerItem(spacer)

        # Buttons giving kick to the pole
        kick_label = QLabel("Kick pole:")
        kick_left_button = QPushButton()
        kick_left_button.setText("Left")
        kick_left_button.adjustSize()
        kick_left_button.clicked.connect(lambda: self.GuiActions.kick_pole("Left"))
        kick_right_button = QPushButton()
        kick_right_button.setText("Right")
        kick_right_button.adjustSize()
        kick_right_button.clicked.connect(lambda: self.GuiActions.kick_pole("Right"))
        ip.addWidget(kick_label)
        ip.addWidget(kick_left_button)
        ip.addWidget(kick_right_button)

        layout.addLayout(ip)

        # endregion

        # region - Text boxes and Combobox to provide settings concerning generation of random experiment
        l_generate_trace = QHBoxLayout()
        l_generate_trace.addWidget(QLabel('Random experiment settings:'))
        l_generate_trace.addWidget(QLabel('Length (s):'))
        self.textbox_length = QLineEdit()
        l_generate_trace.addWidget(self.textbox_length)
        l_generate_trace.addWidget(QLabel('Turning Points (m):'))
        self.textbox_turning_points = QLineEdit()
        l_generate_trace.addWidget(self.textbox_turning_points)
        l_generate_trace.addWidget(QLabel('Interpolation:'))
        self.cb_interpolation = QComboBox()
        self.cb_interpolation.addItems(['0-derivative-smooth', 'linear', 'previous'])
        self.cb_interpolation.currentIndexChanged.connect(self.GuiActions.cb_interpolation_selectionchange)
        self.cb_interpolation.setCurrentText(self.GuiActions.interpolation_type)
        l_generate_trace.addWidget(self.cb_interpolation)

        layout.addLayout(l_generate_trace)

        # endregion

        # region - Textbox to provide csv file name for saving or loading data
        l_text = QHBoxLayout()
        textbox_title = QLabel('CSV file name:')
        self.textbox = QLineEdit()
        l_text.addWidget(textbox_title)
        l_text.addWidget(self.textbox)
        layout.addLayout(l_text)

        # endregion

        # region - Make strip of layout for checkboxes
        l_cb = QHBoxLayout()
        # endregion

        # region - Textbox to provide the target speed-up value
        l_text_speedup = QHBoxLayout()
        tx_speedup_title = QLabel('Speed-up (target):')
        self.tx_speedup = QLineEdit()
        l_text_speedup.addWidget(tx_speedup_title)
        l_text_speedup.addWidget(self.tx_speedup)
        self.tx_speedup.setText(str(self.GuiActions.speedup))
        l_cb.addLayout(l_text_speedup)

        self.wrong_speedup_msg = QMessageBox()
        self.wrong_speedup_msg.setWindowTitle("Speed-up value problem")
        self.wrong_speedup_msg.setIcon(QMessageBox.Icon.Critical)
        # endregion

        # region - Checkboxes

        # region -- Checkbox: Save/don't save experiment recording
        self.cb_save_history = QCheckBox('Save results', main_window)
        if self.GuiActions.save_history:
            self.cb_save_history.toggle()
        self.cb_save_history.toggled.connect(self.GuiActions.cb_save_history_f)
        l_cb.addWidget(self.cb_save_history)
        # endregion

        # region -- Checkbox: Display plots showing dynamic evolution of the system as soon as experiment terminates
        self.cb_show_experiment_summary = QCheckBox('Show experiment summary', main_window)
        if self.GuiActions.show_experiment_summary:
            self.cb_show_experiment_summary.toggle()
        self.cb_show_experiment_summary.toggled.connect(self.GuiActions.cb_show_experiment_summary_f)
        l_cb.addWidget(self.cb_show_experiment_summary)
        # endregion

        # region -- Checkbox: Block pole if it reaches +/-90 deg
        self.cb_stop_at_90_deg = QCheckBox('Stop-at-90-deg', main_window)
        if self.GuiActions.stop_at_90:
            self.cb_stop_at_90_deg.toggle()
        self.cb_stop_at_90_deg.toggled.connect(self.GuiActions.cb_stop_at_90_deg_f)
        l_cb.addWidget(self.cb_stop_at_90_deg)
        # endregion

        # region -- Checkbox: Update slider on click/update slider while hoovering over it
        self.cb_slider_on_click = QCheckBox('Update slider on click', main_window)
        if self.GuiActions.slider_on_click:
            self.cb_slider_on_click.toggle()
        self.cb_slider_on_click.toggled.connect(self.GuiActions.cb_slider_on_click_f)
        l_cb.addWidget(self.cb_slider_on_click)

        # endregion

        # region -- Checkbox: Decide how the cartpole should be displayed
        self.cb_show_hanging_pole = QCheckBox('Show hanging pole', main_window)
        if self.GuiActions.show_hanging_pole:
            self.cb_show_hanging_pole.toggle()
        self.cb_show_hanging_pole.toggled.connect(self.GuiActions.cb_show_hanging_pole_f)
        l_cb.addWidget(self.cb_show_hanging_pole)

        # endregion

        # endregion

        # region - Radio buttons selecting simulator mode: user defined experiment, random experiment, replay

        # List available simulator modes - constant
        if os.getcwd().split(os.sep)[-1] == 'Driver':
            self.available_simulator_modes = ['Slider-Controlled Experiment', 'Random Experiment', 'Replay',
                                              'Physical CP']
        else:
            self.available_simulator_modes = ['Slider-Controlled Experiment', 'Random Experiment', 'Replay']

        self.rbs_simulator_mode = []
        for mode_name in self.available_simulator_modes:
            self.rbs_simulator_mode.append(QRadioButton(mode_name))

        # Ensures that radio buttons are exclusive
        self.simulator_mode_buttons_group = QButtonGroup()
        for button in self.rbs_simulator_mode:
            self.simulator_mode_buttons_group.addButton(button)

        lr_sm = QHBoxLayout()
        lr_sm.addStretch(1)
        lr_sm.addWidget(QLabel('Simulator mode:'))
        for rb in self.rbs_simulator_mode:
            rb.clicked.connect(self.GuiActions.RadioButtons_simulator_mode)
            lr_sm.addWidget(rb)
        lr_sm.addStretch(1)

        self.rbs_simulator_mode[self.available_simulator_modes.index(self.GuiActions.simulator_mode)].setChecked(True)

        l_cb.addStretch(1)
        l_cb.addLayout(lr_sm)
        l_cb.addStretch(1)

        # endregion

        # region - Add checkboxes to layout
        layout.addLayout(l_cb)
        # endregion

        self.layout = layout

    def get_csv_name_from_gui(self):
        self.textbox.text()


def zero_angle_layout():
    # Add the zero_angle_shift textbox with labels
    l_zero_angle_shift = QHBoxLayout()
    l_zero_angle_shift.addWidget(QLabel('0-angle: '))
    textbox_zero_angle_shift = QLineEdit()

    # Set the size based on the font metrics to fit 4 characters
    font_metrics = QFontMetrics(textbox_zero_angle_shift.font())
    character_width = font_metrics.horizontalAdvance('0')  # Width of one '0' character
    textbox_zero_angle_shift.setFixedWidth(character_width * 4 + 10)  # 4 characters + padding

    l_zero_angle_shift.addWidget(textbox_zero_angle_shift)
    l_zero_angle_shift.addWidget(QLabel('deg'))

    return textbox_zero_angle_shift, l_zero_angle_shift
