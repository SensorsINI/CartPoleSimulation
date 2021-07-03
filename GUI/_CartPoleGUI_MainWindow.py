"""
Main window (and main class) of CartPole GUI
"""

# Necessary only for debugging in Visual Studio Code IDE
try:
    import ptvsd
except:
    pass

from CartPole.cartpole_model import TrackHalfLength
import numpy as np
import time

# region Imports needed to create layout of the window in __init__ method

# Import functions from PyQt5 module (creating GUI)
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QApplication, QSlider, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox, \
    QLineEdit, QMessageBox, QComboBox, QButtonGroup
from PyQt5.QtCore import QThreadPool, QTimer, Qt
# The main drawing functionalities are implemented in CartPole Class
# Some more functions needed for interaction of matplotlib with PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# This import mus go before pyplot so also before our scripts
from matplotlib import use, get_backend
# Use Agg if not in scientific mode of Pycharm
if get_backend() != 'module://backend_interagg':
    use('Agg')

# endregion

# Import functions to measure time intervals and to pause a thread for a given time
from time import sleep
import csv

# Import Cart class - the class keeping all the parameters and methods
# related to CartPole which are not related to PyQt5 GUI
from CartPole import CartPole
from CartPole.state_utilities import ANGLED_IDX, ANGLE_IDX, POSITION_IDX, cartpole_state_varname_to_index, create_cartpole_state

from GUI.gui_default_params import *
from GUI.loop_timer import loop_timer
from GUI._CartPoleGUI_worker_template import Worker
from GUI._CartPoleGUI_summary_window import SummaryWindow


# Class implementing the main window of CartPole GUI
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # region Create CartPole instance and load initial settings

        # Create CartPole instance
        self.initial_state = create_cartpole_state()
        self.CartPoleInstance = CartPole(initial_state=self.initial_state)

        # Set timescales
        self.CartPoleInstance.dt_simulation = dt_simulation
        self.CartPoleInstance.dt_controller = controller_update_interval
        self.CartPoleInstance.dt_save = save_interval

        # set other settings
        self.CartPoleInstance.set_controller(controller_init)
        self.CartPoleInstance.stop_at_90 = stop_at_90_init
        self.set_random_experiment_generator_init_params()

        # endregion

        # region Decide whether to save the data in "CartPole memory" or not
        self.save_history = save_history_init
        self.show_experiment_summary = show_experiment_summary_init
        if self.save_history or self.show_experiment_summary:
            self.CartPoleInstance.save_data_in_cart = True
        else:
            self.CartPoleInstance.save_data_in_cart = False

        # endregion

        # region Other variables initial values as provided in gui_default_parameters.py

        # Start user controlled experiment/ start random experiment/ load and replay - on start button
        self.simulator_mode = simulator_mode_init
        self.slider_on_click = slider_on_click_init  # Update slider on click/update slider while hoovering over it
        self.speedup = speedup_init  # Default simulation speed-up

        # endregion

        # region Initialize loop-timer
        # This timer allows to relate the simulation time to user time
        # And (if your computer is fast enough) run simulation
        # slower or faster than real-time by predefined factor (speedup)
        self.looper = loop_timer(dt_target=(self.CartPoleInstance.dt_simulation / self.speedup))
        # endregion

        # region Variables controlling the state of various processes (DO NOT MODIFY)

        self.terminate_experiment_or_replay_thread = False  # True: gives signal causing thread to terminate
        self.pause_experiment_or_replay_thread = False  # True: gives signal causing the thread to pause

        self.run_set_labels_thread = True  # True if gauges (labels) keep being repeatedly updated
        # Stop threads by setting False

        # Flag indicating if the "START! / STOP!" button should act as start or as stop when pressed.
        # Can take values "START!" or "STOP!"
        self.start_or_stop_action = "START!"
        # Flag indicating whether the pause button should pause or unpause.
        self.pause_or_unpause_action = "PAUSE"

        # Flag indicating that saving of experiment recording to csv file has finished
        self.experiment_or_replay_thread_terminated = False

        self.user_time_counter = 0  # Measures the user time

        # Slider instant value (which is draw in GUI) differs from value saved in CartPole instance
        # if the option updating slider "on-click" is enabled.
        self.slider_instant_value = self.CartPoleInstance.slider_value

        # endregion

        # region Create GUI Layout

        # region - Create container for top level layout
        layout = QVBoxLayout()
        # endregion

        # region - Change geometry of the main window
        self.setGeometry(300, 300, 2500, 1000)
        # endregion

        # region - Matplotlib figures (CartPole drawing and Slider)
        # Draw Figure
        self.fig = Figure(figsize=(25, 10))  # Regulates the size of Figure in inches, before scaling to window size.
        self.canvas = FigureCanvas(self.fig)
        self.fig.AxCart = self.canvas.figure.add_subplot(211)
        self.fig.AxSlider = self.canvas.figure.add_subplot(212)
        self.fig.AxSlider.set_ylim(0, 1)

        self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)

        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.canvas)

        # endregion

        # region - Radio buttons selecting current controller
        self.rbs_controllers = []
        for controller_name in self.CartPoleInstance.controller_names:
            self.rbs_controllers.append(QRadioButton(controller_name))

        # Ensures that radio buttons are exclusive
        self.controllers_buttons_group = QButtonGroup()
        for button in self.rbs_controllers:
            self.controllers_buttons_group.addButton(button)

        lr_c = QVBoxLayout()
        lr_c.addStretch(1)
        for rb in self.rbs_controllers:
            rb.clicked.connect(self.RadioButtons_controller_selection)
            lr_c.addWidget(rb)
        lr_c.addStretch(1)

        self.rbs_controllers[self.CartPoleInstance.controller_idx].setChecked(True)

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
        self.timer.setInterval(100) # Tick every 1/10 of the second
        self.timer.timeout.connect(self.set_user_time_label)
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
        self.bss.pressed.connect(self.start_stop_button)
        self.bp = QPushButton("PAUSE")
        self.bp.pressed.connect(self.pause_unpause_button)
        bq = QPushButton("QUIT")
        bq.pressed.connect(self.quit_application)
        lspb = QHBoxLayout()  # Sub-Layout for Start/Stop and Pause Buttons
        lspb.addWidget(self.bss)
        lspb.addWidget(self.bp)

        lb = QVBoxLayout()  # Layout for buttons
        lb.addLayout(lspb)
        lb.addWidget(bq)
        ip = QHBoxLayout()  # Layout for initial position sliders
        self.initial_position_slider = QSlider(orientation=Qt.Horizontal)
        self.initial_position_slider.setRange(-int(float(1000*TrackHalfLength)), int(float(1000*TrackHalfLength)))
        self.initial_position_slider.setValue(0)
        self.initial_position_slider.setSingleStep(1)
        self.initial_position_slider.valueChanged.connect(self.update_initial_position)
        self.initial_angle_slider = QSlider(orientation=Qt.Horizontal)
        self.initial_angle_slider.setRange(-int(float(100*np.pi)), int(float(100*np.pi)))
        self.initial_angle_slider.setValue(0)
        self.initial_angle_slider.setSingleStep(1)
        self.initial_angle_slider.valueChanged.connect(self.update_initial_angle)
        ip.addWidget(QLabel("Initial position:"))
        ip.addWidget(self.initial_position_slider)
        ip.addWidget(QLabel("Initial angle:"))
        ip.addWidget(self.initial_angle_slider)

        kick_label = QLabel("Kick pole:")
        kick_left_button = QPushButton()
        kick_left_button.setText("Left")
        kick_left_button.adjustSize()
        kick_left_button.clicked.connect(self.kick_pole)
        kick_right_button = QPushButton()
        kick_right_button.setText("Right")
        kick_right_button.adjustSize()
        kick_right_button.clicked.connect(self.kick_pole)
        ip.addWidget(kick_label)
        ip.addWidget(kick_left_button)
        ip.addWidget(kick_right_button)

        lb.addLayout(ip)
        layout.addLayout(lb)

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
        self.cb_interpolation.currentIndexChanged.connect(self.cb_interpolation_selectionchange)
        self.cb_interpolation.setCurrentText(self.CartPoleInstance.interpolation_type)
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
        self.tx_speedup.setText(str(self.speedup))
        l_cb.addLayout(l_text_speedup)

        self.wrong_speedup_msg = QMessageBox()
        self.wrong_speedup_msg.setWindowTitle("Speed-up value problem")
        self.wrong_speedup_msg.setIcon(QMessageBox.Critical)
        # endregion

        # region - Checkboxes

        # region -- Checkbox: Save/don't save experiment recording
        self.cb_save_history = QCheckBox('Save results', self)
        if self.save_history:
            self.cb_save_history.toggle()
        self.cb_save_history.toggled.connect(self.cb_save_history_f)
        l_cb.addWidget(self.cb_save_history)
        # endregion

        # region -- Checkbox: Display plots showing dynamic evolution of the system as soon as experiment terminates
        self.cb_show_experiment_summary = QCheckBox('Show experiment summary', self)
        if self.show_experiment_summary:
            self.cb_show_experiment_summary.toggle()
        self.cb_show_experiment_summary.toggled.connect(self.cb_show_experiment_summary_f)
        l_cb.addWidget(self.cb_show_experiment_summary)
        # endregion

        # region -- Checkbox: Block pole if it reaches +/-90 deg
        self.cb_stop_at_90_deg = QCheckBox('Stop-at-90-deg', self)
        if self.CartPoleInstance.stop_at_90:
            self.cb_stop_at_90_deg.toggle()
        self.cb_stop_at_90_deg.toggled.connect(self.cb_stop_at_90_deg_f)
        l_cb.addWidget(self.cb_stop_at_90_deg)
        # endregion

        # region -- Checkbox: Update slider on click/update slider while hoovering over it
        self.cb_slider_on_click = QCheckBox('Update slider on click', self)
        if self.slider_on_click:
            self.cb_slider_on_click.toggle()
        self.cb_slider_on_click.toggled.connect(self.cb_slider_on_click_f)
        l_cb.addWidget(self.cb_slider_on_click)

        # endregion

        # endregion

        # region - Radio buttons selecting simulator mode: user defined experiment, random experiment, replay

        # List available simulator modes - constant
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
            rb.clicked.connect(self.RadioButtons_simulator_mode)
            lr_sm.addWidget(rb)
        lr_sm.addStretch(1)

        self.rbs_simulator_mode[self.available_simulator_modes.index(self.simulator_mode)].setChecked(True)

        l_cb.addStretch(1)
        l_cb.addLayout(lr_sm)
        l_cb.addStretch(1)

        # endregion

        # region - Add checkboxes to layout
        layout.addLayout(l_cb)
        # endregion

        # region - Create an instance of a GUI window
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.show()
        self.setWindowTitle('CartPole Simulator')

        # endregion

        # endregion

        # region Open controller-specific popup windows
        self.open_additional_widgets()
        # endregion

        # region Activate functions capturing mouse movements and clicks over the slider

        # This line links function capturing the mouse position on the canvas of the Figure
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_movement)
        # This line links function capturing the mouse position on the canvas of the Figure click
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)

        # endregion

        # region Introducing multithreading
        # To ensure smooth functioning of the app,
        # the calculations and redrawing of the figures have to be done in a different thread
        # than the one capturing the mouse position and running the animation
        self.threadpool = QThreadPool()
        # endregion

        # region Starts a thread repeatedly redrawing gauges (labels) of the GUI
        # It runs till the QUIT button is pressed
        worker_labels = Worker(self.set_labels_thread)
        self.threadpool.start(worker_labels)
        # endregion

        # region Start animation repeatedly redrawing changing elements of matplotlib figures (CartPole drawing and slider)
        # This animation runs ALWAYS when the GUI is open
        # The buttons of GUI only decide if new parameters are calculated or not
        self.anim = self.CartPoleInstance.run_animation(self.fig)
        # endregion

    def update_initial_position(self, value: str):
        self.initial_state[POSITION_IDX] = float(value) / 1000.0

    def update_initial_angle(self, value: str):
        self.initial_state[ANGLE_IDX] = float(value) / 100.0
    
    def kick_pole(self):
        if self.sender().text() == "Left":
            self.CartPoleInstance.s[ANGLED_IDX] += .6
        elif self.sender().text() == "Right":
            self.CartPoleInstance.s[ANGLED_IDX] -= .6


    # region Thread performing CartPole experiment, slider-controlled or random
    # It iteratively updates  CartPole state and save data to a .csv file
    # It also put simulation time in relation to user time
    def experiment_thread(self):

        # Necessary only for debugging in Visual Studio Code IDE
        try:
            ptvsd.debug_this_thread()
        except:
            pass

        self.looper.start_loop()
        while not self.terminate_experiment_or_replay_thread:
            if self.pause_experiment_or_replay_thread:
                time.sleep(0.1)
            else:
                # Calculations of the Cart state in the next timestep
                self.CartPoleInstance.update_state()

                # Terminate thread if random experiment reached its maximal length
                if (
                        (self.CartPoleInstance.use_pregenerated_target_position is True)
                        and
                        (self.CartPoleInstance.time >= self.CartPoleInstance.t_max_pre)
                ):
                    self.terminate_experiment_or_replay_thread = True

                # FIXME: when Speedup empty in GUI I expected inf speedup but got error Loop timer was not initialized properly
                self.looper.sleep_leftover_time()

        # Save simulation history if user chose to do so at the end of the simulation
        if self.save_history:
            csv_name = self.textbox.text()
            self.CartPoleInstance.save_history_csv(csv_name=csv_name,
                                                   mode='init',
                                                   length_of_experiment=np.around(
                                                       self.CartPoleInstance.dict_history['time'][-1],
                                                       decimals=2))
            self.CartPoleInstance.save_history_csv(csv_name=csv_name,
                                                   mode='save offline')

        self.experiment_or_replay_thread_terminated = True

    # endregion

    # region Thread replaying a saved experiment recording
    def replay_thread(self):

        # Necessary only for debugging in Visual Studio Code IDE
        try:
            ptvsd.debug_this_thread()
        except:
            pass

        # Check what is in the csv textbox
        csv_name = self.textbox.text()

        # Load experiment history
        history_pd, filepath = self.CartPoleInstance.load_history_csv(csv_name=csv_name)

        # Set cartpole in the right mode (just to ensure slider behaves properly)
        with open(filepath) as f:
            reader = csv.reader(f)
            for line in reader:
                line = line[0]
                if line[:len('# Controller: ')] == '# Controller: ':
                    controller_set = self.CartPoleInstance.set_controller(line[len('# Controller: '):].rstrip("\n"))
                    if controller_set:
                        self.rbs_controllers[self.CartPoleInstance.controller_idx].setChecked(True)
                    else:
                        self.rbs_controllers[1].setChecked(True) # Set first, but not manual stabilization
                    break

        # Augment the experiment history with simulation time step size
        dt = []
        row_iterator = history_pd.iterrows()
        _, last = next(row_iterator)  # take first item from row_iterator
        for i, row in row_iterator:
            dt.append(row['time'] - last['time'])
            last = row
        dt.append(dt[-1])
        history_pd['dt'] = np.array(dt)

        # Initialize loop timer (with arbitrary dt)
        replay_looper = loop_timer(dt_target=0.0)

        # Start looping over history
        replay_looper.start_loop()
        for index, row in history_pd.iterrows():
            self.CartPoleInstance.s[cartpole_state_varname_to_index('position')] = row['position']
            self.CartPoleInstance.s[cartpole_state_varname_to_index('positionD')] = row['positionD']
            self.CartPoleInstance.s[cartpole_state_varname_to_index('angle')] = row['angle']
            self.CartPoleInstance.time = row['time']
            self.CartPoleInstance.dt = row['dt']
            try:
                self.CartPoleInstance.u = row['u']
            except KeyError:
                pass
            self.CartPoleInstance.Q = row['Q']
            self.CartPoleInstance.target_position = row['target_position']
            if self.CartPoleInstance.controller_name == 'manual-stabilization':
                self.CartPoleInstance.slider_value = self.CartPoleInstance.Q
            else:
                self.CartPoleInstance.slider_value = self.CartPoleInstance.target_position/self.CartPoleInstance.p.TrackHalfLength

            dt_target = (self.CartPoleInstance.dt / self.speedup)
            replay_looper.dt_target = dt_target

            replay_looper.sleep_leftover_time()

            if self.terminate_experiment_or_replay_thread:  # Means that stop button was pressed
                break

            while self.pause_experiment_or_replay_thread:  # Means that pause button was pressed
                time.sleep(0.1)

        if self.show_experiment_summary:
            self.CartPoleInstance.dict_history = history_pd.loc[:index].to_dict(orient='list')

        self.experiment_or_replay_thread_terminated = True

    # endregion

    # region "START! / STOP!" button -> run/stop slider-controlled experiment, random experiment or replay experiment recording

    # Actions to be taken when "START! / STOP!" button is clicked
    def start_stop_button(self):

        # If "START! / STOP!" button in "START!" mode...
        if self.start_or_stop_action == 'START!':
            self.bss.setText("STOP!")
            self.start_thread()

        # If "START! / STOP!" button in "STOP!" mode...
        elif self.start_or_stop_action == 'STOP!':
            self.bss.setText("START!")
            self.bp.setText("PAUSE")
            # This flag is periodically checked by thread. It terminates if set True.
            self.terminate_experiment_or_replay_thread = True
            # The stop_thread function is called automatically by the thread when it terminates
            # It is implemented this way, because thread my terminate not only due "STOP!" button
            # (e.g. replay thread when whole experiment is replayed)
        
    def pause_unpause_button(self):
        # Only Pause if experiment is running
        if self.pause_or_unpause_action == 'PAUSE' and self.start_or_stop_action == 'STOP!':
            self.pause_or_unpause_action = 'UNPAUSE'
            self.pause_experiment_or_replay_thread = True
            self.bp.setText("UNPAUSE")
        elif self.pause_or_unpause_action == 'UNPAUSE' and self.start_or_stop_action == 'STOP!':
            self.pause_or_unpause_action = 'PAUSE'
            self.pause_experiment_or_replay_thread = False
            self.bp.setText("PAUSE")

    # Run thread. works for all simulator modes.
    def start_thread(self):

        # Check if value provided in speed-up textbox makes sense
        # If not abort start
        speedup_updated = self.get_speedup()
        if not speedup_updated:
            return

        # Disable GUI elements for features which must not be changed in runtime
        # For other features changing in runtime may not cause errors, but will stay without effect for current run
        self.cb_save_history.setEnabled(False)
        for rb in self.rbs_simulator_mode:
            rb.setEnabled(False)
        for rb in self.rbs_controllers:
            rb.setEnabled(False)
        if self.simulator_mode != 'Replay':
            self.cb_show_experiment_summary.setEnabled(False)

        # Set user-provided initial values for state (or its part) of the CartPole
        # Search implementation for more detail
        self.reset_variables(2, s=np.copy(self.initial_state), Q=self.CartPoleInstance.Q, target_position=self.CartPoleInstance.target_position)

        if self.simulator_mode == 'Random Experiment':

            self.CartPoleInstance.use_pregenerated_target_position = True

            if self.textbox_length.text() == '':
                self.CartPoleInstance.length_of_experiment = length_of_experiment_init
            else:
                self.CartPoleInstance.length_of_experiment = float(self.textbox_length.text())

            turning_points_list = []
            if self.textbox_turning_points.text() != '':
                for turning_point in self.textbox_turning_points.text().split(', '):
                    turning_points_list.append(float(turning_point))
            self.CartPoleInstance.turning_points = turning_points_list

            self.CartPoleInstance.setup_cartpole_random_experiment()

        self.looper.dt_target = self.CartPoleInstance.dt_simulation / self.speedup
        # Pass the function to execute
        if self.simulator_mode == "Replay":
            worker = Worker(self.replay_thread)
        elif self.simulator_mode == 'Slider-Controlled Experiment' or self.simulator_mode == 'Random Experiment':
            worker = Worker(self.experiment_thread)
        worker.signals.finished.connect(self.finish_thread)
        # Execute
        self.threadpool.start(worker)


        # Determine what should happen when "START! / STOP!" is pushed NEXT time
        self.start_or_stop_action = "STOP!"

    # finish_threads works for all simulation modes
    # Some lines mya be redundant for replay,
    # however as they do not take much computation time we leave them here
    # As it my code shorter, while hopefully still clear.
    # It is called automatically at the end of experiment_thread
    def finish_thread(self):

        self.CartPoleInstance.use_pregenerated_target_position = False
        self.initial_state = create_cartpole_state()
        self.initial_position_slider.setValue(0)
        self.initial_angle_slider.setValue(0)
        self.CartPoleInstance.s = self.initial_state

        # Some controllers may collect they own statistics about their usage and print it after experiment terminated
        if self.simulator_mode != 'Replay':
            try:
                self.CartPoleInstance.controller.controller_report()
            except:
                pass

        if self.show_experiment_summary:
            self.w_summary = SummaryWindow(summary_plots=self.CartPoleInstance.summary_plots)

        # Some controllers may need reset before being reused in the next experiment without reloading
        if self.simulator_mode != 'Replay':
            try:
                self.CartPoleInstance.controller.controller_reset()
            except:
                pass


        # Reset variables and redraw the figures
        self.reset_variables(0)

        # Draw figures
        self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()

        # Enable back all elements of GUI:
        self.cb_save_history.setEnabled(True)
        self.cb_show_experiment_summary.setEnabled(True)
        for rb in self.rbs_simulator_mode:
            rb.setEnabled(True)
        for rb in self.rbs_controllers:
            rb.setEnabled(True)

        self.start_or_stop_action = "START!"  # What should happen when "START! / STOP!" is pushed NEXT time

    # endregion

    # region Methods: "Get, set, reset, quit"

    # Set parameters from gui_default_parameters related to generating a random experiment target position
    def set_random_experiment_generator_init_params(self):
        self.CartPoleInstance.track_relative_complexity = track_relative_complexity_init
        self.CartPoleInstance.length_of_experiment = length_of_experiment_init
        self.CartPoleInstance.interpolation_type = interpolation_type_init
        self.CartPoleInstance.turning_points_period = turning_points_period_init
        self.CartPoleInstance.start_random_target_position_at = start_random_target_position_at_init
        self.CartPoleInstance.end_random_target_position_at = end_random_target_position_at_init
        self.CartPoleInstance.turning_points = turning_points_init

    # Method resetting variables which change during experimental run
    def reset_variables(self, reset_mode=1, s=None, Q=None, target_position=None):
        self.CartPoleInstance.set_cartpole_state_at_t0(reset_mode, s=s, Q=Q, target_position=target_position)
        self.user_time_counter = 0
        # "Try" because this function is called for the first time during initialisation of the Window
        # when the timer label instance is not yer there.
        try:
            self.labt.setText("Time (s): " + str(float(self.user_time_counter) / 10.0))
        except:
            pass
        self.experiment_or_replay_thread_terminated = False  # This is a flag informing thread terminated
        self.terminate_experiment_or_replay_thread = False  # This is a command to terminate a thread
        self.pause_experiment_or_replay_thread = False  # This is a command to pause a thread
        self.start_or_stop_action = "START!"
        self.pause_or_unpause_action = "PAUSE"
        self.looper.first_call_done = False

    ######################################################################################################

    # (Marcin) Below are methods with less critical functions.

    # A thread redrawing labels (except for timer, which has its own function) of GUI every 0.1 s
    def set_labels_thread(self):
        while (self.run_set_labels_thread):
            self.labSpeed.setText("Speed (m/s): " + str(np.around(self.CartPoleInstance.s[cartpole_state_varname_to_index('positionD')], 2)))
            self.labAngle.setText(
                "Angle (deg): " + str(np.around(self.CartPoleInstance.s[cartpole_state_varname_to_index('angle')] * 360 / (2 * np.pi), 2)))
            self.labMotor.setText("Motor power (Q): {:.3f}".format(np.around(self.CartPoleInstance.Q, 2)))
            if self.CartPoleInstance.controller_name == 'manual-stabilization':
                self.labTargetPosition.setText("")
            else:
                self.labTargetPosition.setText(
                    "Target position (m): " + str(np.around(self.CartPoleInstance.target_position, 2)))

            if self.CartPoleInstance.controller_name == 'manual_stabilization':
                self.labSliderInstant.setText(
                    "Slider instant value (-): " + str(np.around(self.slider_instant_value, 2)))
            else:
                self.labSliderInstant.setText(
                    "Slider instant value (m): " + str(np.around(self.slider_instant_value, 2)))

            self.labTimeSim.setText('Simulation time (s): {:.2f}'.format(self.CartPoleInstance.time))

            mean_dt_real = np.mean(self.looper.circ_buffer_dt_real)
            if mean_dt_real > 0:
                self.labSpeedUp.setText('Speed-up (measured): x{:.2f}'
                                        .format(self.CartPoleInstance.dt_simulation / mean_dt_real))
            sleep(0.1)

    # Function to measure the time of simulation as experienced by user
    # It corresponds to the time of simulation according to equations only if real time mode is on
    # TODO (Marcin) I just retained this function from some example being my starting point
    #   It seems it sometimes counting time to slow. Consider replacing in future
    def set_user_time_label(self):
        # "If": Increment time counter only if simulation is running
        if self.start_or_stop_action == "STOP!": # indicates what start button was pressed and some process is running
            self.user_time_counter += 1
            # The updates are done smoother if the label is updated here
            # and not in the separate thread
            self.labTime.setText("Time (s): " + str(float(self.user_time_counter) / 10.0))

    # The actions which has to be taken to properly terminate the application
    # The method is evoked after QUIT button is pressed
    # TODO: Can we connect it somehow also the the default cross closing the application?
    def quit_application(self):
        # Stops animation (updating changing elements of the Figure)
        self.anim._stop()
        # Stops the two threads updating the GUI labels and updating the state of Cart instance
        self.run_set_labels_thread = False
        self.terminate_experiment_or_replay_thread = True
        self.pause_experiment_or_replay_thread = False
        # Closes the GUI window
        self.close()
        # The standard command
        # It seems however not to be working by its own
        # I don't know how it works
        QApplication.quit()

    # endregion

    # region Mouse interaction

    """
    These are some methods GUI uses to capture mouse effect while hoovering or clicking over/on the charts
    """

    # Function evoked at a mouse movement
    # If the mouse cursor is over the lower chart it reads the corresponding value
    # and updates the slider
    def on_mouse_movement(self, event):
        if self.simulator_mode == 'Slider-Controlled Experiment':
            if event.xdata == None or event.ydata == None:
                pass
            else:
                if event.inaxes == self.fig.AxSlider:
                    self.slider_instant_value = event.xdata
                    if not self.slider_on_click:
                        self.CartPoleInstance.update_slider(mouse_position=event.xdata)

    # Function evoked at a mouse click
    # If the mouse cursor is over the lower chart it reads the corresponding value
    # and updates the slider
    def on_mouse_click(self, event):
        if self.simulator_mode == 'Slider-Controlled Experiment':
            if event.xdata == None or event.ydata == None:
                pass
            else:
                if event.inaxes == self.fig.AxSlider:
                    self.CartPoleInstance.update_slider(mouse_position=event.xdata)

    # endregion

    # region Changing "static" options: radio buttons, text boxes, combo boxes, check boxes

    """
    This section collects methods used to change some ''static option'':
    e.g. change current controller, switch between saving and not saving etc.
    These are functions associated with radio buttons, check boxes, textfilds etc.
    The functions of "START! / STOP!" button is much more complex
    and we put them hence in a separate section.
    """

    # region - Radio buttons

    # Chose the controller method which should be used with the CartPole
    def RadioButtons_controller_selection(self):
        # Change the mode variable depending on the Radiobutton state
        for i in range(len(self.rbs_controllers)):
            if self.rbs_controllers[i].isChecked():
                self.CartPoleInstance.set_controller(controller_idx=i)

        # Reset the state of GUI and of the Cart instance after the mode has changed
        # TODO: Do I need the follwowing lines?
        self.reset_variables(0)
        self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()

        self.open_additional_widgets()

    # Chose the simulator mode - effect of start/stop button
    def RadioButtons_simulator_mode(self):
        # Change the mode variable depending on the Radiobutton state
        for i in range(len(self.rbs_simulator_mode)):
            sleep(0.001)
            if self.rbs_simulator_mode[i].isChecked():
                self.simulator_mode = self.available_simulator_modes[i]

        # Reset the state of GUI and of the Cart instance after the mode has changed
        # TODO: Do I need the follwowing lines?
        self.reset_variables(0)
        self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()

    # endregion

    # region - Text Boxes

    # Read speedup provided by user from appropriate GUI textbox
    def get_speedup(self):
        """
        Get speedup provided by user from appropriate textbox.
        Speed-up gives how many times faster or slower than real time the simulation or replay should run.
        The provided values may not always be reached due to computer speed limitation
        """
        speedup = self.tx_speedup.text()
        if speedup == '':
            self.speedup = np.inf
            return True
        else:
            try:
                speedup = float(speedup)
            except ValueError:
                self.wrong_speedup_msg.setText(
                    'You have provided the input for speed-up which is not convertible to a number')
                x = self.wrong_speedup_msg.exec_()
                return False
            if speedup == 0.0:
                self.wrong_speedup_msg.setText(
                    'You cannot run an experiment with 0 speed-up (stopped time flow)')
                x = self.wrong_speedup_msg.exec_()
                return False
            else:
                self.speedup = speedup
                return True

    # endregion

    # region - Combo Boxes

    # Select how to interpolate between turning points of randomly chosen target positions
    def cb_interpolation_selectionchange(self, i):
        """
        Select interpolation type for random target positions of randomly generated experiment
        """
        self.CartPoleInstance.interpolation_type = self.cb_interpolation.currentText()

    # endregion

    # region - Check boxes

    # Action toggling between saving and not saving simulation results
    def cb_save_history_f(self, state):

        if state:
            self.save_history = 1
        else:
            self.save_history = 0

        if self.save_history or self.show_experiment_summary:
            self.CartPoleInstance.save_data_in_cart = True
        else:
            self.CartPoleInstance.save_data_in_cart = False

    # Action toggling between saving and not saving simulation results
    def cb_show_experiment_summary_f(self, state):

        if state:
            self.show_experiment_summary = 1
        else:
            self.show_experiment_summary = 0

        if self.save_history or self.show_experiment_summary:
            self.CartPoleInstance.save_data_in_cart = True
        else:
            self.CartPoleInstance.save_data_in_cart = False

    # Action toggling between stopping (or not) the pole if it reaches 90 deg
    def cb_stop_at_90_deg_f(self, state):

        if state:
            self.CartPoleInstance.stop_at_90 = True
        else:
            self.CartPoleInstance.stop_at_90 = False

    # Action toggling between updating CarPole slider value on click or by hoovering over it
    def cb_slider_on_click_f(self, state):

        if state:
            self.slider_on_click = True
        else:
            self.slider_on_click = False

    # endregion

    # region - Additional GUI Popups

    def open_additional_widgets(self):
        # Open up additional options widgets depending on the controller type
        if self.CartPoleInstance.controller_name == 'mppi':
            from GUI._ControllerGUI_MPPIOptionsWindow import MPPIOptionsWindow
            self.optionsWidget = MPPIOptionsWindow()
        else:
            try: self.optionsWidget.close()
            except: pass
            self.optionsWidget = None

    # endregion








