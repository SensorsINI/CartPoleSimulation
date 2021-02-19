"""
Main window (and main class) of CartPole GUI
"""

import numpy as np

# region Imports needed to create layout of the window in __init__ method

# Import functions from PyQt5 module (creating GUI)
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QApplication, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox, \
    QLineEdit, QMessageBox, QComboBox, QButtonGroup
from PyQt5.QtCore import QThreadPool, QTimer
# The main drawing functionalities are implemented in CartPole Class
# Some more functions needed for interaction of matplotlib with PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# endregion


# Import functions to measure time intervals and to pause a thread for a given time
from time import sleep

# Import Cart class - the class keeping all the parameters and methods
# related to CartPole which are not related to PyQt5 GUI
from CartPole import CartPole

from CartPoleGUI.gui_default_params import *
from CartPoleGUI.loop_timer import loop_timer
from CartPoleGUI._CartPoleGUI_worker_template import Worker
from CartPoleGUI._CartPoleGUI_summary_window import SummaryWindow


# Class implementing the main window of CartPole GUI
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # region Create CartPole instance and load initial settings

        # Create CartPole instance
        self.CartPoleInstance = CartPole()

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

        self.run_experiment_thread = False  # True if experiment thread is running
        self.run_set_labels_thread = True  # True if gauges (labels) keep being repeatedly updated
        # Stop threads by setting False

        self.saved = False  # Flag indicating that saving of experiment recording to csv file has finished

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
        self.timer.setInterval(100)
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

        # region - Buttons "START! / STOP!", "QUIT"
        bss = QPushButton("START! / STOP!")
        bss.pressed.connect(self.start_stop_button)
        bq = QPushButton("QUIT")
        bq.pressed.connect(self.quit_application)
        lb = QVBoxLayout()  # Layout for buttons
        lb.addWidget(bss)
        lb.addWidget(bq)
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

        # endregion

        # region - Add checkboxes to layout
        l_cb.addStretch(1)
        l_cb.addLayout(lr_sm)
        l_cb.addStretch(1)
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


    # region This method performs CartPole experiment
    # It iteratively updates  CartPole state and save data to a .csv file
    # It also put simulation time in relation to user time
    def experiment_thread(self):
        self.looper.start_loop()
        while (self.run_experiment_thread):

            # Calculations of the Cart state in the next timestep
            self.CartPoleInstance.update_state()

            # Ensure that the animation drawing function can access CartPoleInstance at this moment
            QApplication.processEvents()

            # Terminate thread if random experiment reached it maximal length
            if (
                    (self.CartPoleInstance.use_pregenerated_target_position is True)
                    and
                    (self.CartPoleInstance.time >= self.CartPoleInstance.t_max_pre)
            ):
                self.run_experiment_thread = 0

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
            self.saved = True

    # endregion

    # region "START/STOP" button -> run slider-controlled experiment, random experiment or replay experiment recording

    # Actions to be taken when start/stop button is clicked
    def start_stop_button(self):
        # Start/stop a new user controller experiment
        if self.simulator_mode == 'Slider-Controlled Experiment':
            self.play()
        # Or start a random experiment
        elif self.simulator_mode == 'Random Experiment':
            self.generate_random_experiment()
        # Or launch a replay function (immediately in a new thread)...
        elif self.simulator_mode == 'Replay':
            worker_replay = Worker(self.thread_replay)
            self.threadpool.start(worker_replay)
        else:
            raise ValueError('Not a proper value for self.simulator_mode')


    # Effect of start button if "load data" check box is unchecked
    def play(self):

        # Terminate experiment if it was running
        if self.run_experiment_thread == 1:
            self.run_experiment_thread = 0
            # If user is saving data wait till data is saved
            if self.save_history:
                while not self.saved:
                    sleep(0.001)

            self.CartPoleInstance.use_pregenerated_target_position = False

            # If
            try:
                self.CartPoleInstance.controller.controller_summary()
            except:
                pass

            if self.show_experiment_summary:
                self.CartPoleInstance.summary_plots()
                self.w_summary = SummaryWindow(summary_plots=self.CartPoleInstance.summary_plots)
            # Reset variables and redraw the figures
            self.reset_variables(0)
            # Draw figures
            self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
            self.canvas.draw()
            self.cb_save_history.setEnabled(True)
            self.cb_show_experiment_summary.setEnabled(True)

        # Start experiment if it was not running
        elif self.run_experiment_thread == 0:
            self.cb_save_history.setEnabled(False)
            self.cb_show_experiment_summary.setEnabled(False)
            speedup_updated = self.get_speedup()
            try:
                self.CartPoleInstance.controller.reset()
            except:
                print('Controller reset not done')
            if speedup_updated:
                self.reset_variables(1)
                self.looper.dt_target = self.CartPoleInstance.dt_simulation / self.speedup
                self.run_experiment_thread = 1
                # Pass the function to execute
                worker_calculations = Worker(self.experiment_thread)
                # Execute
                self.threadpool.start(worker_calculations)

    # Effect of start button if "load data" check box is checked
    def thread_replay(self):
        # Check what is in the csv textbox
        csv_name = self.textbox.text()
        history_pd = self.CartPoleInstance.load_history_csv(csv_name=csv_name)

        dt = []
        row_iterator = history_pd.iterrows()
        _, last = next(row_iterator)  # take first item from row_iterator
        for i, row in row_iterator:
            dt.append(row['time'] - last['time'])
            last = row
        dt.append(dt[-1])
        history_pd['dt'] = np.array(dt)

        # Check speedup which user provided with GUI
        self.get_speedup()

        # Define loop timer for now with arbitrary dt
        replay_looper = loop_timer(dt_target=0.0)

        # Start looper
        replay_looper.start_loop()
        for index, row in history_pd.iterrows():
            self.CartPoleInstance.s.position = row['s.position']
            self.CartPoleInstance.s.positionD = row['s.positionD']
            self.CartPoleInstance.s.angle = row['s.angle']
            self.CartPoleInstance.time = row['time']
            self.CartPoleInstance.dt = row['dt']
            self.CartPoleInstance.u = row['u']
            self.CartPoleInstance.Q = row['Q']
            self.CartPoleInstance.target_position = row['target_position']
            self.CartPoleInstance.slider_value = self.CartPoleInstance.target_position

            dt_target = (self.CartPoleInstance.dt / self.speedup)
            replay_looper.dt_target = dt_target

            replay_looper.sleep_leftover_time()

        dict_history = history_pd.to_dict(orient='list')
        self.CartPoleInstance.dict_history = dict_history
        self.CartPoleInstance.summary_plots()
        self.reset_variables(0)

    # Generate experiment with random target position trace
    def generate_random_experiment(self):

        self.cb_save_history.setEnabled(False)
        if self.textbox_length.text() == '':
            self.CartPoleInstance.random_length = random_length_globals
        else:
            self.CartPoleInstance.random_length = float(self.textbox_length.text())

        turning_points_list = []
        if self.textbox_turning_points.text() != '':
            for turning_point in self.textbox_turning_points.text().split(', '):
                turning_points_list.append(float(turning_point))
        self.CartPoleInstance.turning_points = turning_points_list

        self.CartPoleInstance.Generate_Random_Trace_Function()
        if self.run_experiment_thread == 1:
            print('First reset the previous run')
        else:
            self.CartPoleInstance.use_pregenerated_target_position = True
            self.run_experiment_thread = 1
            # Pass the function to execute
            worker_experiment = Worker(self.experiment_thread)
            # Execute
            self.threadpool.start(worker_experiment)
            while (self.run_experiment_thread == 1):
                QApplication.processEvents()
            # If user is saving data wait till data is saved
            if self.save_history:
                while not self.saved:
                    QApplication.processEvents()
                    sleep(0.001)

            self.CartPoleInstance.use_pregenerated_target_position = False
            self.CartPoleInstance.summary_plots()
            self.w_summary = SummaryWindow(summary_plots=self.CartPoleInstance.summary_plots)
            # Reset all the variables which change during an experiment run
            self.reset_variables(0)
            # Draw figures
            self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
            self.canvas.draw()
            self.cb_save_history.setEnabled(True)

    # endregion

    # region Methods: "Get, set, reset, quit"

    # Set parameters from gui_default_parameters related to generating a random experiment target position
    def set_random_experiment_generator_init_params(self):
        self.CartPoleInstance.track_relative_complexity = track_relative_complexity_globals
        self.CartPoleInstance.random_length = random_length_globals
        self.CartPoleInstance.interpolation_type = interpolation_type_globals
        self.CartPoleInstance.turning_points_period = turning_points_period_globals
        self.CartPoleInstance.start_random_target_position_at = start_random_target_position_at_globals
        self.CartPoleInstance.end_random_target_position_at = end_random_target_position_at_globals
        self.CartPoleInstance.turning_points = turning_points_globals

    # Method resetting variables which change during experimental run
    def reset_variables(self, reset_mode=1):
        self.CartPoleInstance.set_cartpole_state_at_t0(reset_mode)
        self.user_time_counter = 0
        # "Try" because this function is called for the first time during initialisation of the Window
        # when the timer label instance is not yer there.
        try:
            self.labt.setText("Time (s): " + str(float(self.user_time_counter) / 10.0))
        except:
            pass
        self.saved = False
        self.looper.first_call_done = False

    ######################################################################################################

    # (Marcin) Below are methods with less critical functions.

    # A thread redrawing labels (except for timer, which has its own function) of GUI every 0.1 s
    def set_labels_thread(self):
        while (self.run_set_labels_thread):
            self.labSpeed.setText("Speed (m/s): " + str(np.around(self.CartPoleInstance.s.positionD, 2)))
            self.labAngle.setText(
                "Angle (deg): " + str(np.around(self.CartPoleInstance.s.angle * 360 / (2 * np.pi), 2)))
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
        if self.run_experiment_thread == 1:
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
        self.run_experiment_thread = False
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

    # endregion








