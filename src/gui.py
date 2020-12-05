# Import functions from PyQt5 module (creating GUI)
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QApplication, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox, \
    QLineEdit, QMessageBox, QComboBox
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, QTimer, Qt
# Import functions to measure time intervals and to pause a thread for a given time
from time import sleep
import timeit
# The main drawing functionalities are implemented in Cart Class in another file
# Some more functions needed for interaction of matplotlib with PyQt5 and
# for running the animation are imported here
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import animation
import matplotlib.pyplot as plt
# Import function from numpy library
from numpy import pi, around, array, inf

import numpy as np
import pandas as pd

# Import Cart class - the class keeping all the parameters and methods
# related to CartPole which are not related to PyQt5 GUI
from src.CartClass import Cart

from src.globals import *
from src.utilis import *


# Window displaying summary (matplotlib plots) of an experiment with CartPole after clicking Stop button
# (if experiment was previously running)
class SummaryWindow(QWidget):
    def __init__(self, summary_plots=None):
        super(SummaryWindow, self).__init__()

        ## Create GUI Layout
        layout = QVBoxLayout()

        self.fig, self.axs = summary_plots()
        for axis in self.axs:
            axis.tick_params(axis='both', which='major', labelsize=9)
            axis.xaxis.label.set_fontsize(11)
            axis.yaxis.label.set_fontsize(11)
        self.fig.subplots_adjust(bottom=0.11)
        self.fig.subplots_adjust(hspace=0.3)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.toolbar)
        lf.addWidget(self.canvas)
        layout.addLayout(lf)


        # add quit button
        bq = QPushButton("QUIT")
        bq.pressed.connect(self.close)
        lb = QVBoxLayout()  # Layout for buttons
        lb.addWidget(bq)
        layout.addLayout(lb)

        self.setLayout(layout)
        self.setGeometry(100, 150, 1500, 800)
        # self.adjustSize()
        self.show()
        self.setWindowTitle('Last experiment summary')



# The following classes WorkerSignals and Worker are a standard tamplete
# used to implement multithreading in PyQt5 context

class WorkerSignals(QObject):
    result = pyqtSignal(object)


class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        result = self.fn(*self.args, **self.kwargs)
        self.signals.result.emit(result)  # Return the result of the processing


# Class implementing the main (and currently the only) window of our GUI
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Stop threads if False
        self.run_thread_calculations = False
        self.run_thread_labels = True
        self.run_replay = False

        self.counter = 0
        self.save_history = save_history_globals
        self.stop_at_90 = stop_at_90_globals
        self.load_recording = load_recording_globals
        self.slider_on_click = slider_on_click_globals
        self.saved = 0
        self.printing_summary = 1
        self.Q_thread_enabled = False

        self.dt_main_simulation = dt_main_simulation_globals
        self.speedup = speedup_globals
        self.looper = loop_timer(dt_target=(self.dt_main_simulation / self.speedup))

        # Create Cart object

        self.MyCart = Cart()
        self.slider_value = self.MyCart.slider_value

        ## Create GUI Layout
        layout = QVBoxLayout()
        # Change the window geometry
        self.setGeometry(300, 300, 2500, 1000)

        # Create layout for Matplotlib figures only
        # Draw Figure
        self.fig = Figure(figsize=(25, 10))  # Regulates the size of Figure in inches, before scaling to window size.
        self.canvas = FigureCanvas(self.fig)
        self.fig.AxCart = self.canvas.figure.add_subplot(211)
        self.fig.AxSlider = self.canvas.figure.add_subplot(212)

        self.MyCart.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)

        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.canvas)

        # Radiobuttons to toggle the mode of operation
        lr = QVBoxLayout()
        self.rbs = []
        for controller_name in self.MyCart.controller_names:
            self.rbs.append(QRadioButton(controller_name))

        lr.addStretch(1)
        for rb in self.rbs:
            rb.clicked.connect(self.RadioButtons)
            lr.addWidget(rb)
        lr.addStretch(1)
        # self.rb_manual.setChecked(True)
        self.rbs[self.MyCart.mode].setChecked(True)

        # Create main part of the layout for Figures and radiobuttons
        # And add it to the whole layout
        lm = QHBoxLayout()
        lm.addLayout(lf)
        lm.addLayout(lr)
        layout.addLayout(lm)

        # Displays of current relevant values:
        # Time(user time, not necesarily time of the Cart (i.e. used in sinmulation)),
        # Speed, Angle and slider value
        ld = QHBoxLayout()
        self.labTime = QLabel("User's time (s): ")
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()
        self.labSpeed = QLabel('Speed (m/s):')
        self.labAngle = QLabel('Angle (deg):')
        self.labMotor = QLabel('')
        self.labTargetPosition = QLabel('')
        ld.addWidget(self.labTime)
        ld.addWidget(self.labSpeed)
        ld.addWidget(self.labAngle)
        ld.addWidget(self.labMotor)
        ld.addWidget(self.labTargetPosition)
        layout.addLayout(ld)

        # Second row of labels
        ld2 = QHBoxLayout()
        self.labTimeSim = QLabel('Simulation Time (s):')
        ld2.addWidget(self.labTimeSim)
        self.labSpeedUp = QLabel('Speed-up (measured):')
        ld2.addWidget(self.labSpeedUp)
        self.labSliderInstant = QLabel('')
        ld2.addWidget(self.labSliderInstant)
        layout.addLayout(ld2)

        # Buttons "START/STOP", "RESET", "QUIT"
        bss = QPushButton("START!/STOP!")
        bss.pressed.connect(self.play)
        bq = QPushButton("QUIT")
        bq.pressed.connect(self.quit_application)
        bt = QPushButton("GENERATE TRACE")
        bt.pressed.connect(self.generate_trace)
        lb = QVBoxLayout()  # Layout for buttons
        lb.addWidget(bss)
        lb.addWidget(bq)
        lb.addWidget(bt)
        layout.addLayout(lb)

        l_generate_trace = QHBoxLayout()
        l_generate_trace.addWidget(QLabel('Generate trace settings:'))
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
        self.cb_interpolation.setCurrentText(self.MyCart.interpolation_type)
        l_generate_trace.addWidget(self.cb_interpolation)

        layout.addLayout(l_generate_trace)

        # Textbox to add provide a file name
        l_text = QHBoxLayout()
        textbox_title = QLabel('CSV file name:')
        self.textbox = QLineEdit()
        l_text.addWidget(textbox_title)
        l_text.addWidget(self.textbox)
        layout.addLayout(l_text)

        # Checkboxs:
        # TODO to decide if to plot simulation history
        # TODO change real-time checkbox to textbox speedup
        l_cb = QHBoxLayout()

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

        self.cb_load_recorded_data = QCheckBox('Load recorded data', self)
        if self.load_recording:
            self.cb_load_recorded_data.toggle()
        self.cb_load_recorded_data.toggled.connect(self.cb_load_recorded_data_f)
        l_cb.addWidget(self.cb_load_recorded_data)

        self.cb_save_history = QCheckBox('Save results', self)
        if self.save_history:
            self.cb_save_history.toggle()
        self.cb_save_history.toggled.connect(self.cb_save_history_f)
        l_cb.addWidget(self.cb_save_history)

        self.cb_stop_at_90_deg = QCheckBox('Stop-at-90-deg', self)
        if self.stop_at_90:
            self.cb_stop_at_90_deg.toggle()
        self.cb_stop_at_90_deg.toggled.connect(self.cb_stop_at_90_deg_f)
        l_cb.addWidget(self.cb_stop_at_90_deg)

        self.cb_slider_on_click = QCheckBox('Update slider on click', self)
        if self.slider_on_click:
            self.cb_slider_on_click.toggle()
        self.cb_slider_on_click.toggled.connect(self.cb_slider_on_click_f)
        l_cb.addWidget(self.cb_slider_on_click)

        self.load_generate_conflict_msg = QMessageBox()
        self.load_generate_conflict_msg.setWindowTitle("Load-generate conflict")
        self.load_generate_conflict_msg.setIcon(QMessageBox.Critical)
        self.load_generate_conflict_msg.setText(
            'You cannot use "GENERATE TRACE" button \nwith "load recorded data" box checked.')


        l_cb.addStretch(1)

        layout.addLayout(l_cb)

        # Create an instance of a GUI window
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.show()
        self.setWindowTitle('CartPole')

        # This line introduces multithreading:
        # to ensure smooth functioning of the app,
        # the calculations and redrawing of the figures have to be done
        # in a different thread thqn the one cupturing the mouse position
        self.threadpool = QThreadPool()

        # This line links function capturing the mouse position on the canvas of the Figure
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_movement)
        # This line links function capturing the mouse position on the canvas of the Figure click
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)

        # Starts a thread constantly redrawing labels of the GUI
        # It runs till the QUIT button is pressed
        worker_labels = Worker(self.thread_labels)
        self.threadpool.start(worker_labels)

        self.reset_variables()
        # Defines a variable holding animation object
        self.anim = None
        # Start redrawing the changing elements of the Figure
        # It runs till the QUIT button is pressed
        self.run_animation()

    # Function evoked at a mouse movement
    # If the mouse cursor is over the lower chart it reads the corresponding value
    # and updates the slider
    def on_mouse_movement(self, event):
        if event.xdata == None or event.ydata == None:
            pass
        else:
            if event.inaxes == self.fig.AxSlider:
                self.slider_value = event.xdata
                if not self.slider_on_click:
                    self.MyCart.update_slider(mouse_position=event.xdata)

    # Function evoked at a mouse click
    # If the mouse cursor is over the lower chart it reads the corresponding value
    # and updates the slider
    def on_mouse_click(self, event):
        if event.xdata == None or event.ydata == None:
            pass
        else:
            if event.inaxes == self.fig.AxSlider:
                self.MyCart.update_slider(mouse_position=event.xdata)

    def cb_interpolation_selectionchange(self, i):
        self.MyCart.interpolation_type = self.cb_interpolation.currentText()

    def get_speedup(self):
        speedup = self.tx_speedup.text()
        if speedup == '':
            self.speedup = inf
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

    # Method resetting variables
    def reset_variables(self):
        self.MyCart.reset_state()
        self.MyCart.reset_dict_history()
        self.counter = 0
        # "Try" because this function is called for the first time during initialisation of the Window
        # when the timer label instance is not yer there.
        try:
            self.labt.setText("Time (s): " + str(float(self.counter) / 10.0))
        except:
            pass
        self.saved = 0
        self.looper.first_call_done = False

    # This method initiate calculation of simulation and iterative updates of Cart state
    # It also measures time intervals for real time simulation
    # implements a termination condition if Pole went out of control
    # and initiate saving to a .csv file
    def thread_calculations(self):

        self.looper.start_loop()
        while (self.run_thread_calculations):

            # Calculations of the Cart state in the next timestep
            self.MyCart.update_state(dt=self.dt_main_simulation)

            # Ensure that the animation drawing function can access MyCart at this moment
            QApplication.processEvents()

            if self.MyCart.use_pregenerated_target_position == True and self.MyCart.time >= self.MyCart.t_max_pre:
                # print('Terminating!')
                self.run_thread_calculations = 0

            # FIXME: when Speedup empty in GUI I expected inf speedup but got error Loop timer was not initialized properly
            self.looper.sleep_leftover_time()

        # print('Welcome')
        # Save simulation history if user chose to do so at the end of the simulation
        if self.save_history:
            csv_name = self.textbox.text()
            self.MyCart.save_history_csv(csv_name=csv_name)
            self.saved = 1

    # We define a separate threat for the controller:
    def thread_control_input(self):
        while self.run_thread_calculations:
            self.MyCart.Update_Q()

    # A thread redrawing labels (except for timer, which has its own function) of GUI every 0.1 s
    def thread_labels(self):
        while (self.run_thread_labels):
            self.labSpeed.setText("Speed (m/s): " + str(around(self.MyCart.s.positionD, 2)))
            self.labAngle.setText("Angle (deg): " + str(around(self.MyCart.s.angle * 360 / (2 * pi), 2)))
            self.labMotor.setText("Motor power (Q): {:.3f}".format(around(self.MyCart.Q, 2)))
            if self.MyCart.mode == 0:
                self.labTargetPosition.setText("")
            else:
                self.labTargetPosition.setText("Target position (m): " + str(around(self.MyCart.slider_value, 2)))

            if self.MyCart.mode == 0:
                self.labSliderInstant.setText("Slider instant value (-): " + str(around(self.slider_value, 2)))
            else:
                self.labSliderInstant.setText("Slider instant value (m): " + str(around(self.slider_value, 2)))

            self.labTimeSim.setText('Simulation time (s): {:.2f}'.format(self.MyCart.time))

            mean_dt_real = np.mean(self.looper.circ_buffer_dt_real)
            if mean_dt_real > 0:
                self.labSpeedUp.setText('Speed-up (measured): x{:.2f}'
                                        .format(self.dt_main_simulation / mean_dt_real))

            sleep(0.1)

    # Actions to be taken when start/stop button is clicked
    def play(self):
        # print('Mode {}'.format(self.MyCart.mode))
        if self.load_recording:
            worker_replay = Worker(self.thread_replay)
            # Execute
            self.threadpool.start(worker_replay)

        else:
            if self.run_thread_calculations == 1:
                self.run_thread_calculations = 0
                # If user is saving data wait till data is saved
                if self.save_history:
                    while (self.saved == 0):
                        sleep(0.001)

                self.MyCart.use_pregenerated_target_position = False
                self.MyCart.summary_plots()
                self.w_summary = SummaryWindow(summary_plots=self.MyCart.summary_plots)
                # Reset variables and redraw the figures
                self.reset_variables()
                # Draw figures
                self.MyCart.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
                self.canvas.draw()
                self.cb_load_recorded_data.setEnabled(True)
                self.cb_save_history.setEnabled(True)

            elif self.run_thread_calculations == 0:
                self.cb_save_history.setEnabled(False)
                self.cb_load_recorded_data.setEnabled(False)
                speedup_updated = self.get_speedup()
                if speedup_updated:
                    self.looper.dt_target = self.dt_main_simulation / self.speedup
                    self.run_thread_calculations = 1
                    # Pass the function to execute
                    worker_calculations = Worker(self.thread_calculations)
                    # Execute
                    self.threadpool.start(worker_calculations)

                    if self.Q_thread_enabled:
                        self.MyCart.Q_thread_enabled = self.Q_thread_enabled
                        worker_control_input = Worker(self.thread_control_input)
                        self.threadpool.start(worker_control_input)
                    else:
                        self.MyCart.Q_thread_enabled = self.Q_thread_enabled

    def thread_replay(self):

        # Check what is in the csv textbox
        csv_name = self.textbox.text()
        history_pd = self.MyCart.load_history_csv(csv_name=csv_name)
        # Shift dt by one up
        history_pd['dt'] = history_pd['dt'].shift(-1)
        history_pd = history_pd[:-1]

        # Check speedup which user provided with GUI
        self.get_speedup()

        # Define loop timer for now with arbitrary dt
        replay_looper = loop_timer(dt_target=0.0)

        # Start looper
        replay_looper.start_loop()
        for index, row in history_pd.iterrows():
            self.MyCart.s.position = row['s.position']
            self.MyCart.s.positionD = row['s.positionD']
            self.MyCart.s.angle = row['s.angle']
            self.MyCart.time = row['time']
            self.MyCart.dt = row['dt']
            self.MyCart.u = row['u']
            self.MyCart.Q = row['Q']
            self.MyCart.target_position = row['target_position']
            self.slider_value = self.MyCart.target_position

            dt_target = (self.MyCart.dt / self.speedup)
            replay_looper.dt_target = dt_target

            replay_looper.sleep_leftover_time()

        self.reset_variables()

    # The acctions which has to be taken to properly terminate the application
    # The method is evoked after QUIT button is pressed
    # TODO: Can we connect it somehow also the the default cross closing the application?
    def quit_application(self):
        # Stops animation (updating changing elements of the Figure)
        self.anim._stop()
        # Stops the two threads updating the GUI labels and updating the state of Cart instance
        self.run_thread_labels = False
        self.run_thread_calculations = False
        # Closes the GUI window
        self.close()
        # The standard command
        # It seems however not to be working by its own
        # I don't know how it works
        QApplication.quit()

    def generate_trace(self):
        if self.load_recording:
            x = self.load_generate_conflict_msg.exec_()
            return
        self.cb_load_recorded_data.setEnabled(False)
        self.cb_save_history.setEnabled(False)
        if self.textbox_length.text() == '':
            self.MyCart.random_length = random_length_globals
        else:
            self.MyCart.random_length = float(self.textbox_length.text())

        turning_points_list = []
        if self.textbox_turning_points.text() != '':
            for turning_point in self.textbox_turning_points.text().split(', '):
                turning_points_list.append(float(turning_point))
        self.MyCart.turning_points = turning_points_list

        self.MyCart.Generate_Random_Trace_Function()
        if self.run_thread_calculations == 1:
            print('First reset the previous run')
        else:
            self.MyCart.use_pregenerated_target_position = True
            self.run_thread_calculations = 1
            # Pass the function to execute
            worker_calculations = Worker(self.thread_calculations)
            # Execute
            self.threadpool.start(worker_calculations)
            while (self.run_thread_calculations == 1):
                QApplication.processEvents()
            # If user is saving data wait till data is saved
            if self.save_history:
                while (self.saved == 0):
                    QApplication.processEvents()
                    sleep(0.001)

            self.MyCart.use_pregenerated_target_position = False
            self.MyCart.summary_plots()
            self.w_summary = SummaryWindow(summary_plots=self.MyCart.summary_plots)
            # Reset variables and redraw the figures
            self.reset_variables()
            # Draw figures
            self.MyCart.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
            self.canvas.draw()
            self.cb_load_recorded_data.setEnabled(True)
            self.cb_save_history.setEnabled(True)

    # Function to measure the time of simulation as experienced by user
    # It corresponds to the time of simulation according to equations only if real time mode is on
    def recurring_timer(self):
        # "If": Increment time counter only if simulation is running
        if self.run_thread_calculations == 1:
            self.counter += 1
            # The updates are done smoother if the label is updated here
            # and not in the separate thread
            self.labTime.setText("Time (s): " + str(float(self.counter) / 10.0))

    # Action to be taken while a radio button is clicked
    # Toggle a mode of simulation: manual or LQR
    def RadioButtons(self):

        # Change the mode variable depending on the Radiobutton state
        for i in range(len(self.rbs)):
            if self.rbs[i].isChecked():
                self.MyCart.set_mode(new_mode=i)

        # Reset the state of GUI and of the Cart instance after the mode has changed
        self.reset_variables()
        self.MyCart.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()

    # Action toggling between saving and not saving simulation results
    def cb_save_history_f(self, state):

        if state:
            self.save_history = 1
        else:
            self.save_history = 0

    # Action toggling between stopping (or not) the pole if it reaches 90 deg
    def cb_stop_at_90_deg_f(self, state):

        if state:
            self.stop_at_90 = True
        else:
            self.stop_at_90 = False

        self.MyCart.stop_at_90 = self.stop_at_90


    def cb_slider_on_click_f(self, state):

        if state:
            self.slider_on_click = True
        else:
            self.slider_on_click = False


    # Action toggling between loading (and/for replaying) recorded data and performing new experiment
    def cb_load_recorded_data_f(self, state):

        if state:
            self.load_recording = 1
        else:
            self.load_recording = 0

    # A function redrawing the changing elements of the Figure
    # This animation runs always when the GUI is open
    # The buttons of GUI only decide if new parameters are calculated or not
    def run_animation(self):

        def init():
            # Adding variable elements to the Figure
            self.fig.AxCart.add_patch(self.MyCart.Mast)
            self.fig.AxCart.add_patch(self.MyCart.Chassis)
            self.fig.AxCart.add_patch(self.MyCart.WheelLeft)
            self.fig.AxCart.add_patch(self.MyCart.WheelRight)
            self.fig.AxSlider.add_patch(self.MyCart.Slider)
            return self.MyCart.Mast, self.MyCart.Chassis, self.MyCart.WheelLeft, self.MyCart.WheelRight, self.MyCart.Slider

        def animationManage(i, MyCart):
            # Updating variable elements
            self.MyCart.update_drawing()
            # Special care has to be taken of the mast rotation
            self.MyCart.t2 = self.MyCart.t2 + self.fig.AxCart.transData
            self.MyCart.Mast.set_transform(self.MyCart.t2)
            return self.MyCart.Mast, self.MyCart.Chassis, self.MyCart.WheelLeft, self.MyCart.WheelRight, self.MyCart.Slider

        # Initialize animation object
        self.anim = animation.FuncAnimation(self.fig, animationManage,
                                            init_func=init,
                                            frames=300,
                                            fargs=(self.MyCart,),
                                            interval=10,
                                            blit=True,
                                            repeat=True)
