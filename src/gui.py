# Import functions from PyQt5 module (creating GUI)
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QApplication, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox, \
    QLineEdit, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, QTimer, Qt
# Import functions to measure time intervals and to pause a thread for a given time
from time import sleep
import timeit
# The main drawing functionalities are implemented in Cart Class in another file
# Some more functions needed for interaction of matplotlib with PyQt5 and
# for running the animation are imported here
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
class SummaryWindow(QMainWindow):
    def __init__(self):
        super(SummaryWindow, self).__init__()
        lbl = QLabel('Summary', self)


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
        self.load_recording = load_recording_globals
        self.saved = 0
        self.printing_summary = 1
        self.Q_thread_enabled = False

        self.dt_main_simulation = dt_main_simulation_globals
        self.speedup = speedup_globals
        self.looper = loop_timer(dt_target=(self.dt_main_simulation / self.speedup))

        # Create Cart object

        self.MyCart = Cart()

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
        self.rb_manual = QRadioButton('Manual Stabilization')
        self.rb_LQR = QRadioButton('LQR-control with adjustable target position')
        self.rb_do_mpc = QRadioButton('do-mpc-control with adjustable target position')
        self.rb_do_mpc_discrete = QRadioButton('do-mpc-discrete-control with adjustable target position')
        self.rbs = [self.rb_manual, self.rb_LQR, self.rb_do_mpc, self.rb_do_mpc_discrete]

        lr.addStretch(1)
        for rb in self.rbs:
            rb.toggled.connect(self.RadioButtons)
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

        # This line links function cupturing the mouse position to the canvas of the Figure
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_movement)

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

    # Function evoked at a mouese movement
    # If the mouse coursor is over the lower chart it reads the coresponding value
    # and updates the slider
    def on_mouse_movement(self, event):
        if event.xdata == None or event.ydata == None:
            pass
        else:
            if event.inaxes == self.fig.AxSlider:
                self.MyCart.update_slider(mouse_position=event.xdata)

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

    # Method printing the parameters of the CartPole over time during the experiment
    def show_summary(self):

        fig, axs = plt.subplots(4, 1, figsize=(18, 14), sharex=True)  # share x axis so zoom zooms all plots

        # Plot angle error
        axs[0].set_ylabel("Angle (deg)", fontsize=18)
        axs[0].plot(array(self.MyCart.dict_history['time']), array(self.MyCart.dict_history['s.angle']) * 180.0 / pi,
                    'b', markersize=12, label='Ground Truth')
        axs[0].tick_params(axis='both', which='major', labelsize=16)

        # Plot position
        axs[1].set_ylabel("position (m)", fontsize=18)
        axs[1].plot(self.MyCart.dict_history['time'], self.MyCart.dict_history['s.position'], 'g', markersize=12,
                    label='Ground Truth')
        axs[1].tick_params(axis='both', which='major', labelsize=16)

        # Plot motor input command
        axs[2].set_ylabel("motor (N)", fontsize=18)
        axs[2].plot(self.MyCart.dict_history['time'], self.MyCart.dict_history['u'], 'r', markersize=12,
                    label='motor')
        axs[2].tick_params(axis='both', which='major', labelsize=16)

        # Plot target position
        axs[3].set_ylabel("position target", fontsize=18)
        axs[3].plot(self.MyCart.dict_history['time'], self.MyCart.dict_history['target_position'], 'k')
        axs[3].tick_params(axis='both', which='major', labelsize=16)

        axs[3].set_xlabel('Time (s)', fontsize=18)

        plt.show()

        print('Max state:')
        print('[x,v,theta, omega]')
        max_state = (max(abs(array(self.MyCart.dict_history['s.position']))),
                     max(abs(array(self.MyCart.dict_history['s.positionD']))),
                     (180 / pi) * max(abs(array(self.MyCart.dict_history['s.angle']))),
                     (180 / pi) * max(abs(array(self.MyCart.dict_history['s.angleD']))))
        print(max_state)

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

            self.looper.sleep_leftover_time()

        # print('Welcome')
        # Save simulation history if user chose to do so at the end of the simulation
        if self.save_history:
            csv_name = self.textbox.text()
            self.MyCart.augment_dict_history()
            self.MyCart.save_history_csv(csv_name=csv_name)
            self.saved = 1

        # plot_summary = True
        # if self.save_history and plot_summary:

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

            self.labTimeSim.setText('Simulation time (s): {:.2f}'.format(self.MyCart.time))

            mean_dt_real = np.mean(self.looper.circ_buffer_dt_real)
            if mean_dt_real > 0:
                self.labSpeedUp.setText('Speed-up (measured): x{:.2f}'
                                        .format(self.dt_main_simulation / mean_dt_real))

            sleep(0.1)

    # Actions to be taken when start/stop button is clicked
    def play(self):
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
                self.show_summary()
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
            self.MyCart.s.position = self.MyCart.s.position + 0.1
            self.MyCart.s.positionD = row['s.positionD']
            self.MyCart.s.angle = row['s.angle']
            self.MyCart.time = row['time']
            self.MyCart.dt = row['dt']/1000.0
            self.MyCart.u = row['u']
            self.MyCart.target_position = row['target_position']

            dt_target = (self.MyCart.dt / self.speedup)
            replay_looper.dt_target = dt_target
            # print(replay_looper.dt_target)

            replay_looper.sleep_leftover_time()

        self.reset_variables()
        # Load either last one if empty or the one with given name
        # You need just a position of the cart, angle and target position (slider)
        # Use looper to wait?
        # Use speed-up also in this mode
        # ...

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
            self.show_summary()
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
        if self.rb_manual.isChecked():
            self.MyCart.set_mode(new_mode=0)
        elif self.rb_LQR.isChecked():
            self.MyCart.set_mode(new_mode=1)
        elif self.rb_do_mpc.isChecked():
            self.MyCart.set_mode(new_mode=2)

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

        print('save history is now {}'.format(self.save_history))

    # Action toggling between loading (and/for replaying) recorded data and performing new experiment
    def cb_load_recorded_data_f(self, state):

        if state:
            self.load_recording = 1
        else:
            self.load_recording = 0

        print('load recording is now {}'.format(self.load_recording))

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
