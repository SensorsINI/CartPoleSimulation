# Import functions from PyQt5 module (creating GUI)
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QApplication, QVBoxLayout,\
                                        QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, QTimer, Qt
from PyQt5.QtGui import QIcon
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
import matplotlib
# Import function from numpy library
from numpy import pi, around, array

# Import Cart class - the class keeping all the parameters and methods
# related to CartPole which are not related to PyQt5 GUI
from CartClass import Cart


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

        ## Initialization of variables
        # Time in simulation
        self.dt_fix = 0.0001  # This is fixed timestep value
        self.dt_real = self.dt_fix
        self.dt = 0.0  # It is NOT the fixed timestep value. It is just a value for the first timestep

        # Stop threads if False
        self.run_thread_calculations = False
        self.run_thread_labels = True

        self.counter = 0
        self.real_time = 1
        self.save_history = True
        self.saved = 0
        self.printing_summary = 1
        self.Q_thread_enabled = False

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
        self.rb_manual.toggled.connect(self.RadioButtons)
        self.rb_LQR.toggled.connect(self.RadioButtons)
        self.rb_do_mpc.toggled.connect(self.RadioButtons)
        lr.addStretch(1)
        lr.addWidget(self.rb_manual)
        lr.addWidget(self.rb_LQR)
        lr.addWidget(self.rb_do_mpc)
        lr.addStretch(1)
        self.rb_manual.setChecked(True)

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
        self.labSpeedUp = QLabel('Speed-up:')
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

        # Checkboxs:
        # to swich between real time and constant dt simulation
        # TODO to decide if to save the simulation history
        # TODO to decide if to plot simulation history
        cb = QCheckBox('Real time simulation', self)
        cb.toggle()
        cb.stateChanged.connect(self.real_time_simulation_f)
        layout.addWidget(cb)

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

    # Method printing the parameters of the CartPole over time during the experiment
    def show_summary(self):

        fig, axs = plt.subplots(4, 1, figsize=(18, 14), sharex=True)  # share x axis so zoom zooms all plots

        # Plot angle error
        axs[0].set_ylabel("Angle (deg)", fontsize=18)
        axs[0].plot(array(self.MyCart.dict_history['time']), array(self.MyCart.dict_history['angleErr']) * 180.0 / pi,
                    'b', markersize=12, label='Ground Truth')
        axs[0].tick_params(axis='both', which='major', labelsize=16)

        # Plot position
        axs[1].set_ylabel("position (m)", fontsize=18)
        axs[1].plot(self.MyCart.dict_history['time'], self.MyCart.dict_history['position'], 'g', markersize=12,
                    label='Ground Truth')
        axs[1].tick_params(axis='both', which='major', labelsize=16)

        # Plot motor input command
        axs[2].set_ylabel("motor (N)", fontsize=18)
        axs[2].plot(self.MyCart.dict_history['time'], self.MyCart.dict_history['motor'], 'r', markersize=12,
                    label='motor')
        axs[2].tick_params(axis='both', which='major', labelsize=16)

        # Plot target position
        axs[3].set_ylabel("position target", fontsize=18)
        axs[3].plot(self.MyCart.dict_history['time'], self.MyCart.dict_history['PositionTarget'], 'k')
        axs[3].tick_params(axis='both', which='major', labelsize=16)

        axs[3].set_xlabel('Time (s)', fontsize=18)

        print('Max state:')
        print('[x,v,theta, omega]')
        max_state = (max(abs(array(self.MyCart.dict_history['position']))),
                     max(abs(array(self.MyCart.dict_history['positionD']))),
                     (180 / pi) * max(abs(array(self.MyCart.dict_history['angleErr']))),
                     (180 / pi) * max(abs(array(self.MyCart.dict_history['angleD']))))
        print(max_state)

    # This method initiate calculation of simulation and iterative updates of Cart state
    # It also measures time intervals for real time simulation
    # implements a termination condition if Pole went out of control
    # and initiate saving to a .csv file
    def thread_calculations(self):

        start = timeit.default_timer()

        while (self.run_thread_calculations):

            # Measuring real-time timestep
            stop = timeit.default_timer()
            self.dt_real = stop - start

            if self.real_time == 1:
                self.dt = self.dt_real
            else:
                self.dt = self.dt_fix

            start = timeit.default_timer()

            # Calculations of the Cart state in the next timestep
            self.MyCart.update_state(dt=self.dt)

            # Finish simulation if angle bigger than 90 deg.
            # if abs(self.MyCart.angle)>pi/2:
            #     self.run_thread_calculations = 0
            #     break

            # Ensure that the animation drawing function can access MyCart at this moment
            QApplication.processEvents()

            if self.MyCart.play_pregenerated == True and self.MyCart.time_total >= self.MyCart.t_max_pre:
                # print('Terminating!')
                self.run_thread_calculations = 0

        # print('Welcome')
        # Save simulation history if user chose to do so at the end of the simulation
        if self.save_history:
            self.MyCart.save_history_csv()
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
            self.labSpeed.setText("Speed (m/s): " + str(around(self.MyCart.s.CartPositionD, 2)))
            self.labAngle.setText("Angle (deg): " + str(around(self.MyCart.s.angle * 360 / (2 * pi), 2)))
            self.labMotor.setText("Motor power (Q): {}".format(around(self.MyCart.Q, 2)))
            if self.MyCart.mode == 0:
                self.labTargetPosition.setText("")
            elif self.MyCart.mode == 1:
                self.labTargetPosition.setText("Target position (m): " + str(around(self.MyCart.slider_value, 2)))

            self.labTimeSim.setText('Simulation time (s): {:.2f}'.format(self.MyCart.time_total))

            if self.real_time:
                speed_up = 1.0
            else:
                speed_up = 0.95*speed_up + 0.05*(self.dt_fix/self.dt_real)
            self.labSpeedUp.setText('Speed-up (average): x{:.2f}'.format(speed_up))
            sleep(0.1)

    # Actions to be taken when start/stop button is clicked
    def play(self):
        if self.run_thread_calculations == 1:
            self.run_thread_calculations = 0
            # If user is saving data wait till data is saved
            if self.save_history:
                while (self.saved == 0):
                    sleep(0.001)

            self.MyCart.play_pregenerated = False
            self.show_summary()
            # Reset variables and redraw the figures
            self.reset_variables()
            # Draw figures
            self.MyCart.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
            self.canvas.draw()

        elif self.run_thread_calculations == 0:
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
        self.MyCart.Generate_Random_Trace_Function()
        if self.run_thread_calculations == 1:
            print('First reset the previous run')
        else:
            self.MyCart.play_pregenerated = True
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

            self.MyCart.play_pregenerated = False
            self.show_summary()
            # Reset variables and redraw the figures
            self.reset_variables()
            # Draw figures
            self.MyCart.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
            self.canvas.draw()

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
            self.MyCart.mode = 0
        elif self.rb_LQR.isChecked():
            self.MyCart.mode = 1
            self.MyCart.controller = self.MyCart.controller_lqr
        elif self.rb_do_mpc.isChecked():
            self.MyCart.mode = 2
            self.MyCart.controller = self.MyCart.controller_do_mpc

        # Reset the state of GUI and of the Cart instance after the mode has changed
        self.reset_variables()
        self.MyCart.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()

    # Action toggling between real time and fix time step mode
    def real_time_simulation_f(self, state):
        if state == Qt.Checked:
            self.real_time = 1
        else:
            self.real_time = 0

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
