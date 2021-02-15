# Import functions from PyQt5 module (creating GUI)
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QApplication, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox, \
    QLineEdit, QMessageBox, QComboBox
from PyQt5.QtCore import QThreadPool, QTimer
# Import functions to measure time intervals and to pause a thread for a given time
from time import sleep
# The main drawing functionalities are implemented in CartPole Class in another file
# Some more functions needed for interaction of matplotlib with PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# Import Cart class - the class keeping all the parameters and methods
# related to CartPole which are not related to PyQt5 GUI
from CartPole import CartPole

from CartPoleGUI.gui_default_params import *
from CartPoleGUI.loop_timer import loop_timer
from CartPoleGUI._CartPoleGUI_summary_window import SummaryWindow
from CartPoleGUI._CartPoleGUI_worker_template import Worker


# Class implementing the main window of CartPole GUI
class MainWindow(QMainWindow):

    from CartPoleGUI._CartPoleGUI_MainWindow_get_set_reset_quit import set_experiment_generator_init_params,\
        set_labels_thread, set_user_time_label, reset_variables, quit_application
    from CartPoleGUI._CartPoleGUI_MainWindow_mouse_interactive import on_mouse_click, on_mouse_movement
    from CartPoleGUI._CartPoleGUI_MainWindow_changing_static_options import RadioButtons
    from CartPoleGUI._CartPoleGUI_MainWindow_changing_static_options import get_speedup
    from CartPoleGUI._CartPoleGUI_MainWindow_changing_static_options import cb_interpolation_selectionchange, cb_save_history_f,\
        cb_show_experiment_summary_f, cb_stop_at_90_deg_f, cb_slider_on_click_f, cb_load_recorded_data_f
    from CartPoleGUI._CartPoleGUI_MainWindow_generate_random_experiment import generate_random_experiment
    from CartPoleGUI._CartPoleGUI_MainWindow_play_and_replay import start_stop_button, thread_replay, play
    from CartPoleGUI._CartPoleGUI_MainWindow_experiment_thread import experiment_thread

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Variables defining mode of operation of CartPole GUI
        # Modify starting values through gui_default_params.py
        self.stop_at_90 = stop_at_90_init
        self.load_recording = load_recording_init
        self.slider_on_click = slider_on_click_init
        self.save_history = save_history_init
        self.show_experiment_summary = show_experiment_summary_init
        if self.save_history or self.show_experiment_summary:
            self.save_data_in_cart =True
        else:
            self.save_data_in_cart = False

        # Variables controlling the state of various processes (DO NOT MODIFY)
        # Stop threads if False
        self.run_experiment_thread = False
        self.run_set_labels_thread = True
        self.run_replay = False

        self.counter = 0
        self.saved = 0
        self.printing_summary = 1
        self.reset_mode = 1 # Set 0 to start with all states 0, 1 to make

        self.dt_main_simulation = dt_main_simulation
        self.speedup = speedup_init
        self.looper = loop_timer(dt_target=(self.dt_main_simulation / self.speedup))

        # Create Cart object and load initial settings
        self.CartPoleInstance = CartPole()

        # Set timescales
        self.CartPoleInstance.dt_simulation = dt_main_simulation
        self.CartPoleInstance.dt_controller = controller_update_interval
        self.CartPoleInstance.dt_save = save_interval

        # set other settings
        self.CartPoleInstance.set_controller(controller_init)
        self.CartPoleInstance.save_data_in_cart = self.save_data_in_cart
        self.CartPoleInstance.stop_at_90 = stop_at_90_init
        self.set_experiment_generator_init_params()
        self.slider_value = self.CartPoleInstance.slider_value

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

        self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)

        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.canvas)

        # Radiobuttons to toggle the mode of operation
        lr = QVBoxLayout()
        self.rbs = []
        for controller_name in self.CartPoleInstance.controller_names:
            self.rbs.append(QRadioButton(controller_name))

        lr.addStretch(1)
        for rb in self.rbs:
            rb.clicked.connect(self.RadioButtons)
            lr.addWidget(rb)
        lr.addStretch(1)
        # self.rb_manual.setChecked(True)
        self.rbs[self.CartPoleInstance.controller_idx].setChecked(True)

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
        self.timer.timeout.connect(self.set_user_time_label)
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
        bss.pressed.connect(self.start_stop_button)
        bq = QPushButton("QUIT")
        bq.pressed.connect(self.quit_application)
        bt = QPushButton("GENERATE TRACE")
        bt.pressed.connect(self.generate_random_experiment)
        lb = QVBoxLayout()  # Layout for buttons
        lb.addWidget(bss)
        lb.addWidget(bq)
        lb.addWidget(bt)
        layout.addLayout(lb)

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

        # Textbox to add provide a file name
        l_text = QHBoxLayout()
        textbox_title = QLabel('CSV file name:')
        self.textbox = QLineEdit()
        l_text.addWidget(textbox_title)
        l_text.addWidget(self.textbox)
        layout.addLayout(l_text)

        # Checkboxs:
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

        self.cb_show_experiment_summary = QCheckBox('Show experiment summary', self)
        if self.show_experiment_summary:
            self.cb_show_experiment_summary.toggle()
        self.cb_show_experiment_summary.toggled.connect(self.cb_show_experiment_summary_f)
        l_cb.addWidget(self.cb_show_experiment_summary)

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
        self.setWindowTitle('CartPole Simulator')

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
        worker_labels = Worker(self.set_labels_thread)
        self.threadpool.start(worker_labels)

        self.reset_variables(0)

        # This animation runs ALWAYS when the GUI is open
        # redrawing the changing elements of the Figure
        # The buttons of GUI only decide if new parameters are calculated or not
        self.anim = self.CartPoleInstance.run_animation(self.fig)




