# Necessary only for debugging in Visual Studio Code IDE
try:
    import ptvsd
except:
    pass

import os
import csv
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from CartPole import CartPole
from CartPole.cartpole_drawer import CartPoleDrawer
from CartPole.cartpole_target_slider import TargetSlider
from CartPole.cartpole_parameters import TrackHalfLength
from CartPole.state_utilities import ANGLED_IDX, ANGLE_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state

from GUI.loop_timer import loop_timer
from GUI._CartPoleGUI_worker_template import Worker
from GUI._CartPoleGUI_summary_window import SummaryWindow
from GUI._ControllerGUI_NoiseOptionsWindow import NoiseOptionsWindow

from others.globals_and_utils import load_config


class CartPole_GuiActions:

    def __init__(self, gui_layout, *args, **kwargs):

        self.gui = gui_layout

        # Import variables from config
        config = load_config('config_gui.yml')

        dt_simulation = config['time_scales']['dt_simulation']
        controller_update_interval = config['time_scales']['controller_update_interval']
        save_interval = config['time_scales']['save_interval']

        controller_init_cpp = config['gui_settings']['controller_init_cpp']
        controller_init_cps = config['gui_settings']['controller_init_cps']

        if os.getcwd().split(os.sep)[-1] == 'Driver':
            controller_init = controller_init_cpp  # Load as default if loaded as part of physical-cartpole
        else:
            controller_init = controller_init_cps  # Load as default if loaded as cartpole simulator stand alone

        save_history_init = config['gui_settings']['save_history_init']
        show_experiment_summary_init = config['gui_settings']['show_experiment_summary_init']
        stop_at_90_init = config['gui_settings']['stop_at_90_init']
        slider_on_click_init = config['gui_settings']['slider_on_click_init']
        simulator_mode_init = config['gui_settings']['simulator_mode_init']
        speedup_init = config['gui_settings']['speedup_init']
        show_hanging_pole_init = config['gui_settings']['show_hanging_pole_init']

        self.track_relative_complexity_init = config['random_trace_generation']['track_relative_complexity_init']
        self.length_of_experiment_init = config['random_trace_generation']['length_of_experiment_init']
        self.interpolation_type_init = config['random_trace_generation']['interpolation_type_init']
        self.turning_points_period_init = config['random_trace_generation']['turning_points_period_init']
        self.start_random_target_position_at_init = config['random_trace_generation']['start_random_target_position_at_init']
        self.end_random_target_position_at_init = config['random_trace_generation']['end_random_target_position_at_init']
        self.turning_points_init = config['random_trace_generation'].get('turning_points_init',
                                                                    None)  # Using .get for optional keys

        self.target_slider = TargetSlider()
        # region Create CartPole instance and load initial settings

        # Create CartPole instance
        self.initial_state = create_cartpole_state()
        self.CartPoleInstance = CartPole(initial_state=self.initial_state, target_slider=self.target_slider)

        # Set timescales
        self.CartPoleInstance.dt_simulation = dt_simulation
        self.CartPoleInstance.dt_controller = controller_update_interval
        self.CartPoleInstance.dt_save = save_interval

        # set other settings
        self.CartPoleInstance.set_controller(controller_init)
        self.CartPoleInstance.stop_at_90 = stop_at_90_init
        self.set_random_experiment_generator_init_params()

        self.PhysicalCartPoleDriverInstance = None

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
        self.CartPoleInstance.show_hanging_pole = show_hanging_pole_init

        # endregion

        self.cp_drawer = CartPoleDrawer(self.CartPoleInstance, self.target_slider)

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
        self.slider_instant_value = self.target_slider.value

        self.noise = 'OFF'
        self.CartPoleInstance.NoiseAdderInstance.noise_mode = self.noise

        self.zero_angle_shift_handler = ZeroAngleShiftHandler(self.CartPoleInstance)

        # region - Matplotlib figures (CartPole drawing and Slider)
        # Draw Figure
        self.fig, self.axes = plt.subplots(2, 1, figsize=(25, 10))  # Regulates the size of Figure in inches, before scaling to window size.
        self.fig.AxCart = self.axes[0]
        self.fig.AxSlider = self.axes[1]
        self.fig.AxSlider.set_ylim(0, 1)

        self.cp_drawer.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)

        self.threadpool = None
        self.canvas = None
        self.anim = None

        # endregion

    def finish_initialization(self):

        self.threadpool = self.gui.threadpool
        self.canvas = self.gui.canvas

        # region Open controller-specific popup windows
        self.open_additional_controller_widget()
        # endregion

        # region Activate functions capturing mouse movements and clicks over the slider

        # This line links function capturing the mouse position on the canvas of the Figure
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_movement)
        # This line links function capturing the mouse position on the canvas of the Figure click
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)

        # endregion

        # endregion

        # region Starts a thread repeatedly redrawing gauges (labels) of the GUI
        # It runs till the QUIT button is pressed
        worker_labels = Worker(self.set_labels_thread)
        self.threadpool.start(worker_labels)
        # endregion

        # region Start animation repeatedly redrawing changing elements of matplotlib figures (CartPole drawing and slider)
        # This animation runs ALWAYS when the GUI is open
        # The buttons of GUI only decide if new parameters are calculated or not
        self.anim = self.cp_drawer.run_animation(self.fig)
        # endregion

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
                        self.CartPoleInstance.use_pregenerated_target_position
                        and
                        (self.CartPoleInstance.time >= self.CartPoleInstance.t_max_pre)
                ):
                    self.terminate_experiment_or_replay_thread = True

                # FIXME: when Speedup empty in GUI I expected inf speedup but got error Loop timer was not initialized properly
                self.looper.sleep_leftover_time()

        # Save simulation history if user chose to do so at the end of the simulation
        if self.save_history:
            csv_name = self.gui.get_csv_name_from_gui()
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
        csv_name = self.gui.get_csv_name_from_gui()

        # Load experiment history
        history_pd, filepath = self.CartPoleInstance.load_history_csv(csv_name=csv_name)

        # Set cartpole in the right mode (just to ensure slider behaves properly)
        with open(filepath, newline='') as f:
            reader = csv.reader(f)
            controller_set = None
            for line in reader:
                line = line[0]
                if line[:len('# Controller: ')] == '# Controller: ':
                    controller_set = self.CartPoleInstance.set_controller(line[len('# Controller: '):].rstrip("\n"))
                    continue
                elif not controller_set:
                    continue
                else:
                    pass

                if line[:len('# Optimizer MPC: ')] == '# Optimizer MPC: ':
                    optimizer_set = self.CartPoleInstance.set_optimizer(line[len('# Optimizer MPC: '):].rstrip("\n"))
                elif controller_set:
                    optimizer_set = ''
                else:
                    continue

                if controller_set:
                    self.gui.rbs_controllers[self.CartPoleInstance.controller_idx].setChecked(True)
                else:
                    self.gui.rbs_controllers[1].setChecked(True)  # Set first, but not manual stabilization
                if optimizer_set:
                    self.gui.rbs_optimizers[self.CartPoleInstance.optimizer_idx].setChecked(True)
                else:
                    self.gui.rbs_optimizers[1].setChecked(True)
                self.update_rbs_optimizers_status(visible=self.CartPoleInstance.controller.has_optimizer)
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
        global L
        for index, row in history_pd.iterrows():
            self.CartPoleInstance.s[POSITION_IDX] = row['position']
            self.CartPoleInstance.s[POSITIOND_IDX] = row['positionD']
            self.CartPoleInstance.s[ANGLE_IDX] = row['angle']
            self.CartPoleInstance.time = row['time']
            self.CartPoleInstance.dt = row['dt']
            try:
                self.CartPoleInstance.u = row['u']
            except KeyError:
                pass
            self.CartPoleInstance.Q = row['Q_applied']
            self.CartPoleInstance.target_position = row['target_position']
            if self.CartPoleInstance.controller_name == 'manual-stabilization':
                self.target_slider.value = self.CartPoleInstance.Q
            else:
                self.target_slider.value = self.CartPoleInstance.target_position / TrackHalfLength

            # TODO: Make it more general for all possible parameters
            try:
                L[...] = row['L']
            except KeyError:
                pass
            except:
                print('Error while assigning L')
                print("Unexpected error:", sys.exc_info()[0])
                print("Unexpected error:", sys.exc_info()[1])

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

    def physical_experiment_thread(self):

        self.gui.bp.setText("Dance!")

        # Necessary only for debugging in Visual Studio Code IDE
        try:
            ptvsd.debug_this_thread()
        except:
            pass

        while not self.PhysicalCartPoleDriverInstance.terminate_experiment and not self.terminate_experiment_or_replay_thread:

            self.PhysicalCartPoleDriverInstance.experiment_sequence()

            if self.CartPoleInstance.controller_name == 'manual-stabilization':
                self.target_slider.value = self.CartPoleInstance.Q
            else:
                self.target_slider.value = self.CartPoleInstance.target_position / TrackHalfLength

        self.PhysicalCartPoleDriverInstance.terminate_experiment = True
        self.terminate_experiment_or_replay_thread = True

        self.PhysicalCartPoleDriverInstance.quit_experiment()

        self.PhysicalCartPoleDriverInstance = None

        self.experiment_or_replay_thread_terminated = True

    # endregion

    # region "START! / STOP!" button -> run/stop slider-controlled experiment, random experiment or replay experiment recording

    # Actions to be taken when "START! / STOP!" button is clicked
    def start_stop_button(self):

        # If "START! / STOP!" button in "START!" mode...
        if self.start_or_stop_action == 'START!':
            self.gui.bss.setText("STOP!")
            if self.simulator_mode == 'Physical CP':
                self.PhysicalCartPoleDriverInstance.switch_on_control()
            else:
                self.start_thread()
            self.start_or_stop_action = "STOP!"

        # If "START! / STOP!" button in "STOP!" mode...
        elif self.start_or_stop_action == 'STOP!':
            self.gui.bss.setText("START!")
            if self.simulator_mode == 'Physical CP':
                self.PhysicalCartPoleDriverInstance.switch_off_control()
            else:
                self.gui.bp.setText("PAUSE")
                # This flag is periodically checked by thread. It terminates if set True.
                self.terminate_experiment_or_replay_thread = True
                # The stop_thread function is called automatically by the thread when it terminates
                # It is implemented this way, because thread my terminate not only due "STOP!" button
                # (e.g. replay thread when whole experiment is replayed)
            self.start_or_stop_action = "START!"

    def pause_unpause_button(self):
        if self.simulator_mode == 'Physical CP':
            if self.pause_or_unpause_action == 'PAUSE' and self.start_or_stop_action == 'STOP!':
                self.PhysicalCartPoleDriverInstance.dancer.danceEnabled = True
                self.pause_or_unpause_action = 'UNPAUSE'
                self.gui.bp.setText("Stop dancing!")
            else:
                self.PhysicalCartPoleDriverInstance.dancer.danceEnabled = False
                self.pause_or_unpause_action = 'PAUSE'
                self.gui.bp.setText("Dance!")
        else:
            # Only Pause if experiment is running
            if self.pause_or_unpause_action == 'PAUSE' and self.start_or_stop_action == 'STOP!':
                self.pause_or_unpause_action = 'UNPAUSE'
                self.pause_experiment_or_replay_thread = True
                self.gui.bp.setText("UNPAUSE")
            elif self.pause_or_unpause_action == 'UNPAUSE' and self.start_or_stop_action == 'STOP!':
                self.pause_or_unpause_action = 'PAUSE'
                self.pause_experiment_or_replay_thread = False
                self.gui.bp.setText("PAUSE")

    # Run thread. works for all simulator modes.
    def start_thread(self):

        # Check if value provided in speed-up textbox makes sense
        # If not abort start
        speedup_updated = self.get_speedup()
        if not speedup_updated:
            return

        # Disable GUI elements for features which must not be changed in runtime
        # For other features changing in runtime may not cause errors, but will stay without effect for current run
        self.gui.cb_save_history.setEnabled(False)
        for rb in self.gui.rbs_simulator_mode:
            rb.setEnabled(False)
        for rb in self.gui.rbs_controllers:
            rb.setEnabled(False)
        if self.simulator_mode != 'Replay':
            self.gui.cb_show_experiment_summary.setEnabled(False)

        # Set user-provided initial values for state (or its part) of the CartPole
        # Search implementation for more detail
        # The following line is important as it let the user to set with the slider the starting target position
        # After the slider was reset at the end of last experiment
        # With the small sliders he can also adjust starting initial_state
        self.reset_variables(2, s=np.copy(self.initial_state), target_position=self.CartPoleInstance.target_position)

        if self.simulator_mode == 'Random Experiment':

            self.CartPoleInstance.use_pregenerated_target_position = True

            if self.gui.textbox_length.text() == '':
                self.CartPoleInstance.length_of_experiment = self.length_of_experiment_init
            else:
                self.CartPoleInstance.length_of_experiment = float(self.gui.textbox_length.text())

            turning_points_list = []
            if self.gui.textbox_turning_points.text() != '':
                for turning_point in self.gui.textbox_turning_points.text().split(', '):
                    turning_points_list.append(float(turning_point))
            self.CartPoleInstance.turning_points = turning_points_list

            self.CartPoleInstance.setup_cartpole_random_experiment()

        self.looper.dt_target = self.CartPoleInstance.dt_simulation / self.speedup
        # Pass the function to execute
        if self.simulator_mode == "Replay":
            worker = Worker(self.replay_thread)
        elif self.simulator_mode == 'Slider-Controlled Experiment' or self.simulator_mode == 'Random Experiment':
            worker = Worker(self.experiment_thread)
        else:
            raise ValueError('Unknown simulator mode')
        worker.signals.finished.connect(self.finish_thread)
        # Execute
        self.threadpool.start(worker)

    # finish_threads works for all simulation modes
    # Some lines mya be redundant for replay,
    # however as they do not take much computation time we leave them here
    # As it my code shorter, while hopefully still clear.
    # It is called automatically at the end of experiment_thread
    def finish_thread(self):

        self.CartPoleInstance.use_pregenerated_target_position = False
        self.initial_state = create_cartpole_state()
        self.gui.initial_position_slider.setValue(0)
        self.gui.initial_angle_slider.setValue(0)
        self.CartPoleInstance.s = self.initial_state

        # Some controllers may collect they own statistics about their usage and print it after experiment terminated
        if self.simulator_mode != 'Replay':
            try:
                self.CartPoleInstance.controller.controller_report()
            except:
                pass

        if self.simulator_mode == 'Physical CP':
            self.PhysicalCartPoleDriverInstance.quit_experiment()
            self.PhysicalCartPoleDriverInstance = None

        if self.show_experiment_summary:
            self.w_summary = SummaryWindow(dict_history=self.CartPoleInstance.dict_history)

        # Reset variables and redraw the figures
        self.reset_variables(0)

        # Draw figures
        self.cp_drawer.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()

        # Enable back all elements of GUI:
        self.gui.cb_save_history.setEnabled(True)
        self.gui.cb_show_experiment_summary.setEnabled(True)
        for rb in self.gui.rbs_simulator_mode:
            rb.setEnabled(True)
        for rb in self.gui.rbs_controllers:
            rb.setEnabled(True)

        self.start_or_stop_action = "START!"  # What should happen when "START! / STOP!" is pushed NEXT time

    # endregion

    # region Methods: "Get, set, reset, quit"

    # Set parameters from gui_default_parameters related to generating a random experiment target position
    def set_random_experiment_generator_init_params(self):
        self.CartPoleInstance.track_relative_complexity = self.track_relative_complexity_init
        self.CartPoleInstance.length_of_experiment = self.length_of_experiment_init
        self.CartPoleInstance.interpolation_type = self.interpolation_type_init
        self.CartPoleInstance.turning_points_period = self.turning_points_period_init
        self.CartPoleInstance.start_random_target_position_at = self.start_random_target_position_at_init
        self.CartPoleInstance.end_random_target_position_at = self.end_random_target_position_at_init
        self.CartPoleInstance.turning_points = self.turning_points_init

    # Method resetting variables which change during experimental run
    def reset_variables(self, reset_mode=1, s=None, target_position=None):
        self.CartPoleInstance.set_cartpole_state_at_t0(reset_mode, s=s, target_position=target_position)
        self.user_time_counter = 0
        # "Try" because this function is called for the first time during initialisation of the Window
        # when the timer label instance is not yer there.
        try:
            self.gui.labt.setText("Time (s): " + str(float(self.user_time_counter) / 10.0))
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
        while self.run_set_labels_thread:
            self.gui.labSpeed.setText("Speed (m/s): " + str(np.around(self.CartPoleInstance.s[POSITIOND_IDX], 2)))
            self.gui.labAngle.setText(
                "Angle (deg): " + str(np.around(self.CartPoleInstance.s[ANGLE_IDX] * 360 / (2 * np.pi), 2)))
            self.gui.labMotor.setText("Motor power (Q): {:.3f}".format(np.around(self.CartPoleInstance.Q, 2)))
            if self.CartPoleInstance.controller_name == 'manual-stabilization':
                self.gui.labTargetPosition.setText("")
            else:
                self.gui.labTargetPosition.setText(
                    "Target position (m): " + str(np.around(self.CartPoleInstance.target_position, 2)))

            if self.CartPoleInstance.controller_name == 'manual_stabilization':
                self.gui.labSliderInstant.setText(
                    "Slider instant value (-): " + str(np.around(self.slider_instant_value, 2)))
            else:
                self.gui.labSliderInstant.setText(
                    "Slider instant value (m): " + str(np.around(self.slider_instant_value, 2)))

            self.gui.labTimeSim.setText('Simulation time (s): {:.2f}'.format(self.CartPoleInstance.time))

            mean_dt_real = np.mean(self.looper.circ_buffer_dt_real)
            if mean_dt_real > 0:
                self.gui.labSpeedUp.setText('Speed-up (measured): x{:.2f}'
                                        .format(self.CartPoleInstance.dt_simulation / mean_dt_real))
            time.sleep(0.1)

    # Function to measure the time of simulation as experienced by user
    # It corresponds to the time of simulation according to equations only if real time mode is on
    # TODO (Marcin) I just retained this function from some example being my starting point
    #   It seems it sometimes counting time to slow. Consider replacing in future
    def set_user_time_label(self):
        # "If": Increment time counter only if simulation is running
        if self.start_or_stop_action == "STOP!":  # indicates what start button was pressed and some process is running
            self.user_time_counter += 1
            # The updates are done smoother if the label is updated here
            # and not in the separate thread
            self.gui.labTime.setText("Time (s): " + str(float(self.user_time_counter) / 10.0))

    # The actions which has to be taken to properly terminate the application
    # The method is evoked after QUIT button is pressed
    # TODO: Can we connect it somehow also the the default cross closing the application?
    def quit_application(self):
        # Stops animation (updating changing elements of the Figure) if the event source exists
        if self.anim and self.anim.event_source:
            self.anim.event_source.remove_callback(self.anim._step)
            self.anim.event_source = None
        # Stops the two threads updating the GUI labels and updating the state of Cart instance
        self.run_set_labels_thread = False
        self.terminate_experiment_or_replay_thread = True
        self.pause_experiment_or_replay_thread = False
        # Closes the GUI window

        self.threadpool.clear()
        self.threadpool.waitForDone()

    def closeEvent(self, event):
        self.quit_application()

    # endregion

    # region Mouse interaction

    """
    These are some methods GUI uses to capture mouse effect while hoovering or clicking over/on the charts
    """

    # Function evoked at a mouse movement
    # If the mouse cursor is over the lower chart it reads the corresponding value
    # and updates the slider
    def on_mouse_movement(self, event):
        condition = self.simulator_mode == 'Slider-Controlled Experiment' or (
                self.simulator_mode == 'Physical CP' and not self.PhysicalCartPoleDriverInstance.dancer.danceEnabled
        )
        if condition:
            if event.xdata == None or event.ydata == None:
                pass
            else:
                if event.inaxes == self.fig.AxSlider:
                    self.slider_instant_value = event.xdata
                    if not self.slider_on_click:
                        self.target_slider.update_slider(mouse_position=event.xdata)

    # Function evoked at a mouse click
    # If the mouse cursor is over the lower chart it reads the corresponding value
    # and updates the slider
    def on_mouse_click(self, event):
        condition = self.simulator_mode == 'Slider-Controlled Experiment' or (
                self.simulator_mode == 'Physical CP' and not self.PhysicalCartPoleDriverInstance.dancer.danceEnabled
        )
        if condition:
            if event.xdata == None or event.ydata == None:
                pass
            else:
                if event.inaxes == self.fig.AxSlider:
                    self.target_slider.update_slider(mouse_position=event.xdata)

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
        if self.simulator_mode != 'Replay':
            # Change the mode variable depending on the Radiobutton state
            for i in range(len(self.gui.rbs_controllers)):
                if self.gui.rbs_controllers[i].isChecked():
                    self.CartPoleInstance.set_controller(controller_idx=i)

            self.update_rbs_optimizers_status(visible=self.CartPoleInstance.controller.has_optimizer)

            self.open_additional_controller_widget()

        # Reset the state of GUI and of the Cart instance after the mode has changed
        # TODO: Do I need the follwowing lines?
        self.reset_variables(0)
        self.cp_drawer.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()

    def update_rbs_optimizers_status(self, visible: bool):
        for rb in self.gui.rbs_optimizers:
            rb.setEnabled(visible)

    def RadioButtons_optimizer_selection(self):
        if self.simulator_mode != 'Replay':
            # Change the mode variable depending on the Radiobutton state
            for i in range(len(self.gui.rbs_optimizers)):
                if self.gui.rbs_optimizers[i].isChecked():
                    self.CartPoleInstance.set_optimizer(optimizer_idx=i)

    # Chose the simulator mode - effect of start/stop button
    def RadioButtons_simulator_mode(self):
        # Change the mode variable depending on the Radiobutton state
        for i in range(len(self.gui.rbs_simulator_mode)):
            time.sleep(0.001)
            if self.gui.rbs_simulator_mode[i].isChecked():
                self.simulator_mode = self.gui.available_simulator_modes[i]

        # Reset the state of GUI and of the Cart instance after the mode has changed
        # TODO: Do I need the follwowing lines?
        self.reset_variables(0)

        if self.simulator_mode == 'Physical CP':
            print("********************************************************************")
            print("\n\nSetting up physical cartpole driver...")
            from DriverFunctions.PhysicalCartPoleDriver import PhysicalCartPoleDriver
            self.PhysicalCartPoleDriverInstance = PhysicalCartPoleDriver(self.CartPoleInstance)
            self.PhysicalCartPoleDriverInstance.setup()
            worker = Worker(self.physical_experiment_thread)
            worker.signals.finished.connect(self.finish_thread)
            # Execute
            self.threadpool.start(worker)

        else:
            if self.PhysicalCartPoleDriverInstance:
                self.PhysicalCartPoleDriverInstance.terminate_experiment = True

        self.cp_drawer.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()

    # Chose the equilibrium - stabilize up or down
    def RadioButtons_equilibrium(self):
        time.sleep(0.001)
        if self.gui.rbs_equilibrium[0].isChecked():
            self.CartPoleInstance.target_equilibrium = 1.0
        else:
            self.CartPoleInstance.target_equilibrium = -1.0

    # Chose the noise mode - effect of start/stop button
    def RadioButtons_noise_on_off(self):
        # Change the mode variable depending on the Radiobutton state
        if self.gui.rbs_noise[0].isChecked():
            self.noise = 'ON'
            self.CartPoleInstance.NoiseAdderInstance.noise_mode = self.noise
        elif self.gui.rbs_noise[1].isChecked():
            self.noise = 'OFF'
            self.CartPoleInstance.NoiseAdderInstance.noise_mode = self.noise
        else:
            raise Exception('Something wrong with ON/OFF button for noise')

        self.open_additional_noise_widget()

    # endregion

    # region - Text Boxes

    # Read speedup provided by user from appropriate GUI textbox
    def get_speedup(self):
        """
        Get speedup provided by user from appropriate textbox.
        Speed-up gives how many times faster or slower than real time the simulation or replay should run.
        The provided values may not always be reached due to computer speed limitation
        """
        speedup = self.gui.tx_speedup.text()
        if speedup == '':
            self.speedup = np.inf
            return True
        else:
            try:
                speedup = float(speedup)
            except ValueError:
                self.gui.wrong_speedup_msg.setText(
                    'You have provided the input for speed-up which is not convertible to a number')
                x = self.gui.wrong_speedup_msg.exec_()
                return False
            if speedup == 0.0:
                self.gui.wrong_speedup_msg.setText(
                    'You cannot run an experiment with 0 speed-up (stopped time flow)')
                x = self.gui.wrong_speedup_msg.exec_()
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
        self.CartPoleInstance.interpolation_type = self.gui.cb_interpolation.currentText()

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

    # Action toggling between showing the ground level and above
    # and showing above and below ground level the length of the pole
    # Second option is good for visualizing swing-up
    def cb_show_hanging_pole_f(self, state):
        if state:
            self.CartPoleInstance.show_hanging_pole = True
        else:
            self.CartPoleInstance.show_hanging_pole = False
        self.cp_drawer.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()

    # endregion

    # region - Additional GUI Popups

    def open_additional_controller_widget(self):
        # Open up additional options widgets depending on the controller type
        if self.CartPoleInstance.controller_name == 'mppi-cartpole':
            try:
                from GUI._ControllerGUI_MPPIOptionsWindow import MPPIOptionsWindow
                self.optionsControllerWidget = MPPIOptionsWindow()
            except:
                pass
        else:
            try:
                self.optionsControllerWidget.close()
            except:
                pass
            self.optionsControllerWidget = None

    def open_additional_noise_widget(self):
        # Open up additional options widgets depending on the controller type
        if self.noise == 'ON':
            self.optionsNoiseWidget = NoiseOptionsWindow()
        else:
            try:
                self.optionsNoiseWidget.close()
            except:
                pass
            self.optionsNoiseWidget = None

    # endregion

    # region - Sliders setting initial position and angle of the CartPole

    def update_initial_position(self, value: str):
        self.initial_state[POSITION_IDX] = float(value) / 1000.0

    def update_initial_angle(self, value: str):
        self.initial_state[ANGLE_IDX] = float(value) / 100.0

    # endregion

    # region - Slider setting latency of the controller

    def update_latency(self, value: str):
        latency_slider = float(value)
        latency = latency_slider * self.CartPoleInstance.LatencyAdderInstance.max_latency / self.gui.LATENCY_SLIDER_RANGE_INT  # latency in seconds
        self.CartPoleInstance.LatencyAdderInstance.set_latency(latency)
        self.gui.labLatency.setText('{:.1f} ms'.format(latency * 1000.0))  # latency in ms

    # endregion

    # region Buttons for providing a kick to the pole

    def kick_pole(self, direction):
        # Adjust the angle based on the direction passed
        if direction == "Left":
            self.CartPoleInstance.s[ANGLED_IDX] += .6  # Adjust angle for left kick
        elif direction == "Right":
            self.CartPoleInstance.s[ANGLED_IDX] -= .6  # Adjust angle for right kick

    # endregion
    # TODO: Porobably bad idea with these properties.
    #  Better initialize Cartpole in MainWindow and pass to both GuiActions and GuiLayout
    @property
    def controller(self):
        return self.CartPoleInstance.controller

    @property
    def controller_names(self):
        return self.CartPoleInstance.controller_names

    @property
    def controller_idx(self):
        return self.CartPoleInstance.controller_idx

    @property
    def optimizer_names(self):
        return self.CartPoleInstance.optimizer_names

    @property
    def optimizer_idx(self):
        return self.CartPoleInstance.optimizer_idx

    @property
    def target_position(self):
        return self.CartPoleInstance.target_position

    @property
    def target_equilibrium(self):
        return self.CartPoleInstance.target_equilibrium

    @property
    def interpolation_type(self):
        return self.CartPoleInstance.interpolation_type

    @property
    def show_hanging_pole(self):
        return self.CartPoleInstance.show_hanging_pole

    @property
    def stop_at_90(self):
        return self.CartPoleInstance.stop_at_90

    @property
    def latency(self):
        return self.CartPoleInstance.LatencyAdderInstance.latency

    @property
    def max_latency(self):
        return self.CartPoleInstance.LatencyAdderInstance.max_latency


class ZeroAngleShiftHandler:
    def __init__(self, cart_pole_instance):
        self.cart_pole_instance = cart_pole_instance
        self.textbox = None

    def connect_textbox(self, textbox):
        self.textbox = textbox

        # Connect the editingFinished signal
        self.textbox.editingFinished.connect(self.on_user_input_finished)

    def on_user_input_finished(self):
        # Now process the input and update the internal value
        self.update_zero_angle_shift()

    def update_zero_angle_shift(self):
        # Get the value from the textbox
        try:
            value = np.deg2rad(float(self.textbox.text()))
        except ValueError:
            # Handle invalid input (non-float) by ignoring it or resetting the value
            return

        # Update the CartPoleInstance's zero_angle_shift
        self.cart_pole_instance.zero_angle_shift = value
