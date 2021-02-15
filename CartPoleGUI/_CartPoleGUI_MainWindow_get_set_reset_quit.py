from CartPoleGUI.gui_default_params import *

from PyQt5.QtWidgets import QApplication

import numpy as np
from time import sleep

def set_experiment_generator_init_params(self):
    self.CartPoleInstance.track_relative_complexity = track_relative_complexity_globals
    self.CartPoleInstance.random_length = random_length_globals
    self.CartPoleInstance.interpolation_type = interpolation_type_globals
    self.CartPoleInstance.turning_points_period = turning_points_period_globals
    self.CartPoleInstance.start_random_target_position_at = start_random_target_position_at_globals
    self.CartPoleInstance.end_random_target_position_at = end_random_target_position_at_globals
    self.CartPoleInstance.turning_points = turning_points_globals

# Method resetting variables
def reset_variables(self, reset_mode=1):
    self.CartPoleInstance.set_cartpole_state_at_t0(reset_mode)
    self.counter = 0
    # "Try" because this function is called for the first time during initialisation of the Window
    # when the timer label instance is not yer there.
    try:
        self.labt.setText("Time (s): " + str(float(self.counter) / 10.0))
    except:
        pass
    self.saved = 0
    self.looper.first_call_done = False

######################################################################################################

# (Marcin) Below are methods with less critical functions.

# A thread redrawing labels (except for timer, which has its own function) of GUI every 0.1 s
def set_labels_thread(self):
    while (self.run_set_labels_thread):
        self.labSpeed.setText("Speed (m/s): " + str(np.around(self.CartPoleInstance.s.positionD, 2)))
        self.labAngle.setText("Angle (deg): " + str(np.around(self.CartPoleInstance.s.angle * 360 / (2 * np.pi), 2)))
        self.labMotor.setText("Motor power (Q): {:.3f}".format(np.around(self.CartPoleInstance.Q, 2)))
        if self.CartPoleInstance.controller_name == 'manual-stabilization':
            self.labTargetPosition.setText("")
        else:
            self.labTargetPosition.setText("Target position (m): " + str(np.around(self.CartPoleInstance.slider_value, 2)))

        if self.CartPoleInstance.controller_name == 'manual_stabilization':
            self.labSliderInstant.setText("Slider instant value (-): " + str(np.around(self.slider_value, 2)))
        else:
            self.labSliderInstant.setText("Slider instant value (m): " + str(np.around(self.slider_value, 2)))

        self.labTimeSim.setText('Simulation time (s): {:.2f}'.format(self.CartPoleInstance.time))

        mean_dt_real = np.mean(self.looper.circ_buffer_dt_real)
        if mean_dt_real > 0:
            self.labSpeedUp.setText('Speed-up (measured): x{:.2f}'
                                    .format(self.dt_main_simulation / mean_dt_real))
        sleep(0.1)

# Function to measure the time of simulation as experienced by user
# It corresponds to the time of simulation according to equations only if real time mode is on
# TODO (Marcin) I just retained this function from some example being my starting point
#   It seems it sometimes counting time to slow. Consider replacing in future
def set_user_time_label(self):
    # "If": Increment time counter only if simulation is running
    if self.run_experiment_thread == 1:
        self.counter += 1
        # The updates are done smoother if the label is updated here
        # and not in the separate thread
        self.labTime.setText("Time (s): " + str(float(self.counter) / 10.0))

# The acctions which has to be taken to properly terminate the application
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
