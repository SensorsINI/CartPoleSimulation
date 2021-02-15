"""
This ''submodule'' collects methods used to change some ''static option'':
e.g. change current controller, switch between saving and not saving etc.
These are functions associated with radio buttons, check boxes, textfilds etc.
The functions of buttons which start some processes (play, replay, quit...) are usually much more complex
and we put them hence in separate file(s).
"""


import numpy as np

###################################################################################################

#   Radiobuttons

# Action to be taken while a radio button is clicked
# Chose the controller method which should be used with the CartPole
def RadioButtons(self):
    # Change the mode variable depending on the Radiobutton state
    for i in range(len(self.rbs)):
        if self.rbs[i].isChecked():
            self.CartPoleInstance.set_controller(controller_idx=i)

    # Reset the state of GUI and of the Cart instance after the mode has changed
    self.reset_variables(0)
    self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
    self.canvas.draw()

###################################################################################################

#   Textboxes

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

###################################################################################################

#   Checkboxes


def cb_interpolation_selectionchange(self, i):
    """
    Select interpolation type for random target positions of randomly generated experiment
    """
    self.CartPoleInstance.interpolation_type = self.cb_interpolation.currentText()


# Action toggling between saving and not saving simulation results
def cb_save_history_f(self, state):

    if state:
        self.save_history = 1
    else:
        self.save_history = 0

    if self.save_history or self.show_experiment_summary:
        self.save_data_in_cart =True
    else:
        self.save_data_in_cart = False

    self.CartPoleInstance.save_data_in_cart = self.save_data_in_cart


# Action toggling between saving and not saving simulation results
def cb_show_experiment_summary_f(self, state):

    if state:
        self.show_experiment_summary = 1
    else:
        self.show_experiment_summary = 0

    if self.save_history or self.show_experiment_summary:
        self.save_data_in_cart =True
    else:
        self.save_data_in_cart = False

    self.CartPoleInstance.save_data_in_cart = self.save_data_in_cart


# Action toggling between stopping (or not) the pole if it reaches 90 deg
def cb_stop_at_90_deg_f(self, state):

    if state:
        self.stop_at_90 = True
    else:
        self.stop_at_90 = False

    self.CartPoleInstance.stop_at_90 = self.stop_at_90


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


###################################################################################################
