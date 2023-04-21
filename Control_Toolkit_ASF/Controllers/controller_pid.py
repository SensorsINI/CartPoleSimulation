"""
A PID controller for the Cartpole using CartpoleSimulator conventions

        We checked that factory-firmware gain values are not working great for our hardware
        Hence we do not provide values recomputed to our software
        Instead we provide a set of values working good with our software at out hardware.
        I leave control-factory.json, but I don't think this is trustworthy.

        Comment of Tobi for position:
        "Naive solution: if too positive (too right), move left (minus on Q_position),
        but this does not produce correct control.
        The correct strategy is that if cart is too positive (too right),
        produce lean to the left by introducing a positive set point angle leaning slightly to left,
        i.e. more positve position_error makes more positive effective ANGLE_TARGET
        End result is that sign of Q_position is flipped
        KD term with "-" resists the motion
        KP and KI with "-" acts attractive towards the target position"

        Comment of Tobi for angle:
        "Assuming gains are positive,
        error growing to the "right"
        (around zero in upright position, this means in fact angle gets negative),
        causes motor to move to the right
        iff a term below has - sign"
"""

import json

from CartPoleSimulation.Control_Toolkit.Controllers import template_controller
from CartPoleSimulation.CartPole.state_utilities import cartpole_state_varname_to_index
from DriverFunctions.json_helpers import get_new_json_filename

from globals import dec, inc, JSON_PATH

import numpy as np

from SI_Toolkit.computation_library import TensorType

# PID params from json
PARAMS_JSON_FILE = JSON_PATH + 'control_PID.json'

# Sensitivity for PID gains - these are hardcoded multiplicative factors for PID gains
# They help to keep gains in json files in user-friendly magnitude.
# To start with a working solution we recommend that whichever sensitivity factor you change,
# you also change the corresponding gain in json file you intend to use
# E.g. if you multiply here 
# For transparency we recommend that you only change sensitivity gains by powers of 10.
sensitivity_pP_gain = 1.0
sensitivity_pI_gain = 1.0
sensitivity_pD_gain = 0.01

sensitivity_aP_gain = 1.0
sensitivity_aI_gain = 1.0
sensitivity_aD_gain = 0.01

class controller_pid(template_controller):
    def configure(self):

        self.time_last = None

        self.PARAMS_JSON_FILE = PARAMS_JSON_FILE

        ########################################################################################################

        # Position PID

        self.POSITION_TARGET = 0

        # Errors
        self.position_error = 0.0
        self.position_error_integral = 0.0
        self.position_error_diff = 0.0

        self.position_error_previous = None

        # Gains
        self.POSITION_KP = 0.0
        self.POSITION_KD = 0.0
        self.POSITION_KI = 0.0

        # "Cost" components:
        # gain * error(or error integral or error difference) * sensitivity factor (see at the top of the file)
        self.pP = 0.0
        self.pI = 0.0
        self.pD = 0.0

        # Motor command - position-PID contribution
        self.Q_position = 0.0

        ########################################################################################################
        
        # Angle PID

        self.ANGLE_TARGET = 0.0

        # Errors
        self.angle_error = 0.0
        self.angle_error_integral = 0.0
        self.angle_error_diff = 0.0

        self.angle_error_previous = None

        # Gains
        self.ANGLE_KP = 0.0
        self.ANGLE_KD = 0.0
        self.ANGLE_KI = 0.0

        # "Cost" components:
        # gain * error(or error integral or error difference) * sensitivity factor (see at the top of the file)
        self.aP = 0.0
        self.aI = 0.0
        self.aD = 0.0

        # Motor command - angle-PID contribution
        self.Q_angle = 0.0

        ########################################################################################################

        # Final motor command - sum of angle-PID and position-PID motor commands
        self.Q = 0


    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_attributes(updated_attributes)

        self.POSITION_TARGET = self.variable_parameters.target_position

        ########################################################################################################

        # Time

        if self.time_last is None:
            time_difference = 0.0
        else:
            time_difference = time - self.time_last

        # Ignore time difference if the difference very big
        # (it would harm integral gain if appears as error and otherwise the system is anyway not stable)
        if time_difference > 0.1:
            time_difference = 0.0

        self.time_last = time

        ########################################################################################################

        # Position PID

        # Error
        self.position_error = (s[cartpole_state_varname_to_index('position')] - self.variable_parameters.target_position)

        # Error difference
        if time_difference > 0.0001 and (self.position_error_previous is not None):
            self.position_error_diff = (self.position_error - self.position_error_previous) / time_difference
        else:
            self.position_error_diff = 0.0

        self.position_error_previous = self.position_error

        # Error integral
        if self.POSITION_KI > 0.0:
            self.position_error_integral += self.position_error * time_difference
            self.position_error_integral = np.clip(self.position_error_integral, -1.0/self.POSITION_KI, 1.0/self.POSITION_KI)  # Makes sure pI is not bigger than 1.0. KI regulates rather the rate of then max value
        else:
            self.position_error_integral = 0.0

        # "Cost" components
        # We split the "cost" components to allow separate printing helping to understand which components are relevant
        self.pP = self.POSITION_KP * self.position_error * sensitivity_pP_gain
        self.pI = self.POSITION_KI * self.position_error_integral * sensitivity_pI_gain
        self.pD = self.POSITION_KD * self.position_error_diff * sensitivity_pD_gain

        # Motor command - position-PID contribution
        self.Q_position = self.pP + self.pI + self.pD

        ########################################################################################################

        # Angle PID

        # Error
        self.angle_error = (s[cartpole_state_varname_to_index('angle')] - self.ANGLE_TARGET)

        # Error difference
        if time_difference > 0.0001 and (self.angle_error_previous is not None):
            self.angle_error_diff = (self.angle_error - self.angle_error_previous) / time_difference # correct for actual sample interval; if interval is too long, reduce diff error
        else:
            self.angle_error_diff = 0.0

        self.angle_error_previous = self.angle_error

        # Error integral
        if self.ANGLE_KI > 0.0:
            self.angle_error_integral += self.angle_error * time_difference
            self.angle_error_integral = np.clip(self.angle_error_integral, -1.0/self.ANGLE_KI, 1.0/self.ANGLE_KI)
        else:
            self.angle_error_integral = 0.0

        # "Cost" components
        # We split the "cost" components to allow separate printing helping to understand which components are relevant
        self.aP = self.ANGLE_KP * self.angle_error * sensitivity_aP_gain
        self.aI = self.ANGLE_KI * self.angle_error_integral * sensitivity_aI_gain
        self.aD = self.ANGLE_KD * self.angle_error_diff * sensitivity_aD_gain

        # Motor command - angle-PID contribution
        self.Q_angle = -self.aP - self.aI  - self.aD   # if too CCW (pos error), move cart left

        ########################################################################################################

        self.Q = self.Q_angle + self.Q_position

        return self.Q

    def printparams(self):
        print("\nAngle PID Control Parameters")
        print("    Set point       {0}".format(self.ANGLE_TARGET))
        print("    P Gain          {0:.2f}".format(self.ANGLE_KP))
        print("    I Gain          {0:.2f}".format(self.ANGLE_KI))
        print("    D Gain          {0:.2f}".format(self.ANGLE_KD))

        print("Position PD Control Parameters")
        print("    Set point       {0}".format(self.POSITION_TARGET))
        print("    P Gain          {0:.2f}".format(self.POSITION_KP))
        print("    I Gain          {0:.2f}".format(self.POSITION_KI))
        print("    D Gain          {0:.2f}".format(self.POSITION_KD))

    def loadparams(self):
        print(f"\nLoading parameters from {self.PARAMS_JSON_FILE}....")
        f = open(self.PARAMS_JSON_FILE)
        try:
            p = json.load(f)
            self.ANGLE_TARGET = p['ANGLE_TARGET']
            self.ANGLE_KP = p['ANGLE_KP']
            self.ANGLE_KI = p['ANGLE_KI']
            self.ANGLE_KD = p['ANGLE_KD']
            self.POSITION_KP = p['POSITION_KP']
            self.POSITION_KI = p['POSITION_KI']
            self.POSITION_KD = p['POSITION_KD']
        except Exception as e:
            print(f"\nsomething went wrong loading parameters: {e}")
            return
        print("success, parameters are")
        self.printparams()

    def saveparams(self):
        json_filepath = get_new_json_filename(self.controller_name)
        print(f"\nSaving parameters to {json_filepath}")

        p = {}
        p['ANGLE_TARGET'] = self.ANGLE_TARGET
        p['ANGLE_KP'] = self.ANGLE_KP
        p['ANGLE_KI'] = self.ANGLE_KI
        p['ANGLE_KD'] = self.ANGLE_KD
        p['POSITION_KP'] = self.POSITION_KP
        p['POSITION_KI'] = self.POSITION_KI
        p['POSITION_KD'] = self.POSITION_KD
        with open(json_filepath, 'w') as f:
            json.dump(p, f)

    def keyboard_input(self, c):
        if c == 'p':
            self.printparams()
        # Angle Gains
        elif c == '2':
            self.ANGLE_KP = inc(self.ANGLE_KP)
            print("\nIncreased angle KP {0}".format(self.ANGLE_KP))
        elif c == '1':
            self.ANGLE_KP = dec(self.ANGLE_KP)
            print("\nDecreased angle KP {0}".format(self.ANGLE_KP))
        elif c == 'w':
            self.ANGLE_KI = inc(self.ANGLE_KI)
            print("\nIncreased angle KI {0}".format(self.ANGLE_KI))
        elif c == 'q':
            self.ANGLE_KI = dec(self.ANGLE_KI)
            print("\nDecreased angle KI {0}".format(self.ANGLE_KI))
        elif c == 's':
            self.ANGLE_KD = inc(self.ANGLE_KD)
            print("\nIncreased angle KD {0}".format(self.ANGLE_KD))
        elif c == 'a':
            self.ANGLE_KD = dec(self.ANGLE_KD)
            print("\nDecreased angle KD {0}".format(self.ANGLE_KD))
        # Position Gains
        elif c == '4':
            self.POSITION_KP = inc(self.POSITION_KP)
            print("\nIncreased position KP {0}".format(self.POSITION_KP))
        elif c == '3':
            self.POSITION_KP = dec(self.POSITION_KP)
            print("\nDecreased position KP {0}".format(self.POSITION_KP))
        elif c == 'r':
            self.POSITION_KI = inc(self.POSITION_KI)
            print("\nIncreased position KI {0}".format(self.POSITION_KI))
        elif c == 'e':
            self.POSITION_KI = dec(self.POSITION_KI)
            print("\nDecreased position KI {0}".format(self.POSITION_KI))
        elif c == 'f':
            self.POSITION_KD = inc(self.POSITION_KD)
            print("\nIncreased position KD {0}".format(self.POSITION_KD))
        elif c == 'd':
            self.POSITION_KD = dec(self.POSITION_KD)
            print("\nDecreased position KD {0}".format(self.POSITION_KD))
        elif c == 'S':
            self.saveparams()
        elif c == 'L':
            self.loadparams()

    def print_help(self):
        print("\n***********************************")
        print("keystroke commands")
        print("ESC quit")
        print("k toggle control on/off (initially off)")
        print("K trigger motor position calibration")
        print("=/- increase/decrease (fine tune) angle deviation value")
        print("[/] increase/decrease position target")
        print("2/1 angle proportional gain")
        print("w/q angle integral gain")
        print("s/a angle derivative gain")
        print("z/x angle smoothing")
        print("4/3 position proportional gain")
        print("r/e position integral gain")
        print("f/d position derivative gain")
        print("p print PID parameters")
        print("l toggle logging data")
        print("S/L Save/Load param values from disk")
        print("D Toggle dance mode")
        print(",./ Turn on motor left zero right")
        print("m Toggle measurement")
        print("j Switch joystick control mode")
        print("b Print angle measurement from sensor")
        print("6 Enable/Disable live plot")
        print("5 Interrupts for histogram plot")
        print("***********************************")

    def controller_reset(self):

        self.time_last = None

        self.position_error_previous = None
        self.angle_error_previous = None

        self.position_error_integral = 0.0
        self.angle_error_integral = 0.0
