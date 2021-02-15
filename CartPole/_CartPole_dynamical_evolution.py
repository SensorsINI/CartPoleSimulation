import numpy as np
from types import SimpleNamespace

from CartPole._CartPole_mathematical_helpers import wrap_angle_rad
from CartPole.cartpole_model import cartpole_ode, Q2u


# This method changes the internal state of the CartPole
# from a state at time t to a state at t+dt
# We assume this function is called for the first time to calculate first time step
# @profile(precision=4)
def update_state(self):

    # Update the total time of the simulation
    self.time = self.time + self.dt_simulation

    # Update target position depending on the mode of operation
    if self.use_pregenerated_target_position == True:
        self.target_position = self.random_track_f(self.time)
        self.slider_value = self.target_position  # Assign target position to slider to display it
    else:
        if self.controller_name == 'manual-stabilization':
            self.target_position = 0.0 # In this case target position is not used. This just fill the corresponding column in history with zeros
        else:
            self.target_position = self.slider_value  # Get target position from slider

    # Calculate the next state
    self.s = cartpole_integration(self.s, self.dt_simulation)

    # Snippet to stop pole at +/- 90 deg if enabled
    zero_DD = None
    if self.stop_at_90:
        if self.s.angle >= np.pi / 2:
            self.s.angle = np.pi / 2
            self.s.angleD = 0.0
            zero_DD = True  # Make also second derivatives 0 after they are calculated
        elif self.s.angle <= -np.pi / 2:
            self.s.angle = -np.pi / 2
            self.s.angleD = 0.0
            zero_DD = True  # Make also second derivatives 0 after they are calculated
        else:
            zero_DD = False

    # Wrap angle to +/-π
    self.s.angle = wrap_angle_rad(self.s.angle)

    # In case in the next step the wheel of the cart
    # went beyond the track
    # Bump elastically into an (invisible) boarder
    if (abs(self.s.position) + self.WheelToMiddle) > self.HalfLength:
        self.s.positionD = -self.s.positionD

    # Determine the dimensionless [-1,1] value of the motor power Q
    self.Update_Q()

    # Convert dimensionless motor power to a physical force acting on the Cart
    self.u = Q2u(self.Q, self.p)

    # Update second derivatives
    self.s.angleDD, self.s.positionDD = cartpole_ode(self.p, self.s, self.u)

    if zero_DD:
        self.s.angleDD = 0.0

    # Calculate time steps from last saving
    # The counter should be initialized at max-1 to start with a control input update
    self.dt_save_steps_counter += 1

    # If update time interval elapsed save current state and zero the counter
    if self.dt_save_steps_counter == self.dt_save_number_of_steps:
        # If user chose to save history of the simulation it is saved now
        # It is saved first internally to a dictionary in the Cart instance
        if self.save_data_in_cart:

            # Saving simulation data
            self.dict_history['time'].append(self.time)
            self.dict_history['s.position'].append(self.s.position)
            self.dict_history['s.positionD'].append(self.s.positionD)
            self.dict_history['s.positionDD'].append(self.s.positionDD)
            self.dict_history['s.angle'].append(self.s.angle)
            self.dict_history['s.angleD'].append(self.s.angleD)
            self.dict_history['s.angleDD'].append(self.s.angleDD)
            self.dict_history['u'].append(self.u)
            self.dict_history['Q'].append(self.Q)
            # The target_position is not always meaningful
            # If it is not meaningful all values in this column are set to 0
            self.dict_history['target_position'].append(self.target_position)

            self.dict_history['s.angle.sin'].append(np.sin(self.s.angle))
            self.dict_history['s.angle.cos'].append(np.cos(self.s.angle))

        else:

            # Create a dict with current state - already rounded
            self.dict_history = {'time': [np.around(self.time, self.rounding_decimals)],
                                     's.position': [np.around(self.s.position, self.rounding_decimals)],
                                     's.positionD': [np.around(self.s.positionD, self.rounding_decimals)],
                                     's.positionDD': [np.around(self.s.positionDD, self.rounding_decimals)],
                                     's.angle': [np.around(self.s.angle, self.rounding_decimals)],
                                     's.angleD': [np.around(self.s.angleD, self.rounding_decimals)],
                                     's.angleDD': [np.around(self.s.angleDD, self.rounding_decimals)],
                                     'u': [np.around(self.u, self.rounding_decimals)],
                                     'Q': [np.around(self.Q, self.rounding_decimals)],
                                     'target_position': [np.around(self.target_position, self.rounding_decimals)],
                                     's.angle.sin': [np.around(np.sin(self.s.angle), self.rounding_decimals)],
                                     's.angle.cos': [np.around(np.cos(self.s.angle), self.rounding_decimals)]}
            self.save_flag = True

        self.dt_save_steps_counter = 0


def cartpole_integration(s, dt):
    """Simple single step integration of CartPole state by dt

    Takes state as SimpleNamespace, but returns as separate variables

    :param s: state of the CartPole (contains: s.position, s.positionD, s.angle and s.angleD)
    :param dt: time step by which the CartPole state should be integrated
    """
    s_next = SimpleNamespace()

    s_next.position = s.position + s.positionD * dt
    s_next.positionD = s.positionD + s.positionDD * dt

    s_next.angle = s.angle + s.angleD * dt
    s_next.angleD = s.angleD + s.angleDD * dt

    return s_next

# Determine the dimensionless [-1,1] value of the motor power Q
# We assume this function is called for the first time to calculate 0th time step
def Update_Q(self):

    # Calculate time steps from last update
    # The counter should be initialized at max-1 to start with a control input update
    self.dt_controller_steps_counter += 1

    # If update time interval elapsed update control input and zero the counter
    if self.dt_controller_steps_counter == self.dt_controller_number_of_steps:

        if self.controller_name == 'manual-stabilization':  # in this case slider corresponds already to the power of the motor
                self.Q = self.slider_value
        else:  # in this case slider gives a target position, lqr regulator
                self.Q = self.controller.step(self.s, self.target_position)

        self.dt_controller_steps_counter = 0
