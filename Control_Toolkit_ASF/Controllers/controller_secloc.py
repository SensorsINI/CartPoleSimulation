
from SI_Toolkit.computation_library import NumpyLibrary, TensorType
import numpy as np
import math
from datetime import datetime
import yaml
import os

from scipy.interpolate import interp1d
from dataclasses import dataclass
from Control_Toolkit.Controllers import template_controller

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
actuator_noise = config["cartpole"]["actuator_noise"]
config_controller = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"), "r"), Loader=yaml.FullLoader)

"""
Python implementation of the theory for Sparse Envent-Based Closed Loop Control (SECLOC):
https://www.frontiersin.org/articles/10.3389/fnins.2019.00827/full
"""
class controller_secloc(template_controller):
    _computation_library = NumpyLibrary
    
    def configure(self):
        log_base = config_controller["secloc"]["log_base"]
        dt = config_controller["secloc"]["dt"]
        ref_period = config_controller["secloc"]["ref_period"]
        dead_band = config_controller["secloc"]["dead_band"]
        pid_Kp = config_controller["secloc"]["pid_Kp"]
        pid_Kd = config_controller["secloc"]["pid_Kd"]
        pid_Ki = config_controller["secloc"]["pid_Ki"]
        
        self.p_Q = actuator_noise
        self.pid = Event_based_PID(Kp=pid_Kp, Kd=pid_Kd, Ki=pid_Ki, sensor_log_base=1.15, disp=True)
        self.potentiometer = Event_based_sensor(pid=self.pid, log_base=log_base, dt=dt, ref_period=ref_period, dead_band=dead_band, disp=True)
        self.potentiometer.set_point = 0
        self.motor_map = 128
        self.interpolation = interp1d([-self.motor_map,self.motor_map], [1,-1])
        self.step_idx = 0

    def step(self, s: np.ndarray, time=None, updated_attributes: dict[str, TensorType]={}):
        self.update_attributes(updated_attributes)
        # Read the cartpole state s = ["angle", "angleD", "angle_cos", "angle_sin", "position", "positionD"]
        # angle: pole UP -> 0, then +/-pi
        print(f"***** Step #{self.step_idx} *****")
        pole_angle = s[0]
        print(f"Current pole angle value: {pole_angle} radians. left/right --> +/- pi. Pole UP --> angle=0")
        print("Update the potentiometer event based sensor using the cartpole sate")
        self.potentiometer.update(signal_in= pole_angle)
        # Get the new control command and return it in the range [-1, 1]
        motor_signal = self.potentiometer.pid.motor_signal
        if motor_signal > self.motor_map:
            motor_signal = self.motor_map
        elif motor_signal < -self.motor_map:
            motor_signal = -self.motor_map
        motor_signal = self.interpolation(motor_signal)
        print(f"Map the motor signal {self.potentiometer.pid.motor_signal} to [-1,1]: motor_action: {motor_signal}")
        Q = np.float32(motor_signal * (1 + self.p_Q))
        # Clip Q
        if Q > 1.0:
            Q = 1.0
        elif Q < -1.0:
            Q = -1.0
        self.step_idx += 1
        print(f"********************")
        print(" ")
        return Q  # normed control input in the range [-1,1]. Move cart left:- right:+ 


@dataclass
class Event:
    time: int
    polarity: int
    change_sign: int
    n_change_base: int


class Motor_control:
    def __init__(self) -> None:
        pass


class Event_based_PID:
    def __init__(self, Kp: float, Ki:float, Kd: float, sensor_log_base: float, disp: bool):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.sensor_log_base = sensor_log_base
        self.disp = disp
        self.t = None
        self.Err = None
        self.I_err = None
        self.D_err = None
        self.c_time_micros = None
        self.motor_signal = 0

    def change_event_received(self, polarity: int, change_sign: int, n_change_base: int):
        elapsed_time = (self.c_time_micros - datetime.now().microsecond)*0.000001
        self.c_time_micros = datetime.now().microsecond
        self.I_err += self.Err * (2.0 + pow(-1.0, change_sign) * (pow(self.sensor_log_base, n_change_base * polarity) - 1.0)) / (elapsed_time * 2.0)
        self.D_err = pow(-1.0, change_sign) * (pow(self.sensor_log_base, n_change_base * polarity) - 1.0) * self.Err / elapsed_time
        self.Err += (pow(self.sensor_log_base, n_change_base * polarity) - 1.0) * self.Err
        self.Err *= pow(-1.0, change_sign)
        self.motor_signal = self.Kp * self.Err + self.Ki * self.I_err + self.Kd * self.D_err
        if self.disp:
            print(f"Change event received -- Err: {self.Err}; I-Err: {self.I_err}; dErr/dt: {self.D_err}")
    
    def init_error(self, e0: float):
        print(f"Initializing the Err: {e0}")
        self.Err = e0
        self.I_err = 0
        self.D_err = 0
        self.c_time_micros = datetime.now().microsecond 


class Event_based_sensor:
    def __init__(self, pid: Event_based_PID, log_base: float, dt: int, ref_period: int, dead_band: float, disp: bool):
        self.dt = dt # Temporal base of the system (in microseconds)
        self.ref_period = ref_period # Refractory period (in microseconds)
        self.log_base = log_base # Base for the computation of the event
        self.dead_band = dead_band # Dead band around the setpoint that is used to avoid reacting to sensor noise.
        self.disp = disp # Verbosity of the library
        self.last_val = None # Last value stored
        self.set_point = None # Sensor setpoint
        self.shift_base = 0.0
        self.has_init = False
        self.internal_timer = None
        self.last_event_time = None
        self.spike_change_sign = None
        self.pid = pid
        self.reset_state()

    def reset_state(self):
        self.internal_timer = 0
        self.last_event_time = 0
        self.spike_change_sign = 0

    def init_sensor(self, x0: int):
        print(f"Initializing the sensor to last_var: {x0}")
        self.last_val = x0
        self.pid.init_error(x0)
        self.has_init = True

    def update(self, signal_in: float):
        signal_shift = self.set_point - signal_in
        print(f"Signal_shift: {signal_shift}; Dead_band: {self.dead_band}")
        if(abs(signal_shift) > self.dead_band):
            self.update_change_event(signal_shift)
        self.internal_timer += 1

    def update_change_event(self, signal_shift: float):
        n_change_base = 1
        polarity = 0
        if self.has_init == False:
            self.init_sensor(signal_shift)
        # Check if the refractory periof if fit
        if (self.internal_timer - self.last_event_time) >= self.ref_period: #TODO check the time makes sense in term of microseconds
            ratio_increase = 0.0
            ratio_decrease = 0.0	
            ratio_sign = 0

            if self.last_val != 0:
                ratio_increase = abs(signal_shift / self.last_val) #abs(signal_shift) / abs(self.last_val)
                ratio_sign = self.sign(signal_shift / self.last_val)
            else:
                ratio_increase = self.log_base
                ratio_sign = self.sign(signal_shift)
                    
            if (signal_shift != 0.0) and (self.last_val != 0.0):
                ratio_decrease = abs(self.last_val / signal_shift) #abs(self.last_val) / abs(signal_shift)
                ratio_sign = self.sign(signal_shift / self.last_val)

            if ratio_sign >= 0:
                ratio_sign = 0
            else:
                ratio_sign = 1

            print(f"Ref_periof reached. ratio_increase: {ratio_increase}; ratio_decrease: {ratio_decrease}; ratio_sign: {ratio_sign}")
            
            if ratio_increase >= self.log_base:
                self.spike_change_sign = ratio_sign
                n_change_base = round(math.log(ratio_increase) / math.log(self.log_base))
                self.last_val = signal_shift
                self.last_event_time = self.internal_timer
                polarity = 1
                self.emitEvent(polarity=polarity, change_sign=self.spike_change_sign, n_change_base=n_change_base)
            
            if ratio_decrease >= self.log_base: #if abs(ratio_decrease) >= self.log_base:
                self.spike_change_sign = ratio_sign
                n_change_base = round(math.log(ratio_decrease) / math.log(self.log_base))
                self.last_val = signal_shift
                self.last_event_time = self.internal_timer
                polarity = -1
                self.emitEvent(polarity=polarity, change_sign=self.spike_change_sign, n_change_base=n_change_base)
        
        else:
            print(f"TRef_periof NOT reached. Time since last event: {self.internal_timer - self.last_event_time}; Ref_period: {self.ref_period}")

        # if self.disp:
        #     self.printInfo(time= self.internal_timer, polarity= polarity, change_sign=self.spike_change_sign, n_change_base= n_change_base)

    def sign(self, x: float):
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0
        else:
            return 0

    def emitEvent(self, polarity: int, change_sign: int, n_change_base: int):
        if self.pid is not None:
            print(f"Emiting and event -- pol: {polarity}; change_sign: {change_sign}; n_change_base: {n_change_base}")
            self.pid.change_event_received(polarity, change_sign, n_change_base)

    def printInfo(self, time: int, polarity: int, change_sign: int, n_change_base: int):
        print(f"Info -- time: {time}; pol: {polarity}; n_change_base: {n_change_base}; change_sign: {change_sign}")
