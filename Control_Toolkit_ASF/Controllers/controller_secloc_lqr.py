
from SI_Toolkit.computation_library import NumpyLibrary, TensorType
import numpy as np
import scipy
import yaml
import os

from CartPole.cartpole_jacobian import cartpole_jacobian

from CartPole.cartpole_parameters import u_max
from CartPole.state_utilities import create_cartpole_state
from CartPole.state_utilities import (ANGLE_IDX, ANGLED_IDX, POSITION_IDX,
                                      POSITIOND_IDX)
from Control_Toolkit.Controllers import template_controller
from others.globals_and_utils import create_rng, load_config
from time import process_time as cpu_time

s0 = create_cartpole_state()
config = load_config("cartpole_physical_parameters.yml")


class controller_secloc_lqr(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):
        seed = self.config_controller["seed"]
        self.rng = create_rng(self.__class__.__name__, seed if seed==None else seed*2)

        s = s0
        s[POSITION_IDX] = 0.0
        s[POSITIOND_IDX] = 0.0
        s[ANGLE_IDX] = 0.0
        s[ANGLED_IDX] = 0.0
        u = 0.0

        jacobian = cartpole_jacobian(s, u)
        A = jacobian[:, :-1]
        B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * u_max

        self.Q = np.diag(self.config_controller["Q"])
        self.R = self.config_controller["R"]

        X = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)

        if np.array(self.R).ndim == 0:
            Ri = 1.0 / self.R
        else:
            Ri = np.linalg.inv(self.R)

        K = np.dot(Ri, (np.dot(B.T, X)))

        eigVals = np.linalg.eigvals(A - np.dot(B, K))

        self.K = K
        self.X = X
        self.eigVals = eigVals

        self.ref_period = self.config_controller["ref_period"]
        self.log_base = self.config_controller["log_base"]
        self.dead_ang = self.config_controller["dead_ang"]
        self.dead_pos = self.config_controller["dead_pos"]
        self.ang_last_shift = 0.0001
        self.pos_last_shift = 0.0001
        self.last_Q = 0
        self.tick_control = 0
        self.tick_total = 0
        self.tick_last = 0

        self.pos_events = 0
        self.ang_events = 0
        self.total_events = 0
        self.max_instructions = 0
        self.count_instructions = 0
        self.sum_instructions = 0

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.tick_last = self.tick_last + 1
        self.tick_total = self.tick_total + 1

        self.update_attributes(updated_attributes)

        state = np.array(
            [[s[POSITION_IDX] - self.variable_parameters.target_position], [s[POSITIOND_IDX]], [s[ANGLE_IDX]], [s[ANGLED_IDX]]])

        ti_inicio = cpu_time()

        if self.tick_last >= self.ref_period:
            spike = 0
            ang_shift = s[ANGLE_IDX]
            ang_shift_sign = 1
            if ang_shift < 0:
                ang_shift_sign = -1
                ang_shift = -ang_shift
            if (ang_shift > self.dead_ang) and (self.ang_last_shift != 0):
                ang_ratio_inc = ang_shift/self.ang_last_shift
                ang_ratio_dec = 1.0/ang_ratio_inc
                if (ang_ratio_inc >= self.log_base) or (ang_ratio_dec >= self.log_base):
                    self.ang_last_shift = ang_shift
                    spike = 1
                    self.ang_events = self.ang_events + 1
            if spike == 0:
                pos_shift = s[POSITION_IDX] - self.variable_parameters.target_position
                pos_shift_sign = 1
                if pos_shift < 0:
                    pos_shift_sign = -1
                    pos_shift = -pos_shift
                if (pos_shift > self.dead_pos) and (self.pos_last_shift != 0):
                    pos_ratio_inc = pos_shift/self.pos_last_shift
                    pos_ratio_dec = 1.0/pos_ratio_inc
                    if (pos_ratio_inc >= self.log_base) or (pos_ratio_dec >= self.log_base):
                        self.pos_last_shift = pos_shift
                        spike = 1
                        self.pos_events = self.pos_events + 1
            if spike==1:
                self.tick_last = 0
                self.tick_control = self.tick_control + 1

        self.count_instructions += 1

        if spike == 1:
            Q = np.dot(-self.K, state).item()
            Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)

            self.last_Q = Q
            self.total_events = self.total_events + 1
            return Q
        return self.last_Q
