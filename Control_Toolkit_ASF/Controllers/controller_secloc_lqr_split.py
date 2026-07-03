"""
Sparse event-triggered LQR controller.

The LQR policy computes a fresh control value. The Secloc gate decides whether
the policy should run on this sample or whether the previous value is reused.
"""

from SI_Toolkit.computation_library import NumpyLibrary, TensorType
import numpy as np
import scipy

from CartPole.cartpole_jacobian import cartpole_jacobian
from CartPole.cartpole_parameters import u_max
from CartPole.state_utilities import create_cartpole_state
from CartPole.state_utilities import (
    ANGLE_IDX,
    ANGLED_IDX,
    POSITION_IDX,
    POSITIOND_IDX,
)
from Control_Toolkit.Controllers import template_controller
from others.globals_and_utils import create_rng, load_config


s0 = create_cartpole_state()
config = load_config("cartpole_physical_parameters.yml")


class LQRPolicy:
    def __init__(self, K, X, eigVals, Q, R):
        self.K = K
        self.X = X
        self.eigVals = eigVals
        self.Q = Q
        self.R = R

    @classmethod
    def from_config(cls, config_controller):
        s = s0
        s[POSITION_IDX] = 0.0
        s[POSITIOND_IDX] = 0.0
        s[ANGLE_IDX] = 0.0
        s[ANGLED_IDX] = 0.0
        u = 0.0

        jacobian = cartpole_jacobian(s, u)
        A = jacobian[:, :-1]
        B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * u_max

        Q = np.diag(config_controller["Q"])
        R = config_controller["R"]
        print("Q:", Q, " - R:", R)

        X = scipy.linalg.solve_continuous_are(A, B, Q, R)

        if np.array(R).ndim == 0:
            Ri = 1.0 / R
        else:
            Ri = np.linalg.inv(R)

        K = np.dot(Ri, (np.dot(B.T, X)))
        eigVals = np.linalg.eigvals(A - np.dot(B, K))

        print(" K:", K)
        return cls(K=K, X=X, eigVals=eigVals, Q=Q, R=R)

    def state_vector(self, s, target_position):
        return np.array(
            [
                [s[POSITION_IDX] - target_position],
                [s[POSITIOND_IDX]],
                [s[ANGLE_IDX]],
                [s[ANGLED_IDX]],
            ]
        )

    def step(self, s, target_position):
        state = self.state_vector(s, target_position)
        Q = np.dot(-self.K, state).item()
        return np.clip(Q, -1.0, 1.0, dtype=np.float32)


class SeclocGate:
    def __init__(self, log_base, ref_period, dead_ang, dead_pos):
        self.log_base = log_base
        self.ref_period = ref_period
        self.dead_ang = dead_ang
        self.dead_pos = dead_pos
        self.reset()

    def reset(self):
        self.ang_last_shift = 0.0001
        self.pos_last_shift = 0.0001
        self.time_last = None

    def should_sample(self, s, target_position, time=None):
        if self.time_last is None:
            time_difference = self.ref_period
        else:
            time_difference = time - self.time_last

        if time_difference + self.ref_period / 20 < self.ref_period:
            return False

        spike = False
        ang_shift = s[ANGLE_IDX]
        pos_shift = s[POSITION_IDX] - target_position

        if ang_shift < 0:
            ang_shift = -ang_shift
        if pos_shift < 0:
            pos_shift = -pos_shift

        if (ang_shift > self.dead_ang) and (self.ang_last_shift != 0):
            ang_ratio_inc = ang_shift / self.ang_last_shift
            ang_ratio_dec = 1.0 / ang_ratio_inc
            if (ang_ratio_inc >= self.log_base) or (ang_ratio_dec >= self.log_base):
                self.ang_last_shift = ang_shift
                spike = True
        elif (pos_shift > self.dead_pos) and (self.pos_last_shift != 0):
            pos_ratio_inc = pos_shift / self.pos_last_shift
            pos_ratio_dec = 1.0 / pos_ratio_inc
            if (pos_ratio_inc >= self.log_base) or (pos_ratio_dec >= self.log_base):
                self.pos_last_shift = pos_shift
                spike = True

        if spike:
            self.time_last = time

        return spike


class controller_secloc_lqr_split(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):
        seed = self.config_controller["seed"]
        self.rng = create_rng(self.__class__.__name__, seed if seed == None else seed * 2)

        self.lqr = LQRPolicy.from_config(self.config_controller)
        self.K = self.lqr.K
        self.X = self.lqr.X
        self.eigVals = self.lqr.eigVals
        self.Q = self.lqr.Q
        self.R = self.lqr.R

        self.secloc = SeclocGate(
            log_base=self.config_controller["log_base"],
            ref_period=self.config_controller["ref_period"],
            dead_ang=self.config_controller["dead_ang"],
            dead_pos=self.config_controller["dead_pos"],
        )
        self.last_Q = 0

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_attributes(updated_attributes)

        target_position = self.variable_parameters.target_position
        if self.secloc.should_sample(s, target_position, time=time):
            self.last_Q = self.lqr.step(s, target_position)
            return self.last_Q

        return self.last_Q
