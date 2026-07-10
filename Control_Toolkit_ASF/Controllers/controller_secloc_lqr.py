"""
This is a linear-quadratic regulator with a built-in (legacy, monolithic) SecLoc event gate.
It assumes that the input relation is u = Q*u_max (no fancy motor model) !

The gate (log-ratio criterion on |angle| and |position - target|) and the LQR are fused in
one controller, mirroring Firmware/Src/General/secloc_lqr.c (SECLOC_LQR_Ops) on the chip.
For the modular gate + inner-controller architecture use the 'secloc' wrapper instead.
"""
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

# Optional instruction-count profiling (Linux perf based). The controller works without it.
try:
    from cirron import Collector
    CIRRON_AVAILABLE = True
except ImportError:
    CIRRON_AVAILABLE = False

    class Collector:  # no-op stand-in so the `with Collector()` blocks stay unchanged
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

s0 = create_cartpole_state()
config = load_config("cartpole_physical_parameters.yml")


class controller_secloc_lqr(template_controller):
    _computation_library = NumpyLibrary()
    
    def configure(self):
        # From https://github.com/markwmuller/controlpy/blob/master/controlpy/synthesis.py#L8
        """Solve the continuous time LQR controller for a continuous time system.

        A and B are system matrices, describing the systems dynamics:
         dx/dt = A x + B u

        The controller minimizes the infinite horizon quadratic cost function:
         cost = integral (x.T*Q*x + u.T*R*u) dt

        where Q is a positive semidefinite matrix, and R is positive definite matrix.

        Returns K, X, eigVals:
        Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
        The optimal input is then computed as:
         input: u = -K*x
        """
        # ref Bertsekas, p.151

        seed = self.config_controller["seed"]
        self.rng = create_rng(self.__class__.__name__, seed if seed==None else seed*2)

        # Calculate Jacobian around equilibrium
        # Set point around which the Jacobian should be linearized
        # It can be here either pole up (all zeros) or pole down
        s = s0
        s[POSITION_IDX] = 0.0
        s[POSITIOND_IDX] = 0.0
        s[ANGLE_IDX] = 0.0
        s[ANGLED_IDX] = 0.0
        u = 0.0

        jacobian = cartpole_jacobian(s, u)
        A = jacobian[:, :-1]
        B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * u_max

        # Cost matrices for LQR controller
        self.Q = np.diag(self.config_controller["Q"]) # How much to punish x, v, theta, omega
        self.R = self.config_controller["R"]  # How much to punish Q
        print('Q:',self.Q,' - R:',self.R)
        # first, try to solve the ricatti equation
        X = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)

        # compute the LQR gain
        if np.array(self.R).ndim == 0:
            Ri = 1.0 / self.R
        else:
            Ri = np.linalg.inv(self.R)

        K = np.dot(Ri, (np.dot(B.T, X)))

        eigVals = np.linalg.eigvals(A - np.dot(B, K))

        self.K = K
        print(' K:', self.K)
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
        self.tick_last = 0  # Big number to assure a first event is raised.

        # FOR TESTING PURPOSES ONLY
        self.pos_events = 0
        self.ang_events = 0
        self.total_events = 0
        self.max_instructions = 0
        self.count_instructions = 0
        self.sum_instructions = 0

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):

        self.tick_last = self.tick_last + 1
        self.tick_total = self.tick_total + 1  # Increases tick count.

        self.update_attributes(updated_attributes)

        state = np.array(
            [[s[POSITION_IDX] - self.variable_parameters.target_position], [s[POSITIOND_IDX]], [s[ANGLE_IDX]], [s[ANGLED_IDX]]])

        ti_inicio = cpu_time()
        spike = 0
        # Start SECLOC checks.
        with Collector() as collectorSECLOC:
            if self.tick_last >= self.ref_period:
                ang_shift = s[ANGLE_IDX]
                ang_shift_sign = 1          # Only for polarity purposes
                if ang_shift < 0:
                    ang_shift_sign = -1
                    ang_shift = -ang_shift
                if (ang_shift > self.dead_ang) and (self.ang_last_shift != 0):		# ang_dead_band cant be 0.
                    ang_ratio_inc = ang_shift/self.ang_last_shift
                    ang_ratio_dec = 1.0/ang_ratio_inc
                    if (ang_ratio_inc >= self.log_base) or (ang_ratio_dec >= self.log_base):
                        self.ang_last_shift = ang_shift
                        spike = 1
                        self.ang_events = self.ang_events + 1
                if spike == 0:      # Change again to == 0 after measurements
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
                    #ti_fin = cpu_time()
                    #print(f"Segundos de CPU SECLOC: {ti_fin - ti_inicio}")
                    #ti_inicio = cpu_time()
        if CIRRON_AVAILABLE:
            if (collectorSECLOC.counters.instruction_count > self.max_instructions):
                self.max_instructions = collectorSECLOC.counters.instruction_count
            self.count_instructions += 1
            self.sum_instructions += collectorSECLOC.counters.instruction_count
            print("SECLOC Instructions: ", collectorSECLOC.counters.instruction_count, " max: ", self.max_instructions, " AVG: ", self.sum_instructions / self.count_instructions)
        if spike == 1:
            with Collector() as collectorLQR:
                Q = np.dot(-self.K, state).item()
                Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)
            if CIRRON_AVAILABLE:
                print("LQR Instructions.", collectorLQR.counters.instruction_count)
            #ti_fin = cpu_time()
            #print(f"Segundos de CPU LQR: {ti_fin - ti_inicio}")
            self.last_Q = Q
            #print("Ticks totales: ", self.tick_total, " - Ticks control: ", self.tick_control, ".\n\t",
            #      self.tick_control * 100 / self.tick_total, "%.")
            #self.total_events = self.total_events + 1
            #print("Total events: ", self.total_events, " - Ang_events: ", self.ang_events, " - Pos_events: ", self.pos_events)
            #print("\n\t", self.ang_events*100/self.total_events, "/", self.pos_events*100/self.total_events, "%.")
            return Q
        return self.last_Q
