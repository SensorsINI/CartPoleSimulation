"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*u_max (no fancy motor model) !
"""

from SI_Toolkit.computation_library import NumpyLibrary, TensorType
import numpy as np
import scipy
import os
import atexit
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from CartPole.cartpole_jacobian import cartpole_jacobian

from CartPole.cartpole_parameters import u_max
from CartPole.state_utilities import create_cartpole_state
from CartPole.state_utilities import (ANGLE_IDX, ANGLED_IDX, POSITION_IDX,
                                      POSITIOND_IDX)
from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.load_and_normalize import load_yaml
from others.globals_and_utils import create_rng, load_config

s0 = create_cartpole_state()
config = load_config("cartpole_physical_parameters.yml")


class LQRConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, controller):
        self.controller = controller
        self.config_path = os.path.abspath(controller.config_path)

    def on_modified(self, event):
        if os.path.abspath(event.src_path) == self.config_path:
            self.controller.reload_controller_config_from_config_flag = True


class controller_lqr(template_controller):
    _computation_library = NumpyLibrary()
    
    def configure(self):
        self._reload_controller_config()
        self._start_config_watcher()

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
        self.X = X
        self.eigVals = eigVals

    def _reload_controller_config(self):
        config_controllers, config_path = load_yaml(
            os.path.join("Control_Toolkit_ASF", "config_controllers.yml"),
            return_path=True,
        )
        self.config_controller = dict(config_controllers[self.controller_name])
        self.config_path = config_path

    def _start_config_watcher(self):
        if hasattr(self, "config_observer"):
            return

        self.reload_controller_config_from_config_flag = False
        self.config_observer = Observer()
        self.config_handler = LQRConfigChangeHandler(self)
        self.config_observer.schedule(
            self.config_handler,
            os.path.dirname(os.path.abspath(self.config_path)),
            recursive=False,
        )
        atexit.register(self.stop_config_watcher)
        self.config_observer.start()

    def stop_config_watcher(self):
        if not hasattr(self, "config_observer"):
            return

        if self.config_observer is not None and self.config_observer.is_alive():
            self.config_observer.stop()
            self.config_observer.join()
        self.config_observer = None

    def update_controller_parameters_from_config(self):
        if not getattr(self, "reload_controller_config_from_config_flag", False):
            return

        self.reload_controller_config_from_config_flag = False
        try:
            self.configure()
            print(f"LQR config reloaded: Q={self.config_controller['Q']}, R={self.config_controller['R']}, K={self.K}")
        except Exception as exc:
            print(f"LQR config reload failed, keeping previous gain: {exc}")

    def __del__(self):
        self.stop_config_watcher()

    def controller_reset(self):
        pass

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_controller_parameters_from_config()
        self.update_attributes(updated_attributes)
        
        state = np.array(
            [[s[POSITION_IDX] - self.variable_parameters.target_position], [s[POSITIOND_IDX]], [s[ANGLE_IDX]], [s[ANGLED_IDX]]])

        Q = np.dot(-self.K, state).item()

        # Clip Q
        Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)
        return Q
