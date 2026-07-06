"""Sparse Event-Based Closed-Loop Control (SECLOC) applied to an arbitrary controller.

On the physical cartpole, set globals.CONTROLLER_NAME to the inner controller (e.g.
"pid", "mpc") and globals.USE_SECLOC = True. The driver wraps that controller via
CartPole.set_controller(..., use_secloc=True).

Inner controller settings come from that controller's entry in config_controllers.yml.
Gate settings come from config_secloc.yml (via secloc_config in the secloc entry).

Theory: https://www.frontiersin.org/articles/10.3389/fnins.2019.00827/full
"""
from SI_Toolkit.computation_library import NumpyLibrary

from Control_Toolkit_ASF.Controllers.secloc_controller_wrapper import SeclocControllerWrapper


class controller_secloc(SeclocControllerWrapper):
    _computation_library = NumpyLibrary()
