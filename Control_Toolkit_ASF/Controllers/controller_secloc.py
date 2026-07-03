"""Sparse Event-Based Closed-Loop Control (SECLOC) applied to an arbitrary controller.

The controller doing the actual computation is chosen by the 'inner_controller' key
of the 'secloc' entry in config_controllers.yml (e.g. "lqr" or "mpc"); the Secloc
gate (see secloc_gate.py, configured via config_secloc.yml) decides on every step
whether that computation is due or whether the previous control value is held.

Theory: https://www.frontiersin.org/articles/10.3389/fnins.2019.00827/full
"""
from SI_Toolkit.computation_library import NumpyLibrary

from Control_Toolkit_ASF.Controllers.secloc_controller_wrapper import SeclocControllerWrapper


class controller_secloc(SeclocControllerWrapper):
    _computation_library = NumpyLibrary()
