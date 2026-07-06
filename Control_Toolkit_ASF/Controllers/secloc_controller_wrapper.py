"""Generic base class combining an arbitrary controller with the Secloc event-based gate.

A concrete secloc controller is a thin subclass, e.g.::

    class controller_secloc_lqr(SeclocControllerWrapper):
        _computation_library = NumpyLibrary()
        inner_controller_name = "lqr"

The wrapped ("inner") controller is resolved with import_controller_by_name and does
the actual control computation; the SeclocGate decides on every step whether that
computation is due or whether the previous control value is held.

For inner controllers that are not template_controllers (e.g. plain computation
classes), override make_inner_controller / inner_step / sync_inner_parameters
instead of setting inner_controller_name.

The wrapper exposes the split interface used by the async driver wrappers:
should_trigger() (cheap gate evaluation, safe to call every loop iteration) and
compute_step() (the potentially expensive computation). step() combines both with a
zero-order hold on the last computed control.

Unknown attribute reads (K, optimizer, predictor, cost_function, ...) are forwarded
to the inner controller, so the wrapper is a drop-in replacement for it.
"""
import os

import numpy as np

from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.globals_and_utils import import_controller_by_name
from Control_Toolkit_ASF.Controllers.secloc_gate import SeclocGate
from SI_Toolkit.computation_library import NumpyLibrary, PyTorchLibrary, TensorFlowLibrary, TensorType
from SI_Toolkit.load_and_normalize import load_yaml

CONFIG_CONTROLLERS_PATH = os.path.join("Control_Toolkit_ASF", "config_controllers.yml")


# Framework plumbing normally created by template_controller.__init__. It is shared
# with the inner controller instead of running its __init__, which would re-derive
# everything (config entry, computation library, variable parameters) from the inner
# controller's own name instead of the wrapper's.
_SHARED_ATTRIBUTES = (
    "config_controller",
    "environment_name",
    "control_limits",
    "action_low",
    "action_high",
    "initial_environment_attributes",
    "variable_parameters",
    "u",
    "controller_logging",
    "save_vars",
    "logs",
    "controller_data_for_csv",
)


class SeclocControllerWrapper(template_controller):
    # Optional default when set_inner_controller_name() is not called (subclasses, tests).
    inner_controller_name: str = None

    @property
    def has_optimizer(self):
        # CartPole reads this before configure() to decide whether to pass an
        # optimizer name, so before the inner controller exists it is derived
        # from the inner controller's class.
        inner = self.__dict__.get("inner")
        if inner is not None:
            return getattr(inner, "has_optimizer", False)
        inner_name = self._resolved_inner_controller_name()
        if inner_name is None:
            return False
        return import_controller_by_name(inner_name)._has_optimizer

    def set_inner_controller_name(self, inner_controller_name):
        """Choose the wrapped controller at runtime; call before configure()."""
        self._inner_controller_override = inner_controller_name

    def _resolved_inner_controller_name(self):
        return getattr(self, "_inner_controller_override", None) or self.inner_controller_name

    def configure(self, *args, **kwargs):
        self.inner = self.make_inner_controller(*args, **kwargs)

        secloc_config_name = self.config_controller.get("secloc_config", "default")
        self.secloc = SeclocGate.from_config_file(secloc_config_name)
        self.secloc.start_config_watcher(config_name=secloc_config_name)
        self.last_Q = 0

        self.controller_data_for_csv = dict(
            getattr(self.inner, "controller_data_for_csv", {})
        )
        self.controller_data_for_csv.update(self.secloc.get_csv_data())

    # ------------------------------------------------------------- inner controller
    def make_inner_controller(self, *args, **kwargs):
        """Instantiate and configure the wrapped controller.

        The inner controller is created without running template_controller.__init__;
        it shares variable parameters and logs with this wrapper but loads its settings
        from its own entry in config_controllers.yml (pid, mpc, lqr, ...).
        """
        inner_name = self._resolved_inner_controller_name()
        if inner_name is None:
            raise ValueError(
                f"{self.__class__.__name__} needs an inner controller: call "
                "set_inner_controller_name() before configure(), or set "
                "inner_controller_name on the class."
            )

        inner_config = dict(load_yaml(CONFIG_CONTROLLERS_PATH)[inner_name])

        Controller = import_controller_by_name(inner_name)
        inner = Controller.__new__(Controller)
        inner._computation_library = self._computation_library
        for attribute in _SHARED_ATTRIBUTES:
            if hasattr(self, attribute):
                setattr(inner, attribute, getattr(self, attribute))
        inner.config_controller = inner_config
        computation_library_name = str(inner.config_controller.get("computation_library", ""))
        if computation_library_name:
            if "tensorflow" in computation_library_name.lower():
                inner._computation_library = TensorFlowLibrary()
            elif "pytorch" in computation_library_name.lower():
                inner._computation_library = PyTorchLibrary()
            elif "numpy" in computation_library_name.lower():
                inner._computation_library = NumpyLibrary()
            else:
                raise ValueError(
                    f"Computation library {computation_library_name!r} could not be interpreted."
                )
        inner.configure(*args, **kwargs)
        return inner

    def sync_inner_parameters(self):
        """Propagate wrapper state to the inner controller before it is queried.

        Shares the variable parameters (target position etc.) and lets the inner
        controller process a pending config-file reload if it supports one.
        Override (e.g. with a no-op) for inners that are not template_controllers.
        """
        self.inner.variable_parameters = self.variable_parameters
        update_from_config = getattr(
            self.inner, "update_controller_parameters_from_config", None
        )
        if callable(update_from_config):
            update_from_config()

    def inner_step(self, s, time=None):
        """The actual control computation. Override for non-standard inners."""
        return self.inner.step(s, time=time, updated_attributes={})

    def __getattr__(self, name):
        # Forward unknown attribute reads (K, optimizer, predictor, ...) to the
        # inner controller so the wrapper is a drop-in replacement for it.
        inner = self.__dict__.get("inner")
        if inner is None or name.startswith("__"):
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}"
            )
        return getattr(inner, name)

    # ------------------------------------------------------------------- lifecycle
    def stop_config_watcher(self):
        inner = self.__dict__.get("inner")
        if inner is not None and hasattr(inner, "stop_config_watcher"):
            inner.stop_config_watcher()
        secloc = self.__dict__.get("secloc")
        if secloc is not None:
            secloc.stop_config_watcher()

    def __del__(self):
        self.stop_config_watcher()

    def controller_reset(self):
        inner = self.__dict__.get("inner")
        inner_reset = getattr(inner, "controller_reset", None)
        if callable(inner_reset):
            try:
                inner_reset()
            except NotImplementedError:
                pass
        secloc = self.__dict__.get("secloc")
        if secloc is not None:
            secloc.reset()
        self.last_Q = 0

    def get_controller_status(self):
        return self.secloc.get_status()

    # ------------------------------------------------------------- secloc tunables
    @property
    def log_base(self):
        return self.secloc.log_base

    @log_base.setter
    def log_base(self, value):
        self.secloc.log_base = value

    @property
    def dead_ang(self):
        return self.secloc.dead_ang

    @dead_ang.setter
    def dead_ang(self, value):
        self.secloc.dead_ang = value

    @property
    def dead_pos(self):
        return self.secloc.dead_pos

    @dead_pos.setter
    def dead_pos(self, value):
        self.secloc.dead_pos = value

    @property
    def ang_last_shift(self):
        return self.secloc.ang_last_shift

    @ang_last_shift.setter
    def ang_last_shift(self, value):
        self.secloc.ang_last_shift = value

    @property
    def pos_last_shift(self):
        return self.secloc.pos_last_shift

    @pos_last_shift.setter
    def pos_last_shift(self, value):
        self.secloc.pos_last_shift = value

    @property
    def time_last(self):
        return self.secloc.time_last

    @time_last.setter
    def time_last(self, value):
        self.secloc.time_last = value

    # ------------------------------------------------------------------ control API
    def should_trigger(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        """Evaluate the Secloc gate (cheap). Returns True if a fresh computation is due.

        Carries the side effects the gate has inside step(): reloading the Secloc
        config if flagged, syncing attributes and inner-controller parameters, and
        updating the gate's internal state and decision statistics via should_sample().
        """
        self.secloc.update_from_config_file_if_needed()
        time_difference = self.secloc.time_difference(time)

        self.update_attributes(updated_attributes)
        self.sync_inner_parameters()

        if "config_controller" in updated_attributes and "ref_period" in self.config_controller:
            self.secloc.update_ref_period_from_config(self.config_controller)

        target_position = self.variable_parameters.target_position
        return self.secloc.should_sample(
            s,
            target_position,
            time=time,
            time_difference=time_difference,
        )

    def compute_step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        """Run the (potentially expensive) controller computation, no gate involved."""
        self.update_attributes(updated_attributes)
        self.sync_inner_parameters()
        self.last_Q = self.inner_step(s, time=time)
        return self.last_Q

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        if self.should_trigger(s, time=time, updated_attributes=updated_attributes):
            return self.compute_step(s, time=time, updated_attributes={})

        return self.last_Q
