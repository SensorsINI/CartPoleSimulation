from SI_Toolkit.computation_library import NumpyLibrary, TensorType
import numpy as np

from Control_Toolkit.Controllers.controller_mpc import controller_mpc
from Control_Toolkit_ASF.Controllers.secloc_gate import SeclocGate
from Control_Toolkit.Controllers import template_controller


class controller_secloc_mpc(template_controller):
    _has_optimizer = True
    _computation_library = NumpyLibrary()

    def configure(self, optimizer_name=None):
        self.mpc = controller_mpc.__new__(controller_mpc)
        self.mpc.config_controller = self.config_controller
        self.mpc.variable_parameters = self.variable_parameters
        self.mpc.control_limits = self.control_limits
        self.mpc.controller_logging = self.controller_logging
        self.mpc.environment_name = self.environment_name
        self.mpc._computation_library = self._computation_library
        self.mpc.logs = self.logs
        self.mpc.save_vars = self.save_vars
        self.mpc.configure(optimizer_name=optimizer_name)
        self.optimizer = self.mpc.optimizer
        self.predictor = self.mpc.predictor
        self.cost_function = self.mpc.cost_function

        self.secloc = SeclocGate.from_config_file(
            self.config_controller.get("secloc_config", "default")
        )
        self.secloc.start_config_watcher(
            config_name=self.config_controller.get("secloc_config", "default"),
        )
        self.last_Q = 0
        self.controller_data_for_csv = self.mpc.controller_data_for_csv

        self.log_base = self.secloc.log_base
        self.dead_ang = self.secloc.dead_ang
        self.dead_pos = self.secloc.dead_pos

    def stop_config_watcher(self):
        if hasattr(self, "secloc"):
            self.secloc.stop_config_watcher()

    def __del__(self):
        self.stop_config_watcher()

    def controller_reset(self):
        if hasattr(self, "mpc"):
            self.mpc.controller_reset()
        if hasattr(self, "secloc"):
            self.secloc.reset()
        self.last_Q = 0

    def get_controller_status(self):
        return self.secloc.get_status()

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

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.secloc.update_from_config_file_if_needed()
        time_difference = self.secloc.time_difference(time)

        self.update_attributes(updated_attributes)
        self.mpc.variable_parameters = self.variable_parameters

        target_position = self.variable_parameters.target_position
        if self.secloc.should_sample(
            s,
            target_position,
            time=time,
            time_difference=time_difference,
        ):
            self.last_Q = self.mpc.step(s, time=time, updated_attributes={})
            self.optimizer = self.mpc.optimizer
            self.logs = self.mpc.logs
            return self.last_Q

        return self.last_Q
