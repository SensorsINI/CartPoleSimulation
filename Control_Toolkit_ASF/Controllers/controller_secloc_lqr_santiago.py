from SI_Toolkit.computation_library import NumpyLibrary, TensorType
import numpy as np

from Control_Toolkit_ASF.Controllers.lqr_santiago import LQRSantiago
from Control_Toolkit_ASF.Controllers.secloc_gate import SeclocGate
from Control_Toolkit.Controllers import template_controller
from others.globals_and_utils import create_rng


class controller_secloc_lqr_santiago(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):
        seed = self.config_controller["seed"]
        self.rng = create_rng(self.__class__.__name__, seed if seed == None else seed * 2)

        self.lqr = LQRSantiago.from_config(self.config_controller)
        self.secloc = SeclocGate.from_config_file(
            self.config_controller.get("secloc_config", "default")
        )
        self.secloc.start_config_watcher(
            config_name=self.config_controller.get("secloc_config", "default"),
        )
        self.last_Q = 0

        # Preserve the public attributes exposed by the original monolithic controller.
        self.K = self.lqr.K
        self.X = self.lqr.X
        self.eigVals = self.lqr.eigVals
        self.Q = self.lqr.Q
        self.R = self.lqr.R
        self.log_base = self.secloc.log_base
        self.dead_ang = self.secloc.dead_ang
        self.dead_pos = self.secloc.dead_pos

    def stop_config_watcher(self):
        self.secloc.stop_config_watcher()

    def __del__(self):
        self.stop_config_watcher()

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

        if "config_controller" in updated_attributes and "ref_period" in self.config_controller:
            self.secloc.update_ref_period_from_config(self.config_controller)

        target_position = self.variable_parameters.target_position
        if self.secloc.should_sample(
            s,
            target_position,
            time=time,
            time_difference=time_difference,
        ):
            self.last_Q = self.lqr.step(s, target_position)
            return self.last_Q

        return self.last_Q
