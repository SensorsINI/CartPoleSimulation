from SI_Toolkit.computation_library import ComputationLibrary, TensorType
from Control_Toolkit.others.globals_and_utils import get_logger

logger = get_logger(__name__)


class cost_function_base:
    def get_terminal_cost(self, s_hor: TensorType):
        raise NotImplementedError()

    def get_stage_cost(self, s: TensorType, u: TensorType, u_prev: TensorType):
        raise NotImplementedError()

    def get_trajectory_cost(
        self, s_hor: TensorType, u: TensorType, u_prev: TensorType = None
    ):
        raise NotImplementedError()

    def set_computation_library(self, ComputationLib: "type[ComputationLibrary]"):
        assert isinstance(ComputationLib, type)
        self.lib = ComputationLib
