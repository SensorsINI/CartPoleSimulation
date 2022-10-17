from Control_Toolkit.others.environment import EnvironmentBatched, TensorType


class cost_function_base:
    environment: EnvironmentBatched
    
    def get_terminal_cost(self, s_hor: TensorType):
        raise NotImplementedError()

    def get_stage_cost(self, s: TensorType, u: TensorType, u_prev: TensorType):
        raise NotImplementedError()

    def get_trajectory_cost(self, s_hor: TensorType, u: TensorType, u_prev: TensorType=None):
        raise NotImplementedError()
