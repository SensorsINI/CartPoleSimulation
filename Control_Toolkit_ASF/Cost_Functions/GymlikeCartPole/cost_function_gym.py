from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Control_Toolkit.others.environment import EnvironmentBatched


class cost_function_gym(cost_function_base):
    """Uses as cost function the get_reward method of environment provided."""

    def __init__(self, env) -> None:
        self.env: EnvironmentBatched = env

    def get_terminal_cost(self, terminal_states: TensorType):
        return 0.0

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        return -self.env.get_reward(states, inputs)

    def get_trajectory_cost(self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None):
        return self.env.lib.sum(
            self.get_stage_cost(state_horizon[:, :-1, :], inputs, previous_input), 1
        ) + self.get_terminal_cost(state_horizon[:, -1, :])
